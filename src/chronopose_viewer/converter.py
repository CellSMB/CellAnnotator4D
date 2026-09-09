from __future__ import annotations

import json
from collections import defaultdict, deque
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Literal

import numpy as np
from skimage.measure import label as connected_components
from skimage.morphology import skeletonize

from .io import load_tiff, save_project
from .model import DEFAULT_COLORS, GraphProject, InstanceGraph


class ConversionError(ValueError):
    """Raised when inference outputs cannot be converted without guessing."""


@dataclass(frozen=True, slots=True)
class InferenceBundle:
    image_path: Path
    masks_path: Path
    lineage_path: Path
    project_path: Path


@dataclass(frozen=True, slots=True)
class ConversionResult:
    bundle: InferenceBundle
    status: Literal["created", "skipped"]
    instances: int = 0
    nodes: int = 0
    edges: int = 0
    lineage_events: int = 0


@dataclass(frozen=True, slots=True)
class _Observation:
    id: str
    frame: int
    track_id: int
    voxel_count: int
    centroid_zyx: tuple[float, float, float]

    @property
    def coordinate_sum(self) -> tuple[int, int, int]:
        raw_sum = np.asarray(self.centroid_zyx) * self.voxel_count
        rounded = np.rint(raw_sum)
        if not np.allclose(raw_sum, rounded, rtol=0, atol=1e-4):
            raise ConversionError(
                f"Observation {self.id} centroid and voxel count do not describe "
                "integer voxel coordinates"
            )
        return tuple(int(value) for value in rounded)


@dataclass(frozen=True, slots=True)
class _MaskComponent:
    id: int
    display_label: int
    coordinates: np.ndarray

    @property
    def voxel_count(self) -> int:
        return len(self.coordinates)

    @property
    def coordinate_sum(self) -> tuple[int, int, int]:
        return tuple(int(value) for value in self.coordinates.sum(axis=0, dtype=np.int64))


@dataclass(frozen=True, slots=True)
class _TerminalBranch:
    length: float
    path: tuple[str, ...]
    removable_nodes: frozenset[str]


_TIFF_SUFFIXES = (".tif", ".tiff")
_NEIGHBOUR_OFFSETS = tuple(
    offset for offset in product((-1, 0, 1), repeat=3) if offset != (0, 0, 0)
)
DEFAULT_MIN_TERMINAL_BRANCH_LENGTH = 5.0


def discover_inference_bundles(source: str | Path) -> list[InferenceBundle]:
    """Find matching image, tracked-mask, and lineage files below *source*."""

    resolved = Path(source).expanduser().resolve()
    if not resolved.exists():
        raise ConversionError(f"Input does not exist: {resolved}")

    lineage_paths: list[Path]
    if resolved.is_dir():
        lineage_paths = sorted(resolved.rglob("*_lineage.json"))
    elif resolved.name.endswith("_lineage.json"):
        lineage_paths = [resolved]
    elif resolved.suffix.lower() in _TIFF_SUFFIXES:
        name = resolved.stem
        if name.endswith("_cp_masks"):
            name = name.removesuffix("_cp_masks")
        lineage = resolved.with_name(f"{name}_lineage.json")
        lineage_paths = [lineage] if lineage.is_file() else []
    elif resolved.name == "finalize_manifest.json":
        lineage_paths = sorted(resolved.parent.glob("*_lineage.json"))
    else:
        lineage_paths = []

    if not lineage_paths:
        raise ConversionError(
            "No *_lineage.json file was found. Pass an inference directory, final "
            "directory, output TIFF, mask TIFF, or lineage JSON."
        )

    bundles = [_bundle_for_lineage(path) for path in lineage_paths]
    project_paths = [bundle.project_path for bundle in bundles]
    if len(project_paths) != len(set(project_paths)):
        raise ConversionError("Multiple lineage files resolve to the same project path")
    return bundles


def _bundle_for_lineage(lineage_path: Path) -> InferenceBundle:
    if not lineage_path.is_file():
        raise ConversionError(f"Lineage JSON does not exist: {lineage_path}")
    try:
        payload = json.loads(lineage_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConversionError(f"Could not read lineage JSON {lineage_path}: {error}") from error
    if not isinstance(payload, dict):
        raise ConversionError(f"Lineage JSON must contain an object: {lineage_path}")

    base_name = lineage_path.name.removesuffix("_lineage.json")
    image_candidates: list[Path] = []
    output_image_name = payload.get("output_image_name")
    if output_image_name:
        image_candidates.append(lineage_path.parent / Path(str(output_image_name)).name)
    image_candidates.extend(
        lineage_path.with_name(f"{base_name}{suffix}") for suffix in _TIFF_SUFFIXES
    )
    image_path = _first_file(image_candidates)
    if image_path is None:
        expected = ", ".join(str(path) for path in _unique_paths(image_candidates))
        raise ConversionError(
            f"No source image TIFF matches {lineage_path.name}; checked: {expected}"
        )

    mask_candidates = [
        image_path.with_name(f"{image_path.stem}_cp_masks{suffix}") for suffix in _TIFF_SUFFIXES
    ]
    mask_candidates.extend(
        lineage_path.with_name(f"{base_name}_cp_masks{suffix}") for suffix in _TIFF_SUFFIXES
    )
    masks_path = _first_file(mask_candidates)
    if masks_path is None:
        expected = ", ".join(str(path) for path in _unique_paths(mask_candidates))
        raise ConversionError(
            f"No tracked mask TIFF matches {lineage_path.name}; checked: {expected}"
        )

    return InferenceBundle(
        image_path=image_path.resolve(),
        masks_path=masks_path.resolve(),
        lineage_path=lineage_path.resolve(),
        project_path=image_path.with_suffix(".cpv.json").resolve(),
    )


def _first_file(paths: list[Path]) -> Path | None:
    return next((path for path in _unique_paths(paths) if path.is_file()), None)


def _unique_paths(paths: list[Path]) -> list[Path]:
    return list(dict.fromkeys(paths))


def convert_inference_path(
    source: str | Path,
    *,
    voxel_size_zyx: tuple[float, float, float] = (1.0, 1.0, 1.0),
    node_spacing: float | None = None,
    min_terminal_branch_length: float = DEFAULT_MIN_TERMINAL_BRANCH_LENGTH,
) -> list[ConversionResult]:
    """Convert every inference bundle found at *source*, without replacing projects."""

    voxel_size = tuple(float(value) for value in voxel_size_zyx)
    if len(voxel_size) != 3 or any(not np.isfinite(value) or value <= 0 for value in voxel_size):
        raise ConversionError("Voxel size must contain three positive finite Z, Y, X values")
    spacing = _validate_node_spacing(node_spacing)
    minimum_branch_length = _validate_min_terminal_branch_length(min_terminal_branch_length)

    return [
        convert_inference_bundle(
            bundle,
            voxel_size_zyx=voxel_size,
            node_spacing=spacing,
            min_terminal_branch_length=minimum_branch_length,
        )
        for bundle in discover_inference_bundles(source)
    ]


def convert_inference_bundle(
    bundle: InferenceBundle,
    *,
    voxel_size_zyx: tuple[float, float, float] = (1.0, 1.0, 1.0),
    node_spacing: float | None = None,
    min_terminal_branch_length: float = DEFAULT_MIN_TERMINAL_BRANCH_LENGTH,
) -> ConversionResult:
    """Convert one matched inference output set to a viewer project."""

    if bundle.project_path.exists():
        return ConversionResult(bundle=bundle, status="skipped")

    project = build_project_from_inference(
        bundle,
        voxel_size_zyx=voxel_size_zyx,
        node_spacing=node_spacing,
        min_terminal_branch_length=min_terminal_branch_length,
    )
    try:
        save_project(project, bundle.project_path, overwrite=False)
    except FileExistsError:
        # Another process completed the same conversion while this one was extracting.
        return ConversionResult(bundle=bundle, status="skipped")

    graphs = [graph for instances in project.frames.values() for graph in instances.values()]
    return ConversionResult(
        bundle=bundle,
        status="created",
        instances=len(graphs),
        nodes=sum(len(graph.nodes) for graph in graphs),
        edges=sum(len(graph.edges) for graph in graphs),
        lineage_events=len(project.lineage_events),
    )


def build_project_from_inference(
    bundle: InferenceBundle,
    *,
    voxel_size_zyx: tuple[float, float, float] = (1.0, 1.0, 1.0),
    node_spacing: float | None = None,
    min_terminal_branch_length: float = DEFAULT_MIN_TERMINAL_BRANCH_LENGTH,
) -> GraphProject:
    spacing = _validate_node_spacing(node_spacing)
    minimum_branch_length = _validate_min_terminal_branch_length(min_terminal_branch_length)
    image = load_tiff(bundle.image_path, load_companion_masks=False)
    masks = load_tiff(bundle.masks_path, load_companion_masks=False)
    if image.shape_tzyx != masks.shape_tzyx:
        raise ConversionError(
            "Source image and tracked masks have different TZYX shapes: "
            f"{image.shape_tzyx} != {masks.shape_tzyx}"
        )
    if not np.issubdtype(masks.data.dtype, np.integer):
        raise ConversionError(f"Tracked masks must use an integer dtype, got {masks.data.dtype}")
    if np.any(masks.data < 0):
        raise ConversionError("Tracked masks cannot contain negative labels")

    try:
        lineage = json.loads(bundle.lineage_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConversionError(
            f"Could not read lineage JSON {bundle.lineage_path}: {error}"
        ) from error
    if not isinstance(lineage, dict):
        raise ConversionError("Lineage JSON must contain an object")

    observations, tracks_by_frame, mask_coordinates = _match_observations_to_masks(
        lineage,
        masks.data,
        image.shape_tzyx,
    )
    project = GraphProject(
        shape_tzyx=image.shape_tzyx,
        source_path=str(image.path),
        source_axes=image.source_axes,
        voxel_size_zyx=voxel_size_zyx,
    )
    for frame in range(project.timepoints):
        for track_id in sorted(tracks_by_frame[frame]):
            graph = project.add_instance(
                frame,
                instance_id=_instance_id(track_id),
                name=f"Track {track_id}",
                color=DEFAULT_COLORS[(track_id - 1) % len(DEFAULT_COLORS)],
            )
            _populate_centerline(
                graph,
                mask_coordinates[frame][track_id],
                node_spacing=spacing,
                min_terminal_branch_length=minimum_branch_length,
            )

    _add_lineage_events(project, lineage, observations, tracks_by_frame)
    project.validate()
    return project


def _validate_node_spacing(node_spacing: float | None) -> float | None:
    if node_spacing is None:
        return None
    spacing = float(node_spacing)
    if not np.isfinite(spacing) or spacing <= 0:
        raise ConversionError("Node spacing must be a positive finite voxel distance")
    return spacing


def _validate_min_terminal_branch_length(minimum_length: float) -> float:
    length = float(minimum_length)
    if not np.isfinite(length) or length < 0:
        raise ConversionError(
            "Minimum terminal branch length must be a non-negative finite voxel distance"
        )
    return length


def _match_observations_to_masks(
    lineage: dict[str, Any],
    masks: np.ndarray,
    shape_tzyx: tuple[int, int, int, int],
) -> tuple[
    dict[str, _Observation],
    list[dict[int, _Observation]],
    list[dict[int, tuple[np.ndarray, ...]]],
]:
    declared_shape = lineage.get("input_mask_shape")
    if declared_shape is not None:
        try:
            lineage_shape = tuple(int(value) for value in declared_shape)
        except (TypeError, ValueError) as error:
            raise ConversionError("input_mask_shape in lineage JSON is invalid") from error
        if lineage_shape != shape_tzyx:
            raise ConversionError(
                "Lineage JSON and TIFF data have different TZYX shapes: "
                f"{lineage_shape} != {shape_tzyx}"
            )
    if lineage.get("frame_indexing", "output") != "output":
        raise ConversionError("Only output-indexed lineage JSON is supported")

    raw_observations = lineage.get("observations")
    if not isinstance(raw_observations, list):
        raise ConversionError("Lineage JSON is missing its observations list")

    observations: dict[str, _Observation] = {}
    tracks_by_frame: list[dict[int, _Observation]] = [{} for _ in range(shape_tzyx[0])]
    for raw in raw_observations:
        if not isinstance(raw, dict):
            raise ConversionError("Every lineage observation must be an object")
        try:
            observation_id = str(raw["id"])
            frame = int(raw["frame"])
            track_id = int(raw["track_id"])
            voxel_count = int(raw["voxel_count"])
            centroid_values = tuple(float(value) for value in raw["centroid_zyx"])
            if len(centroid_values) != 3:
                raise ValueError("centroid_zyx must contain exactly three values")
            centroid_zyx: tuple[float, float, float] = centroid_values
            if not all(np.isfinite(value) for value in centroid_zyx):
                raise ValueError("centroid_zyx values must be finite")
            if voxel_count <= 0:
                raise ValueError("voxel_count must be positive")
        except (KeyError, TypeError, ValueError) as error:
            raise ConversionError(f"Invalid lineage observation: {raw!r}") from error
        if observation_id in observations:
            raise ConversionError(f"Duplicate lineage observation ID: {observation_id}")
        if not 0 <= frame < shape_tzyx[0]:
            raise ConversionError(f"Observation {observation_id} has out-of-range frame {frame}")
        if track_id <= 0:
            raise ConversionError(
                f"Observation {observation_id} has non-positive track ID {track_id}"
            )
        if track_id in tracks_by_frame[frame]:
            raise ConversionError(
                f"Track {track_id} has more than one observation at frame {frame}"
            )
        observation = _Observation(
            observation_id,
            frame,
            track_id,
            voxel_count,
            centroid_zyx,
        )
        # Check that count × centroid reproduces integer voxel-coordinate sums.
        observation.coordinate_sum
        observations[observation_id] = observation
        tracks_by_frame[frame][track_id] = observation

    coordinates_by_frame: list[dict[int, tuple[np.ndarray, ...]]] = []
    for frame, expected_tracks in enumerate(tracks_by_frame):
        matched = _solve_component_assignment(
            frame,
            list(expected_tracks.values()),
            _mask_components(masks[frame]),
        )
        coordinates_by_frame.append(
            {
                observation.track_id: tuple(
                    component.coordinates for component in matched[observation.id]
                )
                for observation in expected_tracks.values()
            }
        )
    return observations, tracks_by_frame, coordinates_by_frame


def _mask_components(frame_masks: np.ndarray) -> list[_MaskComponent]:
    """Split N-colour display labels while retaining each label's disconnected pieces."""

    labelled = connected_components(
        frame_masks,
        background=0,
        connectivity=frame_masks.ndim,
    )
    foreground_coordinates = np.argwhere(labelled)
    if not len(foreground_coordinates):
        return []
    component_ids = labelled[tuple(foreground_coordinates.T)]
    order = np.argsort(component_ids, kind="stable")
    foreground_coordinates = foreground_coordinates[order]
    component_ids = component_ids[order]
    split_points = np.flatnonzero(np.diff(component_ids)) + 1
    coordinate_groups = np.split(foreground_coordinates, split_points)

    components: list[_MaskComponent] = []
    for component_id, coordinates in zip(
        np.unique(component_ids),
        coordinate_groups,
        strict=True,
    ):
        first = tuple(int(value) for value in coordinates[0])
        components.append(
            _MaskComponent(
                id=int(component_id),
                display_label=int(frame_masks[first]),
                coordinates=coordinates,
            )
        )
    return components


def _solve_component_assignment(
    frame: int,
    observations: list[_Observation],
    components: list[_MaskComponent],
) -> dict[str, tuple[_MaskComponent, ...]]:
    if sum(item.voxel_count for item in observations) != sum(
        item.voxel_count for item in components
    ):
        raise ConversionError(
            f"Frame {frame} foreground voxel count does not match the lineage JSON"
        )
    if not observations and not components:
        return {}
    if not observations or not components:
        raise ConversionError(
            f"Frame {frame} mask components do not match its lineage observations"
        )

    solutions: list[dict[str, tuple[_MaskComponent, ...]]] = []

    def search(
        remaining_observations: tuple[_Observation, ...],
        remaining_components: tuple[_MaskComponent, ...],
        assignments: dict[str, tuple[_MaskComponent, ...]],
    ) -> None:
        if len(solutions) >= 2:
            return
        if not remaining_observations:
            if not remaining_components:
                solutions.append(dict(assignments))
            return

        candidates_by_observation = [
            (
                observation,
                _candidate_component_subsets(observation, remaining_components),
            )
            for observation in remaining_observations
        ]
        observation, candidates = min(
            candidates_by_observation,
            key=lambda item: len(item[1]),
        )
        if not candidates:
            return
        later_observations = tuple(
            item for item in remaining_observations if item.id != observation.id
        )
        for candidate in candidates:
            selected_ids = {item.id for item in candidate}
            assignments[observation.id] = candidate
            search(
                later_observations,
                tuple(item for item in remaining_components if item.id not in selected_ids),
                assignments,
            )
            del assignments[observation.id]
            if len(solutions) >= 2:
                return

    search(tuple(observations), tuple(components), {})
    if not solutions:
        raise ConversionError(
            f"Frame {frame} mask shapes cannot be matched to the lineage centroids and voxel counts"
        )
    if len(solutions) > 1:
        raise ConversionError(f"Frame {frame} has an ambiguous mask-to-lineage component mapping")
    return solutions[0]


def _candidate_component_subsets(
    observation: _Observation,
    components: tuple[_MaskComponent, ...],
) -> list[tuple[_MaskComponent, ...]]:
    direct = [
        (component,)
        for component in components
        if component.voxel_count == observation.voxel_count
        and component.coordinate_sum == observation.coordinate_sum
    ]
    if direct:
        return direct

    candidates: list[tuple[_MaskComponent, ...]] = []
    by_display_label: dict[int, list[_MaskComponent]] = defaultdict(list)
    for component in components:
        if component.voxel_count <= observation.voxel_count:
            by_display_label[component.display_label].append(component)
    for label_components in by_display_label.values():
        candidates.extend(_matching_subsets(observation, label_components))
    return candidates


def _matching_subsets(
    observation: _Observation,
    components: list[_MaskComponent],
) -> list[tuple[_MaskComponent, ...]]:
    target_count = observation.voxel_count
    target_sum = np.asarray(observation.coordinate_sum, dtype=np.int64)
    ordered = sorted(
        components,
        key=lambda item: (-item.voxel_count, item.id),
    )
    results: list[tuple[_MaskComponent, ...]] = []

    def search(
        start: int,
        chosen: tuple[_MaskComponent, ...],
        voxel_count: int,
        coordinate_sum: np.ndarray,
    ) -> None:
        if voxel_count == target_count:
            if np.array_equal(coordinate_sum, target_sum):
                results.append(chosen)
            return
        for index in range(start, len(ordered)):
            component = ordered[index]
            new_count = voxel_count + component.voxel_count
            if new_count > target_count:
                continue
            new_sum = coordinate_sum + np.asarray(
                component.coordinate_sum,
                dtype=np.int64,
            )
            if np.any(new_sum > target_sum):
                continue
            search(
                index + 1,
                chosen + (component,),
                new_count,
                new_sum,
            )

    search(0, (), 0, np.zeros(3, dtype=np.int64))
    return results


def _populate_centerline(
    graph: InstanceGraph,
    mask_components: tuple[np.ndarray, ...],
    *,
    node_spacing: float | None = None,
    min_terminal_branch_length: float = DEFAULT_MIN_TERMINAL_BRANCH_LENGTH,
) -> None:
    if not mask_components:
        raise ConversionError(f"Cannot extract an empty mask for {graph.name}")

    for mask_coordinates in mask_components:
        nodes_before = set(graph.nodes)
        lower = mask_coordinates.min(axis=0)
        upper = mask_coordinates.max(axis=0) + 1
        cropped = np.zeros(
            tuple(int(stop - start) for start, stop in zip(lower, upper, strict=True)),
            dtype=bool,
        )
        local_coordinates = mask_coordinates - lower
        cropped[tuple(local_coordinates.T)] = True
        skeleton = skeletonize(cropped, method="lee")
        skeleton_coordinates = np.argwhere(skeleton) + lower

        if not len(skeleton_coordinates):
            _populate_pca_fallback(graph, mask_coordinates)
        else:
            _populate_skeleton_graph(graph, skeleton_coordinates)
        created_nodes = set(graph.nodes) - nodes_before
        if len(mask_coordinates) > 1 and len(created_nodes) < 2:
            for node_id in created_nodes:
                graph.remove_node(node_id)
            _populate_pca_fallback(graph, mask_coordinates)
    _prune_short_terminal_branches(graph, min_terminal_branch_length)
    if node_spacing is None:
        _dissolve_collinear_nodes(graph)
    else:
        _resample_degree_two_branches(graph, node_spacing)


def _populate_pca_fallback(graph: InstanceGraph, coordinates: np.ndarray) -> None:
    """Represent masks that Lee thinning entirely removes with their principal axis."""

    centre = coordinates.mean(axis=0, dtype=np.float64)
    if len(coordinates) == 1:
        graph.add_node(centre, node_id=_next_node_id(graph))
        return
    centred = coordinates.astype(np.float64) - centre
    covariance = centred.T @ centred
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    projections = centred @ axis
    first_position = centre + float(projections.min()) * axis
    last_position = centre + float(projections.max()) * axis
    first = graph.add_node(first_position, node_id=_next_node_id(graph))
    if np.linalg.norm(last_position - first_position) < 1e-8:
        return
    last = graph.add_node(last_position, node_id=_next_node_id(graph))
    graph.add_edge(first, last)


def _populate_skeleton_graph(graph: InstanceGraph, coordinates: np.ndarray) -> None:
    integer_coordinates = [tuple(int(value) for value in row) for row in coordinates]
    coordinate_index = {coordinate: index for index, coordinate in enumerate(integer_coordinates)}
    adjacency = [set() for _ in integer_coordinates]
    for index, coordinate in enumerate(integer_coordinates):
        for offset in _NEIGHBOUR_OFFSETS:
            neighbour = tuple(coordinate[axis] + offset[axis] for axis in range(3))
            neighbour_index = coordinate_index.get(neighbour)
            if neighbour_index is not None and neighbour_index > index:
                adjacency[index].add(neighbour_index)
                adjacency[neighbour_index].add(index)

    junction_voxels = {index for index, neighbours in enumerate(adjacency) if len(neighbours) > 2}
    junction_components = _induced_components(junction_voxels, adjacency)
    token_for_voxel: dict[int, int] = {}
    token_positions: dict[int, np.ndarray] = {}
    next_token = 0
    for component in junction_components:
        token = next_token
        next_token += 1
        for voxel in component:
            token_for_voxel[voxel] = token
        token_positions[token] = coordinates[sorted(component)].mean(axis=0)
    for voxel in range(len(integer_coordinates)):
        if voxel in token_for_voxel:
            continue
        token = next_token
        next_token += 1
        token_for_voxel[voxel] = token
        token_positions[token] = coordinates[voxel].astype(np.float64)

    contracted_edges: set[tuple[int, int]] = set()
    for first, neighbours in enumerate(adjacency):
        for second in neighbours:
            first_token = token_for_voxel[first]
            second_token = token_for_voxel[second]
            if first_token == second_token:
                continue
            contracted_edges.add(
                (first_token, second_token)
                if first_token < second_token
                else (second_token, first_token)
            )

    node_for_token: dict[int, str] = {}
    for ordinal, token in enumerate(
        sorted(token_positions, key=lambda item: tuple(token_positions[item])),
        start=len(graph.nodes) + 1,
    ):
        node_for_token[token] = graph.add_node(
            token_positions[token],
            node_id=f"n{ordinal:06d}",
        )
    for first, second in sorted(contracted_edges):
        graph.add_edge(node_for_token[first], node_for_token[second])


def _next_node_id(graph: InstanceGraph) -> str:
    return f"n{len(graph.nodes) + 1:06d}"


def _induced_components(
    vertices: set[int],
    adjacency: list[set[int]],
) -> list[set[int]]:
    unseen = set(vertices)
    components: list[set[int]] = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        component = {start}
        queue = deque([start])
        while queue:
            current = queue.popleft()
            neighbours = adjacency[current] & unseen
            unseen.difference_update(neighbours)
            component.update(neighbours)
            queue.extend(sorted(neighbours))
        components.append(component)
    return components


def _trace_degree_two_paths(
    graph: InstanceGraph,
) -> tuple[dict[str, set[str]], list[list[str]], list[list[str]]]:
    adjacency = {node_id: graph.neighbours(node_id) for node_id in graph.nodes}
    anchors = {node_id for node_id, neighbours in adjacency.items() if len(neighbours) != 2}
    visited_edges: set[tuple[str, str]] = set()
    branches: list[list[str]] = []

    for start in sorted(anchors):
        for neighbour in sorted(adjacency[start]):
            edge = _edge_key(start, neighbour)
            if edge in visited_edges:
                continue
            visited_edges.add(edge)
            path = [start]
            previous = start
            current = neighbour
            while True:
                path.append(current)
                if current in anchors:
                    break
                next_nodes = adjacency[current] - {previous}
                if len(next_nodes) != 1:
                    raise ConversionError(f"Could not trace a degree-2 branch in {graph.name}")
                next_node = next(iter(next_nodes))
                next_edge = _edge_key(current, next_node)
                if next_edge in visited_edges:
                    raise ConversionError(
                        f"Encountered an inconsistent branch cycle in {graph.name}"
                    )
                visited_edges.add(next_edge)
                previous, current = current, next_node
            branches.append(path)

    cycles: list[list[str]] = []
    for first, second in sorted(graph.edges):
        edge = _edge_key(first, second)
        if edge in visited_edges:
            continue
        visited_edges.add(edge)
        cycle = [first, second]
        previous = first
        current = second
        while True:
            next_nodes = adjacency[current] - {previous}
            if len(next_nodes) != 1:
                raise ConversionError(f"Could not trace a closed branch in {graph.name}")
            next_node = next(iter(next_nodes))
            next_edge = _edge_key(current, next_node)
            if next_node == cycle[0]:
                visited_edges.add(next_edge)
                break
            if next_edge in visited_edges:
                raise ConversionError(f"Encountered an inconsistent closed branch in {graph.name}")
            visited_edges.add(next_edge)
            cycle.append(next_node)
            previous, current = current, next_node
        cycles.append(cycle)

    if visited_edges != set(graph.edges):
        raise ConversionError(f"Could not trace every centreline edge in {graph.name}")
    return adjacency, branches, cycles


def _prune_short_terminal_branches(
    graph: InstanceGraph,
    minimum_length: float,
) -> None:
    """Remove short leaf-to-junction paths without collapsing graph components."""

    if minimum_length <= 0 or len(graph.nodes) < 2 or not graph.edges:
        return

    while True:
        adjacency, branches, _cycles = _trace_degree_two_paths(graph)
        components = graph.connected_components()
        component_for_node = {
            node_id: index for index, component in enumerate(components) for node_id in component
        }
        candidates_by_component: list[list[_TerminalBranch]] = [[] for _component in components]

        for path in branches:
            if path[0] == path[-1]:
                continue
            first_degree = len(adjacency[path[0]])
            last_degree = len(adjacency[path[-1]])
            if first_degree == 1 and last_degree >= 3:
                removable_nodes = path[:-1]
            elif last_degree == 1 and first_degree >= 3:
                removable_nodes = path[1:]
            else:
                # In particular, preserve every leaf-to-leaf component and every
                # path whose endpoints are both topology-changing junctions.
                continue

            positions = np.asarray(
                [graph.nodes[node_id].position for node_id in path],
                dtype=np.float64,
            )
            length = _polyline_length(positions)
            if length >= minimum_length:
                continue
            candidate = _TerminalBranch(
                length=length,
                path=tuple(path),
                removable_nodes=frozenset(removable_nodes),
            )
            candidates_by_component[component_for_node[path[0]]].append(candidate)

        if not any(candidates_by_component):
            return

        nodes_to_remove: set[str] = set()
        for component, candidates in zip(
            components,
            candidates_by_component,
            strict=True,
        ):
            proposed = set().union(*(candidate.removable_nodes for candidate in candidates))
            remaining = component - proposed
            if not _nodes_contain_graph_edge(graph, remaining):
                # If every arm of a tiny branched component is below the cutoff,
                # retain its longest arm. This guarantees at least two nodes and
                # one edge without inventing geometry or touching leaf-to-leaf paths.
                for candidate in sorted(
                    candidates,
                    key=lambda item: (-item.length, item.path),
                ):
                    proposed.difference_update(candidate.removable_nodes)
                    remaining = component - proposed
                    if _nodes_contain_graph_edge(graph, remaining):
                        break
            nodes_to_remove.update(proposed)

        if not nodes_to_remove:
            return
        for node_id in sorted(nodes_to_remove):
            graph.remove_node(node_id)
        graph.validate()


def _nodes_contain_graph_edge(graph: InstanceGraph, node_ids: set[str]) -> bool:
    return any(first in node_ids and second in node_ids for first, second in graph.edges)


def _resample_degree_two_branches(
    graph: InstanceGraph,
    spacing: float,
) -> None:
    """Resample branch polylines while retaining every topology-changing node."""

    if len(graph.nodes) < 2 or not graph.edges:
        return

    adjacency, branches, cycles = _trace_degree_two_paths(graph)
    anchors = {node_id for node_id, neighbours in adjacency.items() if len(neighbours) != 2}

    original_nodes = dict(graph.nodes)
    graph.nodes = {node_id: original_nodes[node_id] for node_id in sorted(anchors)}
    graph.edges = set()
    next_ordinal = 1

    def add_sampled_node(position: np.ndarray) -> str:
        nonlocal next_ordinal
        while f"n{next_ordinal:06d}" in graph.nodes:
            next_ordinal += 1
        node_id = f"n{next_ordinal:06d}"
        next_ordinal += 1
        return graph.add_node(position, node_id=node_id)

    for path in branches:
        positions = np.asarray(
            [original_nodes[node_id].position for node_id in path],
            dtype=np.float64,
        )
        total_length = _polyline_length(positions)
        first_id = path[0]
        last_id = path[-1]
        minimum_segments = 3 if first_id == last_id else 1
        segment_count = _target_segment_count(
            total_length,
            spacing,
            minimum=minimum_segments,
        )
        if (
            first_id != last_id
            and segment_count == 1
            and _edge_key(first_id, last_id) in graph.edges
        ):
            segment_count = 2
        sample_distances = [
            total_length * index / segment_count for index in range(1, segment_count)
        ]

        sampled_positions = _positions_along_polyline(
            positions,
            sample_distances,
        )
        node_ids = [
            first_id,
            *(add_sampled_node(position) for position in sampled_positions),
            last_id,
        ]
        for first_node, second_node in zip(node_ids, node_ids[1:]):
            if first_node != second_node:
                graph.add_edge(first_node, second_node)

    for cycle in cycles:
        positions = np.asarray(
            [original_nodes[node_id].position for node_id in cycle],
            dtype=np.float64,
        )
        closed_positions = np.vstack((positions, positions[0]))
        total_length = _polyline_length(closed_positions)
        sample_count = _target_segment_count(
            total_length,
            spacing,
            minimum=3,
        )
        if total_length > 0:
            sample_distances = (
                np.arange(sample_count, dtype=np.float64) * total_length / sample_count
            )
            sampled_positions = _positions_along_polyline(
                closed_positions,
                sample_distances,
            )
        else:
            sampled_positions = [positions[0]] * sample_count
        node_ids = [add_sampled_node(position) for position in sampled_positions]
        for first_node, second_node in zip(
            node_ids,
            node_ids[1:] + node_ids[:1],
            strict=True,
        ):
            graph.add_edge(first_node, second_node)

    graph.validate()


def _edge_key(first: str, second: str) -> tuple[str, str]:
    return (first, second) if first < second else (second, first)


def _target_segment_count(
    length: float,
    spacing: float,
    *,
    minimum: int,
) -> int:
    if length <= 0:
        return minimum
    ratio = length / spacing
    candidates = {
        minimum,
        max(minimum, int(np.floor(ratio))),
        max(minimum, int(np.ceil(ratio))),
    }
    return min(
        candidates,
        key=lambda count: (abs(length / count - spacing), count),
    )


def _polyline_length(positions: np.ndarray) -> float:
    if len(positions) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())


def _positions_along_polyline(
    positions: np.ndarray,
    distances: list[float] | np.ndarray,
) -> list[np.ndarray]:
    if not len(distances):
        return []
    if len(positions) < 2:
        return [positions[0].copy() for _ in distances]

    segment_vectors = np.diff(positions, axis=0)
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total_length = float(cumulative[-1])
    if total_length == 0:
        return [positions[0].copy() for _ in distances]

    sampled: list[np.ndarray] = []
    for raw_distance in distances:
        distance = min(max(float(raw_distance), 0.0), total_length)
        segment = int(np.searchsorted(cumulative, distance, side="right") - 1)
        segment = min(segment, len(segment_lengths) - 1)
        while segment < len(segment_lengths) - 1 and segment_lengths[segment] == 0:
            segment += 1
        if segment_lengths[segment] == 0:
            sampled.append(positions[segment].copy())
            continue
        fraction = (distance - cumulative[segment]) / segment_lengths[segment]
        sampled.append(positions[segment] + fraction * segment_vectors[segment])
    return sampled


def _dissolve_collinear_nodes(graph: InstanceGraph) -> None:
    changed = True
    while changed:
        changed = False
        for node_id in sorted(graph.nodes):
            neighbours = sorted(graph.neighbours(node_id))
            if len(neighbours) != 2:
                continue
            centre = np.asarray(graph.nodes[node_id].position)
            first = np.asarray(graph.nodes[neighbours[0]].position) - centre
            second = np.asarray(graph.nodes[neighbours[1]].position) - centre
            scale = np.linalg.norm(first) * np.linalg.norm(second)
            if scale == 0:
                continue
            if (
                np.linalg.norm(np.cross(first, second)) <= 1e-8 * scale
                and np.dot(first, second) < 0
            ):
                graph.dissolve_degree_two_node(node_id)
                changed = True
                break


def _add_lineage_events(
    project: GraphProject,
    lineage: dict[str, Any],
    observations: dict[str, _Observation],
    tracks_by_frame: list[dict[int, _Observation]],
) -> None:
    explicit_events = _explicit_event_signatures(lineage, observations)
    used_explicit_ids: set[str] = set()

    links = lineage.get("links", [])
    if not isinstance(links, list):
        raise ConversionError("Lineage JSON links must be a list")
    links_by_transition: dict[tuple[int, int], list[tuple[str, str]]] = defaultdict(list)
    for raw_link in links:
        if not isinstance(raw_link, dict):
            raise ConversionError("Every lineage link must be an object")
        try:
            previous_id = str(raw_link["prev_observation"])
            next_id = str(raw_link["next_observation"])
            previous = observations[previous_id]
            next_observation = observations[next_id]
        except (KeyError, TypeError) as error:
            raise ConversionError(f"Invalid lineage link: {raw_link!r}") from error
        if previous.frame >= next_observation.frame:
            raise ConversionError(
                f"Lineage link {previous_id} -> {next_id} does not move forward in time"
            )
        links_by_transition[(previous.frame, next_observation.frame)].append((previous_id, next_id))

    generated_index = 0
    for (source_time, target_time), transition_links in sorted(links_by_transition.items()):
        for source_ids, target_ids in _link_components(transition_links):
            signature = (frozenset(source_ids), frozenset(target_ids))
            explicit_id = explicit_events.get(signature)
            if explicit_id is not None:
                event_id = explicit_id
                used_explicit_ids.add(explicit_id)
            else:
                generated_index += 1
                event_id = f"link_t{source_time:06d}_t{target_time:06d}_{generated_index:06d}"
            project.add_lineage_event(
                source_time=source_time,
                target_time=target_time,
                sources=(
                    _instance_id(observations[observation_id].track_id)
                    for observation_id in sorted(source_ids)
                ),
                targets=(
                    _instance_id(observations[observation_id].track_id)
                    for observation_id in sorted(target_ids)
                ),
                event_id=event_id,
            )

    for signature, event_id in explicit_events.items():
        if event_id in used_explicit_ids:
            continue
        source_ids, target_ids = signature
        source_times = {observations[item].frame for item in source_ids}
        target_times = {observations[item].frame for item in target_ids}
        source_time = next(iter(source_times))
        target_time = next(iter(target_times))
        try:
            project.add_lineage_event(
                source_time=source_time,
                target_time=target_time,
                sources=(_instance_id(observations[item].track_id) for item in sorted(source_ids)),
                targets=(_instance_id(observations[item].track_id) for item in sorted(target_ids)),
                event_id=event_id,
            )
        except ValueError as error:
            raise ConversionError(
                f"Explicit lineage event {event_id} contradicts the JSON links: {error}"
            ) from error

    # Track IDs themselves encode continuity, so recover a missing simple link when an
    # older lineage JSON omits its detailed links list.
    for frame in range(project.timepoints - 1):
        shared_tracks = sorted(set(tracks_by_frame[frame]) & set(tracks_by_frame[frame + 1]))
        for track_id in shared_tracks:
            instance_id = _instance_id(track_id)
            if (
                project.outgoing_event(frame, instance_id) is None
                and project.incoming_event(frame + 1, instance_id) is None
            ):
                project.add_lineage_event(
                    source_time=frame,
                    target_time=frame + 1,
                    sources=(instance_id,),
                    targets=(instance_id,),
                    event_id=f"continuation_t{frame:06d}_{instance_id}",
                )

    # Complete the representation with roots and leaves derived from absent links.
    for frame in range(project.timepoints):
        for track_id in sorted(tracks_by_frame[frame]):
            instance_id = _instance_id(track_id)
            if project.incoming_event(frame, instance_id) is None:
                project.add_lineage_event(
                    source_time=None,
                    target_time=frame,
                    targets=(instance_id,),
                    event_id=f"start_t{frame:06d}_{instance_id}",
                )
            if project.outgoing_event(frame, instance_id) is None:
                project.add_lineage_event(
                    source_time=frame,
                    target_time=None,
                    sources=(instance_id,),
                    event_id=f"end_t{frame:06d}_{instance_id}",
                )


def _explicit_event_signatures(
    lineage: dict[str, Any],
    observations: dict[str, _Observation],
) -> dict[tuple[frozenset[str], frozenset[str]], str]:
    raw_events = lineage.get("events", [])
    if not isinstance(raw_events, list):
        raise ConversionError("Lineage JSON events must be a list")
    signatures: dict[tuple[frozenset[str], frozenset[str]], str] = {}
    used_ids: set[str] = set()
    for raw_event in raw_events:
        if not isinstance(raw_event, dict):
            raise ConversionError("Every lineage event must be an object")
        try:
            event_id = str(raw_event["id"])
            source_ids = frozenset(str(value) for value in raw_event["parent_observations"])
            target_ids = frozenset(str(value) for value in raw_event["child_observations"])
        except (KeyError, TypeError) as error:
            raise ConversionError(f"Invalid explicit lineage event: {raw_event!r}") from error
        if not source_ids or not target_ids:
            raise ConversionError(f"Explicit lineage event {event_id} has an empty side")
        missing = (source_ids | target_ids) - set(observations)
        if missing:
            raise ConversionError(
                f"Explicit lineage event {event_id} references unknown observations: "
                f"{sorted(missing)}"
            )
        source_times = {observations[item].frame for item in source_ids}
        target_times = {observations[item].frame for item in target_ids}
        if len(source_times) != 1 or len(target_times) != 1:
            raise ConversionError(
                f"Explicit lineage event {event_id} spans multiple source or target frames"
            )
        if next(iter(source_times)) >= next(iter(target_times)):
            raise ConversionError(f"Explicit lineage event {event_id} does not move forward")
        signature = (source_ids, target_ids)
        if signature in signatures or event_id in used_ids:
            raise ConversionError(f"Duplicate explicit lineage event: {event_id}")
        signatures[signature] = event_id
        used_ids.add(event_id)
    return signatures


def _link_components(
    links: list[tuple[str, str]],
) -> list[tuple[set[str], set[str]]]:
    adjacency: dict[tuple[Literal["source", "target"], str], set[tuple[str, str]]] = defaultdict(
        set
    )
    for source_id, target_id in links:
        source_node = ("source", source_id)
        target_node = ("target", target_id)
        adjacency[source_node].add(target_node)
        adjacency[target_node].add(source_node)

    unseen = set(adjacency)
    components: list[tuple[set[str], set[str]]] = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        queue = deque([start])
        source_ids: set[str] = set()
        target_ids: set[str] = set()
        while queue:
            kind, observation_id = queue.popleft()
            (source_ids if kind == "source" else target_ids).add(observation_id)
            neighbours = adjacency[(kind, observation_id)] & unseen
            unseen.difference_update(neighbours)
            queue.extend(sorted(neighbours))
        components.append((source_ids, target_ids))
    return sorted(
        components,
        key=lambda component: (sorted(component[0]), sorted(component[1])),
    )


def _instance_id(track_id: int) -> str:
    return f"track_{track_id:06d}"
