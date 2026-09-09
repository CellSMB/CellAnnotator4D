from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.ndimage import (
    distance_transform_edt,
    find_objects,
    gaussian_filter1d,
    map_coordinates,
)
from skimage.measure import label as connected_components

from .io import MASK_NAME_SUFFIXES, discover_companion_mask, load_mask_tiff, save_project
from .model import GraphProject, InstanceGraph


@dataclass(frozen=True, slots=True)
class MaskRadiusSummary:
    nodes_updated: int
    nodes_unmatched: int
    nodes_skipped: int


@dataclass(frozen=True, slots=True)
class MaskRadiusProjectResult:
    project_path: Path
    nodes_updated: int
    nodes_unmatched: int
    nodes_skipped: int


@dataclass(slots=True)
class _ComponentDistance:
    lower_zyx: np.ndarray
    distance: np.ndarray


def apply_mask_radii(
    project: GraphProject,
    masks_tzyx: np.ndarray,
    *,
    scale: float = 1.0,
    offset: float = 0.0,
    max_search_distance: float = 3.0,
    only_zero: bool = False,
) -> MaskRadiusSummary:
    """Set node radii from each mask component's Euclidean distance transform.

    Radii use voxel-coordinate units, matching graph positions and the viewer's
    swept-sphere overlay. Nodes that land just outside their source component
    can be associated with a nearby component crossed by the same graph.
    """

    masks = np.asarray(masks_tzyx)
    if masks.shape != project.shape_tzyx:
        raise ValueError(
            f"Mask shape does not match the project: {masks.shape} != {project.shape_tzyx}"
        )
    if not np.issubdtype(masks.dtype, np.integer):
        raise ValueError(f"Instance masks must use an integer dtype, got {masks.dtype}")
    if np.any(masks < 0):
        raise ValueError("Instance masks cannot contain negative labels")
    scale = float(scale)
    offset = float(offset)
    max_search_distance = float(max_search_distance)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Radius scale must be positive and finite")
    if not np.isfinite(offset):
        raise ValueError("Radius offset must be finite")
    if not np.isfinite(max_search_distance) or max_search_distance < 0:
        raise ValueError("Maximum component search distance must be non-negative and finite")

    updated = 0
    unmatched = 0
    skipped = 0
    for time in range(project.timepoints):
        graphs = project.instances_at(time)
        if not graphs:
            continue
        labels, count = connected_components(
            masks[time],
            background=0,
            connectivity=3,
            return_num=True,
        )
        labels = np.asarray(labels)
        component_slices = find_objects(labels, max_label=int(count))
        distance_cache: dict[int, _ComponentDistance] = {}

        for graph in graphs.values():
            candidate_components = _graph_component_ids(graph, labels)
            for node in graph.nodes.values():
                if only_zero and node.radius > 0:
                    skipped += 1
                    continue
                component_id = _component_at_position(labels, node.position)
                if component_id == 0:
                    component_id = _nearest_component(
                        labels,
                        node.position,
                        candidate_components,
                        max_search_distance,
                    )
                if component_id == 0:
                    graph.set_node_radius(node.id, 0.0)
                    unmatched += 1
                    continue

                component_distance = distance_cache.get(component_id)
                if component_distance is None:
                    component_slice = component_slices[component_id - 1]
                    if component_slice is None:
                        graph.set_node_radius(node.id, 0.0)
                        unmatched += 1
                        continue
                    component_distance = _build_component_distance(
                        labels,
                        component_id,
                        component_slice,
                    )
                    distance_cache[component_id] = component_distance
                raw_radius = _sample_component_distance(component_distance, node.position)
                graph.set_node_radius(node.id, max(0.0, (raw_radius + offset) * scale))
                updated += 1

    project.validate()
    return MaskRadiusSummary(
        nodes_updated=updated,
        nodes_unmatched=unmatched,
        nodes_skipped=skipped,
    )


def estimate_intensity_radii(
    volume_zyx: np.ndarray,
    graphs: dict[str, InstanceGraph],
    *,
    maximum_radius: float = 15.0,
    threshold_fraction: float = 0.5,
    ray_percentile: float = 10.0,
    smoothing_sigma: float = 0.75,
    ray_count: int = 64,
    sample_step: float = 0.5,
    minimum_contrast_fraction: float = 0.03,
) -> dict[tuple[str, str], float]:
    """Estimate bright-tube radii from radial intensity falloff around each node.

    The result is not applied to the graphs. This lets the GUI turn one rerun
    into a single undoable edit.
    """

    volume = np.asarray(volume_zyx)
    if volume.ndim != 3:
        raise ValueError(f"Expected a ZYX intensity volume, got shape {volume.shape}")
    maximum_radius = float(maximum_radius)
    threshold_fraction = float(threshold_fraction)
    ray_percentile = float(ray_percentile)
    smoothing_sigma = float(smoothing_sigma)
    sample_step = float(sample_step)
    minimum_contrast_fraction = float(minimum_contrast_fraction)
    ray_count = int(ray_count)
    if not np.isfinite(maximum_radius) or maximum_radius <= 0:
        raise ValueError("Maximum radius must be positive and finite")
    if not 0 < threshold_fraction < 1:
        raise ValueError("Intensity threshold fraction must lie strictly between 0 and 1")
    if not 0 < ray_percentile <= 100:
        raise ValueError("Ray percentile must lie in (0, 100]")
    if not np.isfinite(smoothing_sigma) or smoothing_sigma < 0:
        raise ValueError("Smoothing sigma must be non-negative and finite")
    if not np.isfinite(sample_step) or sample_step <= 0:
        raise ValueError("Radial sample step must be positive and finite")
    if not np.isfinite(minimum_contrast_fraction) or minimum_contrast_fraction < 0:
        raise ValueError("Minimum contrast fraction must be non-negative and finite")
    if ray_count < 8:
        raise ValueError("At least eight radial rays are required")

    finite_sample = _finite_sample(volume)
    if not finite_sample.size:
        return {
            (graph.id, node.id): 0.0 for graph in graphs.values() for node in graph.nodes.values()
        }
    low, high = (float(value) for value in np.percentile(finite_sample, (5.0, 99.0)))
    global_background = float(np.percentile(finite_sample, 20.0))
    intensity_span = max(high - low, np.finfo(np.float32).eps)
    minimum_contrast = minimum_contrast_fraction * intensity_span
    distances = np.arange(
        0.0,
        maximum_radius + sample_step * 0.5,
        sample_step,
        dtype=np.float32,
    )
    if distances[-1] < maximum_radius:
        distances = np.append(distances, np.float32(maximum_radius))
    directions = _fibonacci_sphere(ray_count)
    estimates: dict[tuple[str, str], float] = {}

    for graph in graphs.values():
        for node in graph.nodes.values():
            profiles = _radial_profiles(
                volume,
                node.position,
                directions,
                distances,
                cval=global_background,
            )
            inner = profiles[:, distances <= min(1.0, maximum_radius)]
            outer = profiles[:, distances >= maximum_radius * 0.8]
            foreground = float(np.percentile(inner, 80.0))
            background = float(np.percentile(outer, 25.0))
            if not np.isfinite(foreground) or foreground - background < minimum_contrast:
                estimates[(graph.id, node.id)] = 0.0
                continue

            threshold = background + threshold_fraction * (foreground - background)
            if smoothing_sigma > 0:
                profiles = gaussian_filter1d(
                    profiles,
                    sigma=smoothing_sigma / sample_step,
                    axis=1,
                    mode="nearest",
                )
            crossings = np.asarray(
                [_first_threshold_crossing(profile, distances, threshold) for profile in profiles],
                dtype=np.float32,
            )
            estimate = float(np.percentile(crossings, ray_percentile))
            estimates[(graph.id, node.id)] = min(max(estimate, 0.0), maximum_radius)
    return estimates


def discover_radius_project_paths(source: str | Path) -> list[Path]:
    resolved = Path(source).expanduser().resolve()
    if not resolved.exists():
        raise ValueError(f"Input does not exist: {resolved}")
    if resolved.is_dir():
        projects = sorted(resolved.rglob("*.cpv.json"))
    elif resolved.name.endswith(".cpv.json"):
        projects = [resolved]
    elif resolved.name.endswith("_lineage.json"):
        projects = [resolved.with_name(resolved.name.removesuffix("_lineage.json") + ".cpv.json")]
    elif resolved.suffix.casefold() in {".tif", ".tiff"}:
        stem = resolved.stem
        for suffix in MASK_NAME_SUFFIXES:
            if stem.casefold().endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        projects = [resolved.with_name(stem + ".cpv.json")]
    else:
        projects = []
    projects = [path.resolve() for path in projects if path.is_file()]
    if not projects:
        raise ValueError(
            "No .cpv.json project was found. Run the centreline conversion command first."
        )
    return projects


def update_project_radii_from_masks(
    source: str | Path,
    *,
    scale: float = 1.0,
    offset: float = 0.0,
    max_search_distance: float = 3.0,
    only_zero: bool = False,
) -> list[MaskRadiusProjectResult]:
    results: list[MaskRadiusProjectResult] = []
    for project_path in discover_radius_project_paths(source):
        project, masks = _load_project_and_companion_mask(project_path)
        summary = apply_mask_radii(
            project,
            masks,
            scale=scale,
            offset=offset,
            max_search_distance=max_search_distance,
            only_zero=only_zero,
        )
        save_project(project, project_path)
        results.append(
            MaskRadiusProjectResult(
                project_path=project_path,
                nodes_updated=summary.nodes_updated,
                nodes_unmatched=summary.nodes_unmatched,
                nodes_skipped=summary.nodes_skipped,
            )
        )
    return results


def _load_project_and_companion_mask(project_path: Path) -> tuple[GraphProject, np.ndarray]:
    payload = json.loads(project_path.read_text(encoding="utf-8"))
    project = GraphProject.from_dict(payload)
    source_path = Path(project.source_path).expanduser()
    if not source_path.is_absolute():
        source_path = (project_path.parent / source_path).resolve()
    mask_path = discover_companion_mask(source_path)
    if mask_path is None:
        raise ValueError(f"No companion mask was found beside {source_path.name}")
    mask = load_mask_tiff(mask_path, expected_shape=project.shape_tzyx)
    project.source_path = str(source_path)
    return project, mask.data


def _graph_component_ids(graph: InstanceGraph, labels: np.ndarray) -> set[int]:
    positions: list[np.ndarray] = [
        np.asarray(node.position, dtype=float)[np.newaxis, :] for node in graph.nodes.values()
    ]
    for first_id, second_id in graph.edges:
        first = np.asarray(graph.nodes[first_id].position, dtype=float)
        second = np.asarray(graph.nodes[second_id].position, dtype=float)
        samples = max(2, int(np.ceil(np.max(np.abs(second - first)) * 2)) + 1)
        positions.append(np.linspace(first, second, samples))
    if not positions:
        return set()
    coordinates = np.floor(np.concatenate(positions, axis=0) + 0.5).astype(np.intp)
    shape = np.asarray(labels.shape, dtype=np.intp)
    coordinates = coordinates[np.all((coordinates >= 0) & (coordinates < shape), axis=1)]
    if not len(coordinates):
        return set()
    values = labels[tuple(coordinates.T)]
    return {int(value) for value in np.unique(values) if value}


def _component_at_position(labels: np.ndarray, position_zyx: tuple[float, float, float]) -> int:
    coordinate = np.floor(np.asarray(position_zyx) + 0.5).astype(np.intp)
    coordinate = np.clip(coordinate, 0, np.asarray(labels.shape) - 1)
    return int(labels[tuple(coordinate)])


def _nearest_component(
    labels: np.ndarray,
    position_zyx: tuple[float, float, float],
    candidates: set[int],
    maximum_distance: float,
) -> int:
    if maximum_distance <= 0:
        return 0
    position = np.asarray(position_zyx, dtype=float)
    margin = int(np.ceil(maximum_distance))
    lower = np.maximum(np.floor(position).astype(int) - margin, 0)
    upper = np.minimum(np.ceil(position).astype(int) + margin + 1, labels.shape)
    local = labels[tuple(slice(int(start), int(stop)) for start, stop in zip(lower, upper))]
    if candidates:
        foreground = np.isin(local, np.fromiter(candidates, dtype=labels.dtype))
    else:
        foreground = local != 0
    coordinates = np.argwhere(foreground)
    if not len(coordinates):
        return 0
    coordinates = coordinates + lower
    distances = np.linalg.norm(coordinates - position, axis=1)
    closest = int(np.argmin(distances))
    if float(distances[closest]) > maximum_distance:
        return 0
    return int(labels[tuple(coordinates[closest])])


def _build_component_distance(
    labels: np.ndarray,
    component_id: int,
    component_slice: tuple[slice, ...],
) -> _ComponentDistance:
    lower = np.asarray([part.start or 0 for part in component_slice], dtype=float)
    component = labels[component_slice] == component_id
    padded = np.pad(component, 1, mode="constant", constant_values=False)
    distance = distance_transform_edt(padded).astype(np.float32, copy=False)
    return _ComponentDistance(lower_zyx=lower, distance=distance)


def _sample_component_distance(
    component: _ComponentDistance,
    position_zyx: tuple[float, float, float],
) -> float:
    local = np.asarray(position_zyx, dtype=float) - component.lower_zyx + 1.0
    sampled = map_coordinates(
        component.distance,
        local[:, np.newaxis],
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    return float(sampled[0])


def _finite_sample(volume: np.ndarray, maximum_values: int = 1_000_000) -> np.ndarray:
    flat = volume.reshape(-1)
    if flat.size > maximum_values:
        flat = flat[:: max(1, flat.size // maximum_values)]
    return np.asarray(flat[np.isfinite(flat)], dtype=np.float32)


@lru_cache(maxsize=8)
def _fibonacci_sphere(count: int) -> np.ndarray:
    indices = np.arange(count, dtype=np.float32)
    golden_angle = np.float32(np.pi * (3.0 - np.sqrt(5.0)))
    z = 1.0 - 2.0 * (indices + 0.5) / count
    radial = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    angle = indices * golden_angle
    directions = np.column_stack((z, radial * np.sin(angle), radial * np.cos(angle)))
    directions.setflags(write=False)
    return directions


def _radial_profiles(
    volume: np.ndarray,
    position_zyx: tuple[float, float, float],
    directions: np.ndarray,
    distances: np.ndarray,
    *,
    cval: float,
) -> np.ndarray:
    centre = np.asarray(position_zyx, dtype=np.float32)
    coordinates = (
        centre[np.newaxis, np.newaxis, :]
        + directions[:, np.newaxis, :] * distances[np.newaxis, :, np.newaxis]
    )
    sampled = map_coordinates(
        volume,
        np.moveaxis(coordinates, -1, 0).reshape(3, -1),
        order=1,
        mode="constant",
        cval=float(cval),
        prefilter=False,
    )
    return np.asarray(sampled, dtype=np.float32).reshape(len(directions), len(distances))


def _first_threshold_crossing(
    profile: np.ndarray,
    distances: np.ndarray,
    threshold: float,
) -> float:
    candidates = np.flatnonzero(profile[1:] <= threshold) + 1
    if not candidates.size:
        return float(distances[-1])
    index = int(candidates[0])
    previous_value = float(profile[index - 1])
    current_value = float(profile[index])
    previous_distance = float(distances[index - 1])
    current_distance = float(distances[index])
    if previous_value <= threshold or previous_value == current_value:
        return current_distance
    fraction = (previous_value - threshold) / (previous_value - current_value)
    return previous_distance + min(max(fraction, 0.0), 1.0) * (current_distance - previous_distance)
