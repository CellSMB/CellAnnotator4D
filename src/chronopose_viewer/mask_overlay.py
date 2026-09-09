from __future__ import annotations

import colorsys
import hashlib
from dataclasses import dataclass

import numpy as np
from skimage.measure import label as connected_components

from .model import InstanceGraph


@dataclass(frozen=True, slots=True)
class MaskFrameOverlay:
    """Cached component labels and their current display colors for one frame."""

    components: np.ndarray
    palette_rgba: np.ndarray
    graph_signature: tuple[object, ...]

    @property
    def component_count(self) -> int:
        return len(self.palette_rgba) - 1

    def colorize(self, component_values: np.ndarray) -> np.ndarray:
        return self.palette_rgba[np.asarray(component_values)]


@dataclass(slots=True)
class _ComponentFrame:
    components: np.ndarray
    count: int


class InstanceMaskColorizer:
    """Match cached 3D mask components to graph colors by centerline overlap."""

    def __init__(self, masks_tzyx: np.ndarray, *, seed: str = "") -> None:
        masks = np.asarray(masks_tzyx)
        if masks.ndim != 4:
            raise ValueError(f"Expected TZYX instance masks, got shape {masks.shape}")
        if not np.issubdtype(masks.dtype, np.integer):
            raise ValueError(f"Instance masks must use an integer dtype, got {masks.dtype}")
        if np.any(masks < 0):
            raise ValueError("Instance masks cannot contain negative labels")
        self._masks = masks
        self._seed = seed
        self._component_frames: dict[int, _ComponentFrame] = {}
        self._overlays: dict[int, MaskFrameOverlay] = {}

    def frame(
        self,
        time: int,
        graphs: dict[str, InstanceGraph],
    ) -> MaskFrameOverlay:
        if not 0 <= time < self._masks.shape[0]:
            raise IndexError(f"Mask timepoint {time} is out of range")
        signature = _graph_signature(graphs)
        cached = self._overlays.get(time)
        if cached is not None and cached.graph_signature == signature:
            return cached

        component_frame = self._components_at(time)
        palette = self._match_component_colors(
            time,
            component_frame.components,
            component_frame.count,
            graphs,
        )
        overlay = MaskFrameOverlay(
            components=component_frame.components,
            palette_rgba=palette,
            graph_signature=signature,
        )
        self._overlays[time] = overlay
        return overlay

    def _components_at(self, time: int) -> _ComponentFrame:
        cached = self._component_frames.get(time)
        if cached is not None:
            return cached

        labels, count = connected_components(
            self._masks[time],
            background=0,
            connectivity=3,
            return_num=True,
        )
        components = np.ascontiguousarray(labels, dtype=_component_dtype(int(count)))
        components.setflags(write=False)
        cached = _ComponentFrame(components=components, count=int(count))
        self._component_frames[time] = cached
        return cached

    def _match_component_colors(
        self,
        time: int,
        components: np.ndarray,
        component_count: int,
        graphs: dict[str, InstanceGraph],
    ) -> np.ndarray:
        palette = np.zeros((component_count + 1, 4), dtype=np.uint8)
        if component_count == 0:
            return palette

        best_counts = np.zeros(component_count + 1, dtype=np.int64)
        matched_colors: list[str | None] = [None] * (component_count + 1)
        for graph in graphs.values():
            coordinates = _graph_voxels(graph, components.shape)
            if not len(coordinates):
                continue
            sampled = components[tuple(coordinates.T)].astype(np.int64, copy=False)
            counts = np.bincount(sampled, minlength=component_count + 1)
            improved = counts > best_counts
            improved[0] = False
            for component_id in np.flatnonzero(improved):
                matched_colors[int(component_id)] = graph.color
            best_counts[improved] = counts[improved]

        for component_id in range(1, component_count + 1):
            graph_color = matched_colors[component_id]
            if graph_color is None:
                rgb = _fallback_rgb(self._seed, time, component_id)
            else:
                rgb = _hex_to_rgb(graph_color)
            palette[component_id, :3] = rgb
            palette[component_id, 3] = 255
        palette.setflags(write=False)
        return palette


def _graph_signature(graphs: dict[str, InstanceGraph]) -> tuple[object, ...]:
    return tuple(
        (
            instance_id,
            graph.color,
            tuple((node_id, node.position) for node_id, node in graph.nodes.items()),
            tuple(sorted(graph.edges)),
        )
        for instance_id, graph in graphs.items()
    )


def _graph_voxels(
    graph: InstanceGraph,
    shape_zyx: tuple[int, ...],
) -> np.ndarray:
    sampled_positions: list[np.ndarray] = []
    for node in graph.nodes.values():
        sampled_positions.append(np.asarray(node.position, dtype=float)[np.newaxis, :])
    for first_id, second_id in graph.edges:
        first = np.asarray(graph.nodes[first_id].position, dtype=float)
        second = np.asarray(graph.nodes[second_id].position, dtype=float)
        maximum_delta = float(np.max(np.abs(second - first)))
        sample_count = max(2, int(np.ceil(maximum_delta * 2.0)) + 1)
        sampled_positions.append(np.linspace(first, second, sample_count))
    if not sampled_positions:
        return np.empty((0, 3), dtype=np.intp)

    positions = np.concatenate(sampled_positions, axis=0)
    coordinates = np.floor(positions + 0.5).astype(np.intp)
    shape = np.asarray(shape_zyx, dtype=np.intp)
    in_bounds = np.all((coordinates >= 0) & (coordinates < shape), axis=1)
    coordinates = coordinates[in_bounds]
    if not len(coordinates):
        return np.empty((0, 3), dtype=np.intp)
    return np.unique(coordinates, axis=0)


def _component_dtype(component_count: int) -> np.dtype:
    if component_count <= np.iinfo(np.uint8).max:
        return np.dtype(np.uint8)
    if component_count <= np.iinfo(np.uint16).max:
        return np.dtype(np.uint16)
    if component_count <= np.iinfo(np.uint32).max:
        return np.dtype(np.uint32)
    return np.dtype(np.uint64)


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    value = color.lstrip("#")
    if len(value) == 6:
        try:
            return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4))  # type: ignore[return-value]
        except ValueError:
            pass
    return 76, 201, 240


def _fallback_rgb(seed: str, time: int, component_id: int) -> tuple[int, int, int]:
    digest = hashlib.blake2b(
        f"{seed}:{time}:{component_id}".encode(),
        digest_size=8,
    ).digest()
    hue = int.from_bytes(digest[:4], "big") / (2**32)
    saturation = 0.62 + digest[4] / 255 * 0.25
    value = 0.82 + digest[5] / 255 * 0.16
    red, green, blue = colorsys.hsv_to_rgb(hue, saturation, value)
    return tuple(round(channel * 255) for channel in (red, green, blue))  # type: ignore[return-value]
