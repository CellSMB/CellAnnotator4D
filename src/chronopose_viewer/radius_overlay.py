from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .geometry import Plane, plane_shape, project_position
from .model import InstanceGraph


@dataclass(frozen=True, slots=True)
class RadiusSliceOverlay:
    image_rgba: np.ndarray
    data_key: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class RadiusVolumeOverlay:
    components: np.ndarray
    palette_rgba: np.ndarray
    steps_zyx: tuple[int, int, int]
    data_key: tuple[object, ...]


class RadiusOverlayRenderer:
    """Rasterize node balls and linearly swept edge balls with small bounded caches."""

    def __init__(self, *, slice_cache_size: int = 32, volume_cache_size: int = 4) -> None:
        self._slice_cache_size = max(1, int(slice_cache_size))
        self._volume_cache_size = max(1, int(volume_cache_size))
        self._slice_cache: OrderedDict[tuple[object, ...], RadiusSliceOverlay] = OrderedDict()
        self._volume_cache: OrderedDict[tuple[object, ...], RadiusVolumeOverlay] = OrderedDict()

    def clear(self) -> None:
        self._slice_cache.clear()
        self._volume_cache.clear()

    @staticmethod
    def has_radii(graphs: dict[str, InstanceGraph]) -> bool:
        return any(node.radius > 0 for graph in graphs.values() for node in graph.nodes.values())

    def slice(
        self,
        graphs: dict[str, InstanceGraph],
        shape_zyx: tuple[int, int, int],
        plane: Plane,
        index: int,
    ) -> RadiusSliceOverlay:
        signature = radius_graph_signature(graphs)
        key = ("slice", tuple(shape_zyx), plane.value, int(index), signature)
        cached = self._slice_cache.get(key)
        if cached is not None:
            self._slice_cache.move_to_end(key)
            return cached

        height, width = plane_shape(shape_zyx, plane)
        labels = np.zeros((height, width), dtype=_label_dtype(len(graphs)))
        palette = _graph_palette(graphs)
        for label_id, graph in enumerate(graphs.values(), start=1):
            for first_id, second_id in graph.edges:
                first = graph.nodes[first_id]
                second = graph.nodes[second_id]
                _draw_swept_edge_slice(
                    labels,
                    project_position(plane, first.position),
                    first.radius,
                    project_position(plane, second.position),
                    second.radius,
                    float(index),
                    label_id,
                )
            for node in graph.nodes.values():
                _draw_sphere_slice(
                    labels,
                    project_position(plane, node.position),
                    node.radius,
                    float(index),
                    label_id,
                )
        image = np.ascontiguousarray(palette[labels])
        image.setflags(write=False)
        overlay = RadiusSliceOverlay(image_rgba=image, data_key=key)
        self._remember(self._slice_cache, key, overlay, self._slice_cache_size)
        return overlay

    def volume(
        self,
        graphs: dict[str, InstanceGraph],
        shape_zyx: tuple[int, int, int],
        *,
        maximum_axis: int = 256,
    ) -> RadiusVolumeOverlay:
        maximum_axis = max(2, int(maximum_axis))
        signature = radius_graph_signature(graphs)
        key = ("volume", tuple(shape_zyx), maximum_axis, signature)
        cached = self._volume_cache.get(key)
        if cached is not None:
            self._volume_cache.move_to_end(key)
            return cached

        steps = tuple(max(1, int(np.ceil(size / maximum_axis))) for size in shape_zyx)
        render_shape = tuple(int(np.ceil(size / step)) for size, step in zip(shape_zyx, steps))
        labels = np.zeros(render_shape, dtype=_label_dtype(len(graphs)))
        # Conservative cell coverage keeps thin non-zero radii visible after downsampling.
        sampling_margin = 0.5 * float(
            np.linalg.norm(np.maximum(np.asarray(steps, dtype=float) - 1.0, 0.0))
        )
        for label_id, graph in enumerate(graphs.values(), start=1):
            for first_id, second_id in graph.edges:
                first = graph.nodes[first_id]
                second = graph.nodes[second_id]
                if first.radius <= 0 and second.radius <= 0:
                    continue
                _draw_swept_edge_volume(
                    labels,
                    first.position,
                    first.radius + sampling_margin,
                    second.position,
                    second.radius + sampling_margin,
                    steps,
                    label_id,
                )
            for node in graph.nodes.values():
                if node.radius <= 0:
                    continue
                _draw_sphere_volume(
                    labels,
                    node.position,
                    node.radius + sampling_margin,
                    steps,
                    label_id,
                )
        labels = np.ascontiguousarray(labels)
        labels.setflags(write=False)
        palette = _graph_palette(graphs)
        palette.setflags(write=False)
        overlay = RadiusVolumeOverlay(
            components=labels,
            palette_rgba=palette,
            steps_zyx=steps,
            data_key=key,
        )
        self._remember(self._volume_cache, key, overlay, self._volume_cache_size)
        return overlay

    @staticmethod
    def _remember(
        cache: OrderedDict,
        key: tuple[object, ...],
        value: object,
        maximum_size: int,
    ) -> None:
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > maximum_size:
            cache.popitem(last=False)


def radius_graph_signature(graphs: dict[str, InstanceGraph]) -> tuple[object, ...]:
    return tuple(
        (
            instance_id,
            graph.color,
            tuple((node_id, node.position, node.radius) for node_id, node in graph.nodes.items()),
            tuple(sorted(graph.edges)),
        )
        for instance_id, graph in graphs.items()
    )


def _graph_palette(graphs: dict[str, InstanceGraph]) -> np.ndarray:
    palette = np.zeros((len(graphs) + 1, 4), dtype=np.uint8)
    for label_id, graph in enumerate(graphs.values(), start=1):
        palette[label_id, :3] = _hex_to_rgb(graph.color)
        palette[label_id, 3] = 255
    return palette


def _label_dtype(label_count: int) -> np.dtype:
    if label_count <= np.iinfo(np.uint8).max:
        return np.dtype(np.uint8)
    if label_count <= np.iinfo(np.uint16).max:
        return np.dtype(np.uint16)
    return np.dtype(np.uint32)


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    value = color.lstrip("#")
    if len(value) == 6:
        try:
            return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4))  # type: ignore[return-value]
        except ValueError:
            pass
    return 76, 201, 240


def _draw_sphere_slice(
    labels: np.ndarray,
    centre_hvd: Sequence[float],
    radius: float,
    slice_depth: float,
    label_id: int,
) -> None:
    radius = float(radius)
    if radius <= 0:
        return
    horizontal, vertical, depth = (float(value) for value in centre_hvd)
    depth_delta = slice_depth - depth
    squared_radius = radius * radius - depth_delta * depth_delta
    if squared_radius < 0:
        return
    cross_radius = float(np.sqrt(max(0.0, squared_radius)))
    bounds = _bounds_2d(labels.shape, horizontal, vertical, cross_radius)
    if bounds is None:
        return
    y_slice, x_slice = bounds
    yy, xx = np.ogrid[y_slice, x_slice]
    inside = (xx - horizontal) ** 2 + (yy - vertical) ** 2 <= squared_radius
    target = labels[y_slice, x_slice]
    target[inside] = label_id


def _draw_swept_edge_slice(
    labels: np.ndarray,
    first_hvd: Sequence[float],
    first_radius: float,
    second_hvd: Sequence[float],
    second_radius: float,
    slice_depth: float,
    label_id: int,
) -> None:
    if first_radius <= 0 and second_radius <= 0:
        return
    first = np.asarray(first_hvd, dtype=float)
    second = np.asarray(second_hvd, dtype=float)
    maximum_radius = max(float(first_radius), float(second_radius))
    if (
        slice_depth < min(first[2], second[2]) - maximum_radius
        or slice_depth > max(first[2], second[2]) + maximum_radius
    ):
        return
    horizontal_min = min(first[0], second[0]) - maximum_radius
    horizontal_max = max(first[0], second[0]) + maximum_radius
    vertical_min = min(first[1], second[1]) - maximum_radius
    vertical_max = max(first[1], second[1]) + maximum_radius
    bounds = _rect_bounds(
        labels.shape,
        horizontal_min,
        horizontal_max,
        vertical_min,
        vertical_max,
    )
    if bounds is None:
        return
    y_slice, x_slice = bounds
    yy, xx = np.ogrid[y_slice, x_slice]
    points = (
        xx - first[0],
        yy - first[1],
        np.asarray(slice_depth - first[2]),
    )
    inside = _inside_swept_spheres(
        points,
        second - first,
        float(first_radius),
        float(second_radius),
    )
    target = labels[y_slice, x_slice]
    target[inside] = label_id


def _draw_sphere_volume(
    labels: np.ndarray,
    centre_zyx: Sequence[float],
    radius: float,
    steps_zyx: tuple[int, int, int],
    label_id: int,
) -> None:
    centre = np.asarray(centre_zyx, dtype=float)
    bounds = _volume_bounds(labels.shape, centre - radius, centre + radius, steps_zyx)
    if bounds is None:
        return
    z_slice, y_slice, x_slice = bounds
    y_values = np.arange(y_slice.start, y_slice.stop) * steps_zyx[1]
    x_values = np.arange(x_slice.start, x_slice.stop) * steps_zyx[2]
    yy, xx = np.meshgrid(y_values, x_values, indexing="ij")
    squared_radius = radius * radius
    for z_index in range(z_slice.start, z_slice.stop):
        z_value = z_index * steps_zyx[0]
        inside = (z_value - centre[0]) ** 2 + (yy - centre[1]) ** 2 + (
            xx - centre[2]
        ) ** 2 <= squared_radius
        target = labels[z_index, y_slice, x_slice]
        target[inside] = label_id


def _draw_swept_edge_volume(
    labels: np.ndarray,
    first_zyx: Sequence[float],
    first_radius: float,
    second_zyx: Sequence[float],
    second_radius: float,
    steps_zyx: tuple[int, int, int],
    label_id: int,
) -> None:
    first = np.asarray(first_zyx, dtype=float)
    second = np.asarray(second_zyx, dtype=float)
    maximum_radius = max(float(first_radius), float(second_radius))
    bounds = _volume_bounds(
        labels.shape,
        np.minimum(first, second) - maximum_radius,
        np.maximum(first, second) + maximum_radius,
        steps_zyx,
    )
    if bounds is None:
        return
    z_slice, y_slice, x_slice = bounds
    y_values = np.arange(y_slice.start, y_slice.stop) * steps_zyx[1]
    x_values = np.arange(x_slice.start, x_slice.stop) * steps_zyx[2]
    yy, xx = np.meshgrid(y_values, x_values, indexing="ij")
    direction = second - first
    for z_index in range(z_slice.start, z_slice.stop):
        z_value = z_index * steps_zyx[0]
        points = (
            np.asarray(z_value - first[0]),
            yy - first[1],
            xx - first[2],
        )
        inside = _inside_swept_spheres(
            points,
            direction,
            float(first_radius),
            float(second_radius),
        )
        target = labels[z_index, y_slice, x_slice]
        target[inside] = label_id


def _inside_swept_spheres(
    point_minus_first: tuple[np.ndarray, np.ndarray, np.ndarray],
    direction: np.ndarray,
    first_radius: float,
    second_radius: float,
) -> np.ndarray:
    q_dot_direction = sum(point_minus_first[axis] * direction[axis] for axis in range(3))
    q_squared = sum(value * value for value in point_minus_first)
    radius_delta = second_radius - first_radius
    quadratic = radius_delta * radius_delta - float(direction @ direction)
    linear = first_radius * radius_delta + q_dot_direction
    constant = first_radius * first_radius - q_squared
    maximum = np.maximum(constant, quadratic + 2.0 * linear + constant)
    if quadratic < -1e-12:
        fraction = np.clip(-linear / quadratic, 0.0, 1.0)
        maximum = np.maximum(
            maximum,
            quadratic * fraction * fraction + 2.0 * linear * fraction + constant,
        )
    return maximum >= 0.0


def _bounds_2d(
    shape_yx: tuple[int, int],
    horizontal: float,
    vertical: float,
    radius: float,
) -> tuple[slice, slice] | None:
    return _rect_bounds(
        shape_yx,
        horizontal - radius,
        horizontal + radius,
        vertical - radius,
        vertical + radius,
    )


def _rect_bounds(
    shape_yx: tuple[int, int],
    horizontal_min: float,
    horizontal_max: float,
    vertical_min: float,
    vertical_max: float,
) -> tuple[slice, slice] | None:
    x_start = max(0, int(np.ceil(horizontal_min)))
    x_stop = min(shape_yx[1], int(np.floor(horizontal_max)) + 1)
    y_start = max(0, int(np.ceil(vertical_min)))
    y_stop = min(shape_yx[0], int(np.floor(vertical_max)) + 1)
    if x_start >= x_stop or y_start >= y_stop:
        return None
    return slice(y_start, y_stop), slice(x_start, x_stop)


def _volume_bounds(
    render_shape: tuple[int, int, int],
    lower_zyx: np.ndarray,
    upper_zyx: np.ndarray,
    steps_zyx: tuple[int, int, int],
) -> tuple[slice, slice, slice] | None:
    lower = np.maximum(
        np.ceil(lower_zyx / np.asarray(steps_zyx)).astype(int),
        0,
    )
    upper = np.minimum(
        np.floor(upper_zyx / np.asarray(steps_zyx)).astype(int) + 1,
        render_shape,
    )
    if np.any(lower >= upper):
        return None
    return tuple(slice(int(start), int(stop)) for start, stop in zip(lower, upper))  # type: ignore[return-value]
