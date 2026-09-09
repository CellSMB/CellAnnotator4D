from __future__ import annotations

from enum import StrEnum
from typing import Sequence

import numpy as np

from .model import Position


class Plane(StrEnum):
    XY = "xy"
    XZ = "xz"
    YZ = "yz"

    @property
    def title(self) -> str:
        return self.value.upper()

    @property
    def depth_axis(self) -> int:
        return {Plane.XY: 0, Plane.XZ: 1, Plane.YZ: 2}[self]

    @property
    def depth_label(self) -> str:
        return {Plane.XY: "Z", Plane.XZ: "Y", Plane.YZ: "X"}[self]

    @property
    def horizontal_label(self) -> str:
        return {Plane.XY: "X", Plane.XZ: "X", Plane.YZ: "Y"}[self]

    @property
    def vertical_label(self) -> str:
        return {Plane.XY: "Y", Plane.XZ: "Z", Plane.YZ: "Z"}[self]


def project_position(plane: Plane, position_zyx: Sequence[float]) -> tuple[float, float, float]:
    z, y, x = (float(value) for value in position_zyx)
    if plane is Plane.XY:
        return x, y, z
    if plane is Plane.XZ:
        return x, z, y
    return y, z, x


def unproject_position(plane: Plane, horizontal: float, vertical: float, depth: float) -> Position:
    if plane is Plane.XY:
        return float(depth), float(vertical), float(horizontal)
    if plane is Plane.XZ:
        return float(vertical), float(depth), float(horizontal)
    return float(vertical), float(horizontal), float(depth)


def move_in_plane(
    plane: Plane, position_zyx: Sequence[float], horizontal: float, vertical: float
) -> Position:
    _, _, depth = project_position(plane, position_zyx)
    return unproject_position(plane, horizontal, vertical, depth)


def extract_slice(volume_zyx: np.ndarray, plane: Plane, index: int) -> np.ndarray:
    if plane is Plane.XY:
        return volume_zyx[index, :, :]
    if plane is Plane.XZ:
        return volume_zyx[:, index, :]
    return volume_zyx[:, :, index]


def plane_shape(shape_zyx: Sequence[int], plane: Plane) -> tuple[int, int]:
    z, y, x = (int(value) for value in shape_zyx)
    if plane is Plane.XY:
        return y, x
    if plane is Plane.XZ:
        return z, x
    return z, y


def clamp_position(position_zyx: Sequence[float], shape_zyx: Sequence[int]) -> Position:
    return tuple(
        min(max(float(value), 0.0), float(size - 1))
        for value, size in zip(position_zyx, shape_zyx, strict=True)
    )  # type: ignore[return-value]


def closest_fraction_on_segment_2d(
    point: tuple[float, float],
    first: tuple[float, float],
    second: tuple[float, float],
    *,
    scale: tuple[float, float] = (1.0, 1.0),
) -> tuple[float, float]:
    scale_x = max(abs(float(scale[0])), 1e-12)
    scale_y = max(abs(float(scale[1])), 1e-12)
    point_scaled = np.array((point[0] / scale_x, point[1] / scale_y), dtype=float)
    first_scaled = np.array((first[0] / scale_x, first[1] / scale_y), dtype=float)
    second_scaled = np.array((second[0] / scale_x, second[1] / scale_y), dtype=float)
    direction = second_scaled - first_scaled
    denominator = float(direction @ direction)
    if denominator == 0:
        return 0.0, float(np.linalg.norm(point_scaled - first_scaled))
    fraction = float((point_scaled - first_scaled) @ direction / denominator)
    fraction = min(max(fraction, 0.0), 1.0)
    closest = first_scaled + fraction * direction
    return fraction, float(np.linalg.norm(point_scaled - closest))


def interpolate_position(
    first_zyx: Sequence[float], second_zyx: Sequence[float], fraction: float
) -> Position:
    fraction = min(max(float(fraction), 0.0), 1.0)
    return tuple(
        float(first) + fraction * (float(second) - float(first))
        for first, second in zip(first_zyx, second_zyx, strict=True)
    )  # type: ignore[return-value]
