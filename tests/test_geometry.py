from __future__ import annotations

import numpy as np
import pytest

from chronopose_viewer.geometry import (
    Plane,
    closest_fraction_on_segment_2d,
    extract_slice,
    move_in_plane,
    project_position,
    unproject_position,
)


@pytest.mark.parametrize("plane", list(Plane))
def test_projection_round_trip(plane: Plane) -> None:
    position = (3.0, 5.0, 7.0)
    horizontal, vertical, depth = project_position(plane, position)
    assert unproject_position(plane, horizontal, vertical, depth) == position


def test_each_plane_extracts_expected_slice() -> None:
    volume = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    assert np.array_equal(extract_slice(volume, Plane.XY, 1), volume[1, :, :])
    assert np.array_equal(extract_slice(volume, Plane.XZ, 1), volume[:, 1, :])
    assert np.array_equal(extract_slice(volume, Plane.YZ, 1), volume[:, :, 1])


def test_dragging_in_plane_preserves_hidden_coordinate() -> None:
    assert move_in_plane(Plane.XY, (4, 5, 6), 10, 11) == (4, 11, 10)
    assert move_in_plane(Plane.XZ, (4, 5, 6), 10, 11) == (11, 5, 10)
    assert move_in_plane(Plane.YZ, (4, 5, 6), 10, 11) == (11, 10, 6)


def test_edge_pick_returns_clamped_fraction_and_pixel_distance() -> None:
    fraction, distance = closest_fraction_on_segment_2d((5, 2), (0, 0), (10, 0), scale=(1, 2))
    assert fraction == pytest.approx(0.5)
    assert distance == pytest.approx(1.0)
