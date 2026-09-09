from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

from chronopose_viewer.app import build_mask_radii_parser
from chronopose_viewer.geometry import Plane
from chronopose_viewer.io import load_project, save_project
from chronopose_viewer.model import GraphProject, InstanceGraph
from chronopose_viewer.radius_estimation import (
    apply_mask_radii,
    estimate_intensity_radii,
    update_project_radii_from_masks,
)
from chronopose_viewer.radius_overlay import RadiusOverlayRenderer


def test_swept_radius_overlay_fills_slices_and_caches_rasters() -> None:
    graph = InstanceGraph("graph", "graph", "#123456")
    first = graph.add_node((5, 5, 3), node_id="first", radius=2.0)
    second = graph.add_node((5, 5, 7), node_id="second", radius=1.0)
    graph.add_edge(first, second)
    renderer = RadiusOverlayRenderer()
    graphs = {graph.id: graph}

    xy = renderer.slice(graphs, (11, 11, 11), Plane.XY, 5)

    assert xy.image_rgba.shape == (11, 11, 4)
    assert tuple(xy.image_rgba[5, 5]) == (0x12, 0x34, 0x56, 255)
    assert xy.image_rgba[5, 6, 3] == 255
    assert xy.image_rgba[0, 0, 3] == 0
    assert renderer.slice(graphs, (11, 11, 11), Plane.XY, 5) is xy

    volume = renderer.volume(graphs, (11, 11, 11))
    assert volume.components.shape == (11, 11, 11)
    assert volume.steps_zyx == (1, 1, 1)
    assert volume.components[5, 5, 5] == 1
    assert renderer.volume(graphs, (11, 11, 11)) is volume

    separated = InstanceGraph("separated", "separated", "#ffffff")
    left = separated.add_node((5, 5, 2), radius=1.0)
    right = separated.add_node((5, 5, 8), radius=1.0)
    separated.add_edge(left, right)
    separated_graphs = {separated.id: separated}
    assert renderer.volume(separated_graphs, (11, 11, 11)).components[5, 5, 5] == 1

    graph.set_node_radius(first, 0)
    graph.set_node_radius(second, 0)
    assert not renderer.has_radii(graphs)
    empty = renderer.slice(graphs, (11, 11, 11), Plane.XY, 5)
    assert not np.any(empty.image_rgba[..., 3])


def test_mask_distance_transform_populates_radii_and_can_preserve_manual_values() -> None:
    project = GraphProject(shape_tzyx=(1, 9, 9, 9), source_path="source.tif")
    graph = project.add_instance(0, instance_id="graph")
    centre = graph.add_node((4, 4, 4), node_id="centre")
    boundary = graph.add_node((2, 4, 4), node_id="boundary")
    graph.add_edge(centre, boundary)
    masks = np.zeros(project.shape_tzyx, dtype=np.uint8)
    masks[0, 2:7, 2:7, 2:7] = 4

    summary = apply_mask_radii(project, masks)

    assert summary.nodes_updated == 2
    assert summary.nodes_unmatched == 0
    assert graph.nodes[centre].radius == pytest.approx(3.0)
    assert graph.nodes[boundary].radius == pytest.approx(1.0)

    graph.set_node_radius(centre, 9.0)
    summary = apply_mask_radii(project, masks, scale=2.0, only_zero=True)
    assert summary.nodes_skipped == 2
    assert graph.nodes[centre].radius == 9.0


def test_intensity_estimator_recovers_bright_sphere_and_zeros_flat_background() -> None:
    zz, yy, xx = np.ogrid[:41, :41, :41]
    volume = np.zeros((41, 41, 41), dtype=np.float32)
    volume[(zz - 20) ** 2 + (yy - 20) ** 2 + (xx - 20) ** 2 <= 5**2] = 100
    graph = InstanceGraph("graph", "graph", "#ffffff")
    bright = graph.add_node((20, 20, 20), node_id="bright")
    dark = graph.add_node((3, 3, 3), node_id="dark")

    estimates = estimate_intensity_radii(
        volume,
        {graph.id: graph},
        maximum_radius=10,
        smoothing_sigma=0,
    )

    assert estimates[(graph.id, bright)] == pytest.approx(5.0, abs=0.5)
    assert estimates[(graph.id, dark)] == 0.0


def test_post_conversion_mask_radius_command_updates_existing_project(tmp_path: Path) -> None:
    source = tmp_path / "sample.tif"
    masks_path = tmp_path / "sample_cp_masks.tif"
    data = np.zeros((1, 7, 7, 7), dtype=np.uint8)
    masks = np.zeros_like(data)
    masks[0, 1:6, 1:6, 1:6] = 1
    tifffile.imwrite(source, data, metadata={"axes": "TZYX"}, photometric="minisblack")
    tifffile.imwrite(
        masks_path,
        masks,
        metadata={"axes": "TZYX"},
        photometric="minisblack",
    )
    project = GraphProject(shape_tzyx=data.shape, source_path=str(source))
    graph = project.add_instance(0, instance_id="graph")
    graph.add_node((3, 3, 3), node_id="centre")
    project_path = save_project(project, tmp_path / "sample.cpv.json")

    [result] = update_project_radii_from_masks(project_path)

    assert result.nodes_updated == 1
    restored, _ = load_project(project_path)
    assert restored.instances_at(0)["graph"].nodes["centre"].radius == pytest.approx(3.0)
    arguments = build_mask_radii_parser().parse_args(
        [str(project_path), "--scale", "1.2", "--only-zero"]
    )
    assert arguments.scale == 1.2
    assert arguments.only_zero
