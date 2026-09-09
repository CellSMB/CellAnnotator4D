from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

from chronopose_viewer.app import build_convert_parser
from chronopose_viewer.converter import (
    ConversionError,
    _populate_centerline,
    _prune_short_terminal_branches,
    _resample_degree_two_branches,
    convert_inference_path,
    discover_inference_bundles,
)
from chronopose_viewer.io import load_project
from chronopose_viewer.model import EventKind, InstanceGraph


def _observation(
    observation_id: str,
    frame: int,
    track_id: int,
    coordinates: list[tuple[int, int, int]],
) -> dict[str, object]:
    points = np.asarray(coordinates)
    return {
        "id": observation_id,
        "frame": frame,
        "track_id": track_id,
        "voxel_count": len(points),
        "centroid_zyx": points.mean(axis=0).tolist(),
    }


def _write_inference_fixture(root: Path) -> Path:
    final = root / "final"
    final.mkdir(parents=True)
    shape = (2, 5, 8, 10)
    image = np.zeros(shape, dtype=np.uint8)
    masks = np.zeros(shape, dtype=np.uint8)

    track_10_t0 = [(1, 1, 1), (1, 1, 2), (1, 1, 3), (1, 2, 5)]
    track_20_t0 = [(3, 6, 6), (3, 6, 7), (3, 6, 8)]
    track_20_t1 = [(3, 6, 5), (3, 6, 6), (3, 6, 7)]
    track_30_t1 = [(1, 1, 1), (1, 1, 2), (1, 1, 3)]
    track_40_t1 = [(1, 3, 4), (1, 3, 5)]

    # Deliberately use N-colour display labels that have no relationship to track
    # IDs. Track 10 is disconnected and shares display label 7 with track 20.
    for point in track_10_t0 + track_20_t0:
        masks[(0, *point)] = 7
    for point in track_20_t1:
        masks[(1, *point)] = 9
    for point in track_30_t1 + track_40_t1:
        masks[(1, *point)] = 4

    image_path = final / "sample.tif"
    masks_path = final / "sample_cp_masks.tif"
    tifffile.imwrite(
        image_path,
        image,
        metadata={"axes": "TZYX"},
        photometric="minisblack",
    )
    tifffile.imwrite(
        masks_path,
        masks,
        metadata={"axes": "TZYX"},
        photometric="minisblack",
    )

    observations = [
        _observation("t000000_o000000", 0, 10, track_10_t0),
        _observation("t000000_o000001", 0, 20, track_20_t0),
        _observation("t000001_o000000", 1, 20, track_20_t1),
        _observation("t000001_o000001", 1, 30, track_30_t1),
        _observation("t000001_o000002", 1, 40, track_40_t1),
    ]
    lineage = {
        "version": 1,
        "frame_indexing": "output",
        "input_mask_shape": list(shape),
        "output_image_name": "/container/output/sample.tif",
        "observations": observations,
        "links": [
            {
                "prev_observation": "t000000_o000000",
                "next_observation": "t000001_o000001",
            },
            {
                "prev_observation": "t000000_o000000",
                "next_observation": "t000001_o000002",
            },
            {
                "prev_observation": "t000000_o000001",
                "next_observation": "t000001_o000000",
            },
        ],
        "events": [
            {
                "id": "event_000001",
                "type": "split",
                "parent_observations": ["t000000_o000000"],
                "child_observations": [
                    "t000001_o000001",
                    "t000001_o000002",
                ],
            }
        ],
    }
    (final / "sample_lineage.json").write_text(
        json.dumps(lineage),
        encoding="utf-8",
    )
    return final


def test_converts_ncolor_masks_to_centerlines_and_lineages(tmp_path) -> None:
    final = _write_inference_fixture(tmp_path / "inference")
    bundles = discover_inference_bundles(final.parent)
    assert len(bundles) == 1
    assert bundles[0].project_path == final / "sample.cpv.json"

    [result] = convert_inference_path(final.parent, voxel_size_zyx=(2.0, 0.5, 0.5))
    assert result.status == "created"
    assert result.instances == 5
    assert result.lineage_events == 7

    project, volume = load_project(result.bundle.project_path)
    assert volume.shape_tzyx == (2, 5, 8, 10)
    assert project.voxel_size_zyx == (2.0, 0.5, 0.5)
    assert set(project.instances_at(0)) == {"track_000010", "track_000020"}
    assert set(project.instances_at(1)) == {
        "track_000020",
        "track_000030",
        "track_000040",
    }
    assert all(graph.nodes for frame in project.frames.values() for graph in frame.values())
    split_components = project.instances_at(0)["track_000010"].connected_components()
    assert sorted(len(component) for component in split_components) == [1, 2]

    split = next(event for event in project.lineage_events if event.id == "event_000001")
    assert split.kind is EventKind.FISSION
    assert split.sources == ("track_000010",)
    assert set(split.targets) == {"track_000030", "track_000040"}
    continuation = next(
        event for event in project.lineage_events if event.kind is EventKind.ONE_TO_ONE
    )
    assert continuation.sources == continuation.targets == ("track_000020",)

    payload = json.loads(result.bundle.project_path.read_text(encoding="utf-8"))
    assert payload["source"]["path"] == "sample.tif"


def test_existing_project_is_skipped_without_modification(tmp_path) -> None:
    final = _write_inference_fixture(tmp_path / "inference")
    [created] = convert_inference_path(final)
    assert created.status == "created"
    original = created.bundle.project_path.read_bytes()

    [skipped] = convert_inference_path(final)
    assert skipped.status == "skipped"
    assert skipped.bundle.project_path.read_bytes() == original


def test_resamples_open_branch_at_requested_voxel_spacing() -> None:
    graph = InstanceGraph("line", "line", "#ffffff")
    node_ids = [graph.add_node((0, 0, x), node_id=f"original_{x}") for x in range(11)]
    for first, second in zip(node_ids, node_ids[1:]):
        graph.add_edge(first, second)

    _resample_degree_two_branches(graph, 3.0)

    x_positions = sorted(node.position[2] for node in graph.nodes.values())
    assert x_positions == pytest.approx([0, 10 / 3, 20 / 3, 10])
    edge_lengths = sorted(
        np.linalg.norm(
            np.asarray(graph.nodes[first].position) - np.asarray(graph.nodes[second].position)
        )
        for first, second in graph.edges
    )
    assert edge_lengths == pytest.approx([10 / 3, 10 / 3, 10 / 3])
    assert sorted(graph.degree(node_id) for node_id in graph.nodes) == [1, 1, 2, 2]


def test_resampling_preserves_junctions_leaves_isolated_nodes_and_loops() -> None:
    branched = InstanceGraph("branch", "branch", "#ffffff")
    junction = branched.add_node((0, 0, 0), node_id="junction")
    leaves: list[str] = []
    for axis in range(3):
        previous = junction
        for distance in range(1, 7):
            position = [0.0, 0.0, 0.0]
            position[axis] = distance
            node_id = branched.add_node(position)
            branched.add_edge(previous, node_id)
            previous = node_id
        leaves.append(previous)
    isolated = branched.add_node((9, 9, 9), node_id="isolated")

    _resample_degree_two_branches(branched, 4.0)

    assert branched.nodes[junction].position == (0.0, 0.0, 0.0)
    assert branched.degree(junction) == 3
    assert all(branched.degree(node_id) == 1 for node_id in leaves)
    assert [branched.nodes[node_id].position for node_id in leaves] == [
        (6.0, 0.0, 0.0),
        (0.0, 6.0, 0.0),
        (0.0, 0.0, 6.0),
    ]
    assert branched.degree(isolated) == 0
    assert len(branched.nodes) == 8

    loop = InstanceGraph("loop", "loop", "#ffffff")
    positions = (
        [(0, 0, x) for x in range(5)]
        + [(0, y, 4) for y in range(1, 5)]
        + [(0, 4, x) for x in range(3, -1, -1)]
        + [(0, y, 0) for y in range(3, 0, -1)]
    )
    loop_nodes = [
        loop.add_node(position, node_id=f"loop_{index}") for index, position in enumerate(positions)
    ]
    for first, second in zip(
        loop_nodes,
        loop_nodes[1:] + loop_nodes[:1],
        strict=True,
    ):
        loop.add_edge(first, second)

    _resample_degree_two_branches(loop, 4.0)

    assert len(loop.nodes) == 4
    assert len(loop.edges) == 4
    assert all(loop.degree(node_id) == 2 for node_id in loop.nodes)


def test_each_nontrivial_mask_component_has_at_least_two_nodes() -> None:
    graph = InstanceGraph("small", "small", "#ffffff")
    multi_voxel_blob = np.argwhere(np.ones((2, 2, 2), dtype=bool))
    split_multi_voxel = np.asarray([(4, 4, 4), (4, 4, 5)])
    single_voxel = np.asarray([(7, 7, 7)])

    _populate_centerline(
        graph,
        (multi_voxel_blob, split_multi_voxel, single_voxel),
        node_spacing=100,
    )

    component_sizes = sorted(len(component) for component in graph.connected_components())
    assert graph.id == "small"
    assert component_sizes == [1, 2, 2]
    for component in graph.connected_components():
        if len(component) > 1:
            assert any(first in component and second in component for first, second in graph.edges)


def test_prunes_only_short_leaf_to_junction_branches() -> None:
    graph = InstanceGraph("mixed", "mixed", "#ffffff")
    junction = graph.add_node((0, 0, 0), node_id="junction")
    left = graph.add_node((0, 0, -10), node_id="left")
    right = graph.add_node((0, 0, 10), node_id="right")
    nub = graph.add_node((0, 2, 0), node_id="nub")
    for leaf in (left, right, nub):
        graph.add_edge(junction, leaf)

    tiny_first = graph.add_node((10, 0, 0), node_id="tiny_first")
    tiny_second = graph.add_node((10, 0, 1), node_id="tiny_second")
    graph.add_edge(tiny_first, tiny_second)

    loop_nodes = [
        graph.add_node(position, node_id=f"loop_{index}")
        for index, position in enumerate(((20, 0, 0), (20, 0, 1), (20, 1, 0)))
    ]
    for first, second in zip(
        loop_nodes,
        loop_nodes[1:] + loop_nodes[:1],
        strict=True,
    ):
        graph.add_edge(first, second)

    _prune_short_terminal_branches(graph, 5.0)

    assert nub not in graph.nodes
    assert {junction, left, right} <= graph.nodes.keys()
    assert graph.neighbours(tiny_first) == {tiny_second}
    assert all(graph.degree(node_id) == 2 for node_id in loop_nodes)


def test_pruning_an_all_short_star_retains_two_nodes_and_one_edge() -> None:
    graph = InstanceGraph("tiny_star", "tiny_star", "#ffffff")
    junction = graph.add_node((0, 0, 0), node_id="junction")
    for index, position in enumerate(((1, 0, 0), (0, 1, 0), (0, 0, 1))):
        leaf = graph.add_node(position, node_id=f"leaf_{index}")
        graph.add_edge(junction, leaf)

    _prune_short_terminal_branches(graph, 5.0)

    assert len(graph.nodes) == 2
    assert len(graph.edges) == 1
    assert sorted(graph.degree(node_id) for node_id in graph.nodes) == [1, 1]


def test_pruning_repeats_when_removing_ticks_exposes_a_short_stem() -> None:
    graph = InstanceGraph("nested_tick", "nested_tick", "#ffffff")
    main_junction = graph.add_node((0, 0, 0), node_id="main_junction")
    left = graph.add_node((0, 0, -10), node_id="left")
    right = graph.add_node((0, 0, 10), node_id="right")
    fork = graph.add_node((0, 3, 0), node_id="fork")
    first_tick = graph.add_node((0, 3, -1), node_id="first_tick")
    second_tick = graph.add_node((0, 3, 1), node_id="second_tick")
    for first, second in (
        (main_junction, left),
        (main_junction, right),
        (main_junction, fork),
        (fork, first_tick),
        (fork, second_tick),
    ):
        graph.add_edge(first, second)

    _prune_short_terminal_branches(graph, 5.0)

    assert set(graph.nodes) == {main_junction, left, right}
    assert graph.edges == {
        tuple(sorted((main_junction, left))),
        tuple(sorted((main_junction, right))),
    }


def test_spacing_flag_alias_and_validation(tmp_path) -> None:
    arguments = build_convert_parser().parse_args([str(tmp_path), "--spacing", "4.5"])
    assert arguments.node_spacing == 4.5
    assert arguments.min_terminal_branch_length == 5.0

    arguments = build_convert_parser().parse_args([str(tmp_path), "--min-branch-length", "0"])
    assert arguments.min_terminal_branch_length == 0

    with pytest.raises(ConversionError, match="positive finite"):
        convert_inference_path(tmp_path, node_spacing=0)
    with pytest.raises(ConversionError, match="non-negative finite"):
        convert_inference_path(tmp_path, min_terminal_branch_length=-1)
