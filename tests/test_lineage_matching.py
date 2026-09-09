from __future__ import annotations

from chronopose_viewer.lineage_matching import suggest_maximal_one_to_one_matches
from chronopose_viewer.model import GraphProject


def _add_point_instance(
    project: GraphProject,
    time: int,
    instance_id: str,
    x: float,
) -> None:
    graph = project.add_instance(time, name=instance_id, instance_id=instance_id)
    graph.add_node((0, 0, x), node_id=f"{instance_id}-node")


def test_suggests_global_maximal_matches_using_centerline_geometry() -> None:
    project = GraphProject(shape_tzyx=(2, 1, 1, 101), source_path="source.tif")
    _add_point_instance(project, 0, "source-left", 2)
    _add_point_instance(project, 0, "source-middle", 48)
    _add_point_instance(project, 0, "source-right", 97)
    # Reverse the insertion order relative to the geometrically corresponding sources.
    _add_point_instance(project, 1, "target-right", 96)
    _add_point_instance(project, 1, "target-left", 3)

    suggestions = suggest_maximal_one_to_one_matches(project)

    assert len(suggestions) == 2
    assert {(match.source_instance_id, match.target_instance_id) for match in suggestions} == {
        ("source-left", "target-left"),
        ("source-right", "target-right"),
    }


def test_existing_connections_and_start_end_markers_are_not_candidates() -> None:
    project = GraphProject(shape_tzyx=(2, 1, 1, 20), source_path="source.tif")
    for time, prefix in ((0, "source"), (1, "target")):
        for index, x in enumerate((2, 10, 18)):
            _add_point_instance(project, time, f"{prefix}-{index}", x)
    existing = project.connect_lineage_instances(0, "source-0", 1, "target-0")
    end = project.mark_lineage_end(0, "source-1")
    start = project.mark_lineage_start(1, "target-1")

    suggestions = suggest_maximal_one_to_one_matches(project)

    assert [(match.source_instance_id, match.target_instance_id) for match in suggestions] == [
        ("source-2", "target-2")
    ]
    assert project.lineage_events == [existing, end, start]
