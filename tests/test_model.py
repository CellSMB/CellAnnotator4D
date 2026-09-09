from __future__ import annotations

import pytest

from chronopose_viewer.model import EventKind, GraphProject, InstanceGraph, LineageEvent


def test_arbitrary_graph_editing_operations() -> None:
    graph = InstanceGraph(id="g", name="graph", color="#ffffff")
    first = graph.add_node((0, 1, 2), node_id="a", radius=1.0)
    middle = graph.add_node((1, 2, 3), node_id="b")
    last = graph.add_node((2, 3, 4), node_id="c", radius=3.0)
    isolated = graph.add_node((9, 9, 9), node_id="isolated")
    graph.add_edge(first, middle)
    graph.add_edge(middle, last)
    graph.add_edge(last, first)

    graph.move_node(isolated, (8, 7, 6))
    assert graph.nodes[isolated].position == (8.0, 7.0, 6.0)
    graph.remove_edge(first, last)
    replacement = graph.dissolve_degree_two_node(middle)
    assert replacement == ("a", "c")
    assert set(graph.nodes) == {"a", "c", "isolated"}

    inserted = graph.split_edge("a", "c", node_id="inserted")
    assert graph.nodes[inserted].position == (1.0, 2.0, 3.0)
    assert graph.nodes[inserted].radius == 2.0
    assert graph.edges == {("a", "inserted"), ("c", "inserted")}


def test_lineage_event_kinds_support_empty_one_and_many_sides() -> None:
    project = GraphProject(shape_tzyx=(3, 4, 5, 6), source_path="/tmp/source.tif")
    a = project.add_instance(0, instance_id="a")
    b = project.add_instance(0, instance_id="b")
    c = project.add_instance(1, instance_id="c")
    d = project.add_instance(1, instance_id="d")
    e = project.add_instance(2, instance_id="e")

    start = project.add_lineage_event(source_time=None, target_time=0, targets=[a.id])
    fission_fusion = project.add_lineage_event(
        source_time=0,
        target_time=1,
        sources=[a.id, b.id],
        targets=[c.id, d.id],
    )
    fusion = project.add_lineage_event(
        source_time=1, target_time=2, sources=[c.id, d.id], targets=[e.id]
    )
    end = project.add_lineage_event(source_time=2, target_time=None, sources=[e.id])

    assert start.kind is EventKind.START
    assert fission_fusion.kind is EventKind.FISSION_FUSION
    assert fusion.kind is EventKind.FUSION
    assert end.kind is EventKind.END


@pytest.mark.parametrize(
    ("sources", "targets", "expected"),
    [
        (("a",), ("b",), EventKind.ONE_TO_ONE),
        (("a",), ("b", "c"), EventKind.FISSION),
        (("a", "b"), ("c",), EventKind.FUSION),
        (("a", "b"), ("c", "d"), EventKind.FISSION_FUSION),
    ],
)
def test_matched_event_kind_is_inferred_from_cardinality(
    sources: tuple[str, ...], targets: tuple[str, ...], expected: EventKind
) -> None:
    event = LineageEvent("event", 0, 1, sources, targets)
    assert event.kind is expected


def test_instance_cannot_be_in_two_events_for_same_transition() -> None:
    project = GraphProject(shape_tzyx=(2, 1, 1, 1), source_path="source.tif")
    source = project.add_instance(0)
    first_target = project.add_instance(1)
    second_target = project.add_instance(1)
    project.add_lineage_event(
        source_time=0, target_time=1, sources=[source.id], targets=[first_target.id]
    )
    with pytest.raises(ValueError, match="already has"):
        project.add_lineage_event(
            source_time=0, target_time=1, sources=[source.id], targets=[second_target.id]
        )


def test_start_and_end_events_are_exclusive_with_frame_matches() -> None:
    project = GraphProject(shape_tzyx=(3, 1, 1, 1), source_path="source.tif")
    source = project.add_instance(0)
    target = project.add_instance(1)
    later = project.add_instance(2)
    project.add_lineage_event(
        source_time=0, target_time=1, sources=[source.id], targets=[target.id]
    )
    with pytest.raises(ValueError, match="already has"):
        project.add_lineage_event(source_time=None, target_time=1, targets=[target.id])
    project.add_lineage_event(source_time=1, target_time=2, sources=[target.id], targets=[later.id])
    with pytest.raises(ValueError, match="already has"):
        project.add_lineage_event(source_time=1, target_time=None, sources=[target.id])


def test_project_validation_rejects_bad_loaded_state() -> None:
    project = GraphProject(shape_tzyx=(2, 3, 4, 5), source_path="source.tif")
    source = project.add_instance(0)
    target = project.add_instance(1)
    source.add_node((3, 0, 0))
    with pytest.raises(ValueError, match="outside"):
        project.validate()

    source.nodes.clear()
    first = project.add_lineage_event(
        source_time=0, target_time=1, sources=[source.id], targets=[target.id]
    )
    project.lineage_events.append(
        type(first)(
            id="conflict",
            source_time=None,
            target_time=1,
            sources=(),
            targets=(target.id,),
        )
    )
    with pytest.raises(ValueError, match="contradictory"):
        project.validate()


def test_project_round_trip() -> None:
    project = GraphProject(shape_tzyx=(2, 3, 4, 5), source_path="source.tif")
    graph = project.add_instance(0, name="branched", instance_id="instance")
    graph.add_node((1, 2, 3), node_id="node", radius=2.75)
    project.add_lineage_event(source_time=None, target_time=0, targets=[graph.id])

    restored = GraphProject.from_dict(project.to_dict())
    assert restored.to_dict() == project.to_dict()
    assert restored.instances_at(0)["instance"].nodes["node"].radius == 2.75

    legacy = project.to_dict()
    del legacy["frames"]["0"][0]["nodes"][0]["radius"]
    assert GraphProject.from_dict(legacy).instances_at(0)["instance"].nodes["node"].radius == 0.0


def test_copy_instance_preserves_geometry_and_topology_with_fresh_ids() -> None:
    project = GraphProject(shape_tzyx=(3, 10, 10, 10), source_path="source.tif")
    source = project.add_instance(1, name="network", color="#123456")
    first = source.add_node((1, 2, 3), radius=1.25)
    second = source.add_node((4, 5, 6), radius=2.5)
    source.add_edge(first, second)

    copied = project.copy_instance(1, source.id, 2)

    assert copied.id != source.id
    assert copied.name == source.name
    assert copied.color == source.color
    assert set(copied.nodes).isdisjoint(source.nodes)
    assert sorted(node.position for node in copied.nodes.values()) == [
        (1.0, 2.0, 3.0),
        (4.0, 5.0, 6.0),
    ]
    assert len(copied.edges) == 1
    assert sorted(node.radius for node in copied.nodes.values()) == [1.25, 2.5]


def test_node_radius_validation_rejects_negative_or_non_finite_values() -> None:
    graph = InstanceGraph(id="g", name="graph", color="#ffffff")
    with pytest.raises(ValueError, match="non-negative"):
        graph.add_node((0, 0, 0), radius=-1)
    node_id = graph.add_node((0, 0, 0))
    with pytest.raises(ValueError, match="non-negative"):
        graph.set_node_radius(node_id, float("nan"))


def test_pairwise_lineage_connections_merge_into_many_to_many_event() -> None:
    project = GraphProject(shape_tzyx=(2, 1, 1, 1), source_path="source.tif")
    first_source = project.add_instance(0, instance_id="s1")
    second_source = project.add_instance(0, instance_id="s2")
    first_target = project.add_instance(1, instance_id="t1")
    second_target = project.add_instance(1, instance_id="t2")

    project.connect_lineage_instances(0, first_source.id, 1, first_target.id)
    project.connect_lineage_instances(0, second_source.id, 1, first_target.id)
    event = project.connect_lineage_instances(0, first_source.id, 1, second_target.id)

    assert len(project.lineage_events) == 1
    assert set(event.sources) == {"s1", "s2"}
    assert set(event.targets) == {"t1", "t2"}
    assert event.kind is EventKind.FISSION_FUSION


def test_connecting_replaces_compatible_start_and_end_markers() -> None:
    project = GraphProject(shape_tzyx=(2, 1, 1, 1), source_path="source.tif")
    source = project.add_instance(0, instance_id="source")
    target = project.add_instance(1, instance_id="target")
    project.mark_lineage_end(0, source.id)
    project.mark_lineage_start(1, target.id)

    event = project.connect_lineage_instances(0, source.id, 1, target.id)

    assert len(project.lineage_events) == 1
    assert event.kind is EventKind.ONE_TO_ONE
    assert event.sources == (source.id,)
    assert event.targets == (target.id,)


def test_connect_instances_merges_topology_and_compatible_lineage_events() -> None:
    project = GraphProject(shape_tzyx=(3, 10, 10, 10), source_path="source.tif")
    earlier_a = project.add_instance(0, instance_id="earlier-a")
    earlier_b = project.add_instance(0, instance_id="earlier-b")
    first = project.add_instance(
        1,
        name="kept name",
        color="#123456",
        instance_id="first",
    )
    second = project.add_instance(1, instance_id="second")
    later_a = project.add_instance(2, instance_id="later-a")
    later_b = project.add_instance(2, instance_id="later-b")
    first_node = first.add_node((1, 1, 1), node_id="first-node")
    first_tip = first.add_node((2, 2, 2), node_id="first-tip")
    first.add_edge(first_node, first_tip)
    second_node = second.add_node((7, 7, 7), node_id="second-node")
    second_tip = second.add_node((8, 8, 8), node_id="second-tip")
    second.add_edge(second_node, second_tip)
    project.add_lineage_event(
        source_time=0,
        target_time=1,
        sources=(earlier_a.id,),
        targets=(first.id,),
    )
    project.add_lineage_event(
        source_time=0,
        target_time=1,
        sources=(earlier_b.id,),
        targets=(second.id,),
    )
    project.add_lineage_event(
        source_time=1,
        target_time=2,
        sources=(first.id,),
        targets=(later_a.id,),
    )
    project.add_lineage_event(
        source_time=1,
        target_time=2,
        sources=(second.id,),
        targets=(later_b.id,),
    )

    result = project.connect_instances(1, first.id, first_tip, second.id, second_node)

    assert result.kept_instance_id == first.id
    assert result.removed_instance_id == second.id
    assert first.name == "kept name"
    assert first.color == "#123456"
    assert set(project.instances_at(1)) == {first.id}
    assert len(first.nodes) == 4
    assert len(first.edges) == 3
    assert result.connecting_edge in first.edges
    incoming = project.incoming_event(1, first.id)
    outgoing = project.outgoing_event(1, first.id)
    assert incoming is not None
    assert outgoing is not None
    assert set(incoming.sources) == {earlier_a.id, earlier_b.id}
    assert incoming.targets == (first.id,)
    assert outgoing.sources == (first.id,)
    assert set(outgoing.targets) == {later_a.id, later_b.id}
    assert all(
        second.id not in event.sources and second.id not in event.targets
        for event in project.lineage_events
    )
    project.validate()
    restored = GraphProject.from_dict(project.to_dict())
    assert restored.to_dict() == project.to_dict()


def test_connect_instances_rejects_incompatible_lineage_without_mutating_graphs() -> None:
    project = GraphProject(shape_tzyx=(4, 2, 2, 2), source_path="source.tif")
    source_at_zero = project.add_instance(0)
    source_at_one = project.add_instance(1)
    first = project.add_instance(2, instance_id="first")
    second = project.add_instance(2, instance_id="second")
    first_node = first.add_node((0, 0, 0))
    second_node = second.add_node((1, 1, 1))
    project.add_lineage_event(
        source_time=0,
        target_time=2,
        sources=(source_at_zero.id,),
        targets=(first.id,),
    )
    project.add_lineage_event(
        source_time=1,
        target_time=2,
        sources=(source_at_one.id,),
        targets=(second.id,),
    )

    with pytest.raises(ValueError, match="different frames"):
        project.connect_instances(2, first.id, first_node, second.id, second_node)

    assert set(project.instances_at(2)) == {first.id, second.id}
    assert len(first.nodes) == 1
    assert len(second.nodes) == 1


def test_connect_instances_reconciles_start_and_end_markers_without_losing_other_members() -> None:
    project = GraphProject(shape_tzyx=(3, 2, 2, 2), source_path="source.tif")
    earlier = project.add_instance(0, instance_id="earlier")
    first = project.add_instance(1, instance_id="first")
    second = project.add_instance(1, instance_id="second")
    other_start = project.add_instance(1, instance_id="other-start")
    other_end = project.add_instance(1, instance_id="other-end")
    later = project.add_instance(2, instance_id="later")
    first_node = first.add_node((0, 0, 0))
    second_node = second.add_node((1, 1, 1))
    project.add_lineage_event(
        source_time=None,
        target_time=1,
        targets=(first.id, other_start.id),
    )
    project.add_lineage_event(
        source_time=0,
        target_time=1,
        sources=(earlier.id,),
        targets=(second.id,),
    )
    project.add_lineage_event(
        source_time=1,
        target_time=None,
        sources=(first.id, other_end.id),
    )
    project.add_lineage_event(
        source_time=1,
        target_time=2,
        sources=(second.id,),
        targets=(later.id,),
    )

    project.connect_instances(1, first.id, first_node, second.id, second_node)

    incoming = project.incoming_event(1, first.id)
    outgoing = project.outgoing_event(1, first.id)
    assert incoming is not None and incoming.sources == (earlier.id,)
    assert outgoing is not None and outgoing.targets == (later.id,)
    other_start_event = project.incoming_event(1, other_start.id)
    other_end_event = project.outgoing_event(1, other_end.id)
    assert other_start_event is not None
    assert other_start_event.source_time is None
    assert other_start_event.targets == (other_start.id,)
    assert other_end_event is not None
    assert other_end_event.target_time is None
    assert other_end_event.sources == (other_end.id,)
    project.validate()


def test_connect_instances_remaps_colliding_node_ids() -> None:
    project = GraphProject(shape_tzyx=(1, 2, 2, 2), source_path="source.tif")
    first = project.add_instance(0, instance_id="first")
    second = project.add_instance(0, instance_id="second")
    first.add_node((0, 0, 0), node_id="same")
    second.add_node((1, 1, 1), node_id="same")

    result = project.connect_instances(0, first.id, "same", second.id, "same")

    assert result.second_node_id != "same"
    assert set(first.nodes) == {"same", result.second_node_id}
    assert result.connecting_edge in first.edges


def test_split_instance_on_bridge_preserves_both_sides_in_lineage() -> None:
    project = GraphProject(shape_tzyx=(3, 5, 5, 5), source_path="source.tif")
    earlier = project.add_instance(0)
    graph = project.add_instance(1, name="branched", instance_id="graph")
    later = project.add_instance(2)
    first = graph.add_node((1, 1, 1), node_id="a")
    middle = graph.add_node((2, 2, 2), node_id="b")
    last = graph.add_node((3, 3, 3), node_id="c")
    graph.add_edge(first, middle)
    bridge = graph.add_edge(middle, last)
    project.add_lineage_event(
        source_time=0,
        target_time=1,
        sources=(earlier.id,),
        targets=(graph.id,),
    )
    project.add_lineage_event(
        source_time=1,
        target_time=2,
        sources=(graph.id,),
        targets=(later.id,),
    )

    result = project.split_instance_on_edge(1, graph.id, bridge)

    assert set(project.instances_at(1)) == {graph.id, result.new_instance_id}
    split_graph = project.instances_at(1)[result.new_instance_id]
    assert set(graph.nodes) | set(split_graph.nodes) == {first, middle, last}
    assert set(graph.nodes).isdisjoint(split_graph.nodes)
    assert bridge not in graph.edges
    assert bridge not in split_graph.edges
    incoming = project.incoming_event(1, graph.id)
    outgoing = project.outgoing_event(1, graph.id)
    assert incoming is not None
    assert outgoing is not None
    assert set(incoming.targets) == {graph.id, split_graph.id}
    assert set(outgoing.sources) == {graph.id, split_graph.id}
    assert project.incoming_event(1, split_graph.id) is incoming
    assert project.outgoing_event(1, split_graph.id) is outgoing
    project.validate()
    restored = GraphProject.from_dict(project.to_dict())
    restored.remove_instance(1, split_graph.id)
    assert restored.incoming_event(1, graph.id) is not None
    assert restored.outgoing_event(1, graph.id) is not None
    restored.validate()


def test_split_instance_propagates_start_and_end_markers() -> None:
    project = GraphProject(shape_tzyx=(1, 3, 3, 3), source_path="source.tif")
    graph = project.add_instance(0, instance_id="graph")
    first = graph.add_node((0, 0, 0))
    second = graph.add_node((1, 1, 1))
    bridge = graph.add_edge(first, second)
    project.mark_lineage_start(0, graph.id)
    project.mark_lineage_end(0, graph.id)

    result = project.split_instance_on_edge(0, graph.id, bridge)

    start = project.incoming_event(0, graph.id)
    end = project.outgoing_event(0, graph.id)
    assert start is not None
    assert end is not None
    assert set(start.targets) == {graph.id, result.new_instance_id}
    assert set(end.sources) == {graph.id, result.new_instance_id}
    project.validate()


def test_split_instance_refuses_loop_edge_or_more_than_two_resulting_components() -> None:
    project = GraphProject(shape_tzyx=(1, 3, 3, 3), source_path="source.tif")
    loop = project.add_instance(0)
    a = loop.add_node((0, 0, 0), node_id="a")
    b = loop.add_node((1, 1, 1), node_id="b")
    c = loop.add_node((2, 2, 2), node_id="c")
    loop.add_edge(a, b)
    loop.add_edge(b, c)
    loop.add_edge(c, a)
    with pytest.raises(ValueError, match="exactly two"):
        project.split_instance_on_edge(0, loop.id, ("a", "b"))

    disconnected = project.add_instance(0)
    d = disconnected.add_node((0, 0, 0), node_id="d")
    e = disconnected.add_node((1, 1, 1), node_id="e")
    disconnected.add_node((2, 2, 2), node_id="isolated")
    bridge = disconnected.add_edge(d, e)
    with pytest.raises(ValueError, match="already disconnected"):
        project.split_instance_on_edge(0, disconnected.id, bridge)
    assert not project.can_split_instance_on_edge(0, disconnected.id, bridge)

    assert len(project.instances_at(0)) == 2
