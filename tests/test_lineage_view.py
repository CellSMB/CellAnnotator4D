from __future__ import annotations

from PySide6 import QtCore

from chronopose_viewer.lineage_view import LineageEdgeItem, LineageView
from chronopose_viewer.model import GraphProject


def make_project() -> GraphProject:
    project = GraphProject(shape_tzyx=(3, 1, 1, 1), source_path="source.tif")
    project.add_instance(0, name="source", instance_id="source")
    project.add_instance(1, name="target-a", instance_id="target-a")
    project.add_instance(1, name="target-b", instance_id="target-b")
    project.add_instance(2, name="later", instance_id="later")
    return project


def test_lineage_view_lays_time_out_downward_and_requests_pair_connection(qtbot) -> None:
    view = LineageView()
    qtbot.addWidget(view)
    project = make_project()
    view.set_project(project, current_time=1)
    requests: list[tuple[int, str, int, str]] = []
    view.connection_requested.connect(lambda *values: requests.append(values))

    source_item = view._instance_items[(0, "source")]
    target_item = view._instance_items[(1, "target-a")]
    assert source_item.scenePos().y() < target_item.scenePos().y()

    view._instance_clicked(0, "source")
    assert view.selected_instance == (0, "source")
    view._instance_clicked(1, "target-a")
    assert requests == [(0, "source", 1, "target-a")]


def test_lineage_edges_are_clickable_event_objects(qtbot) -> None:
    view = LineageView()
    qtbot.addWidget(view)
    project = make_project()
    event = project.connect_lineage_instances(0, "source", 1, "target-a")
    view.set_project(project, current_time=0)

    edges = [item for item in view.scene().items() if isinstance(item, LineageEdgeItem)]
    assert edges
    assert {edge.event_id for edge in edges} == {event.id}
    removed: list[str] = []
    view.event_remove_requested.connect(removed.append)
    view.resize(600, 500)
    view.show()
    view.reset_view()
    edge_position = edges[0].mapToScene(edges[0].path.pointAtPercent(0.5))
    viewport_position = view.mapFromScene(edge_position)

    qtbot.mouseClick(
        view.viewport(),
        QtCore.Qt.MouseButton.LeftButton,
        pos=viewport_position,
    )

    assert removed == [event.id]


def test_escape_clears_lineage_selection(qtbot) -> None:
    view = LineageView()
    qtbot.addWidget(view)
    view.set_project(make_project(), current_time=0)
    view.show()
    view.setFocus()
    view._instance_clicked(0, "source")

    qtbot.keyClick(view, QtCore.Qt.Key.Key_Escape)

    assert view.selected_instance is None


def test_active_instance_has_a_separate_highlight_from_lineage_selection(qtbot) -> None:
    view = LineageView()
    qtbot.addWidget(view)
    project = make_project()
    view.set_project(project, current_time=1, active_instance_id="target-a")

    assert view._instance_items[(1, "target-a")]._active
    assert not view._instance_items[(1, "target-a")]._selected

    view._instance_clicked(1, "target-a")
    assert view._instance_items[(1, "target-a")]._active
    assert view._instance_items[(1, "target-a")]._selected

    view.set_active_instance(2, "later")
    assert not view._instance_items[(1, "target-a")]._active
    assert view._instance_items[(2, "later")]._active
    assert view._instance_items[(1, "target-a")]._selected


def test_arrange_aligns_connected_instances_and_default_order_restores_input(qtbot) -> None:
    view = LineageView()
    qtbot.addWidget(view)
    project = GraphProject(shape_tzyx=(2, 1, 1, 1), source_path="source.tif")
    project.add_instance(0, name="a", instance_id="a")
    project.add_instance(0, name="b", instance_id="b")
    project.add_instance(1, name="b-next", instance_id="b-next")
    project.add_instance(1, name="a-next", instance_id="a-next")
    project.connect_lineage_instances(0, "a", 1, "a-next")
    project.connect_lineage_instances(0, "b", 1, "b-next")
    view.set_project(project, current_time=0)

    def displayed_order(time: int) -> list[str]:
        return [
            instance_id
            for (_, instance_id), _item in sorted(
                ((key, item) for key, item in view._instance_items.items() if key[0] == time),
                key=lambda pair: pair[1].scenePos().x(),
            )
        ]

    assert displayed_order(1) == ["b-next", "a-next"]

    view.arrange_by_connections()

    assert displayed_order(0) == ["a", "b"]
    assert displayed_order(1) == ["a-next", "b-next"]

    view.reset_instance_order()

    assert displayed_order(0) == ["a", "b"]
    assert displayed_order(1) == ["b-next", "a-next"]
