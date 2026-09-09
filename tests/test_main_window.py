from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PySide6 import QtCore, QtTest, QtWidgets

from chronopose_viewer.geometry import Plane
from chronopose_viewer.interaction import PickResult, PointerButton
from chronopose_viewer.io import MaskData, VolumeData, new_project_for_volume
from chronopose_viewer.main_window import (
    EDIT_TAB,
    INSTANCE_TAB,
    LINEAGE_TAB,
    OPACITY_TAB,
    EditMode,
    MainWindow,
    RadiusDisplayMode,
    mode_help_text,
)


def install_test_volume(window: MainWindow) -> None:
    volume = VolumeData(
        path=Path("/tmp/test-volume.tif"),
        data=np.arange(2 * 4 * 5 * 6, dtype=np.uint16).reshape(2, 4, 5, 6),
        source_axes="TZYX",
        axes_inferred=False,
    )
    window._install_document(new_project_for_volume(volume), volume, project_path=None)


def test_mask_overlay_controls_and_graph_matched_color(qtbot) -> None:
    masks = np.zeros((2, 4, 5, 6), dtype=np.uint8)
    masks[0, 2, 2, 3] = 4
    volume = VolumeData(
        path=Path("/tmp/masked-volume.tif"),
        data=np.arange(2 * 4 * 5 * 6, dtype=np.uint16).reshape(2, 4, 5, 6),
        source_axes="TZYX",
        axes_inferred=False,
        masks=MaskData(
            path=Path("/tmp/masked-volume_cp_masks.tif"),
            data=masks,
            source_axes="TZYX",
            axes_inferred=False,
        ),
    )
    project = new_project_for_volume(volume)
    graph = project.add_instance(0, color="#123456")
    graph.add_node((2, 2, 3))
    window = MainWindow()
    qtbot.addWidget(window)

    window._install_document(project, volume, project_path=None)

    assert window.mask_visibility_checkbox.isEnabled()
    assert window.mask_visibility_checkbox.isChecked()
    assert window.mask_opacity_slider.isEnabled()
    assert window.mask_source_label.text() == "masked-volume_cp_masks.tif"
    xy_mask = window.ortho_views[Plane.XY].mask_item
    assert xy_mask.isVisible()
    assert tuple(xy_mask.image[2, 3]) == (18, 52, 86, 255)

    window.mask_opacity_slider.setValue(35)
    assert xy_mask.opacity() == 0.35
    assert window.mask_opacity_label.text() == "35%"

    window.mask_visibility_checkbox.setChecked(False)
    assert not xy_mask.isVisible()
    assert not window.mask_opacity_slider.isEnabled()
    assert not window.dirty


def test_near_slice_node_emphasis_toggle_refreshes_only_2d_display(qtbot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    install_test_volume(window)

    assert window.near_slice_nodes_checkbox.isEnabled()
    assert not window.near_slice_nodes_checkbox.isChecked()
    assert not window.near_slice_settings_widget.isEnabled()
    assert window.near_slice_opacity_range.lowerValue() == 0
    assert window.near_slice_opacity_range.upperValue() == 100
    assert window.near_slice_near_distance_spin.value() == 0.0
    assert window.near_slice_far_distance_spin.value() == 4.0
    assert all(view._graph_key[1] is False for view in window.ortho_views.values())

    window.near_slice_nodes_checkbox.setChecked(True)

    assert window.near_slice_settings_widget.isEnabled()
    assert all(view._graph_key[1] is True for view in window.ortho_views.values())

    window.near_slice_opacity_range.setValues(20, 80)
    window.near_slice_far_distance_spin.setValue(6.0)
    window.near_slice_near_distance_spin.setValue(1.0)

    assert window.near_slice_opacity_range_label.text() == "20–80%"
    assert window.near_slice_near_distance_label.text() == "80% at ≤"
    assert window.near_slice_far_distance_label.text() == "20% at ≥"
    assert all(view._graph_key[2:6] == (0.2, 0.8, 1.0, 6.0) for view in window.ortho_views.values())

    window.near_slice_near_distance_spin.setValue(7.0)
    assert window.near_slice_far_distance_spin.value() == 7.0
    window.near_slice_far_distance_spin.setValue(3.0)
    assert window.near_slice_near_distance_spin.value() == 3.0
    assert "enabled" in window.statusBar().currentMessage()
    assert not window.dirty


def test_radius_tool_previews_without_mutation_then_commits_on_release(qtbot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    install_test_volume(window)
    assert window.project is not None
    graph = window.project.add_instance(0, instance_id="graph", color="#123456")
    node_id = graph.add_node((2, 2, 2), node_id="node")
    window.active_instance_id = graph.id
    window._refresh_timepoint()
    window.activate_tool(EditMode.EDIT_RADIUS)
    control = QtCore.Qt.KeyboardModifier.ControlModifier

    window._on_2d_drag(Plane.XY, PointerButton.LEFT, "start", 2, 2, control)
    window._on_2d_drag(Plane.XY, PointerButton.LEFT, "move", 5, 2, control)

    assert graph.nodes[node_id].radius == 0.0
    assert window.radius_preview == (graph.id, node_id, 3.0)
    assert not window.ortho_views[Plane.XY].radius_item.isVisible()
    preview_circles = [
        item
        for item in window.ortho_views[Plane.XY]._highlight_items
        if isinstance(item, QtWidgets.QGraphicsEllipseItem)
    ]
    assert [circle.pen().color().name() for circle in preview_circles] == [
        "#000000",
        "#ffffff",
    ]

    window._on_2d_drag(Plane.XY, PointerButton.LEFT, "finish", 5, 2, control)

    assert graph.nodes[node_id].radius == 3.0
    assert window.radius_preview is None
    assert window.ortho_views[Plane.XY].radius_item.isVisible()
    assert window.dirty

    window.undo()
    restored = window.project.instances_at(0)["graph"]
    assert restored.nodes[node_id].radius == 0.0


def test_radius_ctrl_drag_works_through_real_qt_mouse_events(qtbot, monkeypatch) -> None:
    discard_unsaved_on_close(monkeypatch)
    window = MainWindow()
    qtbot.addWidget(window)
    install_test_volume(window)
    assert window.project is not None
    graph = window.project.add_instance(0, instance_id="graph")
    node_id = graph.add_node((2, 2, 2), node_id="node")
    window.active_instance_id = graph.id
    window._refresh_timepoint()
    window.activate_tool(EditMode.EDIT_RADIUS)
    window.resize(1100, 760)
    window.show()
    qtbot.waitExposed(window)

    drag_data_positions(
        window,
        Plane.XY,
        [(2.0, 2.0), (5.0, 2.0)],
        QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.KeyboardModifier.ControlModifier,
    )

    assert graph.nodes[node_id].radius == pytest.approx(3.0, abs=0.12)
    assert window.radius_preview is None
    assert window.undo_stack.count() == 1
    window._set_dirty(False)


def test_escape_cancels_radius_preview_and_overlay_controls_are_display_only(qtbot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    install_test_volume(window)
    assert window.project is not None
    graph = window.project.add_instance(0, instance_id="graph")
    node_id = graph.add_node((2, 2, 2), node_id="node")
    window.active_instance_id = graph.id
    window._refresh_timepoint()
    window.activate_tool(EditMode.EDIT_RADIUS)
    control = QtCore.Qt.KeyboardModifier.ControlModifier

    window._on_2d_drag(Plane.XY, PointerButton.LEFT, "start", 2, 2, control)
    window._on_2d_drag(Plane.XY, PointerButton.LEFT, "move", 4, 2, control)
    window.cancel_current_operation()

    assert graph.nodes[node_id].radius == 0.0
    assert window.radius_preview is None
    assert "radius drag preview" in window.statusBar().currentMessage()
    window.radius_display_buttons[RadiusDisplayMode.NONE].click()
    window.radius_opacity_slider.setValue(20)
    window.radius_sphere_display_buttons[RadiusDisplayMode.NONE].click()
    window.radius_sphere_opacity_slider.setValue(40)
    assert not window.dirty


def test_radius_display_modes_node_spheres_and_opacity_controls(qtbot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    install_test_volume(window)
    assert window.project is not None
    selected = window.project.add_instance(0, instance_id="selected", color="#123456")
    selected.add_node((2, 1, 1), node_id="selected-node", radius=2.0)
    other = window.project.add_instance(0, instance_id="other", color="#ff2200")
    other.add_node((2, 3, 4), node_id="other-node", radius=1.0)
    window.active_instance_id = selected.id
    window.cursor_zyx = (2.0, 1.0, 1.0)
    window._refresh_timepoint()
    xy = window.ortho_views[Plane.XY]

    assert window.tabs.tabText(OPACITY_TAB) == "Opacity"
    opacity_tab = window.tabs.widget(OPACITY_TAB)
    assert opacity_tab.isAncestorOf(window.mask_opacity_slider)
    assert opacity_tab.isAncestorOf(window.radius_opacity_slider)
    assert opacity_tab.isAncestorOf(
        window.radius_sphere_display_buttons[RadiusDisplayMode.ALL]
    )
    assert opacity_tab.isAncestorOf(window.radius_sphere_opacity_slider)
    assert opacity_tab.isAncestorOf(window.near_slice_opacity_range)

    assert window._radius_display_mode() is RadiusDisplayMode.ALL
    assert window._radius_sphere_display_mode() is RadiusDisplayMode.SELECTED
    assert tuple(xy.radius_item.image[1, 1, :3]) == (0x12, 0x34, 0x56)
    assert tuple(xy.radius_item.image[3, 4, :3]) == (0xFF, 0x22, 0x00)
    assert len(xy._radius_sphere_items) == 2
    assert [item.pen().color().name() for item in xy._radius_sphere_items] == [
        "#000000",
        "#ffffff",
    ]
    assert [item.pen().widthF() for item in xy._radius_sphere_items] == [4.5, 1.5]
    assert all(item.isVisible() for item in xy._radius_sphere_items)

    window.radius_display_buttons[RadiusDisplayMode.SELECTED].click()
    assert xy.radius_item.image[1, 1, 3] == 255
    assert xy.radius_item.image[3, 4, 3] == 0
    assert all(item.isVisible() for item in xy._radius_sphere_items)

    window.radius_sphere_display_buttons[RadiusDisplayMode.ALL].click()
    assert len(xy._radius_sphere_items) == 4
    assert [item.pen().color().name() for item in xy._radius_sphere_items] == [
        "#000000",
        "#ffffff",
        "#000000",
        "#ffffff",
    ]

    window.radius_sphere_display_buttons[RadiusDisplayMode.SELECTED].click()
    assert len(xy._radius_sphere_items) == 2
    assert all(item.isVisible() for item in xy._radius_sphere_items)

    window.radius_sphere_opacity_slider.setValue(42)
    assert all(item.opacity() == 0.42 for item in xy._radius_sphere_items)
    assert window.radius_sphere_opacity_label.text() == "42%"

    window.radius_display_buttons[RadiusDisplayMode.NONE].click()
    assert not xy.radius_item.isVisible()
    assert all(item.isVisible() for item in xy._radius_sphere_items)
    assert not window.radius_opacity_slider.isEnabled()
    assert window.radius_sphere_opacity_slider.isEnabled()

    window.radius_sphere_display_buttons[RadiusDisplayMode.NONE].click()
    assert not any(item.isVisible() for item in xy._radius_sphere_items)
    assert not window.radius_sphere_opacity_slider.isEnabled()
    assert not window.dirty


def test_intensity_radius_generation_is_rerunnable_and_undoable(qtbot) -> None:
    zz, yy, xx = np.ogrid[:21, :21, :21]
    frame = np.zeros((21, 21, 21), dtype=np.float32)
    frame[(zz - 10) ** 2 + (yy - 10) ** 2 + (xx - 10) ** 2 <= 4**2] = 100
    volume = VolumeData(
        path=Path("/tmp/intensity-radius.tif"),
        data=frame[np.newaxis, ...],
        source_axes="TZYX",
        axes_inferred=False,
    )
    project = new_project_for_volume(volume)
    graph = project.add_instance(0, instance_id="graph")
    node_id = graph.add_node((10, 10, 10), node_id="node")
    window = MainWindow()
    qtbot.addWidget(window)
    window._install_document(project, volume, project_path=None)
    window.active_instance_id = graph.id
    window.intensity_max_radius_spin.setValue(8)
    window.intensity_smoothing_spin.setValue(0)

    window.estimate_current_intensity_radii(active_only=True)

    assert graph.nodes[node_id].radius == pytest.approx(4.0, abs=0.6)
    first_radius = graph.nodes[node_id].radius
    window.intensity_threshold_spin.setValue(80)
    window.estimate_current_intensity_radii(active_only=True)
    assert graph.nodes[node_id].radius < first_radius
    window.undo()
    assert window.project.instances_at(0)["graph"].nodes[node_id].radius == first_radius
    window._set_dirty(False)


def test_create_extend_split_move_and_delete_graph(qtbot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    install_test_volume(window)

    window.begin_new_instance()
    window._on_2d_click(
        Plane.XY,
        PointerButton.LEFT,
        2.0,
        3.0,
        QtCore.Qt.KeyboardModifier.NoModifier,
    )
    graph = window._active_graph()
    assert graph is not None
    first = next(iter(graph.nodes))
    assert graph.nodes[first].position == (2.0, 3.0, 2.0)

    window._on_2d_click(
        Plane.XY,
        PointerButton.LEFT,
        4.0,
        1.0,
        QtCore.Qt.KeyboardModifier.NoModifier,
    )
    assert len(graph.nodes) == 2
    assert len(graph.edges) == 1
    edge = next(iter(graph.edges))

    window._set_mode(EditMode.SPLIT_EDGE)
    window._handle_edge_click(PickResult("edge", graph.id, edge=edge, fraction=0.25))
    assert len(graph.nodes) == 3
    assert len(graph.edges) == 2
    inserted = window.selected_node
    assert inserted is not None

    window.position_spins[0].setValue(1.5)
    window.position_spins[1].setValue(2.5)
    window.position_spins[2].setValue(3.5)
    window.apply_selected_node_position()
    assert graph.nodes[inserted[1]].position == (1.5, 2.5, 3.5)

    window._set_mode(EditMode.DELETE_NODE)
    window._handle_node_click(PickResult("node", graph.id, node_id=inserted[1]))
    assert inserted[1] not in graph.nodes
    assert not graph.edges
    window._set_dirty(False)


def test_many_to_many_lineage_scene_connections(qtbot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    install_test_volume(window)
    assert window.project is not None
    for time in (0, 1):
        window.project.add_instance(time, name=f"first-{time}")
        window.project.add_instance(time, name=f"second-{time}")

    sources = list(window.project.instances_at(0))
    targets = list(window.project.instances_at(1))
    window._connect_lineage_instances(0, sources[0], 1, targets[0])
    window._connect_lineage_instances(0, sources[1], 1, targets[0])
    window._connect_lineage_instances(0, sources[0], 1, targets[1])

    assert len(window.project.lineage_events) == 1
    event = window.project.lineage_events[0]
    assert len(event.sources) == 2
    assert len(event.targets) == 2
    assert event.label == "fission + fusion"
    window._set_dirty(False)


def test_lineage_active_highlight_tracks_current_frame_instance(qtbot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    install_test_volume(window)
    assert window.project is not None
    first = window.project.add_instance(0, name="first", instance_id="first")
    second = window.project.add_instance(0, name="second", instance_id="second")
    later = window.project.add_instance(1, name="later", instance_id="later")
    window.active_instance_id = first.id
    window._refresh_timepoint()

    assert window.lineage_view._instance_items[(0, first.id)]._active

    window.active_instance_id = second.id
    window._refresh_graphs()

    assert not window.lineage_view._instance_items[(0, first.id)]._active
    assert window.lineage_view._instance_items[(0, second.id)]._active

    window._activate_lineage_instance(1, later.id)

    assert window.time_index == 1
    assert window.active_instance_id == later.id
    assert window.lineage_view._instance_items[(1, later.id)]._active
    assert not window.lineage_view._instance_items[(0, second.id)]._active


def test_auto_match_adds_only_best_free_one_to_one_events(qtbot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    install_test_volume(window)
    assert window.project is not None
    for time, values in ((0, (("left", 1), ("right", 5))), (1, (("right", 5), ("left", 1)))):
        for label, x in values:
            instance_id = f"{label}-{time}"
            graph = window.project.add_instance(
                time,
                name=instance_id,
                instance_id=instance_id,
            )
            graph.add_node((1, 2, x), node_id=f"{instance_id}-node")
    window._refresh_timepoint()
    window.undo_stack.clear()
    window._set_dirty(False)

    window.auto_match_lineages()

    assert {(event.sources, event.targets) for event in window.project.lineage_events} == {
        (("left-0",), ("left-1",)),
        (("right-0",), ("right-1",)),
    }
    assert all(
        event.source_time == 0 and event.target_time == 1 for event in window.project.lineage_events
    )
    assert window.dirty
    window._set_dirty(False)


def test_clicking_node_recentres_all_slice_axes(qtbot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    install_test_volume(window)
    assert window.project is not None
    graph = window.project.add_instance(0)
    node_id = graph.add_node((1.0, 2.0, 4.0))

    window._navigate_to_pick(PickResult("node", graph.id, node_id=node_id))

    assert window.ortho_views[Plane.XY].slice_index == 1
    assert window.ortho_views[Plane.XZ].slice_index == 2
    assert window.ortho_views[Plane.YZ].slice_index == 4


def test_clicking_edge_recentres_on_interpolated_3d_position(qtbot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    install_test_volume(window)
    assert window.project is not None
    graph = window.project.add_instance(0)
    first = graph.add_node((0.0, 0.0, 0.0))
    second = graph.add_node((2.0, 4.0, 4.0))
    edge = graph.add_edge(first, second)

    window._navigate_to_pick(PickResult("edge", graph.id, edge=edge, fraction=0.5))

    assert window.cursor_zyx == (1.0, 2.0, 2.0)
    assert window.ortho_views[Plane.XY].slice_index == 1
    assert window.ortho_views[Plane.XZ].slice_index == 2
    assert window.ortho_views[Plane.YZ].slice_index == 2


def test_remove_edge_preserves_crosshair_slices_and_2d_view_positions(qtbot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    install_test_volume(window)
    assert window.project is not None
    graph = window.project.add_instance(0)
    first = graph.add_node((0, 0, 0))
    second = graph.add_node((3, 4, 5))
    edge = graph.add_edge(first, second)
    window.active_instance_id = graph.id
    window.tabs.setCurrentIndex(EDIT_TAB)
    window._set_mode(EditMode.DELETE_EDGE)
    window.cursor_zyx = (1.0, 1.0, 1.0)
    window._refresh_images()
    window._refresh_graphs()
    cursor_before = window.cursor_zyx
    slices_before = {plane: view.slice_index for plane, view in window.ortho_views.items()}
    ranges_before = {
        plane: np.asarray(view.view_box.viewRange(), dtype=float)
        for plane, view in window.ortho_views.items()
    }

    window._handle_edge_click(PickResult("edge", graph.id, edge=edge, fraction=0.75))

    assert edge not in graph.edges
    assert window.cursor_zyx == cursor_before
    assert {plane: view.slice_index for plane, view in window.ortho_views.items()} == slices_before
    for plane, view in window.ortho_views.items():
        assert np.allclose(view.view_box.viewRange(), ranges_before[plane])
    assert "without changing" in window.statusBar().currentMessage()
    window._set_dirty(False)


def test_right_click_selects_a_different_node_without_dragging(qtbot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    install_test_volume(window)
    assert window.project is not None
    graph = window.project.add_instance(0)
    first = graph.add_node((2.0, 1.0, 1.0))
    second = graph.add_node((2.0, 3.0, 4.0))
    window.active_instance_id = graph.id
    window.selected_node = (graph.id, first)
    window._refresh_timepoint()
    window.show()
    qtbot.waitExposed(window)

    click_data_position(qtbot, window, Plane.XY, 4.0, 3.0, QtCore.Qt.MouseButton.RightButton)

    assert window.selected_node == (graph.id, second)
    assert window.cursor_zyx == (2.0, 3.0, 4.0)


def test_right_clicking_2d_background_moves_crosshair_from_any_tab(qtbot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    install_test_volume(window)
    window.tabs.setCurrentIndex(2)
    window._on_slice_requested(Plane.XY, 1)
    window.show()
    qtbot.waitExposed(window)

    click_data_position(
        qtbot,
        window,
        Plane.XY,
        5.0,
        0.0,
        QtCore.Qt.MouseButton.RightButton,
    )

    # QTest clicks integer screen pixels; the data-to-screen transform differs
    # with platform font metrics and display scaling. Background navigation is
    # continuous, so allow one screen pixel rather than expecting voxel snapping.
    pixel_size = max(abs(value) for value in window.ortho_views[Plane.XY].view_box.viewPixelSize())
    assert window.cursor_zyx[0] == 1.0
    assert window.cursor_zyx[1:] == pytest.approx((0.0, 5.0), abs=pixel_size)
    assert window.ortho_views[Plane.XY].slice_index == 1
    assert window.ortho_views[Plane.XZ].slice_index == 0
    assert window.ortho_views[Plane.YZ].slice_index == 5


def test_ctrl_z_and_redo_shortcuts_restore_annotations_but_not_navigation(
    qtbot, monkeypatch
) -> None:
    discard_unsaved_on_close(monkeypatch)
    window = MainWindow()
    qtbot.addWidget(window)
    install_test_volume(window)
    assert window.project is not None
    project_before = window.project.to_dict()
    window.show()
    qtbot.waitExposed(window)
    window.activateWindow()
    window.setFocus()

    window.begin_new_instance()
    window._on_2d_click(
        Plane.XY,
        PointerButton.LEFT,
        2,
        3,
        QtCore.Qt.KeyboardModifier.NoModifier,
    )
    assert window.project is not None
    project_after = window.project.to_dict()
    assert project_after != project_before
    assert window.undo_stack.count() == 1
    assert window.dirty
    assert window.undo_action.isEnabled()

    window._set_mode(EditMode.DELETE_EDGE)
    window._centre_on((1.0, 2.0, 3.0))
    cursor_before = window.cursor_zyx
    slices_before = {plane: view.slice_index for plane, view in window.ortho_views.items()}
    ranges_before = {
        plane: np.asarray(view.view_box.viewRange(), dtype=float)
        for plane, view in window.ortho_views.items()
    }

    viewport = window.ortho_views[Plane.XY].plot_widget.viewport()
    viewport.setFocus()
    qtbot.keyClick(
        viewport,
        QtCore.Qt.Key.Key_Z,
        QtCore.Qt.KeyboardModifier.ControlModifier,
    )

    assert window.project.to_dict() == project_before
    assert window.cursor_zyx == cursor_before
    assert window.edit_mode is EditMode.DELETE_EDGE
    assert {plane: view.slice_index for plane, view in window.ortho_views.items()} == slices_before
    for plane, view in window.ortho_views.items():
        assert np.allclose(view.view_box.viewRange(), ranges_before[plane])
    assert not window.dirty
    assert window.redo_action.isEnabled()

    qtbot.keyClick(
        viewport,
        QtCore.Qt.Key.Key_Z,
        QtCore.Qt.KeyboardModifier.ControlModifier | QtCore.Qt.KeyboardModifier.ShiftModifier,
    )
    assert window.project.to_dict() == project_after
    assert window.cursor_zyx == cursor_before
    assert window.edit_mode is EditMode.DELETE_EDGE
    assert window.dirty

    qtbot.keyClick(
        viewport,
        QtCore.Qt.Key.Key_Z,
        QtCore.Qt.KeyboardModifier.ControlModifier,
    )
    qtbot.keyClick(
        viewport,
        QtCore.Qt.Key.Key_Y,
        QtCore.Qt.KeyboardModifier.ControlModifier,
    )
    assert window.project.to_dict() == project_after

    window._set_dirty(False)
    assert not window.dirty
    window.undo()
    assert window.dirty
    window.redo()
    assert not window.dirty

    install_test_volume(window)
    assert window.undo_stack.count() == 0
    assert not window.undo_action.isEnabled()
    assert not window.redo_action.isEnabled()
    assert not window.dirty


def test_undo_preserves_lineage_pan_and_zoom(qtbot, monkeypatch) -> None:
    discard_unsaved_on_close(monkeypatch)
    window = MainWindow()
    qtbot.addWidget(window)
    install_test_volume(window)
    assert window.project is not None
    graph = window.project.add_instance(0)
    graph.add_node((2, 1, 1))
    window.active_instance_id = graph.id
    window.tabs.setCurrentIndex(EDIT_TAB)
    window._set_mode(EditMode.ADD_ISOLATED)
    window._handle_empty_click(Plane.XY, 4.0, 3.0)
    assert window.undo_stack.count() == 1

    window.tabs.setCurrentIndex(LINEAGE_TAB)
    window._refresh_lineage_controls()
    window.lineage_view.reset_view()
    window.lineage_view.scale(1.4, 1.4)
    transform_before = window.lineage_view.transform()
    assert window.lineage_view._has_been_fitted

    window.undo()

    assert window.lineage_view.transform() == transform_before
    assert window.lineage_view._has_been_fitted
    assert window.tabs.currentIndex() == LINEAGE_TAB
    window._set_dirty(False)


def test_empty_space_right_drag_tracks_cursor_without_snapping_or_history(
    qtbot, monkeypatch
) -> None:
    discard_unsaved_on_close(monkeypatch)
    window = MainWindow()
    qtbot.addWidget(window)
    install_test_volume(window)
    assert window.project is not None
    graph = window.project.add_instance(0)
    node_id = graph.add_node((2.0, 2.0, 3.0))
    window.active_instance_id = graph.id
    window.selected_node = None
    window._refresh_timepoint()
    window._centre_on((2.0, 0.0, 0.0))
    window.undo_stack.clear()
    window._set_dirty(False)
    window.show()
    qtbot.waitExposed(window)

    view = window.ortho_views[Plane.XY]
    qtbot.wait(50)
    QtWidgets.QApplication.processEvents()
    range_before = np.asarray(view.view_box.viewRange(), dtype=float)
    start = data_viewport_position(view, 0.2, 0.2)
    near_node = data_viewport_position(view, 3.15, 2.15)
    finish = data_viewport_position(view, 5.0, 4.0)
    viewport = view.plot_widget.viewport()

    QtTest.QTest.mousePress(viewport, QtCore.Qt.MouseButton.RightButton, pos=start)
    QtTest.QTest.mouseMove(viewport, near_node, delay=35)
    QtWidgets.QApplication.processEvents()

    assert np.allclose(window.cursor_zyx, (2.0, 2.15, 3.15), atol=0.12)
    assert not np.allclose(window.cursor_zyx, graph.nodes[node_id].position, atol=0.03)
    assert window.selected_node is None
    assert window.undo_stack.count() == 0
    assert not window.dirty

    QtTest.QTest.mouseMove(viewport, finish, delay=35)
    QtWidgets.QApplication.processEvents()
    assert np.allclose(window.cursor_zyx, (2.0, 4.0, 5.0), atol=0.12)
    QtTest.QTest.mouseRelease(viewport, QtCore.Qt.MouseButton.RightButton, pos=finish)
    QtWidgets.QApplication.processEvents()

    assert window.cursor_drag_plane is None
    assert window.selected_node is None
    assert np.allclose(view.view_box.viewRange(), range_before)
    assert window.undo_stack.count() == 0

    cursor_before_node_drag = window.cursor_zyx
    drag_data_position(
        window,
        Plane.XY,
        3.0,
        2.0,
        0.0,
        4.0,
        QtCore.Qt.MouseButton.RightButton,
    )
    assert window.cursor_zyx == cursor_before_node_drag
    assert window.selected_node is None
    assert window.undo_stack.count() == 0


def test_node_drag_is_one_undo_step_and_does_not_restore_cursor_or_tool(qtbot, monkeypatch) -> None:
    discard_unsaved_on_close(monkeypatch)
    window = MainWindow()
    qtbot.addWidget(window)
    install_test_volume(window)
    assert window.project is not None
    graph = window.project.add_instance(0, instance_id="dragged")
    node_id = graph.add_node((2.0, 1.0, 1.0), node_id="node")
    window.active_instance_id = graph.id
    window.tabs.setCurrentIndex(EDIT_TAB)
    window._set_mode(EditMode.DRAG)
    window._refresh_timepoint()
    window.undo_stack.clear()
    window._set_dirty(False)
    project_before = window.project.to_dict()
    window.show()
    qtbot.waitExposed(window)

    drag_data_positions(
        window,
        Plane.XY,
        [(1.0, 1.0), (2.0, 2.0), (3.0, 2.5), (4.0, 3.0)],
        QtCore.Qt.MouseButton.LeftButton,
    )

    assert window.project is not None
    project_after = window.project.to_dict()
    assert project_after != project_before
    assert window.undo_stack.count() == 1
    assert window.undo_stack.undoText() == "Move node"
    assert np.allclose(
        window.project.instances_at(0)[graph.id].nodes[node_id].position,
        (2.0, 3.0, 4.0),
        atol=0.12,
    )

    window._set_mode(EditMode.ADD_ISOLATED)
    window._centre_on((0.0, 0.0, 0.0))
    window.undo()

    assert window.project.to_dict() == project_before
    assert window.cursor_zyx == (0.0, 0.0, 0.0)
    assert window.edit_mode is EditMode.ADD_ISOLATED
    assert not window.dirty

    window.redo()
    assert window.project.to_dict() == project_after
    assert window.cursor_zyx == (0.0, 0.0, 0.0)
    assert window.edit_mode is EditMode.ADD_ISOLATED
    assert window.dirty
    window._set_dirty(False)


def test_graph_topology_edits_undo_and_redo_in_order(qtbot, monkeypatch) -> None:
    discard_unsaved_on_close(monkeypatch)
    window = MainWindow()
    qtbot.addWidget(window)
    install_test_volume(window)
    assert window.project is not None
    graph = window.project.add_instance(0, instance_id="graph")
    first = graph.add_node((2, 1, 1), node_id="first")
    middle = graph.add_node((2, 2, 2), node_id="middle")
    last = graph.add_node((2, 3, 3), node_id="last")
    first_edge = graph.add_edge(first, middle)
    graph.add_edge(middle, last)
    window.active_instance_id = graph.id
    window.tabs.setCurrentIndex(EDIT_TAB)
    window._refresh_timepoint()
    window.undo_stack.clear()
    window._set_dirty(False)
    project_before = window.project.to_dict()

    window._set_mode(EditMode.ADD_CONNECTED)
    window._handle_node_click(PickResult("node", graph.id, node_id=last))
    window._handle_empty_click(Plane.XY, 4.0, 4.0)
    connected = window.selected_node
    assert connected is not None

    window._set_mode(EditMode.ADD_ISOLATED)
    window._handle_empty_click(Plane.XY, 5.0, 0.0)
    isolated = window.selected_node
    assert isolated is not None

    window._set_mode(EditMode.CONNECT)
    window._handle_node_click(PickResult("node", graph.id, node_id=first))
    window._handle_node_click(PickResult("node", graph.id, node_id=isolated[1]))

    window._set_mode(EditMode.SPLIT_EDGE)
    window._handle_edge_click(PickResult("edge", graph.id, edge=first_edge, fraction=0.5))
    inserted = window.selected_node
    assert inserted is not None

    window._set_mode(EditMode.DISSOLVE_NODE)
    window._handle_node_click(PickResult("node", graph.id, node_id=inserted[1]))

    window._set_mode(EditMode.DELETE_NODE)
    window._handle_node_click(PickResult("node", graph.id, node_id=connected[1]))

    assert window.project is not None
    project_after = window.project.to_dict()
    assert project_after != project_before
    assert window.undo_stack.count() == 6

    for _ in range(6):
        window.undo()
    assert window.project.to_dict() == project_before
    assert not window.dirty

    for _ in range(6):
        window.redo()
    assert window.project.to_dict() == project_after
    assert window.dirty
    window._set_dirty(False)


def test_connect_and_split_instance_edits_are_undoable(qtbot, monkeypatch) -> None:
    discard_unsaved_on_close(monkeypatch)
    window = MainWindow()
    qtbot.addWidget(window)
    install_test_volume(window)
    assert window.project is not None
    first_graph = window.project.add_instance(0, name="first", instance_id="first")
    second_graph = window.project.add_instance(0, name="second", instance_id="second")
    first_node = first_graph.add_node((2, 1, 1), node_id="first-node")
    second_node = second_graph.add_node((2, 3, 4), node_id="second-node")
    window.active_instance_id = first_graph.id
    window.tabs.setCurrentIndex(EDIT_TAB)
    window._refresh_timepoint()
    window.undo_stack.clear()
    window._set_dirty(False)
    before_connect = window.project.to_dict()

    window._set_mode(EditMode.CONNECT_INSTANCES)
    window._handle_node_click(PickResult("node", first_graph.id, node_id=first_node))
    window._handle_node_click(PickResult("node", second_graph.id, node_id=second_node))

    assert window.project is not None
    after_connect = window.project.to_dict()
    assert set(window.project.instances_at(0)) == {first_graph.id}
    assert window.undo_stack.undoText() == "Connect instances"
    window.undo()
    assert window.project.to_dict() == before_connect
    window.redo()
    assert window.project.to_dict() == after_connect

    merged = window.project.instances_at(0)[first_graph.id]
    bridge = next(iter(merged.edges))
    window._set_mode(EditMode.SPLIT_INSTANCE)
    window._handle_edge_click(PickResult("edge", merged.id, edge=bridge, fraction=0.5))

    after_split = window.project.to_dict()
    assert len(window.project.instances_at(0)) == 2
    assert window.undo_stack.undoText() == "Split instance"
    window.undo()
    assert window.project.to_dict() == after_connect
    window.redo()
    assert window.project.to_dict() == after_split
    window._set_dirty(False)


def test_instance_and_lineage_mutations_each_enter_undo_history(qtbot, monkeypatch) -> None:
    discard_unsaved_on_close(monkeypatch)
    window = MainWindow()
    qtbot.addWidget(window)
    install_test_volume(window)
    assert window.project is not None
    source = window.project.add_instance(0, name="source", instance_id="source")
    target = window.project.add_instance(1, name="target", instance_id="target")
    source.add_node((1, 1, 1), node_id="source-node")
    target.add_node((1, 2, 2), node_id="target-node")
    window.active_instance_id = source.id
    window._refresh_timepoint()
    window.undo_stack.clear()
    window._set_dirty(False)

    def assert_round_trip(label: str, operation) -> None:
        assert window.project is not None
        before = window.project.to_dict()
        count = window.undo_stack.count()
        operation()
        assert window.project is not None
        after = window.project.to_dict()
        assert after != before
        assert window.undo_stack.count() == count + 1
        assert window.undo_stack.undoText() == label
        window.undo()
        assert window.project is not None
        assert window.project.to_dict() == before
        window.redo()
        assert window.project.to_dict() == after

    def rename() -> None:
        window.instance_name_edit.setText("renamed source")
        window.rename_active_instance()

    assert_round_trip("Rename instance", rename)
    assert_round_trip("Copy instance", lambda: window.copy_active_instance(1))
    assert_round_trip(
        "Connect lineage instances",
        lambda: window._connect_lineage_instances(0, source.id, 1, target.id),
    )

    assert window.project is not None
    event_id = window.project.lineage_events[0].id
    assert_round_trip("Remove lineage event", lambda: window._remove_lineage_event(event_id))

    def mark_start() -> None:
        window._refresh_lineage_controls()
        window.lineage_view._instance_clicked(0, source.id)
        window.mark_selected_lineage_start()

    assert_round_trip("Mark lineage start", mark_start)

    def mark_end() -> None:
        window._refresh_lineage_controls()
        window.lineage_view._instance_clicked(1, target.id)
        window.mark_selected_lineage_end()

    assert_round_trip("Mark lineage end", mark_end)

    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *args, **kwargs: QtWidgets.QMessageBox.StandardButton.Yes,
    )
    window.active_instance_id = source.id
    assert_round_trip("Delete instance", window.delete_active_instance)
    window._set_dirty(False)


def test_right_selection_is_independent_from_creation_and_connection_sources(qtbot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    install_test_volume(window)
    assert window.project is not None
    graph = window.project.add_instance(0)
    first = graph.add_node((2, 1, 1))
    second = graph.add_node((2, 3, 4))
    window.active_instance_id = graph.id
    window.tabs.setCurrentIndex(EDIT_TAB)
    window._set_mode(EditMode.ADD_CONNECTED)
    window._refresh_timepoint()

    window._on_2d_click(
        Plane.XY,
        PointerButton.RIGHT,
        1,
        1,
        QtCore.Qt.KeyboardModifier.NoModifier,
    )
    assert window.selected_node == (graph.id, first)
    assert window.creation_anchor is None

    window._on_2d_click(
        Plane.XY,
        PointerButton.LEFT,
        1,
        1,
        QtCore.Qt.KeyboardModifier.NoModifier,
    )
    assert window.creation_anchor == (graph.id, first)
    assert window._action_source() == (graph.id, first)

    window._on_2d_click(
        Plane.XY,
        PointerButton.RIGHT,
        4,
        3,
        QtCore.Qt.KeyboardModifier.NoModifier,
    )
    assert window.selected_node == (graph.id, second)
    assert window.creation_anchor == (graph.id, first)

    window._set_mode(EditMode.CONNECT)
    window._on_2d_click(
        Plane.XY,
        PointerButton.LEFT,
        1,
        1,
        QtCore.Qt.KeyboardModifier.NoModifier,
    )
    assert window.connect_anchor == (graph.id, first)
    assert window.selected_node == (graph.id, first)
    assert window._action_source() == (graph.id, first)

    window.cancel_current_operation()
    assert window.connect_anchor is None
    assert window.creation_anchor is None
    assert window._action_source() is None

    window._set_mode(EditMode.DRAG)
    window.drag_target = (graph.id, first, Plane.XY)
    assert window._action_source() is None

    window._set_mode(EditMode.CONNECT_INSTANCES)
    window.connect_anchor = (graph.id, first)
    assert window._action_source() == (graph.id, first)


def test_escape_from_add_connected_prevents_accidental_isolated_node(qtbot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    install_test_volume(window)
    assert window.project is not None
    graph = window.project.add_instance(0)
    source = graph.add_node((2, 1, 1))
    window.active_instance_id = graph.id
    window.tabs.setCurrentIndex(EDIT_TAB)
    window._set_mode(EditMode.ADD_CONNECTED)
    window._refresh_timepoint()
    window._handle_node_click(PickResult("node", graph.id, node_id=source))
    assert window.creation_anchor == (graph.id, source)
    window.show()
    window.activateWindow()
    viewport = window.ortho_views[Plane.XY].plot_widget.viewport()
    viewport.setFocus()

    qtbot.keyClick(viewport, QtCore.Qt.Key.Key_Escape)

    assert window.edit_mode is EditMode.ADD_CONNECTED
    assert window.selected_node == (graph.id, source)
    assert window.creation_anchor is None

    window._handle_node_click(PickResult("node", graph.id, node_id=source))
    assert window.creation_anchor == (graph.id, source)
    window.setFocus()
    qtbot.keyClick(window, QtCore.Qt.Key.Key_Escape)
    assert window.creation_anchor is None

    node_count = len(graph.nodes)
    window._on_2d_click(
        Plane.XY,
        PointerButton.LEFT,
        5,
        4,
        QtCore.Qt.KeyboardModifier.NoModifier,
    )
    assert len(graph.nodes) == node_count
    assert "Choose a source node" in window.statusBar().currentMessage()

    window._handle_node_click(PickResult("node", graph.id, node_id=source))
    window._on_2d_click(
        Plane.XY,
        PointerButton.LEFT,
        5,
        4,
        QtCore.Qt.KeyboardModifier.NoModifier,
    )
    assert len(graph.nodes) == node_count + 1
    assert len(graph.edges) == 1
    window._set_dirty(False)


def test_escape_clears_staged_tools_new_instance_and_lineage_but_keeps_selection(qtbot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    install_test_volume(window)
    assert window.project is not None
    graph = window.project.add_instance(0, instance_id="first")
    other = window.project.add_instance(0, instance_id="other")
    first = graph.add_node((2, 1, 1))
    second = graph.add_node((2, 2, 2))
    other_node = other.add_node((2, 3, 4))
    window.active_instance_id = graph.id
    window.selected_node = (graph.id, first)
    window.tabs.setCurrentIndex(EDIT_TAB)
    window._refresh_timepoint()

    window._set_mode(EditMode.CONNECT)
    window._handle_node_click(PickResult("node", graph.id, node_id=first))
    assert window.connect_anchor == (graph.id, first)
    window.cancel_current_operation()
    assert window.connect_anchor is None
    assert window.edit_mode is EditMode.CONNECT
    assert window.selected_node == (graph.id, first)
    assert not graph.edges

    window._set_mode(EditMode.CONNECT_INSTANCES)
    window._handle_node_click(PickResult("node", graph.id, node_id=second))
    assert window.connect_anchor == (graph.id, second)
    window.cancel_current_operation()
    assert window.connect_anchor is None
    assert window.edit_mode is EditMode.CONNECT_INSTANCES
    assert set(window.project.instances_at(0)) == {graph.id, other.id}
    assert other_node in other.nodes

    window._set_mode(EditMode.DRAG)
    window.drag_target = (graph.id, first, Plane.XY)
    window.cancel_current_operation()
    assert window.drag_target is None
    assert window.edit_mode is EditMode.DRAG

    window._refresh_lineage_controls()
    window.lineage_view._instance_clicked(0, graph.id)
    assert window.lineage_view.selected_instance == (0, graph.id)
    window.cancel_current_operation()
    assert window.lineage_view.selected_instance is None
    assert window.selected_node == (graph.id, second)

    window.begin_new_instance()
    assert window.pending_new_instance
    window.cancel_current_operation()
    assert not window.pending_new_instance
    assert window.new_instance_button.text() == "Create new"
    assert window.selected_node == (graph.id, second)


def test_every_tool_help_explains_escape_and_escape_keeps_single_step_tools(qtbot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    install_test_volume(window)
    assert window.project is not None
    graph = window.project.add_instance(0)
    node_id = graph.add_node((2, 2, 2))
    window.active_instance_id = graph.id
    window.selected_node = (graph.id, node_id)
    window.tabs.setCurrentIndex(EDIT_TAB)
    window._refresh_timepoint()
    topology_before = graph.to_dict()

    for mode in EditMode:
        window._set_mode(mode)
        assert "Escape" in mode_help_text(mode)
        assert "Escape" in window.mode_help_label.text()
        selected_before = window.selected_node
        window.cancel_current_operation()
        assert window.edit_mode is mode
        assert window.selected_node == selected_before
        assert graph.to_dict() == topology_before


def test_left_click_reaches_destructive_node_tools_without_wiggle(qtbot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    install_test_volume(window)
    assert window.project is not None
    graph = window.project.add_instance(0)
    node_id = graph.add_node((2.0, 2.0, 3.0))
    window.active_instance_id = graph.id
    window.tabs.setCurrentIndex(EDIT_TAB)
    window._set_mode(EditMode.DELETE_NODE)
    window._refresh_timepoint()
    window.show()
    qtbot.waitExposed(window)

    click_data_position(qtbot, window, Plane.XY, 3.0, 2.0, QtCore.Qt.MouseButton.LeftButton)

    assert node_id not in graph.nodes
    window._set_dirty(False)


def test_2d_right_drag_does_not_zoom_and_middle_drag_pans(qtbot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    install_test_volume(window)
    window.show()
    qtbot.waitExposed(window)
    qtbot.wait(100)
    view = window.ortho_views[Plane.XY]
    viewport = view.plot_widget.viewport()
    start = viewport.rect().center()
    end = start + QtCore.QPoint(40, 25)

    before_right = np.asarray(view.view_box.viewRange(), dtype=float)
    QtTest.QTest.mousePress(viewport, QtCore.Qt.MouseButton.RightButton, pos=start)
    QtTest.QTest.mouseMove(viewport, end, delay=25)
    QtTest.QTest.mouseRelease(viewport, QtCore.Qt.MouseButton.RightButton, pos=end)
    after_right = np.asarray(view.view_box.viewRange(), dtype=float)
    assert np.allclose(before_right, after_right)

    before_middle = after_right
    QtTest.QTest.mousePress(viewport, QtCore.Qt.MouseButton.MiddleButton, pos=start)
    QtTest.QTest.mouseMove(viewport, end, delay=25)
    QtTest.QTest.mouseRelease(viewport, QtCore.Qt.MouseButton.MiddleButton, pos=end)
    after_middle = np.asarray(view.view_box.viewRange(), dtype=float)
    assert np.linalg.norm(after_middle.mean(axis=1) - before_middle.mean(axis=1)) > 0.1
    assert np.allclose(
        np.ptp(after_middle, axis=1),
        np.ptp(before_middle, axis=1),
        rtol=0.02,
        atol=0.1,
    )


def test_drag_tool_uses_left_drag_and_right_drag_never_moves_a_node(qtbot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    install_test_volume(window)
    assert window.project is not None
    graph = window.project.add_instance(0)
    node_id = graph.add_node((2, 1, 1))
    other_node = graph.add_node((2, 4, 0))
    window.active_instance_id = graph.id
    window.selected_node = (graph.id, other_node)
    window.tabs.setCurrentIndex(EDIT_TAB)
    window._set_mode(EditMode.DRAG)
    window._refresh_timepoint()
    window.show()
    qtbot.waitExposed(window)

    drag_data_position(
        window,
        Plane.XY,
        1,
        1,
        4,
        3,
        QtCore.Qt.MouseButton.RightButton,
    )
    assert graph.nodes[node_id].position == (2.0, 1.0, 1.0)
    assert window.selected_node == (graph.id, other_node)

    click_data_position(
        qtbot,
        window,
        Plane.XY,
        1,
        1,
        QtCore.Qt.MouseButton.LeftButton,
    )
    assert window.selected_node == (graph.id, node_id)
    assert graph.nodes[node_id].position == (2.0, 1.0, 1.0)

    window.selected_node = (graph.id, other_node)
    window._refresh_graphs()

    drag_data_position(
        window,
        Plane.XY,
        1,
        1,
        4,
        3,
        QtCore.Qt.MouseButton.LeftButton,
    )
    assert np.allclose(graph.nodes[node_id].position, (2.0, 3.0, 4.0), atol=0.08)
    assert window.selected_node == (graph.id, node_id)
    window._set_dirty(False)


def test_stationary_left_click_in_drag_tool_does_not_select_another_instance(qtbot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    install_test_volume(window)
    assert window.project is not None
    first = window.project.add_instance(0, instance_id="first")
    second = window.project.add_instance(0, instance_id="second")
    first_node = first.add_node((2, 1, 1))
    second_node = second.add_node((2, 3, 4))
    window.active_instance_id = first.id
    window.selected_node = (first.id, first_node)
    window.tabs.setCurrentIndex(EDIT_TAB)
    window._set_mode(EditMode.DRAG)
    window._refresh_timepoint()

    window._on_2d_click(
        Plane.XY,
        PointerButton.LEFT,
        4,
        3,
        QtCore.Qt.KeyboardModifier.NoModifier,
    )
    assert window.active_instance_id == first.id
    assert window.selected_node == (first.id, first_node)

    window._on_2d_click(
        Plane.XY,
        PointerButton.RIGHT,
        4,
        3,
        QtCore.Qt.KeyboardModifier.NoModifier,
    )
    assert window.active_instance_id == second.id
    assert window.selected_node == (second.id, second_node)


def test_connect_and_dissolve_tools_work_with_left_click_dispatch(qtbot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    install_test_volume(window)
    assert window.project is not None
    graph = window.project.add_instance(0)
    first = graph.add_node((2, 1, 1), node_id="a")
    middle = graph.add_node((2, 2, 2), node_id="b")
    last = graph.add_node((2, 3, 3), node_id="c")
    window.active_instance_id = graph.id
    window.tabs.setCurrentIndex(EDIT_TAB)
    window._set_mode(EditMode.CONNECT)

    window._on_2d_click(Plane.XY, PointerButton.LEFT, 1, 1, QtCore.Qt.KeyboardModifier.NoModifier)
    window._on_2d_click(Plane.XY, PointerButton.LEFT, 2, 2, QtCore.Qt.KeyboardModifier.NoModifier)
    window._on_2d_click(Plane.XY, PointerButton.LEFT, 3, 3, QtCore.Qt.KeyboardModifier.NoModifier)
    window._on_2d_click(Plane.XY, PointerButton.LEFT, 2, 2, QtCore.Qt.KeyboardModifier.NoModifier)
    assert graph.degree(middle) == 2

    window._set_mode(EditMode.DISSOLVE_NODE)
    window._on_2d_click(Plane.XY, PointerButton.LEFT, 2, 2, QtCore.Qt.KeyboardModifier.NoModifier)
    assert middle not in graph.nodes
    assert graph.edges == {tuple(sorted((first, last)))}
    window._set_dirty(False)


def test_edit_tool_is_inactive_on_instances_tab_but_right_navigation_still_works(qtbot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    install_test_volume(window)
    assert window.project is not None
    graph = window.project.add_instance(0)
    node_id = graph.add_node((2, 2, 2))
    window.active_instance_id = graph.id
    window._set_mode(EditMode.DELETE_NODE)
    window.tabs.setCurrentIndex(INSTANCE_TAB)
    window._refresh_timepoint()

    window._on_2d_click(Plane.XY, PointerButton.LEFT, 2, 2, QtCore.Qt.KeyboardModifier.NoModifier)
    assert node_id in graph.nodes
    window._on_2d_click(Plane.XY, PointerButton.RIGHT, 2, 2, QtCore.Qt.KeyboardModifier.NoModifier)
    assert window.selected_node == (graph.id, node_id)


def test_hover_only_highlights_targets_the_current_tool_can_use(qtbot, monkeypatch) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    install_test_volume(window)
    assert window.project is not None
    graph = window.project.add_instance(0)
    first = graph.add_node((2, 1, 1))
    middle = graph.add_node((2, 2, 2))
    last = graph.add_node((2, 3, 3))
    graph.add_edge(first, middle)
    graph.add_edge(middle, last)
    window.tabs.setCurrentIndex(EDIT_TAB)
    window._set_mode(EditMode.DISSOLVE_NODE)

    assert not window._hover_matches_tool(PickResult("node", graph.id, node_id=first))
    assert window._hover_matches_tool(PickResult("node", graph.id, node_id=middle))

    def fail_on_full_graph_refresh() -> None:
        raise AssertionError("Hover should only update highlight visuals")

    monkeypatch.setattr(window, "_refresh_graphs", fail_on_full_graph_refresh)
    window._on_view_hover(Plane.XY, PickResult("node", graph.id, node_id=middle))
    assert window.hovered_pick == PickResult("node", graph.id, node_id=middle)

    window.pending_new_instance = True
    assert not window._hover_matches_tool(PickResult("node", graph.id, node_id=middle))


def test_right_navigation_also_works_while_lineage_tab_is_open(qtbot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    install_test_volume(window)
    assert window.project is not None
    graph = window.project.add_instance(0)
    node_id = graph.add_node((2, 3, 4))
    window.active_instance_id = graph.id
    window.tabs.setCurrentIndex(2)
    window._refresh_timepoint()

    window._on_2d_click(Plane.XY, PointerButton.RIGHT, 4, 3, QtCore.Qt.KeyboardModifier.NoModifier)

    assert window.selected_node == (graph.id, node_id)
    assert window.cursor_zyx == (2.0, 3.0, 4.0)


def test_number_shortcut_switches_tool_and_opens_edit_tab(qtbot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    install_test_volume(window)
    window.tabs.setCurrentIndex(INSTANCE_TAB)
    window.show()
    window.activateWindow()
    window.setFocus()

    qtbot.keyClick(window, QtCore.Qt.Key.Key_4)

    assert window.edit_mode is list(EditMode)[3]
    assert window.tabs.currentIndex() == EDIT_TAB


def test_zero_shortcut_selects_tenth_tool(qtbot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    install_test_volume(window)
    window.tabs.setCurrentIndex(INSTANCE_TAB)
    window.show()
    window.activateWindow()
    window.setFocus()

    qtbot.keyClick(window, QtCore.Qt.Key.Key_0)

    assert len(EditMode) == 11
    assert window.edit_mode is EditMode.SPLIT_INSTANCE
    assert window.tabs.currentIndex() == EDIT_TAB


def test_r_shortcut_selects_radius_tool(qtbot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    install_test_volume(window)
    window.tabs.setCurrentIndex(INSTANCE_TAB)
    window.show()
    window.activateWindow()
    window.setFocus()

    qtbot.keyClick(window, QtCore.Qt.Key.Key_R)

    assert window.edit_mode is EditMode.EDIT_RADIUS
    assert window.tabs.currentIndex() == EDIT_TAB


def test_connect_instances_tool_updates_instance_and_lineage_views(qtbot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    install_test_volume(window)
    assert window.project is not None
    first = window.project.add_instance(0, name="first", instance_id="first")
    second = window.project.add_instance(0, name="second", instance_id="second")
    later_first = window.project.add_instance(1, instance_id="later-first")
    later_second = window.project.add_instance(1, instance_id="later-second")
    first_node = first.add_node((2, 1, 1), node_id="first-node")
    second_node = second.add_node((2, 3, 4), node_id="second-node")
    window.project.add_lineage_event(
        source_time=0,
        target_time=1,
        sources=(first.id,),
        targets=(later_first.id,),
    )
    window.project.add_lineage_event(
        source_time=0,
        target_time=1,
        sources=(second.id,),
        targets=(later_second.id,),
    )
    window.tabs.setCurrentIndex(EDIT_TAB)
    window._set_mode(EditMode.CONNECT_INSTANCES)
    window._refresh_timepoint()

    window._handle_node_click(PickResult("node", first.id, node_id=first_node))
    assert window.connect_anchor == (first.id, first_node)
    assert window._action_source() == (first.id, first_node)
    assert not window._hover_matches_tool(PickResult("node", first.id, node_id=first_node))
    assert window._hover_matches_tool(PickResult("node", second.id, node_id=second_node))
    window._handle_node_click(PickResult("node", second.id, node_id=second_node))

    assert set(window.project.instances_at(0)) == {first.id}
    assert len(first.nodes) == 2
    assert len(first.edges) == 1
    assert window.instance_list.count() == 1
    assert (0, second.id) not in window.lineage_view._instance_items
    outgoing = window.project.outgoing_event(0, first.id)
    assert outgoing is not None
    assert set(outgoing.targets) == {later_first.id, later_second.id}
    assert window.active_instance_id == first.id
    window._set_dirty(False)


def test_split_instance_tool_updates_ui_and_reports_non_splitting_edges(qtbot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    install_test_volume(window)
    assert window.project is not None
    graph = window.project.add_instance(0, name="path", instance_id="path")
    first = graph.add_node((2, 1, 1), node_id="a")
    middle = graph.add_node((2, 2, 2), node_id="b")
    last = graph.add_node((2, 3, 3), node_id="c")
    graph.add_edge(first, middle)
    bridge = graph.add_edge(middle, last)
    window.active_instance_id = graph.id
    window.tabs.setCurrentIndex(EDIT_TAB)
    window._set_mode(EditMode.SPLIT_INSTANCE)
    window._refresh_timepoint()
    bridge_pick = PickResult("edge", graph.id, edge=bridge, fraction=0.5)
    assert window._hover_matches_tool(bridge_pick)

    window._handle_edge_click(bridge_pick)

    assert len(window.project.instances_at(0)) == 2
    assert window.instance_list.count() == 2
    assert len([key for key in window.lineage_view._instance_items if key[0] == 0]) == 2

    loop = window.project.add_instance(0, name="loop")
    loop_a = loop.add_node((2, 0, 0), node_id="loop-a")
    loop_b = loop.add_node((2, 0, 1), node_id="loop-b")
    loop_c = loop.add_node((2, 1, 1), node_id="loop-c")
    loop_edge = loop.add_edge(loop_a, loop_b)
    loop.add_edge(loop_b, loop_c)
    loop.add_edge(loop_c, loop_a)
    window._refresh_timepoint()
    loop_pick = PickResult("edge", loop.id, edge=loop_edge, fraction=0.5)
    assert not window._hover_matches_tool(loop_pick)
    count_before = len(window.project.instances_at(0))

    window._handle_edge_click(loop_pick)

    assert len(window.project.instances_at(0)) == count_before
    assert "Instance not split" in window.statusBar().currentMessage()
    window._set_dirty(False)


def test_copy_buttons_copy_active_graph_to_adjacent_frame(qtbot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    install_test_volume(window)
    assert window.project is not None
    graph = window.project.add_instance(0, name="copy me")
    first = graph.add_node((1, 1, 1))
    second = graph.add_node((2, 2, 2))
    graph.add_edge(first, second)
    window.active_instance_id = graph.id
    window._refresh_timepoint()

    window.copy_active_instance(1)

    copies = list(window.project.instances_at(1).values())
    assert len(copies) == 1
    assert copies[0].name == "copy me"
    assert len(copies[0].nodes) == 2
    assert len(copies[0].edges) == 1
    window._set_dirty(False)


def click_data_position(
    qtbot,
    window: MainWindow,
    plane: Plane,
    horizontal: float,
    vertical: float,
    button: QtCore.Qt.MouseButton,
) -> None:
    view = window.ortho_views[plane]
    QtWidgets.QApplication.processEvents()
    viewport_position = data_viewport_position(view, horizontal, vertical)
    qtbot.mouseClick(view.plot_widget.viewport(), button, pos=viewport_position)
    QtWidgets.QApplication.processEvents()


def data_viewport_position(view, horizontal: float, vertical: float) -> QtCore.QPoint:
    scene_position = view.view_box.mapViewToScene(QtCore.QPointF(horizontal, vertical))
    viewport_position = view.plot_widget.mapFromScene(scene_position)
    if isinstance(viewport_position, QtCore.QPointF):
        return viewport_position.toPoint()
    return viewport_position


def discard_unsaved_on_close(monkeypatch) -> None:
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "warning",
        lambda *args, **kwargs: QtWidgets.QMessageBox.StandardButton.Discard,
    )


def drag_data_position(
    window: MainWindow,
    plane: Plane,
    start_horizontal: float,
    start_vertical: float,
    end_horizontal: float,
    end_vertical: float,
    button: QtCore.Qt.MouseButton,
    modifiers: QtCore.Qt.KeyboardModifier = QtCore.Qt.KeyboardModifier.NoModifier,
) -> None:
    drag_data_positions(
        window,
        plane,
        [(start_horizontal, start_vertical), (end_horizontal, end_vertical)],
        button,
        modifiers,
    )


def drag_data_positions(
    window: MainWindow,
    plane: Plane,
    positions: list[tuple[float, float]],
    button: QtCore.Qt.MouseButton,
    modifiers: QtCore.Qt.KeyboardModifier = QtCore.Qt.KeyboardModifier.NoModifier,
) -> None:
    assert len(positions) >= 2
    view = window.ortho_views[plane]
    QtWidgets.QApplication.processEvents()
    viewport_positions = [
        data_viewport_position(view, horizontal, vertical) for horizontal, vertical in positions
    ]
    start = viewport_positions[0]
    end = viewport_positions[-1]
    viewport = view.plot_widget.viewport()
    QtTest.QTest.mousePress(viewport, button, modifiers, start)
    for position in viewport_positions[1:]:
        QtTest.QTest.mouseMove(viewport, position, delay=20)
    QtTest.QTest.mouseRelease(viewport, button, modifiers, end)
    QtWidgets.QApplication.processEvents()
