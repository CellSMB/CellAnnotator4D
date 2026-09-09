from __future__ import annotations

import math
from enum import StrEnum
from pathlib import Path
from typing import Any

from PySide6 import QtCore, QtGui, QtWidgets

from .geometry import (
    Plane,
    clamp_position,
    extract_slice,
    interpolate_position,
    move_in_plane,
    project_position,
    unproject_position,
)
from .interaction import PickResult, PointerButton
from .io import (
    VolumeData,
    load_project,
    load_tiff,
    new_project_for_volume,
    robust_intensity_limits,
    save_project,
)
from .lineage_matching import suggest_maximal_one_to_one_matches
from .lineage_view import LineageView
from .mask_overlay import InstanceMaskColorizer, MaskFrameOverlay
from .mip_view import Mip3DView
from .model import Edge, GraphProject, InstanceGraph, canonical_edge
from .ortho_view import OrthoView
from .radius_estimation import estimate_intensity_radii
from .radius_overlay import RadiusOverlayRenderer
from .range_slider import RangeSlider


class EditMode(StrEnum):
    DRAG = "drag"
    ADD_CONNECTED = "add_connected"
    ADD_ISOLATED = "add_isolated"
    CONNECT = "connect"
    SPLIT_EDGE = "split_edge"
    DELETE_EDGE = "delete_edge"
    DELETE_NODE = "delete_node"
    DISSOLVE_NODE = "dissolve_node"
    CONNECT_INSTANCES = "connect_instances"
    SPLIT_INSTANCE = "split_instance"
    EDIT_RADIUS = "edit_radius"


class RadiusDisplayMode(StrEnum):
    ALL = "all"
    SELECTED = "selected"
    NONE = "none"


MODE_LABELS = {
    EditMode.DRAG: "Drag node",
    EditMode.ADD_CONNECTED: "Add connected node",
    EditMode.ADD_ISOLATED: "Add isolated node",
    EditMode.CONNECT: "Connect two nodes",
    EditMode.SPLIT_EDGE: "Split edge with node",
    EditMode.DELETE_EDGE: "Remove edge",
    EditMode.DELETE_NODE: "Remove node + edges",
    EditMode.DISSOLVE_NODE: "Dissolve degree-2 node",
    EditMode.CONNECT_INSTANCES: "Connect two instances",
    EditMode.SPLIT_INSTANCE: "Split instance at edge",
    EditMode.EDIT_RADIUS: "Edit node radius",
}

MODE_HELP = {
    EditMode.DRAG: (
        "Left click to select or left drag a node of the active instance in a 2D pane. "
        "Right click first to select a different instance."
    ),
    EditMode.ADD_CONNECTED: (
        "Left click a node to mark the magenta source, then click empty 2D space to extend it."
    ),
    EditMode.ADD_ISOLATED: "Click empty space to add a disconnected node.",
    EditMode.CONNECT: "Click the first node, then the second. Cycles are allowed.",
    EditMode.SPLIT_EDGE: "Click an edge to insert a node at that position.",
    EditMode.DELETE_EDGE: "Click an edge to remove it without removing its nodes.",
    EditMode.DELETE_NODE: "Click a node to remove it and all incident edges.",
    EditMode.DISSOLVE_NODE: "Click a degree-2 node to remove it and join its neighbours.",
    EditMode.CONNECT_INSTANCES: (
        "Click a node to mark the first instance's magenta source, then click a node "
        "in a different instance. The first instance keeps its identity and lineage."
    ),
    EditMode.SPLIT_INSTANCE: (
        "Click a bridge edge that divides one graph into exactly two connected components."
    ),
    EditMode.EDIT_RADIUS: (
        "Click a node to select it. Ctrl+left drag it in a 2D pane to preview a radius; "
        "the filled volume updates only when the button is released."
    ),
}

ESCAPE_MODE_HELP = {
    EditMode.DRAG: "stops an active node drag",
    EditMode.ADD_CONNECTED: (
        "clears the magenta source; empty-space clicks then wait for another source"
    ),
    EditMode.CONNECT: "clears the first connection node",
    EditMode.CONNECT_INSTANCES: "clears the first magenta instance connection node",
    EditMode.EDIT_RADIUS: "cancels an uncommitted radius drag preview",
}

INSTANCE_TAB = 0
EDIT_TAB = 1
LINEAGE_TAB = 2
OPACITY_TAB = 3
DEFAULT_MASK_OPACITY = 0.7
DEFAULT_RADIUS_OPACITY = 0.35
DEFAULT_RADIUS_SPHERE_OPACITY = 0.9


def tool_shortcut(number: int) -> str:
    if number == 10:
        return "0"
    if number == 11:
        return "R"
    return str(number)


def mode_help_text(mode: EditMode) -> str:
    mode_escape = ESCAPE_MODE_HELP.get(mode, "clears any staged action")
    return (
        f"{MODE_HELP[mode]}\n\nEscape {mode_escape}. It also cancels pending new-instance "
        "placement and lineage selection, clears hover, and keeps the active tool and "
        "right-click selection."
    )


ProjectSnapshot = dict[str, Any]


class _ProjectSnapshotCommand(QtGui.QUndoCommand):
    """Undo one already-applied project mutation using annotation-only snapshots."""

    def __init__(
        self,
        window: MainWindow,
        text: str,
        before: ProjectSnapshot,
        after: ProjectSnapshot,
    ) -> None:
        super().__init__(text)
        self._window = window
        self._before = before
        self._after = after
        self._first_redo = True

    def undo(self) -> None:
        self._window._restore_project_snapshot(self._before)
        self._window.statusBar().showMessage(f"Undid: {self.text()}", 4000)

    def redo(self) -> None:
        # QUndoStack.push() calls redo(), but the edit has already been applied.
        if self._first_redo:
            self._first_redo = False
            return
        self._window._restore_project_snapshot(self._after)
        self._window.statusBar().showMessage(f"Redid: {self.text()}", 4000)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, initial_path: str | Path | None = None) -> None:
        super().__init__()
        self.volume: VolumeData | None = None
        self.project: GraphProject | None = None
        self.project_path: Path | None = None
        self.time_index = 0
        self.cursor_zyx = (0.0, 0.0, 0.0)
        self.active_instance_id: str | None = None
        self.selected_node: tuple[str, str] | None = None
        self.selected_edge: tuple[str, Edge] | None = None
        self.creation_anchor: tuple[str, str] | None = None
        self.connect_anchor: tuple[str, str] | None = None
        self.drag_target: tuple[str, str, Plane] | None = None
        self.radius_drag_target: tuple[str, str, Plane] | None = None
        self.radius_preview: tuple[str, str, float] | None = None
        self.cursor_drag_plane: Plane | None = None
        self._node_drag_before: ProjectSnapshot | None = None
        self._radius_drag_before: ProjectSnapshot | None = None
        self.edit_mode = EditMode.DRAG
        self.hovered_pick: PickResult | None = None
        self.pending_new_instance = False
        self.dirty = False
        self.full_intensity_range = (0.0, 1.0)
        self.contrast_levels = (0.0, 1.0)
        self.mask_colorizer: InstanceMaskColorizer | None = None
        self.radius_renderer = RadiusOverlayRenderer()
        self._updating_controls = False
        self.undo_stack = QtGui.QUndoStack(self)
        self.undo_stack.setUndoLimit(100)
        self.undo_stack.cleanChanged.connect(self._undo_clean_changed)

        self.setWindowTitle("CellAnnotator4D")
        self.resize(1540, 980)
        self.setAcceptDrops(True)
        self._build_actions()
        self._build_ui()
        self._set_project_controls_enabled(False)
        self.statusBar().showMessage("Open a TZYX TIFF or CellAnnotator4D project")

        if initial_path is not None:
            QtCore.QTimer.singleShot(0, lambda: self.open_path(initial_path))

    def _build_actions(self) -> None:
        self.open_tiff_action = QtGui.QAction("Open TIFF…", self)
        self.open_tiff_action.setShortcut(QtGui.QKeySequence.StandardKey.Open)
        self.open_tiff_action.triggered.connect(self.choose_tiff)

        self.open_project_action = QtGui.QAction("Open project…", self)
        self.open_project_action.setShortcut("Ctrl+Shift+O")
        self.open_project_action.triggered.connect(self.choose_project)

        self.save_action = QtGui.QAction("Save", self)
        self.save_action.setShortcut(QtGui.QKeySequence.StandardKey.Save)
        self.save_action.triggered.connect(self.save)

        self.save_as_action = QtGui.QAction("Save as…", self)
        self.save_as_action.setShortcut(QtGui.QKeySequence.StandardKey.SaveAs)
        self.save_as_action.triggered.connect(self.save_as)

        self.undo_action = QtGui.QAction("&Undo", self)
        self.undo_action.setShortcuts([QtGui.QKeySequence("Ctrl+Z")])
        self.undo_action.setShortcutContext(QtCore.Qt.ShortcutContext.WindowShortcut)
        self.undo_action.setEnabled(False)
        self.undo_action.triggered.connect(self.undo)
        self.undo_stack.canUndoChanged.connect(self.undo_action.setEnabled)
        self.undo_stack.undoTextChanged.connect(self._undo_text_changed)
        self.addAction(self.undo_action)

        self.redo_action = QtGui.QAction("&Redo", self)
        self.redo_action.setShortcuts(
            [
                QtGui.QKeySequence("Ctrl+Y"),
                QtGui.QKeySequence("Ctrl+Shift+Z"),
            ]
        )
        self.redo_action.setShortcutContext(QtCore.Qt.ShortcutContext.WindowShortcut)
        self.redo_action.setEnabled(False)
        self.redo_action.triggered.connect(self.redo)
        self.undo_stack.canRedoChanged.connect(self.redo_action.setEnabled)
        self.undo_stack.redoTextChanged.connect(self._redo_text_changed)
        self.addAction(self.redo_action)

        self.quit_action = QtGui.QAction("Quit", self)
        self.quit_action.setShortcut(QtGui.QKeySequence.StandardKey.Quit)
        self.quit_action.triggered.connect(self.close)

        self.reset_3d_action = QtGui.QAction("Reset 3D camera", self)
        self.reset_3d_action.setShortcut("Ctrl+R")
        self.reset_3d_action.triggered.connect(lambda: self.mip_view.reset_camera())

        self.fit_2d_action = QtGui.QAction("Fit all 2D views", self)
        self.fit_2d_action.setShortcut("Ctrl+0")
        self.fit_2d_action.triggered.connect(self.fit_all_2d_views)

        self.escape_action = QtGui.QAction(self)
        self.escape_action.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Escape))
        self.escape_action.triggered.connect(self.cancel_current_operation)
        self.addAction(self.escape_action)

        self.tool_actions: dict[EditMode, QtGui.QAction] = {}
        for number, mode in enumerate(EditMode, start=1):
            action = QtGui.QAction(f"{number}. {MODE_LABELS[mode]}", self)
            action.setShortcut(tool_shortcut(number))
            action.setShortcutContext(
                QtCore.Qt.ShortcutContext.WidgetShortcut
                if mode is EditMode.EDIT_RADIUS
                else QtCore.Qt.ShortcutContext.WindowShortcut
            )
            action.triggered.connect(
                lambda checked=False, selected_mode=mode: self.activate_tool(selected_mode)
            )
            self.addAction(action)
            self.tool_actions[mode] = action

        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction(self.open_tiff_action)
        file_menu.addAction(self.open_project_action)
        file_menu.addSeparator()
        file_menu.addAction(self.save_action)
        file_menu.addAction(self.save_as_action)
        file_menu.addSeparator()
        file_menu.addAction(self.quit_action)

        edit_menu = self.menuBar().addMenu("&Edit")
        edit_menu.addAction(self.undo_action)
        edit_menu.addAction(self.redo_action)

        view_menu = self.menuBar().addMenu("&View")
        view_menu.addAction(self.fit_2d_action)
        view_menu.addAction(self.reset_3d_action)

    def _build_ui(self) -> None:
        central = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        central.setChildrenCollapsible(False)
        self.setCentralWidget(central)

        viewer_widget = QtWidgets.QWidget()
        viewer_layout = QtWidgets.QVBoxLayout(viewer_widget)
        viewer_layout.setContentsMargins(4, 4, 4, 4)
        viewer_layout.setSpacing(4)

        grid = QtWidgets.QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(4)
        self.ortho_views = {plane: OrthoView(plane) for plane in Plane}
        grid.addWidget(self.ortho_views[Plane.XY], 0, 0)
        grid.addWidget(self.ortho_views[Plane.XZ], 0, 1)
        grid.addWidget(self.ortho_views[Plane.YZ], 1, 0)
        self.mip_view = Mip3DView()
        grid.addWidget(self.mip_view, 1, 1)
        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        viewer_layout.addLayout(grid, 1)

        timeline_box = QtWidgets.QGroupBox("Time")
        timeline_layout = QtWidgets.QHBoxLayout(timeline_box)
        timeline_layout.addWidget(QtWidgets.QLabel("T"))
        self.time_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.time_slider.setRange(0, 0)
        timeline_layout.addWidget(self.time_slider, 1)
        self.time_spin = QtWidgets.QSpinBox()
        self.time_spin.setRange(0, 0)
        self.time_spin.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
        timeline_layout.addWidget(self.time_spin)
        self.time_count_label = QtWidgets.QLabel("/ 0")
        timeline_layout.addWidget(self.time_count_label)
        viewer_layout.addWidget(timeline_box)

        contrast_box = QtWidgets.QGroupBox("Intensity limits (2D slices and 3D MIP)")
        contrast_layout = QtWidgets.QGridLayout(contrast_box)
        self.minimum_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.maximum_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.minimum_slider.setRange(0, 1000)
        self.maximum_slider.setRange(0, 1000)
        self.minimum_value_label = QtWidgets.QLabel("0")
        self.maximum_value_label = QtWidgets.QLabel("1")
        self.minimum_value_label.setMinimumWidth(90)
        self.maximum_value_label.setMinimumWidth(90)
        contrast_layout.addWidget(QtWidgets.QLabel("Minimum"), 0, 0)
        contrast_layout.addWidget(self.minimum_slider, 0, 1)
        contrast_layout.addWidget(self.minimum_value_label, 0, 2)
        contrast_layout.addWidget(QtWidgets.QLabel("Maximum"), 1, 0)
        contrast_layout.addWidget(self.maximum_slider, 1, 1)
        contrast_layout.addWidget(self.maximum_value_label, 1, 2)
        viewer_layout.addWidget(contrast_box)

        central.addWidget(viewer_widget)
        side_panel = self._build_side_panel()
        central.addWidget(side_panel)
        central.setStretchFactor(0, 1)
        central.setStretchFactor(1, 0)
        central.setSizes([1180, 340])

        for view in self.ortho_views.values():
            view.pointer_clicked.connect(self._on_2d_click)
            view.pointer_dragged.connect(self._on_2d_drag)
            view.slice_requested.connect(self._on_slice_requested)
            view.hover_changed.connect(self._on_view_hover)
        self.mip_view.pointer_clicked.connect(self._on_3d_click)
        self.mip_view.hover_changed.connect(self._on_view_hover)
        self.time_slider.valueChanged.connect(self._time_slider_changed)
        self.time_spin.valueChanged.connect(self._time_spin_changed)
        self.minimum_slider.valueChanged.connect(self._contrast_slider_changed)
        self.maximum_slider.valueChanged.connect(self._contrast_slider_changed)
        self.mask_visibility_checkbox.toggled.connect(self._mask_visibility_changed)
        self.mask_opacity_slider.valueChanged.connect(self._mask_opacity_changed)
        self.radius_display_group.buttonClicked.connect(self._radius_display_mode_changed)
        self.radius_opacity_slider.valueChanged.connect(self._radius_opacity_changed)
        self.radius_sphere_display_group.buttonClicked.connect(
            self._radius_sphere_display_mode_changed
        )
        self.radius_sphere_opacity_slider.valueChanged.connect(self._radius_sphere_opacity_changed)
        self.near_slice_nodes_checkbox.toggled.connect(self._near_slice_node_emphasis_changed)
        self.near_slice_opacity_range.valuesChanged.connect(self._near_slice_opacity_range_changed)
        self.near_slice_near_distance_spin.valueChanged.connect(self._near_slice_distance_changed)
        self.near_slice_far_distance_spin.valueChanged.connect(self._near_slice_distance_changed)

    def _build_side_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        panel.setMinimumWidth(320)
        panel.setMaximumWidth(430)
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self._build_instances_tab(), "Instances")
        self.tabs.addTab(self._build_edit_tab(), "Edit graph")
        self.tabs.addTab(self._build_lineage_tab(), "Lineage")
        self.tabs.addTab(self._build_opacity_tab(), "Opacity")
        self.tabs.currentChanged.connect(self._tab_changed)
        layout.addWidget(self.tabs, 1)

        self.selection_label = QtWidgets.QLabel("No graph selection")
        self.selection_label.setWordWrap(True)
        self.selection_label.setStyleSheet(
            "background: #e9eef6; color: #263247; border-radius: 3px; padding: 5px;"
        )
        layout.addWidget(self.selection_label)
        return panel

    def _build_opacity_tab(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        outer_layout = QtWidgets.QVBoxLayout(tab)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        content = QtWidgets.QWidget()
        content.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )
        layout = QtWidgets.QVBoxLayout(content)
        scroll.setWidget(content)
        outer_layout.addWidget(scroll)

        intro = QtWidgets.QLabel(
            "Visibility and opacity controls are display-only and are not saved as "
            "annotation edits."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        mask_box = QtWidgets.QGroupBox("Instance mask overlay")
        mask_layout = QtWidgets.QGridLayout(mask_box)
        self.mask_visibility_checkbox = QtWidgets.QCheckBox("Show masks")
        self.mask_visibility_checkbox.setChecked(False)
        mask_layout.addWidget(self.mask_visibility_checkbox, 0, 0)
        self.mask_source_label = QtWidgets.QLabel("No companion mask found")
        self.mask_source_label.setWordWrap(True)
        self.mask_source_label.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        mask_layout.addWidget(self.mask_source_label, 0, 1, 1, 2)
        mask_layout.addWidget(QtWidgets.QLabel("Opacity"), 1, 0)
        self.mask_opacity_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.mask_opacity_slider.setRange(0, 100)
        self.mask_opacity_slider.setValue(round(DEFAULT_MASK_OPACITY * 100))
        mask_layout.addWidget(self.mask_opacity_slider, 1, 1)
        self.mask_opacity_label = QtWidgets.QLabel(f"{round(DEFAULT_MASK_OPACITY * 100)}%")
        self.mask_opacity_label.setMinimumWidth(42)
        mask_layout.addWidget(self.mask_opacity_label, 1, 2)
        layout.addWidget(mask_box)

        radius_box = QtWidgets.QGroupBox("Node radii")
        radius_layout = QtWidgets.QGridLayout(radius_box)
        radius_layout.addWidget(QtWidgets.QLabel("Connected fill"), 0, 0)
        radius_mode_row = QtWidgets.QHBoxLayout()
        radius_mode_row.setContentsMargins(0, 0, 0, 0)
        self.radius_display_group = QtWidgets.QButtonGroup(self)
        self.radius_display_group.setExclusive(True)
        self.radius_display_buttons: dict[RadiusDisplayMode, QtWidgets.QRadioButton] = {}
        for mode, label in (
            (RadiusDisplayMode.ALL, "All"),
            (RadiusDisplayMode.SELECTED, "Selected"),
            (RadiusDisplayMode.NONE, "None"),
        ):
            button = QtWidgets.QRadioButton(label)
            button.setProperty("radius_display_mode", mode.value)
            self.radius_display_group.addButton(button)
            self.radius_display_buttons[mode] = button
            radius_mode_row.addWidget(button)
        self.radius_display_buttons[RadiusDisplayMode.ALL].setChecked(True)
        radius_mode_row.addStretch(1)
        radius_layout.addLayout(radius_mode_row, 0, 1, 1, 2)
        radius_mode_help = QtWidgets.QLabel(
            "All shows every filled volume. Selected restricts the fill to the active "
            "mitochondrion. None hides the connected fill."
        )
        radius_mode_help.setWordWrap(True)
        radius_layout.addWidget(radius_mode_help, 1, 0, 1, 3)
        radius_layout.addWidget(QtWidgets.QLabel("Fill opacity"), 2, 0)
        self.radius_opacity_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.radius_opacity_slider.setRange(0, 100)
        self.radius_opacity_slider.setValue(round(DEFAULT_RADIUS_OPACITY * 100))
        radius_layout.addWidget(self.radius_opacity_slider, 2, 1)
        self.radius_opacity_label = QtWidgets.QLabel(f"{round(DEFAULT_RADIUS_OPACITY * 100)}%")
        self.radius_opacity_label.setMinimumWidth(42)
        radius_layout.addWidget(self.radius_opacity_label, 2, 2)

        radius_layout.addWidget(QtWidgets.QLabel("Node circles"), 3, 0)
        radius_sphere_mode_row = QtWidgets.QHBoxLayout()
        radius_sphere_mode_row.setContentsMargins(0, 0, 0, 0)
        self.radius_sphere_display_group = QtWidgets.QButtonGroup(self)
        self.radius_sphere_display_group.setExclusive(True)
        self.radius_sphere_display_buttons: dict[
            RadiusDisplayMode, QtWidgets.QRadioButton
        ] = {}
        for mode, label in (
            (RadiusDisplayMode.ALL, "All"),
            (RadiusDisplayMode.SELECTED, "Selected"),
            (RadiusDisplayMode.NONE, "None"),
        ):
            button = QtWidgets.QRadioButton(label)
            button.setProperty("radius_display_mode", mode.value)
            self.radius_sphere_display_group.addButton(button)
            self.radius_sphere_display_buttons[mode] = button
            radius_sphere_mode_row.addWidget(button)
        self.radius_sphere_display_buttons[RadiusDisplayMode.SELECTED].setChecked(True)
        radius_sphere_mode_row.addStretch(1)
        radius_layout.addLayout(radius_sphere_mode_row, 3, 1, 1, 2)
        radius_sphere_mode_help = QtWidgets.QLabel(
            "Draw each node radius independently as a black-backed white circle, "
            "without connecting neighbouring nodes."
        )
        radius_sphere_mode_help.setWordWrap(True)
        radius_layout.addWidget(radius_sphere_mode_help, 4, 0, 1, 3)
        radius_layout.addWidget(QtWidgets.QLabel("Circle opacity"), 5, 0)
        self.radius_sphere_opacity_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.radius_sphere_opacity_slider.setRange(0, 100)
        self.radius_sphere_opacity_slider.setValue(round(DEFAULT_RADIUS_SPHERE_OPACITY * 100))
        radius_layout.addWidget(self.radius_sphere_opacity_slider, 5, 1)
        self.radius_sphere_opacity_label = QtWidgets.QLabel(
            f"{round(DEFAULT_RADIUS_SPHERE_OPACITY * 100)}%"
        )
        self.radius_sphere_opacity_label.setMinimumWidth(42)
        radius_layout.addWidget(self.radius_sphere_opacity_label, 5, 2)
        layout.addWidget(radius_box)

        graph_display_box = QtWidgets.QGroupBox("2D graph display")
        graph_display_layout = QtWidgets.QVBoxLayout(graph_display_box)
        self.near_slice_nodes_checkbox = QtWidgets.QCheckBox(
            "Adjust node opacity by slice distance"
        )
        self.near_slice_nodes_checkbox.setToolTip(
            "Use the absolute node opacities and distance thresholds configured below."
        )
        graph_display_layout.addWidget(self.near_slice_nodes_checkbox)

        self.near_slice_settings_widget = QtWidgets.QWidget()
        near_slice_settings_layout = QtWidgets.QGridLayout(self.near_slice_settings_widget)
        near_slice_settings_layout.setContentsMargins(0, 0, 0, 0)
        near_slice_settings_layout.addWidget(QtWidgets.QLabel("Opacity"), 0, 0)
        self.near_slice_opacity_range = RangeSlider()
        self.near_slice_opacity_range.setRange(0, 100)
        self.near_slice_opacity_range.setValues(0, 100)
        self.near_slice_opacity_range.setAccessibleName(
            "Minimum and maximum near-slice node opacity"
        )
        self.near_slice_opacity_range.setToolTip(
            "Absolute lower and upper node opacity. 0% is invisible; 100% is fully opaque."
        )
        near_slice_settings_layout.addWidget(self.near_slice_opacity_range, 0, 1)
        self.near_slice_opacity_range_label = QtWidgets.QLabel("0–100%")
        self.near_slice_opacity_range_label.setMinimumWidth(58)
        near_slice_settings_layout.addWidget(
            self.near_slice_opacity_range_label,
            0,
            2,
        )

        distance_row = QtWidgets.QHBoxLayout()
        distance_row.setContentsMargins(0, 0, 0, 0)
        self.near_slice_near_distance_label = QtWidgets.QLabel("100% at ≤")
        distance_row.addWidget(self.near_slice_near_distance_label)
        self.near_slice_near_distance_spin = self._distance_threshold_spin(0.0)
        self.near_slice_near_distance_spin.setToolTip(
            "Nodes at or closer than this distance use the upper opacity."
        )
        distance_row.addWidget(self.near_slice_near_distance_spin)
        self.near_slice_far_distance_label = QtWidgets.QLabel("0% at ≥")
        distance_row.addWidget(self.near_slice_far_distance_label)
        self.near_slice_far_distance_spin = self._distance_threshold_spin(4.0)
        self.near_slice_far_distance_spin.setToolTip(
            "Nodes at or farther than this distance use the lower opacity. "
            "Opacity changes linearly between the two distances."
        )
        distance_row.addWidget(self.near_slice_far_distance_spin)
        distance_row.addWidget(QtWidgets.QLabel("slices"))
        near_slice_settings_layout.addLayout(distance_row, 1, 0, 1, 3)
        graph_display_layout.addWidget(self.near_slice_settings_widget)
        layout.addWidget(graph_display_box)
        layout.addStretch(1)
        return tab

    @staticmethod
    def _distance_threshold_spin(value: float) -> QtWidgets.QDoubleSpinBox:
        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(0.0, 9999.0)
        spin.setDecimals(1)
        spin.setSingleStep(0.5)
        spin.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
        spin.setValue(value)
        spin.setFixedWidth(52)
        return spin

    def _build_instances_tab(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        intro = QtWidgets.QLabel(
            "Each item is one arbitrary 3D graph at the current timepoint. "
            "All items remain visible; the active one is highlighted."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self.instance_list = QtWidgets.QListWidget()
        self.instance_list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        layout.addWidget(self.instance_list, 1)

        button_row = QtWidgets.QHBoxLayout()
        self.new_instance_button = QtWidgets.QPushButton("Create new")
        self.delete_instance_button = QtWidgets.QPushButton("Delete")
        button_row.addWidget(self.new_instance_button)
        button_row.addWidget(self.delete_instance_button)
        layout.addLayout(button_row)

        copy_row = QtWidgets.QHBoxLayout()
        self.copy_previous_button = QtWidgets.QPushButton("Copy to previous")
        self.copy_next_button = QtWidgets.QPushButton("Copy to next")
        copy_row.addWidget(self.copy_previous_button)
        copy_row.addWidget(self.copy_next_button)
        layout.addLayout(copy_row)

        rename_form = QtWidgets.QFormLayout()
        self.instance_name_edit = QtWidgets.QLineEdit()
        self.instance_name_edit.setPlaceholderText("Active instance name")
        rename_form.addRow("Name", self.instance_name_edit)
        layout.addLayout(rename_form)

        self.instance_summary = QtWidgets.QLabel()
        self.instance_summary.setWordWrap(True)
        layout.addWidget(self.instance_summary)

        self.instance_list.currentItemChanged.connect(self._instance_selection_changed)
        self.new_instance_button.clicked.connect(self.begin_new_instance)
        self.delete_instance_button.clicked.connect(self.delete_active_instance)
        self.copy_previous_button.clicked.connect(lambda: self.copy_active_instance(-1))
        self.copy_next_button.clicked.connect(lambda: self.copy_active_instance(1))
        self.instance_name_edit.editingFinished.connect(self.rename_active_instance)
        return tab

    def _build_edit_tab(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        outer_layout = QtWidgets.QVBoxLayout(tab)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        content = QtWidgets.QWidget()
        content.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )
        layout = QtWidgets.QVBoxLayout(content)
        scroll.setWidget(content)
        outer_layout.addWidget(scroll)
        self.mode_group = QtWidgets.QButtonGroup(self)
        self.mode_group.setExclusive(True)
        for number, mode in enumerate(EditMode, start=1):
            button = QtWidgets.QToolButton()
            button.setText(f"{number}. {MODE_LABELS[mode]}")
            button.setCheckable(True)
            button.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextOnly)
            button.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Expanding,
                QtWidgets.QSizePolicy.Policy.Fixed,
            )
            self.mode_group.addButton(button, number)
            button.setProperty("edit_mode", mode.value)
            shortcut = tool_shortcut(number)
            button.setToolTip(f"Keyboard shortcut: {shortcut}\n{mode_help_text(mode)}")
            layout.addWidget(button)
            if mode is EditMode.DRAG:
                button.setChecked(True)
        self.mode_group.buttonClicked.connect(self._mode_button_clicked)

        self.mode_help_label = QtWidgets.QLabel(mode_help_text(EditMode.DRAG))
        self.mode_help_label.setWordWrap(True)
        self.mode_help_label.setStyleSheet(
            "background: #202733; color: #f4f7ff; border-radius: 3px; padding: 7px;"
        )
        layout.addWidget(self.mode_help_label)

        navigation_help = QtWidgets.QLabel(
            "Always available: right click selects/recentres; right click empty 2D space "
            "moves the crosshair; right drag from empty 2D space moves it continuously "
            "without snapping to nodes; middle drag pans; wheel zooms; Shift+wheel changes "
            "slice."
        )
        navigation_help.setWordWrap(True)
        navigation_help.setStyleSheet(
            "background: #e9eef6; color: #263247; border-radius: 3px; padding: 6px;"
        )
        layout.addWidget(navigation_help)

        state_legend = QtWidgets.QLabel(
            "<b>Node rings:</b> "
            "<span style='color:#0089a8'><b>cyan</b> = selected</span> · "
            "<span style='color:#c0009d'><b>magenta</b> = tool 2/4/9 source</span> · "
            "<span style='color:#3b8f00'><b>lime</b> = hovered</span>"
        )
        state_legend.setWordWrap(True)
        state_legend.setStyleSheet(
            "background: #f5f7fb; color: #263247; border: 1px solid #ccd5e3; "
            "border-radius: 3px; padding: 5px;"
        )
        layout.addWidget(state_legend)

        coordinate_box = QtWidgets.QGroupBox("Selected node position (ZYX)")
        coordinate_form = QtWidgets.QFormLayout(coordinate_box)
        self.position_spins: list[QtWidgets.QDoubleSpinBox] = []
        for axis in "ZYX":
            spin = QtWidgets.QDoubleSpinBox()
            spin.setDecimals(2)
            spin.setRange(0.0, 0.0)
            spin.setSingleStep(0.25)
            coordinate_form.addRow(axis, spin)
            self.position_spins.append(spin)
        self.apply_position_button = QtWidgets.QPushButton("Apply position")
        coordinate_form.addRow(self.apply_position_button)
        self.selected_radius_label = QtWidgets.QLabel("—")
        self.selected_radius_label.setToolTip(
            "Radius is stored in voxel-coordinate units. Use tool 11 / R to change it."
        )
        coordinate_form.addRow("Radius", self.selected_radius_label)
        layout.addWidget(coordinate_box)

        intensity_box = QtWidgets.QGroupBox("Estimate radii from intensity")
        intensity_layout = QtWidgets.QFormLayout(intensity_box)
        self.intensity_threshold_spin = QtWidgets.QSpinBox()
        self.intensity_threshold_spin.setRange(5, 95)
        self.intensity_threshold_spin.setValue(50)
        self.intensity_threshold_spin.setSuffix("%")
        self.intensity_threshold_spin.setToolTip(
            "Boundary intensity between local background (0%) and the bright centre (100%). "
            "Higher values produce smaller radii."
        )
        intensity_layout.addRow("Boundary level", self.intensity_threshold_spin)
        self.intensity_max_radius_spin = QtWidgets.QDoubleSpinBox()
        self.intensity_max_radius_spin.setRange(0.5, 9999.0)
        self.intensity_max_radius_spin.setValue(15.0)
        self.intensity_max_radius_spin.setDecimals(1)
        self.intensity_max_radius_spin.setSingleStep(1.0)
        self.intensity_max_radius_spin.setSuffix(" vox")
        intensity_layout.addRow("Maximum radius", self.intensity_max_radius_spin)
        self.intensity_ray_percentile_spin = QtWidgets.QSpinBox()
        self.intensity_ray_percentile_spin.setRange(5, 95)
        self.intensity_ray_percentile_spin.setValue(10)
        self.intensity_ray_percentile_spin.setSuffix("%")
        self.intensity_ray_percentile_spin.setToolTip(
            "Percentile of radial boundary crossings. Lower values follow the narrowest "
            "cross-section and resist bright signal along the tube axis."
        )
        intensity_layout.addRow("Ray percentile", self.intensity_ray_percentile_spin)
        self.intensity_smoothing_spin = QtWidgets.QDoubleSpinBox()
        self.intensity_smoothing_spin.setRange(0.0, 5.0)
        self.intensity_smoothing_spin.setValue(0.75)
        self.intensity_smoothing_spin.setDecimals(2)
        self.intensity_smoothing_spin.setSingleStep(0.25)
        self.intensity_smoothing_spin.setSuffix(" vox")
        intensity_layout.addRow("Radial smoothing", self.intensity_smoothing_spin)
        intensity_buttons = QtWidgets.QVBoxLayout()
        self.estimate_active_radii_button = QtWidgets.QPushButton("Active instance")
        self.estimate_frame_radii_button = QtWidgets.QPushButton("All at current T")
        intensity_buttons.addWidget(self.estimate_active_radii_button)
        intensity_buttons.addWidget(self.estimate_frame_radii_button)
        intensity_layout.addRow(intensity_buttons)
        layout.addWidget(intensity_box)
        layout.addStretch(1)

        self.apply_position_button.clicked.connect(self.apply_selected_node_position)
        self.estimate_active_radii_button.clicked.connect(
            lambda: self.estimate_current_intensity_radii(active_only=True)
        )
        self.estimate_frame_radii_button.clicked.connect(
            lambda: self.estimate_current_intensity_radii(active_only=False)
        )
        return tab

    def _build_lineage_tab(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        intro = QtWidgets.QLabel(
            "Time increases downward. Left click one instance, then an instance on another "
            "row to connect them. Connections merge into fission, fusion, or n–n events. "
            "Cyan marks the active instance; yellow marks a connection source. "
            "Click an edge to remove its event; Escape cancels selection."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self.lineage_view = LineageView()
        layout.addWidget(self.lineage_view, 1)
        endpoint_row = QtWidgets.QHBoxLayout()
        self.mark_start_button = QtWidgets.QPushButton("Mark START")
        self.mark_end_button = QtWidgets.QPushButton("Mark END")
        self.reset_lineage_view_button = QtWidgets.QPushButton("Fit view")
        endpoint_row.addWidget(self.mark_start_button)
        endpoint_row.addWidget(self.mark_end_button)
        endpoint_row.addWidget(self.reset_lineage_view_button)
        layout.addLayout(endpoint_row)

        automation_row = QtWidgets.QHBoxLayout()
        self.arrange_lineage_button = QtWidgets.QPushButton("Arrange")
        self.reset_lineage_order_button = QtWidgets.QPushButton("Default order")
        self.auto_match_lineage_button = QtWidgets.QPushButton("Auto 1→1")
        self.arrange_lineage_button.setToolTip(
            "Reorder each time row to reduce crossings and align connected lineages."
        )
        self.reset_lineage_order_button.setToolTip(
            "Restore each time row to the project's original instance order."
        )
        self.auto_match_lineage_button.setToolTip(
            "Globally match free endpoints between each pair of adjacent frames using "
            "centerline position and shape. Existing events and START/END markers are "
            "left unchanged."
        )
        automation_row.addWidget(self.arrange_lineage_button)
        automation_row.addWidget(self.reset_lineage_order_button)
        automation_row.addWidget(self.auto_match_lineage_button)
        layout.addLayout(automation_row)

        self.lineage_selection_label = QtWidgets.QLabel("No lineage instance selected")
        self.lineage_selection_label.setStyleSheet(
            "background: #202733; color: #f4f7ff; border-radius: 3px; padding: 6px;"
        )
        self.lineage_selection_label.setWordWrap(True)
        layout.addWidget(self.lineage_selection_label)

        self.lineage_view.connection_requested.connect(self._connect_lineage_instances)
        self.lineage_view.event_remove_requested.connect(self._remove_lineage_event)
        self.lineage_view.instance_activated.connect(self._activate_lineage_instance)
        self.lineage_view.selection_changed.connect(self._lineage_selection_changed)
        self.lineage_view.message.connect(
            lambda message: self.statusBar().showMessage(message, 7000)
        )
        self.mark_start_button.clicked.connect(self.mark_selected_lineage_start)
        self.mark_end_button.clicked.connect(self.mark_selected_lineage_end)
        self.reset_lineage_view_button.clicked.connect(self.lineage_view.reset_view)
        self.arrange_lineage_button.clicked.connect(self.lineage_view.arrange_by_connections)
        self.reset_lineage_order_button.clicked.connect(self.lineage_view.reset_instance_order)
        self.auto_match_lineage_button.clicked.connect(self.auto_match_lineages)
        return tab

    def _set_project_controls_enabled(self, enabled: bool) -> None:
        self.tabs.setEnabled(enabled)
        self.time_slider.setEnabled(enabled)
        self.time_spin.setEnabled(enabled)
        self.minimum_slider.setEnabled(enabled)
        self.maximum_slider.setEnabled(enabled)
        self.save_action.setEnabled(enabled)
        self.save_as_action.setEnabled(enabled)
        self.reset_3d_action.setEnabled(enabled)
        self.fit_2d_action.setEnabled(enabled)
        masks_available = enabled and self.volume is not None and self.volume.masks is not None
        self.mask_visibility_checkbox.setEnabled(masks_available)
        self.mask_opacity_slider.setEnabled(
            masks_available and self.mask_visibility_checkbox.isChecked()
        )
        for button in self.radius_display_buttons.values():
            button.setEnabled(enabled)
        self.radius_opacity_slider.setEnabled(
            enabled and self._radius_display_mode() is not RadiusDisplayMode.NONE
        )
        for button in self.radius_sphere_display_buttons.values():
            button.setEnabled(enabled)
        self.radius_sphere_opacity_slider.setEnabled(
            enabled and self._radius_sphere_display_mode() is not RadiusDisplayMode.NONE
        )
        self.near_slice_nodes_checkbox.setEnabled(enabled)
        self.near_slice_settings_widget.setEnabled(
            enabled and self.near_slice_nodes_checkbox.isChecked()
        )
        self.estimate_active_radii_button.setEnabled(enabled)
        self.estimate_frame_radii_button.setEnabled(enabled)

    def choose_tiff(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open TZYX TIFF",
            "",
            "TIFF images (*.tif *.tiff);;All files (*)",
        )
        if path:
            self.open_tiff(path)

    def choose_project(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open CellAnnotator4D project",
            "",
            "CellAnnotator4D projects (*.cpv.json *.json);;All files (*)",
        )
        if path:
            self.open_annotation_project(path)

    def open_path(self, path: str | Path) -> None:
        path = Path(path)
        if path.name.endswith(".cpv.json") or path.suffix.lower() == ".json":
            self.open_annotation_project(path)
        else:
            self.open_tiff(path)

    def open_tiff(self, path: str | Path) -> None:
        if not self._discard_or_save_changes():
            return
        try:
            with _wait_cursor():
                volume = load_tiff(path)
        except Exception as error:
            self._show_error("Could not open TIFF", error)
            return
        project = new_project_for_volume(volume)
        self._install_document(project, volume, project_path=None)
        inference = " (axes inferred from dimensionality)" if volume.axes_inferred else ""
        self.statusBar().showMessage(
            f"Opened {volume.path.name}: {volume.shape_tzyx} TZYX, "
            f"source axes {volume.source_axes or 'unknown'}{inference}"
            f"{self._mask_status_suffix(volume)}",
            12000,
        )

    def open_annotation_project(self, path: str | Path) -> None:
        if not self._discard_or_save_changes():
            return
        try:
            with _wait_cursor():
                project, volume = load_project(path)
        except Exception as error:
            self._show_error("Could not open project", error)
            return
        self._install_document(project, volume, project_path=Path(path).resolve())
        self.statusBar().showMessage(
            f"Opened project {Path(path).name}{self._mask_status_suffix(volume)}",
            12000,
        )

    def _install_document(
        self,
        project: GraphProject,
        volume: VolumeData,
        *,
        project_path: Path | None,
    ) -> None:
        self.project = project
        self.volume = volume
        self.project_path = project_path
        self.mask_colorizer = (
            InstanceMaskColorizer(volume.masks.data, seed=volume.masks.path.name)
            if volume.masks is not None
            else None
        )
        visibility_blocker = QtCore.QSignalBlocker(self.mask_visibility_checkbox)
        self.mask_visibility_checkbox.setChecked(volume.masks is not None)
        del visibility_blocker
        if volume.masks is not None:
            self.mask_source_label.setText(volume.masks.path.name)
            self.mask_source_label.setToolTip(str(volume.masks.path))
        elif volume.mask_error is not None:
            self.mask_source_label.setText("Companion mask could not be loaded")
            self.mask_source_label.setToolTip(volume.mask_error)
        else:
            self.mask_source_label.setText("No companion mask found")
            self.mask_source_label.setToolTip(
                "Looked beside the source TIFF for _cp_masks, _masks, and _mask files."
            )
        self.mip_view.clear_mask_overlay()
        self.mip_view.clear_radius_overlay()
        self.mip_view.clear_radius_spheres()
        self.radius_renderer.clear()
        for view in self.ortho_views.values():
            view.clear_mask_overlay()
            view.clear_radius_overlay()
            view.clear_radius_spheres()
        self.time_index = 0
        z_size, y_size, x_size = project.shape_tzyx[1:]
        self.cursor_zyx = (
            (z_size - 1) / 2,
            (y_size - 1) / 2,
            (x_size - 1) / 2,
        )
        self.active_instance_id = None
        self.selected_node = None
        self.selected_edge = None
        self.creation_anchor = None
        self.connect_anchor = None
        self.drag_target = None
        self.radius_drag_target = None
        self.radius_preview = None
        self.cursor_drag_plane = None
        self._node_drag_before = None
        self._radius_drag_before = None
        self.hovered_pick = None
        self.pending_new_instance = False
        self.undo_stack.clear()
        self._set_mode(EditMode.DRAG)
        full_min, full_max, low, high = robust_intensity_limits(volume.data)
        self.full_intensity_range = (full_min, full_max)
        self.contrast_levels = (low, high)
        self._configure_dimensions()
        self._set_contrast_sliders_from_levels()
        self._set_project_controls_enabled(True)
        self._refresh_timepoint(reset_camera=True)
        self._set_dirty(False)

    @staticmethod
    def _mask_status_suffix(volume: VolumeData) -> str:
        if volume.masks is not None:
            return f"; masks {volume.masks.path.name}"
        if volume.mask_error is not None:
            return f"; companion mask ignored ({volume.mask_error})"
        return ""

    def _project_snapshot(self) -> ProjectSnapshot:
        assert self.project is not None
        return self.project.to_dict()

    def _commit_project_edit(
        self,
        text: str,
        before: ProjectSnapshot,
    ) -> bool:
        if self.project is None:
            return False
        after = self._project_snapshot()
        if before == after:
            return False
        self.undo_stack.push(_ProjectSnapshotCommand(self, text, before, after))
        return True

    def _restore_project_snapshot(self, snapshot: ProjectSnapshot) -> None:
        self.project = GraphProject.from_dict(snapshot)
        instances = self.project.instances_at(self.time_index)

        if self.active_instance_id not in instances:
            self.active_instance_id = next(iter(instances), None)

        if self.selected_node is not None:
            instance_id, node_id = self.selected_node
            graph = instances.get(instance_id)
            if graph is None or node_id not in graph.nodes:
                self.selected_node = None

        if self.selected_edge is not None:
            instance_id, edge = self.selected_edge
            graph = instances.get(instance_id)
            if graph is None or edge not in graph.edges:
                self.selected_edge = None

        def valid_anchor(anchor: tuple[str, str] | None) -> bool:
            if anchor is None:
                return True
            graph = instances.get(anchor[0])
            return graph is not None and anchor[1] in graph.nodes

        if not valid_anchor(self.creation_anchor):
            self.creation_anchor = None
        if not valid_anchor(self.connect_anchor):
            self.connect_anchor = None

        if self.hovered_pick is not None:
            graph = instances.get(self.hovered_pick.instance_id)
            if (
                graph is None
                or (
                    self.hovered_pick.kind == "node"
                    and self.hovered_pick.node_id not in graph.nodes
                )
                or (self.hovered_pick.kind == "edge" and self.hovered_pick.edge not in graph.edges)
            ):
                self._clear_hover()

        self.drag_target = None
        self.radius_drag_target = None
        self.radius_preview = None
        self.cursor_drag_plane = None
        self._node_drag_before = None
        self._radius_drag_before = None
        self._refresh_timepoint(preserve_lineage_view=True)

    def _finish_node_drag_history(self) -> None:
        before = self._node_drag_before
        self._node_drag_before = None
        if before is not None:
            self._commit_project_edit("Move node", before)

    def undo(self) -> None:
        self._cancel_radius_drag()
        self._finish_node_drag_history()
        self.undo_stack.undo()

    def redo(self) -> None:
        self._cancel_radius_drag()
        self._finish_node_drag_history()
        self.undo_stack.redo()

    def _undo_clean_changed(self, clean: bool) -> None:
        self._update_dirty_state(not clean)

    def _undo_text_changed(self, text: str) -> None:
        self.undo_action.setText(f"&Undo {text}" if text else "&Undo")

    def _redo_text_changed(self, text: str) -> None:
        self.redo_action.setText(f"&Redo {text}" if text else "&Redo")

    def _configure_dimensions(self) -> None:
        assert self.project is not None
        timepoints, z_size, y_size, x_size = self.project.shape_tzyx
        blockers = (QtCore.QSignalBlocker(self.time_slider), QtCore.QSignalBlocker(self.time_spin))
        self.time_slider.setRange(0, timepoints - 1)
        self.time_spin.setRange(0, timepoints - 1)
        self.time_count_label.setText(f"/ {timepoints - 1}")
        del blockers
        for view in self.ortho_views.values():
            view.configure_shape((z_size, y_size, x_size))
        for spin, size in zip(self.position_spins, (z_size, y_size, x_size), strict=True):
            spin.setRange(0.0, float(size - 1))

    def save(self) -> bool:
        if self.project is None:
            return False
        self._finish_node_drag_history()
        if self.project_path is None:
            return self.save_as()
        try:
            save_project(self.project, self.project_path)
        except Exception as error:
            self._show_error("Could not save project", error)
            return False
        self.undo_stack.setClean()
        self._set_dirty(False)
        self.statusBar().showMessage(f"Saved {self.project_path.name}", 5000)
        return True

    def save_as(self) -> bool:
        if self.project is None:
            return False
        source = Path(self.project.source_path)
        suggested = (
            self.project_path
            if self.project_path is not None
            else source.with_name(source.stem + ".cpv.json")
        )
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save CellAnnotator4D project",
            str(suggested),
            "CellAnnotator4D projects (*.cpv.json)",
        )
        if not path:
            return False
        if not path.endswith(".cpv.json"):
            path += ".cpv.json"
        self.project_path = Path(path).resolve()
        return self.save()

    def _refresh_timepoint(
        self,
        *,
        reset_camera: bool = False,
        preserve_lineage_view: bool = False,
    ) -> None:
        if self.project is None or self.volume is None:
            return
        instances = self.project.instances_at(self.time_index)
        if self.active_instance_id not in instances:
            self.active_instance_id = next(iter(instances), None)
            self.selected_node = None
            self.selected_edge = None
        frame = self.volume.data[self.time_index]
        self.mip_view.set_volume(frame, self.contrast_levels, reset_camera=reset_camera)
        self._refresh_images()
        self._refresh_graphs()
        self._refresh_instance_controls()
        self._refresh_lineage_controls(preserve_view=preserve_lineage_view)
        self._refresh_time_controls()

    def fit_all_2d_views(self) -> None:
        for view in self.ortho_views.values():
            view.fit_to_data()

    def _refresh_images(self) -> None:
        if self.project is None or self.volume is None:
            return
        frame = self.volume.data[self.time_index]
        for plane, view in self.ortho_views.items():
            depth = int(self.cursor_zyx[plane.depth_axis] + 0.5)
            view.set_slice_index(depth)
            view.set_image(extract_slice(frame, plane, depth), self.contrast_levels)
            horizontal, vertical, _ = project_position(plane, self.cursor_zyx)
            view.set_cursor(horizontal, vertical)
        self.mip_view.set_cursor(self.cursor_zyx)
        self._refresh_mask_overlays()
        self._refresh_radius_overlays(update_3d=False)

    def _current_mask_overlay(self) -> MaskFrameOverlay | None:
        if (
            self.project is None
            or self.mask_colorizer is None
            or not self.mask_visibility_checkbox.isChecked()
        ):
            return None
        return self.mask_colorizer.frame(
            self.time_index,
            self.project.instances_at(self.time_index),
        )

    def _refresh_mask_overlays(self) -> None:
        overlay = self._current_mask_overlay()
        visible = overlay is not None
        if not visible:
            for view in self.ortho_views.values():
                view.set_mask_visible(False)
            self.mip_view.set_mask_visible(False)
            return

        assert overlay is not None
        opacity = self.mask_opacity_slider.value() / 100.0
        for plane, view in self.ortho_views.items():
            component_slice = extract_slice(
                overlay.components,
                plane,
                view.slice_index,
            )
            view.set_mask_overlay(
                overlay.colorize(component_slice),
                opacity=opacity,
                visible=True,
            )
        self.mip_view.set_mask_overlay(
            overlay.components,
            overlay.palette_rgba,
            opacity=opacity,
            visible=True,
            data_key=(self.time_index, id(overlay.components)),
        )

    def _radius_display_mode(self) -> RadiusDisplayMode:
        checked = self.radius_display_group.checkedButton()
        if checked is None:
            return RadiusDisplayMode.ALL
        return RadiusDisplayMode(checked.property("radius_display_mode"))

    def _radius_sphere_display_mode(self) -> RadiusDisplayMode:
        checked = self.radius_sphere_display_group.checkedButton()
        if checked is None:
            return RadiusDisplayMode.SELECTED
        return RadiusDisplayMode(checked.property("radius_display_mode"))

    def _refresh_radius_overlays(self, *, update_3d: bool = True) -> None:
        fill_mode = self._radius_display_mode()
        sphere_mode = self._radius_sphere_display_mode()
        if self.project is None:
            for view in self.ortho_views.values():
                view.set_radius_visible(False)
                view.set_radius_spheres_visible(False)
            self.mip_view.set_radius_visible(False)
            self.mip_view.set_radius_spheres_visible(False)
            return

        graphs = self.project.instances_at(self.time_index)
        active_graph = graphs.get(self.active_instance_id)
        selected_graphs = {} if active_graph is None else {active_graph.id: active_graph}
        filled_graphs = (
            graphs
            if fill_mode is RadiusDisplayMode.ALL
            else selected_graphs
            if fill_mode is RadiusDisplayMode.SELECTED
            else {}
        )
        sphere_graphs = (
            graphs
            if sphere_mode is RadiusDisplayMode.ALL
            else selected_graphs
            if sphere_mode is RadiusDisplayMode.SELECTED
            else {}
        )
        shape_zyx = self.project.shape_tzyx[1:]

        fill_visible = self.radius_renderer.has_radii(filled_graphs)
        fill_opacity = self.radius_opacity_slider.value() / 100.0
        if fill_visible:
            for plane, view in self.ortho_views.items():
                overlay = self.radius_renderer.slice(
                    filled_graphs,
                    shape_zyx,
                    plane,
                    view.slice_index,
                )
                view.set_radius_overlay(
                    overlay.image_rgba,
                    opacity=fill_opacity,
                    visible=True,
                )
        else:
            for view in self.ortho_views.values():
                view.set_radius_visible(False)
            self.mip_view.set_radius_visible(False)

        spheres_visible = self.radius_renderer.has_radii(sphere_graphs)
        sphere_opacity = self.radius_sphere_opacity_slider.value() / 100.0
        if spheres_visible:
            for view in self.ortho_views.values():
                view.set_radius_spheres(
                    sphere_graphs,
                    opacity=sphere_opacity,
                    visible=True,
                )
        else:
            for view in self.ortho_views.values():
                view.set_radius_spheres_visible(False)
            self.mip_view.set_radius_spheres_visible(False)

        if not update_3d:
            if self.drag_target is not None:
                self.mip_view.set_radius_visible(False)
                self.mip_view.set_radius_spheres_visible(False)
            return

        if fill_visible:
            overlay_3d = self.radius_renderer.volume(filled_graphs, shape_zyx)
            self.mip_view.set_radius_overlay(
                overlay_3d.components,
                overlay_3d.palette_rgba,
                steps_zyx=overlay_3d.steps_zyx,
                opacity=fill_opacity,
                visible=True,
                data_key=(self.time_index, overlay_3d.data_key),
            )
        if spheres_visible:
            self.mip_view.set_radius_spheres(
                sphere_graphs,
                opacity=sphere_opacity,
                visible=True,
            )

    def _refresh_graphs(self) -> None:
        if self.project is None:
            return
        instances = self.project.instances_at(self.time_index)
        action_source = self._action_source()
        for view in self.ortho_views.values():
            view.set_graphs(
                instances,
                active_instance=self.active_instance_id,
                selected_node=self.selected_node,
                selected_edge=self.selected_edge,
                action_source=action_source,
                hovered=self.hovered_pick,
                radius_preview=self.radius_preview,
                emphasize_near_slice_nodes=self.near_slice_nodes_checkbox.isChecked(),
                near_slice_minimum_opacity=(self.near_slice_opacity_range.lowerValue() / 100.0),
                near_slice_maximum_opacity=(self.near_slice_opacity_range.upperValue() / 100.0),
                near_slice_near_distance=self.near_slice_near_distance_spin.value(),
                near_slice_far_distance=self.near_slice_far_distance_spin.value(),
            )
        self.mip_view.set_graphs(
            instances,
            active_instance=self.active_instance_id,
            selected_node=self.selected_node,
            selected_edge=self.selected_edge,
            action_source=action_source,
            hovered=self.hovered_pick,
        )
        self.lineage_view.set_active_instance(self.time_index, self.active_instance_id)
        self._refresh_selection_controls()
        self._refresh_mask_overlays()
        self._refresh_radius_overlays(update_3d=self.drag_target is None)

    def _refresh_graph_highlights(self) -> None:
        action_source = self._action_source()
        for view in self.ortho_views.values():
            view.set_highlights(
                selected_node=self.selected_node,
                selected_edge=self.selected_edge,
                action_source=action_source,
                hovered=self.hovered_pick,
                radius_preview=self.radius_preview,
            )
        self.mip_view.set_highlights(
            selected_node=self.selected_node,
            selected_edge=self.selected_edge,
            action_source=action_source,
            hovered=self.hovered_pick,
        )

    def _refresh_time_controls(self) -> None:
        blockers = (QtCore.QSignalBlocker(self.time_slider), QtCore.QSignalBlocker(self.time_spin))
        self.time_slider.setValue(self.time_index)
        self.time_spin.setValue(self.time_index)
        del blockers

    def _refresh_instance_controls(self) -> None:
        if self.project is None:
            return
        instances = self.project.instances_at(self.time_index)
        blocker = QtCore.QSignalBlocker(self.instance_list)
        self.instance_list.clear()
        selected_item = None
        for graph in instances.values():
            item = QtWidgets.QListWidgetItem()
            item.setText(f"{graph.name}   ({len(graph.nodes)} nodes, {len(graph.edges)} edges)")
            item.setData(QtCore.Qt.ItemDataRole.UserRole, graph.id)
            item.setForeground(QtGui.QColor(graph.color))
            self.instance_list.addItem(item)
            if graph.id == self.active_instance_id:
                selected_item = item
        if selected_item is not None:
            self.instance_list.setCurrentItem(selected_item)
        del blocker
        active = self._active_graph()
        name_blocker = QtCore.QSignalBlocker(self.instance_name_edit)
        self.instance_name_edit.setText(active.name if active is not None else "")
        del name_blocker
        self.instance_name_edit.setEnabled(active is not None)
        self.delete_instance_button.setEnabled(active is not None)
        self.copy_previous_button.setEnabled(active is not None and self.time_index > 0)
        self.copy_next_button.setEnabled(
            active is not None and self.time_index < self.project.timepoints - 1
        )
        total_nodes = sum(len(graph.nodes) for graph in instances.values())
        total_edges = sum(len(graph.edges) for graph in instances.values())
        self.instance_summary.setText(
            f"t={self.time_index}: {len(instances)} instances, "
            f"{total_nodes} nodes, {total_edges} edges"
        )

    def _refresh_selection_controls(self) -> None:
        graph = self._active_graph()
        node = None
        if graph is not None and self.selected_node is not None:
            instance_id, node_id = self.selected_node
            if instance_id == graph.id:
                node = graph.nodes.get(node_id)
        blockers = [QtCore.QSignalBlocker(spin) for spin in self.position_spins]
        if node is None:
            for spin in self.position_spins:
                spin.setEnabled(False)
                spin.setValue(0)
            self.apply_position_button.setEnabled(False)
            self.selected_radius_label.setText("—")
        else:
            for spin, value in zip(self.position_spins, node.position, strict=True):
                spin.setEnabled(True)
                spin.setValue(value)
            self.apply_position_button.setEnabled(True)
            self.selected_radius_label.setText(f"{node.radius:.3g} vox")
        del blockers
        instances = self.project.instances_at(self.time_index) if self.project is not None else {}
        self.estimate_active_radii_button.setEnabled(graph is not None and bool(graph.nodes))
        self.estimate_frame_radii_button.setEnabled(
            any(candidate.nodes for candidate in instances.values())
        )

        if node is not None and graph is not None:
            self.selection_label.setText(
                f"{graph.name} · node {node.id[:8]} · degree {graph.degree(node.id)} · "
                f"ZYX {tuple(round(value, 2) for value in node.position)} · "
                f"radius {node.radius:.3g} vox"
            )
        elif self.selected_edge is not None and graph is not None:
            _, edge = self.selected_edge
            self.selection_label.setText(f"{graph.name} · edge {edge[0][:6]} — {edge[1][:6]}")
        elif graph is not None:
            self.selection_label.setText(f"Active: {graph.name}")
        else:
            self.selection_label.setText("No graph selection")

    def _refresh_lineage_controls(self, *, preserve_view: bool = False) -> None:
        if self.project is None:
            return
        self.lineage_view.set_project(
            self.project,
            self.time_index,
            active_instance_id=self.active_instance_id,
            preserve_view=preserve_view,
        )
        self._lineage_selection_changed(self.lineage_view.selected_instance)

    def _time_slider_changed(self, value: int) -> None:
        blocker = QtCore.QSignalBlocker(self.time_spin)
        self.time_spin.setValue(value)
        del blocker
        self.set_time(value)

    def _time_spin_changed(self, value: int) -> None:
        blocker = QtCore.QSignalBlocker(self.time_slider)
        self.time_slider.setValue(value)
        del blocker
        self.set_time(value)

    def set_time(self, time: int) -> None:
        if self.project is None or time == self.time_index:
            return
        self._finish_node_drag_history()
        self._cancel_radius_drag()
        self.time_index = min(max(int(time), 0), self.project.timepoints - 1)
        self.selected_node = None
        self.selected_edge = None
        self.creation_anchor = None
        self.connect_anchor = None
        self.drag_target = None
        self.radius_drag_target = None
        self.radius_preview = None
        self.cursor_drag_plane = None
        self.hovered_pick = None
        self.pending_new_instance = False
        self._refresh_timepoint()

    def _on_slice_requested(self, plane: Plane, index: int) -> None:
        if self.project is None:
            return
        position = list(self.cursor_zyx)
        position[plane.depth_axis] = float(index)
        self.cursor_zyx = tuple(position)  # type: ignore[assignment]
        self._refresh_images()
        self._refresh_graphs()

    def _on_2d_click(
        self,
        plane: Plane,
        button: PointerButton,
        horizontal: float,
        vertical: float,
        modifiers: object,
    ) -> None:
        if self.project is None:
            return
        view = self.ortho_views[plane]
        pick = view.pick(horizontal, vertical)

        if button is PointerButton.RIGHT:
            if pick is not None:
                self._navigate_to_pick(pick)
            else:
                depth = float(view.slice_index)
                position = clamp_position(
                    unproject_position(plane, horizontal, vertical, depth),
                    self.project.shape_tzyx[1:],
                )
                self._centre_on(position)
            return
        if button is not PointerButton.LEFT:
            return
        if self.pending_new_instance:
            self._create_instance_at(plane, horizontal, vertical)
            return
        if not self._edit_tools_active():
            return

        if pick is None:
            self._handle_empty_click(plane, horizontal, vertical)
        elif pick.kind == "node":
            self._handle_node_click(pick)
        else:
            self._handle_edge_click(pick)

    def _create_instance_at(self, plane: Plane, horizontal: float, vertical: float) -> None:
        assert self.project is not None
        before = self._project_snapshot()
        depth = float(self.ortho_views[plane].slice_index)
        position = clamp_position(
            unproject_position(plane, horizontal, vertical, depth),
            self.project.shape_tzyx[1:],
        )
        graph = self.project.add_instance(self.time_index)
        node_id = graph.add_node(position)
        self.active_instance_id = graph.id
        self.selected_node = (graph.id, node_id)
        self.selected_edge = None
        self.pending_new_instance = False
        self.new_instance_button.setText("Create new")
        self.cursor_zyx = position
        self._set_mode(EditMode.ADD_CONNECTED)
        self.creation_anchor = (graph.id, node_id)
        self._commit_project_edit("Create instance", before)
        self._refresh_images()
        self._refresh_graphs()
        self._refresh_instance_controls()
        self._refresh_lineage_controls()
        self.statusBar().showMessage(
            "Instance created. Continue clicking empty space to extend it.", 6000
        )

    def _handle_empty_click(self, plane: Plane, horizontal: float, vertical: float) -> None:
        assert self.project is not None
        depth = float(self.ortho_views[plane].slice_index)
        position = clamp_position(
            unproject_position(plane, horizontal, vertical, depth),
            self.project.shape_tzyx[1:],
        )
        if self.edit_mode not in (EditMode.ADD_CONNECTED, EditMode.ADD_ISOLATED):
            return
        graph = self._active_graph()
        if graph is None:
            self.statusBar().showMessage("Create or select an instance first", 5000)
            return
        if self.edit_mode is EditMode.ADD_CONNECTED and (
            self.creation_anchor is None
            or self.creation_anchor[0] != graph.id
            or self.creation_anchor[1] not in graph.nodes
        ):
            self.statusBar().showMessage(
                "Choose a source node before adding a connected node. "
                "Use Add isolated node for a disconnected node.",
                6000,
            )
            return
        before = self._project_snapshot()
        node_id = graph.add_node(position)
        if self.edit_mode is EditMode.ADD_CONNECTED:
            assert self.creation_anchor is not None
            graph.add_edge(self.creation_anchor[1], node_id)
        self.creation_anchor = (
            (graph.id, node_id) if self.edit_mode is EditMode.ADD_CONNECTED else None
        )
        self.selected_node = (graph.id, node_id)
        self.selected_edge = None
        self.cursor_zyx = position
        label = (
            "Add connected node"
            if self.edit_mode is EditMode.ADD_CONNECTED
            else "Add isolated node"
        )
        self._commit_project_edit(label, before)
        self._refresh_images()
        self._refresh_graphs()
        self._refresh_instance_controls()

    def _handle_node_click(self, pick: PickResult) -> None:
        assert self.project is not None and pick.node_id is not None
        if self.edit_mode is EditMode.DRAG:
            graph = self.project.instances_at(self.time_index).get(pick.instance_id)
            if graph is None or pick.node_id not in graph.nodes:
                return
            if graph.id != self.active_instance_id:
                self.statusBar().showMessage(
                    "Right click the instance first, then select or drag one of its nodes.",
                    5000,
                )
                return
            self.selected_node = (graph.id, pick.node_id)
            self.selected_edge = None
            self._refresh_graphs()
            self._refresh_instance_controls()
            return
        if self.edit_mode is EditMode.EDIT_RADIUS:
            graph = self.project.instances_at(self.time_index).get(pick.instance_id)
            if graph is None or pick.node_id not in graph.nodes:
                return
            self.active_instance_id = graph.id
            self.selected_node = (graph.id, pick.node_id)
            self.selected_edge = None
            self._centre_on(graph.nodes[pick.node_id].position)
            self._refresh_instance_controls()
            self.statusBar().showMessage(
                "Node selected. Ctrl+left drag it in a 2D pane to set its radius.",
                5000,
            )
            return
        if self.edit_mode not in {
            EditMode.ADD_CONNECTED,
            EditMode.CONNECT,
            EditMode.DELETE_NODE,
            EditMode.DISSOLVE_NODE,
            EditMode.CONNECT_INSTANCES,
        }:
            return
        if self.edit_mode is EditMode.CONNECT_INSTANCES:
            self._handle_instance_connection_node(pick)
            return
        self.active_instance_id = pick.instance_id
        graph = self._active_graph()
        assert graph is not None
        node_id = pick.node_id

        if self.edit_mode is EditMode.ADD_CONNECTED:
            self.connect_anchor = None
            self.creation_anchor = (graph.id, node_id)
            self.selected_node = (graph.id, node_id)
            self.selected_edge = None
            self._centre_on(graph.nodes[node_id].position)
            self._refresh_instance_controls()
            self.statusBar().showMessage(
                "Parent node selected. Click empty space in a 2D pane to extend it.",
                5000,
            )
            return

        if self.edit_mode is EditMode.CONNECT:
            if self.connect_anchor is None or self.connect_anchor[0] != graph.id:
                self.connect_anchor = (graph.id, node_id)
                self.selected_node = (graph.id, node_id)
                self.selected_edge = None
                self._centre_on(graph.nodes[node_id].position)
                self._refresh_instance_controls()
                self.statusBar().showMessage("Now click the second node to connect", 5000)
                return
            first_id = self.connect_anchor[1]
            if first_id == node_id:
                self.statusBar().showMessage("Choose a different second node", 4000)
                return
            before = self._project_snapshot()
            try:
                graph.add_edge(first_id, node_id)
            except ValueError as error:
                self.statusBar().showMessage(str(error), 5000)
            else:
                self._commit_project_edit("Connect nodes", before)
            self.connect_anchor = None
            self.selected_node = (graph.id, node_id)
            self.selected_edge = None
            self._centre_on(graph.nodes[node_id].position)
            self._refresh_graphs()
            self._refresh_instance_controls()
            return

        if self.edit_mode is EditMode.DELETE_NODE:
            position = graph.nodes[node_id].position
            before = self._project_snapshot()
            graph.remove_node(node_id)
            self.selected_node = None
            self.selected_edge = None
            self._commit_project_edit("Remove node", before)
            self._centre_on(position)
            self._refresh_graphs()
            self._refresh_instance_controls()
            return

        if self.edit_mode is EditMode.DISSOLVE_NODE:
            position = graph.nodes[node_id].position
            before = self._project_snapshot()
            try:
                graph.dissolve_degree_two_node(node_id)
            except ValueError as error:
                self.statusBar().showMessage(str(error), 5000)
                return
            self.selected_node = None
            self.selected_edge = None
            self._commit_project_edit("Dissolve node", before)
            self._centre_on(position)
            self._refresh_graphs()
            self._refresh_instance_controls()
            return

    def _handle_instance_connection_node(self, pick: PickResult) -> None:
        assert self.project is not None and pick.node_id is not None
        instances = self.project.instances_at(self.time_index)
        graph = instances.get(pick.instance_id)
        if graph is None or pick.node_id not in graph.nodes:
            return
        if self.connect_anchor is None:
            self.connect_anchor = (graph.id, pick.node_id)
            self.active_instance_id = graph.id
            self.selected_node = (graph.id, pick.node_id)
            self.selected_edge = None
            self._centre_on(graph.nodes[pick.node_id].position)
            self._refresh_instance_controls()
            self.statusBar().showMessage(
                "First instance source selected. Click a node in a different instance.",
                6000,
            )
            return

        first_instance_id, first_node_id = self.connect_anchor
        if first_instance_id == graph.id:
            self.statusBar().showMessage("Choose a node in a different instance", 5000)
            return
        first_graph = instances.get(first_instance_id)
        if first_graph is None or first_node_id not in first_graph.nodes:
            self.connect_anchor = None
            self.statusBar().showMessage("The first instance source no longer exists", 5000)
            self._refresh_graphs()
            return
        position = graph.nodes[pick.node_id].position
        first_name = first_graph.name
        second_name = graph.name
        before = self._project_snapshot()
        try:
            result = self.project.connect_instances(
                self.time_index,
                first_instance_id,
                first_node_id,
                graph.id,
                pick.node_id,
            )
        except (KeyError, ValueError) as error:
            self.statusBar().showMessage(str(error), 7000)
            return

        self.active_instance_id = result.kept_instance_id
        self.selected_node = (result.kept_instance_id, result.second_node_id)
        self.selected_edge = None
        self.creation_anchor = None
        self.connect_anchor = None
        self._commit_project_edit("Connect instances", before)
        self._centre_on(position)
        self._refresh_instance_controls()
        self._refresh_lineage_controls()
        self.statusBar().showMessage(
            f"Connected {first_name!r} and {second_name!r}; {first_name!r} keeps the "
            "instance identity and reconciled lineage.",
            8000,
        )

    def _handle_edge_click(self, pick: PickResult) -> None:
        assert self.project is not None and pick.edge is not None
        if self.edit_mode not in {
            EditMode.SPLIT_EDGE,
            EditMode.DELETE_EDGE,
            EditMode.SPLIT_INSTANCE,
        }:
            return
        self.active_instance_id = pick.instance_id
        graph = self._active_graph()
        assert graph is not None
        edge = pick.edge
        position = interpolate_position(
            graph.nodes[edge[0]].position,
            graph.nodes[edge[1]].position,
            pick.fraction,
        )
        if self.edit_mode is EditMode.SPLIT_INSTANCE:
            graph_name = graph.name
            before = self._project_snapshot()
            try:
                result = self.project.split_instance_on_edge(
                    self.time_index,
                    graph.id,
                    edge,
                )
            except (KeyError, ValueError) as error:
                self.statusBar().showMessage(f"Instance not split: {error}", 7000)
                return
            split_graph = self.project.instances_at(self.time_index)[result.new_instance_id]
            self.active_instance_id = result.retained_instance_id
            self.selected_node = None
            self.selected_edge = None
            self.creation_anchor = None
            self.connect_anchor = None
            self._commit_project_edit("Split instance", before)
            self._centre_on(position)
            self._refresh_instance_controls()
            self._refresh_lineage_controls()
            self.statusBar().showMessage(
                f"Split {graph_name!r} into {graph_name!r} and {split_graph.name!r}; "
                "lineage endpoints now include both instances.",
                8000,
            )
            return
        if self.edit_mode is EditMode.SPLIT_EDGE:
            before = self._project_snapshot()
            node_id = graph.split_edge(edge[0], edge[1], position)
            self.selected_node = (graph.id, node_id)
            self.selected_edge = None
            self._commit_project_edit("Split edge", before)
            self._centre_on(position)
            self._refresh_graphs()
            self._refresh_instance_controls()
            return
        if self.edit_mode is EditMode.DELETE_EDGE:
            before = self._project_snapshot()
            graph.remove_edge(*edge)
            self.selected_edge = None
            self.selected_node = None
            self._clear_hover()
            self._commit_project_edit("Remove edge", before)
            self._refresh_graphs()
            self._refresh_instance_controls()
            self.statusBar().showMessage(
                "Edge removed without changing the crosshair or view position.",
                5000,
            )
            return

    def _on_2d_drag(
        self,
        plane: Plane,
        button: PointerButton,
        phase: str,
        horizontal: float,
        vertical: float,
        modifiers: object,
    ) -> None:
        if self.project is None:
            return
        view = self.ortho_views[plane]

        if button is PointerButton.RIGHT:
            if phase == "start":
                self.cursor_drag_plane = plane if view.pick(horizontal, vertical) is None else None
            if self.cursor_drag_plane is not plane:
                return
            depth = float(view.slice_index)
            position = clamp_position(
                unproject_position(plane, horizontal, vertical, depth),
                self.project.shape_tzyx[1:],
            )
            self._centre_on(position)
            if phase == "finish":
                self.cursor_drag_plane = None
            return

        if (
            button is PointerButton.LEFT
            and self._edit_tools_active()
            and self.edit_mode is EditMode.EDIT_RADIUS
        ):
            if phase == "start":
                if not self._control_modifier_active(modifiers):
                    self._cancel_radius_drag()
                    return
                pick = view.pick(horizontal, vertical)
                if pick is None or pick.kind != "node" or pick.node_id is None:
                    self._cancel_radius_drag()
                    return
                graph = self.project.instances_at(self.time_index).get(pick.instance_id)
                if graph is None or pick.node_id not in graph.nodes:
                    self._cancel_radius_drag()
                    return
                self.radius_drag_target = (graph.id, pick.node_id, plane)
                self._radius_drag_before = self._project_snapshot()
                self.active_instance_id = graph.id
                self.selected_node = (graph.id, pick.node_id)
                self.selected_edge = None
                self._centre_on(graph.nodes[pick.node_id].position)
                self._update_radius_preview(plane, horizontal, vertical)
                self._refresh_instance_controls()
                return
            if self.radius_drag_target is None or self.radius_drag_target[2] is not plane:
                return
            self._update_radius_preview(plane, horizontal, vertical)
            if phase == "finish":
                self._commit_radius_drag()
            return

        if (
            button is not PointerButton.LEFT
            or not self._edit_tools_active()
            or self.edit_mode is not EditMode.DRAG
        ):
            return
        if phase == "start":
            pick = view.pick(horizontal, vertical)
            if pick is None or pick.kind != "node" or pick.node_id is None:
                self.drag_target = None
                self._node_drag_before = None
                return
            if pick.instance_id != self.active_instance_id:
                self.drag_target = None
                self._node_drag_before = None
                self.statusBar().showMessage(
                    "Right click the instance first, then left drag one of its nodes.",
                    5000,
                )
                return
            self.drag_target = (pick.instance_id, pick.node_id, plane)
            self.selected_node = (pick.instance_id, pick.node_id)
            self.selected_edge = None
            self._node_drag_before = self._project_snapshot()
            self._refresh_graphs()
            self._refresh_instance_controls()
            return
        if self.drag_target is None or self.drag_target[2] is not plane:
            return
        instance_id, node_id, _ = self.drag_target
        graph = self.project.instances_at(self.time_index).get(instance_id)
        if graph is None or node_id not in graph.nodes:
            self.drag_target = None
            self._node_drag_before = None
            return
        position = clamp_position(
            move_in_plane(plane, graph.nodes[node_id].position, horizontal, vertical),
            self.project.shape_tzyx[1:],
        )
        graph.move_node(node_id, position)
        self.cursor_zyx = position
        self._refresh_images()
        self._refresh_graphs()
        if phase == "finish":
            self.drag_target = None
            self._finish_node_drag_history()
            self._refresh_graphs()
            self._refresh_instance_controls()

    @staticmethod
    def _control_modifier_active(modifiers: object) -> bool:
        try:
            return bool(modifiers & QtCore.Qt.KeyboardModifier.ControlModifier)
        except TypeError:
            return False

    def _update_radius_preview(
        self,
        plane: Plane,
        horizontal: float,
        vertical: float,
    ) -> None:
        if self.project is None or self.radius_drag_target is None:
            return
        instance_id, node_id, target_plane = self.radius_drag_target
        if target_plane is not plane:
            return
        graph = self.project.instances_at(self.time_index).get(instance_id)
        if graph is None or node_id not in graph.nodes:
            self._cancel_radius_drag()
            return
        node_horizontal, node_vertical, _ = project_position(
            plane,
            graph.nodes[node_id].position,
        )
        radius = math.hypot(horizontal - node_horizontal, vertical - node_vertical)
        maximum = math.sqrt(sum(float(size - 1) ** 2 for size in self.project.shape_tzyx[1:]))
        radius = min(max(radius, 0.0), maximum)
        self.radius_preview = (instance_id, node_id, radius)
        self._refresh_graph_highlights()
        self.selected_radius_label.setText(f"{radius:.3g} vox (preview)")
        self.statusBar().showMessage(
            f"Radius preview: {radius:.3g} vox — release to apply; Escape cancels.",
            3000,
        )

    def _commit_radius_drag(self) -> None:
        if self.project is None or self.radius_drag_target is None or self.radius_preview is None:
            self._cancel_radius_drag()
            return
        instance_id, node_id, _ = self.radius_drag_target
        graph = self.project.instances_at(self.time_index).get(instance_id)
        before = self._radius_drag_before
        radius = self.radius_preview[2]
        self.radius_drag_target = None
        self.radius_preview = None
        self._radius_drag_before = None
        if graph is None or node_id not in graph.nodes or before is None:
            self._refresh_graphs()
            return
        if radius < 0.05:
            radius = 0.0
        graph.set_node_radius(node_id, radius)
        changed = self._commit_project_edit("Change node radius", before)
        self._refresh_graphs()
        self._refresh_instance_controls()
        if changed:
            self.statusBar().showMessage(f"Node radius set to {radius:.3g} vox", 5000)
        else:
            self.statusBar().showMessage("Node radius unchanged", 3000)

    def _cancel_radius_drag(self) -> None:
        had_preview = self.radius_drag_target is not None or self.radius_preview is not None
        self.radius_drag_target = None
        self.radius_preview = None
        self._radius_drag_before = None
        if had_preview and self.project is not None:
            self._refresh_graph_highlights()
            self._refresh_selection_controls()

    def _on_3d_click(self, button: PointerButton, pick: PickResult | None) -> None:
        if self.project is None or pick is None:
            return
        if button is PointerButton.RIGHT:
            self._navigate_to_pick(pick)
        elif button is PointerButton.LEFT and self._edit_tools_active():
            if pick.kind == "node":
                self._handle_node_click(pick)
            else:
                self._handle_edge_click(pick)

    def _navigate_to_pick(self, pick: PickResult) -> None:
        if self.project is None:
            return
        graph = self.project.instances_at(self.time_index).get(pick.instance_id)
        if graph is None:
            return
        self.active_instance_id = graph.id
        if pick.kind == "node" and pick.node_id in graph.nodes:
            self.selected_node = (graph.id, pick.node_id)
            self.selected_edge = None
            self._centre_on(graph.nodes[pick.node_id].position)
        elif pick.kind == "edge" and pick.edge in graph.edges:
            self.selected_node = None
            self.selected_edge = (graph.id, pick.edge)
            position = interpolate_position(
                graph.nodes[pick.edge[0]].position,
                graph.nodes[pick.edge[1]].position,
                pick.fraction,
            )
            self._centre_on(position)
        self._refresh_instance_controls()

    def _centre_on(self, position: tuple[float, float, float]) -> None:
        self.cursor_zyx = position
        self._refresh_images()
        self._refresh_graphs()

    def begin_new_instance(self) -> None:
        if self.project is None:
            return
        if self.pending_new_instance:
            self.pending_new_instance = False
            self.new_instance_button.setText("Create new")
            self.statusBar().showMessage("New instance creation cancelled", 3000)
            return
        self.pending_new_instance = True
        self.new_instance_button.setText("Cancel new")
        self._set_mode(EditMode.ADD_CONNECTED)
        self.tabs.setCurrentIndex(1)
        self.statusBar().showMessage(
            "Click the first node location in any 2D view. Its hidden coordinate is the slice.",
            8000,
        )

    def delete_active_instance(self) -> None:
        if self.project is None or self.active_instance_id is None:
            return
        graph = self._active_graph()
        if graph is None:
            return
        answer = QtWidgets.QMessageBox.question(
            self,
            "Delete instance",
            f"Delete {graph.name!r}, all its nodes and edges, and its lineage references?",
        )
        if answer != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        before = self._project_snapshot()
        self.project.remove_instance(self.time_index, graph.id)
        self.active_instance_id = None
        self.selected_node = None
        self.selected_edge = None
        self.creation_anchor = None
        self.connect_anchor = None
        self.drag_target = None
        self.radius_drag_target = None
        self.radius_preview = None
        self.cursor_drag_plane = None
        self._node_drag_before = None
        self._radius_drag_before = None
        self._commit_project_edit("Delete instance", before)
        self._refresh_timepoint()

    def copy_active_instance(self, offset: int) -> None:
        if self.project is None or self.active_instance_id is None:
            return
        target_time = self.time_index + offset
        if not 0 <= target_time < self.project.timepoints:
            return
        graph = self._active_graph()
        if graph is None:
            return
        before = self._project_snapshot()
        try:
            copied = self.project.copy_instance(
                self.time_index,
                graph.id,
                target_time,
            )
        except (KeyError, ValueError) as error:
            self.statusBar().showMessage(str(error), 7000)
            return
        self._commit_project_edit("Copy instance", before)
        self._refresh_lineage_controls()
        self.statusBar().showMessage(
            f"Copied {graph.name!r} to t={target_time} as a new instance "
            f"({len(copied.nodes)} nodes, {len(copied.edges)} edges).",
            7000,
        )

    def rename_active_instance(self) -> None:
        graph = self._active_graph()
        if graph is None:
            return
        name = self.instance_name_edit.text().strip()
        if not name:
            blocker = QtCore.QSignalBlocker(self.instance_name_edit)
            self.instance_name_edit.setText(graph.name)
            del blocker
            return
        if name != graph.name:
            before = self._project_snapshot()
            graph.name = name
            self._commit_project_edit("Rename instance", before)
            self._refresh_instance_controls()
            self._refresh_lineage_controls()

    def _instance_selection_changed(
        self,
        current: QtWidgets.QListWidgetItem | None,
        previous: QtWidgets.QListWidgetItem | None,
    ) -> None:
        if current is None:
            return
        instance_id = current.data(QtCore.Qt.ItemDataRole.UserRole)
        if instance_id == self.active_instance_id:
            return
        self.active_instance_id = str(instance_id)
        self.selected_node = None
        self.selected_edge = None
        self.creation_anchor = None
        self.connect_anchor = None
        self._finish_node_drag_history()
        self._cancel_radius_drag()
        self.drag_target = None
        self.cursor_drag_plane = None
        self._refresh_graphs()
        self._refresh_instance_controls()

    def _mode_button_clicked(self, button: QtWidgets.QAbstractButton) -> None:
        self._set_mode(EditMode(button.property("edit_mode")))

    def activate_tool(self, mode: EditMode) -> None:
        self.tabs.setCurrentIndex(EDIT_TAB)
        self._set_mode(mode)

    def _set_mode(self, mode: EditMode) -> None:
        self._finish_node_drag_history()
        self._cancel_radius_drag()
        self.edit_mode = mode
        self.creation_anchor = None
        self.connect_anchor = None
        self.drag_target = None
        self.cursor_drag_plane = None
        self._clear_hover()
        if self.project is not None:
            self._refresh_graphs()
        self.mode_help_label.setText(mode_help_text(mode))
        for button in self.mode_group.buttons():
            if button.property("edit_mode") == mode.value:
                button.setChecked(True)
                break
        self.statusBar().showMessage(MODE_HELP[mode], 5000)

    def cancel_current_operation(self) -> None:
        cancelled: list[str] = []
        if self.pending_new_instance:
            cancelled.append("new-instance placement")
        if self.creation_anchor is not None:
            cancelled.append("connected-node source")
        if self.connect_anchor is not None:
            cancelled.append(
                "instance connection source"
                if self.edit_mode is EditMode.CONNECT_INSTANCES
                else "connection source"
            )
        if self.drag_target is not None:
            cancelled.append("active node drag")
        if self.radius_drag_target is not None or self.radius_preview is not None:
            cancelled.append("radius drag preview")
        if self.cursor_drag_plane is not None:
            cancelled.append("cursor navigation drag")
        if self.lineage_view.selected_instance is not None:
            cancelled.append("lineage source")

        self.pending_new_instance = False
        self.new_instance_button.setText("Create new")
        self.creation_anchor = None
        self.connect_anchor = None
        self._finish_node_drag_history()
        self.drag_target = None
        self._cancel_radius_drag()
        self.cursor_drag_plane = None
        self._clear_hover()
        self.lineage_view.clear_selection()
        if self.project is not None:
            self._refresh_graphs()
        if cancelled:
            self.statusBar().showMessage(
                f"Escape cancelled: {', '.join(cancelled)}. "
                "The active tool and right-click selection are unchanged.",
                6000,
            )
        else:
            self.statusBar().showMessage(
                "Escape: no staged action. The active tool and right-click selection "
                "are unchanged.",
                5000,
            )

    def apply_selected_node_position(self) -> None:
        if self.project is None or self.selected_node is None:
            return
        instance_id, node_id = self.selected_node
        graph = self.project.instances_at(self.time_index).get(instance_id)
        if graph is None or node_id not in graph.nodes:
            return
        position = clamp_position(
            tuple(spin.value() for spin in self.position_spins),
            self.project.shape_tzyx[1:],
        )
        before = self._project_snapshot()
        graph.move_node(node_id, position)
        self.cursor_zyx = position
        self._commit_project_edit("Move node", before)
        self._refresh_images()
        self._refresh_graphs()

    def estimate_current_intensity_radii(self, *, active_only: bool) -> None:
        if self.project is None or self.volume is None:
            return
        instances = self.project.instances_at(self.time_index)
        if active_only:
            graph = self._active_graph()
            targets = {} if graph is None else {graph.id: graph}
        else:
            targets = {graph.id: graph for graph in instances.values() if graph.nodes}
        if not targets or not any(graph.nodes for graph in targets.values()):
            self.statusBar().showMessage("There are no graph nodes to estimate at this time.", 5000)
            return

        self._cancel_radius_drag()
        self._finish_node_drag_history()
        before = self._project_snapshot()
        try:
            with _wait_cursor():
                estimates = estimate_intensity_radii(
                    self.volume.data[self.time_index],
                    targets,
                    maximum_radius=self.intensity_max_radius_spin.value(),
                    threshold_fraction=self.intensity_threshold_spin.value() / 100.0,
                    ray_percentile=float(self.intensity_ray_percentile_spin.value()),
                    smoothing_sigma=self.intensity_smoothing_spin.value(),
                )
                for (instance_id, node_id), radius in estimates.items():
                    target = instances.get(instance_id)
                    if target is not None and node_id in target.nodes:
                        target.set_node_radius(node_id, radius)
        except ValueError as error:
            self.statusBar().showMessage(f"Could not estimate radii: {error}", 7000)
            return

        self._commit_project_edit("Estimate radii from intensity", before)
        if self._radius_display_mode() is RadiusDisplayMode.NONE:
            visible_mode = RadiusDisplayMode.SELECTED if active_only else RadiusDisplayMode.ALL
            self.radius_display_buttons[visible_mode].setChecked(True)
            self._update_radius_control_states()
        self._refresh_graphs()
        self._refresh_instance_controls()
        positive = sum(radius > 0 for radius in estimates.values())
        scope = "active instance" if active_only else f"all instances at t={self.time_index}"
        self.statusBar().showMessage(
            f"Estimated {len(estimates)} node radii for {scope}; {positive} are non-zero. "
            "Adjust the settings and rerun if needed.",
            8000,
        )

    def _connect_lineage_instances(
        self,
        first_time: int,
        first_instance: str,
        second_time: int,
        second_instance: str,
    ) -> None:
        if self.project is None:
            return
        before = self._project_snapshot()
        try:
            event = self.project.connect_lineage_instances(
                first_time,
                first_instance,
                second_time,
                second_instance,
            )
        except (KeyError, ValueError) as error:
            self.statusBar().showMessage(str(error), 7000)
            return
        self._commit_project_edit("Connect lineage instances", before)
        self.lineage_view.clear_selection()
        self._refresh_lineage_controls()
        self.statusBar().showMessage(f"Recorded {event.label} lineage event", 5000)

    def _remove_lineage_event(self, event_id: str) -> None:
        if self.project is None:
            return
        before = self._project_snapshot()
        try:
            event = self.project.remove_lineage_event(event_id)
        except KeyError as error:
            self.statusBar().showMessage(str(error), 5000)
            return
        self._commit_project_edit("Remove lineage event", before)
        self._refresh_lineage_controls()
        self.statusBar().showMessage(f"Removed {event.label} lineage event", 5000)

    def auto_match_lineages(self) -> None:
        if self.project is None:
            return
        suggestions = suggest_maximal_one_to_one_matches(self.project)
        if not suggestions:
            self.statusBar().showMessage(
                "No free adjacent-frame lineage endpoints are available to match.",
                6000,
            )
            return

        before = self._project_snapshot()
        try:
            for suggestion in suggestions:
                self.project.add_lineage_event(
                    source_time=suggestion.source_time,
                    target_time=suggestion.target_time,
                    sources=(suggestion.source_instance_id,),
                    targets=(suggestion.target_instance_id,),
                )
        except (KeyError, ValueError) as error:
            self.project = GraphProject.from_dict(before)
            self._refresh_timepoint(preserve_lineage_view=True)
            self.statusBar().showMessage(f"Could not auto-match lineages: {error}", 7000)
            return

        self._commit_project_edit("Auto-match lineages", before)
        self._refresh_lineage_controls(preserve_view=True)
        transitions = len({suggestion.source_time for suggestion in suggestions})
        self.statusBar().showMessage(
            f"Added {len(suggestions)} one-to-one match"
            f"{'es' if len(suggestions) != 1 else ''} across {transitions} adjacent "
            f"frame pair{'s' if transitions != 1 else ''}; existing events were unchanged.",
            8000,
        )

    def mark_selected_lineage_start(self) -> None:
        if self.project is None or self.lineage_view.selected_instance is None:
            return
        time, instance_id = self.lineage_view.selected_instance
        before = self._project_snapshot()
        try:
            self.project.mark_lineage_start(time, instance_id)
        except (KeyError, ValueError) as error:
            self.statusBar().showMessage(str(error), 7000)
            return
        self._commit_project_edit("Mark lineage start", before)
        self.lineage_view.clear_selection()
        self._refresh_lineage_controls()
        self.statusBar().showMessage(f"Marked instance at t={time} as a lineage start", 5000)

    def mark_selected_lineage_end(self) -> None:
        if self.project is None or self.lineage_view.selected_instance is None:
            return
        time, instance_id = self.lineage_view.selected_instance
        before = self._project_snapshot()
        try:
            self.project.mark_lineage_end(time, instance_id)
        except (KeyError, ValueError) as error:
            self.statusBar().showMessage(str(error), 7000)
            return
        self._commit_project_edit("Mark lineage end", before)
        self.lineage_view.clear_selection()
        self._refresh_lineage_controls()
        self.statusBar().showMessage(f"Marked instance at t={time} as a lineage end", 5000)

    def _activate_lineage_instance(self, time: int, instance_id: str) -> None:
        if self.project is None:
            return
        self.set_time(time)
        if instance_id not in self.project.instances_at(time):
            return
        self.active_instance_id = instance_id
        self.selected_node = None
        self.selected_edge = None
        self._refresh_graphs()
        self._refresh_instance_controls()

    def _lineage_selection_changed(self, selection: object) -> None:
        if not isinstance(selection, tuple) or len(selection) != 2 or self.project is None:
            self.lineage_selection_label.setText("No lineage instance selected")
            self.mark_start_button.setEnabled(False)
            self.mark_end_button.setEnabled(False)
            return
        time, instance_id = int(selection[0]), str(selection[1])
        graph = self.project.instances_at(time).get(instance_id)
        if graph is None:
            self.lineage_selection_label.setText("No lineage instance selected")
            self.mark_start_button.setEnabled(False)
            self.mark_end_button.setEnabled(False)
            return
        self.lineage_selection_label.setText(f"Selected: t={time} · {graph.name}")
        self.mark_start_button.setEnabled(True)
        self.mark_end_button.setEnabled(True)

    def _tab_changed(self, index: int) -> None:
        self._finish_node_drag_history()
        self._cancel_radius_drag()
        self.creation_anchor = None
        self.connect_anchor = None
        self.drag_target = None
        self.cursor_drag_plane = None
        self._clear_hover()
        if index != EDIT_TAB and self.pending_new_instance:
            self.pending_new_instance = False
            self.new_instance_button.setText("Create new")
        if index == LINEAGE_TAB and self.project is not None:
            self._refresh_lineage_controls()
            QtCore.QTimer.singleShot(0, self.lineage_view.ensure_view_fitted)
        self._refresh_graphs()

    def _edit_tools_active(self) -> bool:
        return self.project is not None and self.tabs.currentIndex() == EDIT_TAB

    def _on_view_hover(self, *arguments: object) -> None:
        pick = arguments[-1] if arguments else None
        in_2d = len(arguments) > 1 and isinstance(arguments[0], Plane)
        if not isinstance(pick, PickResult) or not self._hover_matches_tool(
            pick,
            in_2d=in_2d,
        ):
            pick = None
        old_identity = None if self.hovered_pick is None else self.hovered_pick.identity
        new_identity = None if pick is None else pick.identity
        if old_identity == new_identity:
            return
        self.hovered_pick = pick
        self._refresh_graph_highlights()

    def _hover_matches_tool(self, pick: PickResult, *, in_2d: bool = True) -> bool:
        if not self._edit_tools_active() or self.pending_new_instance or self.project is None:
            return False
        graph = self.project.instances_at(self.time_index).get(pick.instance_id)
        if graph is None:
            return False
        if pick.kind == "node":
            if pick.node_id is None or pick.node_id not in graph.nodes:
                return False
            if self.edit_mode is EditMode.DRAG:
                return in_2d and graph.id == self.active_instance_id
            if self.edit_mode is EditMode.DISSOLVE_NODE:
                return graph.degree(pick.node_id) == 2
            if self.edit_mode is EditMode.CONNECT:
                if self.connect_anchor is None or self.connect_anchor[0] != graph.id:
                    return True
                first_node_id = self.connect_anchor[1]
                if first_node_id == pick.node_id or first_node_id not in graph.nodes:
                    return False
                return canonical_edge(first_node_id, pick.node_id) not in graph.edges
            if self.edit_mode is EditMode.CONNECT_INSTANCES:
                return self.connect_anchor is None or self.connect_anchor[0] != graph.id
            return self.edit_mode in {
                EditMode.ADD_CONNECTED,
                EditMode.DELETE_NODE,
                EditMode.EDIT_RADIUS,
            }
        if pick.edge is None or pick.edge not in graph.edges:
            return False
        if self.edit_mode is EditMode.SPLIT_INSTANCE:
            return self.project.can_split_instance_on_edge(
                self.time_index,
                graph.id,
                pick.edge,
            )
        return self.edit_mode in {EditMode.SPLIT_EDGE, EditMode.DELETE_EDGE}

    def _clear_hover(self) -> None:
        for view in self.ortho_views.values():
            view.reset_hover_tracking()
        self.mip_view.reset_hover_tracking()
        self.hovered_pick = None

    def _active_graph(self) -> InstanceGraph | None:
        if self.project is None or self.active_instance_id is None:
            return None
        return self.project.instances_at(self.time_index).get(self.active_instance_id)

    def _action_source(self) -> tuple[str, str] | None:
        if not self._edit_tools_active():
            return None
        if self.edit_mode is EditMode.ADD_CONNECTED:
            return self.creation_anchor
        if self.edit_mode in (EditMode.CONNECT, EditMode.CONNECT_INSTANCES):
            return self.connect_anchor
        return None

    def _mask_visibility_changed(self, visible: bool) -> None:
        masks_available = self.volume is not None and self.volume.masks is not None
        self.mask_opacity_slider.setEnabled(masks_available and visible)
        if not masks_available:
            return
        if visible:
            self._refresh_mask_overlays()
            mask_name = self.volume.masks.path.name
            self.statusBar().showMessage(f"Showing instance masks from {mask_name}", 4000)
        else:
            for view in self.ortho_views.values():
                view.set_mask_visible(False)
            self.mip_view.set_mask_visible(False)
            self.statusBar().showMessage("Instance masks hidden", 3000)

    def _mask_opacity_changed(self, value: int) -> None:
        opacity = min(max(int(value), 0), 100) / 100.0
        self.mask_opacity_label.setText(f"{round(opacity * 100)}%")
        for view in self.ortho_views.values():
            view.set_mask_opacity(opacity)
        self.mip_view.set_mask_opacity(opacity)

    def _update_radius_control_states(self) -> None:
        available = self.project is not None
        self.radius_opacity_slider.setEnabled(
            available and self._radius_display_mode() is not RadiusDisplayMode.NONE
        )
        for button in self.radius_sphere_display_buttons.values():
            button.setEnabled(available)
        self.radius_sphere_opacity_slider.setEnabled(
            available and self._radius_sphere_display_mode() is not RadiusDisplayMode.NONE
        )

    def _radius_display_mode_changed(self, _button: QtWidgets.QAbstractButton) -> None:
        self._update_radius_control_states()
        if self.project is None:
            return
        mode = self._radius_display_mode()
        self._refresh_radius_overlays()
        messages = {
            RadiusDisplayMode.ALL: "Showing all node-radius volumes",
            RadiusDisplayMode.SELECTED: ("Showing the selected mitochondrion's node-radius volume"),
            RadiusDisplayMode.NONE: "Connected node-radius fill hidden",
        }
        self.statusBar().showMessage(messages[mode], 3000)

    def _radius_opacity_changed(self, value: int) -> None:
        opacity = min(max(int(value), 0), 100) / 100.0
        self.radius_opacity_label.setText(f"{round(opacity * 100)}%")
        for view in self.ortho_views.values():
            view.set_radius_opacity(opacity)
        self.mip_view.set_radius_opacity(opacity)

    def _radius_sphere_display_mode_changed(
        self,
        _button: QtWidgets.QAbstractButton,
    ) -> None:
        self._update_radius_control_states()
        if self.project is None:
            return
        mode = self._radius_sphere_display_mode()
        self._refresh_radius_overlays()
        messages = {
            RadiusDisplayMode.ALL: "Showing all node-radius circles",
            RadiusDisplayMode.SELECTED: (
                "Showing the selected mitochondrion's node-radius circles"
            ),
            RadiusDisplayMode.NONE: "Node-radius circles hidden",
        }
        self.statusBar().showMessage(messages[mode], 3000)

    def _radius_sphere_opacity_changed(self, value: int) -> None:
        opacity = min(max(int(value), 0), 100) / 100.0
        self.radius_sphere_opacity_label.setText(f"{round(opacity * 100)}%")
        for view in self.ortho_views.values():
            view.set_radius_sphere_opacity(opacity)
        self.mip_view.set_radius_sphere_opacity(opacity)

    def _near_slice_node_emphasis_changed(self, enabled: bool) -> None:
        self.near_slice_settings_widget.setEnabled(enabled and self.project is not None)
        if self.project is None:
            return
        self._refresh_graphs()
        state = "enabled" if enabled else "disabled"
        self.statusBar().showMessage(f"Distance-based node opacity {state}", 3000)

    def _near_slice_opacity_range_changed(self, lower: int, upper: int) -> None:
        self.near_slice_opacity_range_label.setText(f"{lower}–{upper}%")
        self.near_slice_near_distance_label.setText(f"{upper}% at ≤")
        self.near_slice_far_distance_label.setText(f"{lower}% at ≥")
        if self.project is not None and self.near_slice_nodes_checkbox.isChecked():
            self._refresh_graphs()

    def _near_slice_distance_changed(self, _value: float) -> None:
        near_distance = self.near_slice_near_distance_spin.value()
        far_distance = self.near_slice_far_distance_spin.value()
        sender = self.sender()
        if near_distance > far_distance:
            if sender is self.near_slice_near_distance_spin:
                blocker = QtCore.QSignalBlocker(self.near_slice_far_distance_spin)
                self.near_slice_far_distance_spin.setValue(near_distance)
            else:
                blocker = QtCore.QSignalBlocker(self.near_slice_near_distance_spin)
                self.near_slice_near_distance_spin.setValue(far_distance)
            del blocker
        if self.project is not None and self.near_slice_nodes_checkbox.isChecked():
            self._refresh_graphs()

    def _contrast_slider_changed(self) -> None:
        if self._updating_controls:
            return
        minimum_position = self.minimum_slider.value()
        maximum_position = self.maximum_slider.value()
        if minimum_position >= maximum_position:
            sender = self.sender()
            if sender is self.minimum_slider:
                maximum_position = min(1000, minimum_position + 1)
                blocker = QtCore.QSignalBlocker(self.maximum_slider)
                self.maximum_slider.setValue(maximum_position)
                del blocker
            else:
                minimum_position = max(0, maximum_position - 1)
                blocker = QtCore.QSignalBlocker(self.minimum_slider)
                self.minimum_slider.setValue(minimum_position)
                del blocker
        self.contrast_levels = (
            self._slider_to_intensity(minimum_position),
            self._slider_to_intensity(maximum_position),
        )
        self._update_contrast_labels()
        if self.volume is not None:
            self._refresh_images()
            self.mip_view.set_contrast(self.contrast_levels)

    def _set_contrast_sliders_from_levels(self) -> None:
        self._updating_controls = True
        try:
            self.minimum_slider.setValue(self._intensity_to_slider(self.contrast_levels[0]))
            self.maximum_slider.setValue(self._intensity_to_slider(self.contrast_levels[1]))
        finally:
            self._updating_controls = False
        self._update_contrast_labels()

    def _slider_to_intensity(self, value: int) -> float:
        minimum, maximum = self.full_intensity_range
        return minimum + (maximum - minimum) * value / 1000

    def _intensity_to_slider(self, value: float) -> int:
        minimum, maximum = self.full_intensity_range
        if maximum <= minimum:
            return 0
        return round(1000 * (value - minimum) / (maximum - minimum))

    def _update_contrast_labels(self) -> None:
        self.minimum_value_label.setText(f"{self.contrast_levels[0]:.5g}")
        self.maximum_value_label.setText(f"{self.contrast_levels[1]:.5g}")

    def _update_dirty_state(self, dirty: bool) -> None:
        self.dirty = dirty
        source_name = self.volume.path.name if self.volume is not None else "CellAnnotator4D"
        project_name = self.project_path.name if self.project_path is not None else source_name
        self.setWindowTitle(f"{'*' if dirty else ''}{project_name} — CellAnnotator4D")

    def _set_dirty(self, dirty: bool) -> None:
        """Set the save point; retained as a small compatibility helper for callers/tests."""
        if not dirty:
            self.undo_stack.setClean()
        self._update_dirty_state(dirty)

    def _discard_or_save_changes(self) -> bool:
        self._cancel_radius_drag()
        self._finish_node_drag_history()
        if not self.dirty:
            return True
        answer = QtWidgets.QMessageBox.warning(
            self,
            "Unsaved annotations",
            "Save the current annotation project before continuing?",
            QtWidgets.QMessageBox.StandardButton.Save
            | QtWidgets.QMessageBox.StandardButton.Discard
            | QtWidgets.QMessageBox.StandardButton.Cancel,
            QtWidgets.QMessageBox.StandardButton.Save,
        )
        if answer == QtWidgets.QMessageBox.StandardButton.Cancel:
            return False
        if answer == QtWidgets.QMessageBox.StandardButton.Save:
            return self.save()
        return True

    def _show_error(self, title: str, error: Exception) -> None:
        QtWidgets.QMessageBox.critical(self, title, str(error))
        self.statusBar().showMessage(str(error), 10000)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self._discard_or_save_changes():
            event.accept()
        else:
            event.ignore()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        modifiers = event.modifiers()
        if (
            event.key() == QtCore.Qt.Key.Key_Z
            and modifiers == QtCore.Qt.KeyboardModifier.ControlModifier
        ):
            self.undo()
            event.accept()
            return
        if (
            event.key() == QtCore.Qt.Key.Key_Y
            and modifiers == QtCore.Qt.KeyboardModifier.ControlModifier
        ) or (
            event.key() == QtCore.Qt.Key.Key_Z
            and modifiers
            == (
                QtCore.Qt.KeyboardModifier.ControlModifier
                | QtCore.Qt.KeyboardModifier.ShiftModifier
            )
        ):
            self.redo()
            event.accept()
            return
        if (
            event.key() == QtCore.Qt.Key.Key_Escape
            and modifiers == QtCore.Qt.KeyboardModifier.NoModifier
        ):
            self.cancel_current_operation()
            event.accept()
            return
        if (
            event.key() == QtCore.Qt.Key.Key_R
            and modifiers == QtCore.Qt.KeyboardModifier.NoModifier
        ):
            self.activate_tool(EditMode.EDIT_RADIUS)
            event.accept()
            return
        number_keys = {
            QtCore.Qt.Key.Key_1: 0,
            QtCore.Qt.Key.Key_2: 1,
            QtCore.Qt.Key.Key_3: 2,
            QtCore.Qt.Key.Key_4: 3,
            QtCore.Qt.Key.Key_5: 4,
            QtCore.Qt.Key.Key_6: 5,
            QtCore.Qt.Key.Key_7: 6,
            QtCore.Qt.Key.Key_8: 7,
            QtCore.Qt.Key.Key_9: 8,
            QtCore.Qt.Key.Key_0: 9,
        }
        index = number_keys.get(event.key())
        if index is not None and modifiers == QtCore.Qt.KeyboardModifier.NoModifier:
            self.activate_tool(list(EditMode)[index])
            event.accept()
            return
        super().keyPressEvent(event)

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        urls = event.mimeData().urls()
        if len(urls) == 1 and urls[0].isLocalFile():
            suffix = Path(urls[0].toLocalFile()).suffix.lower()
            if suffix in {".tif", ".tiff", ".json"}:
                event.acceptProposedAction()

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        urls = event.mimeData().urls()
        if len(urls) == 1 and urls[0].isLocalFile():
            self.open_path(urls[0].toLocalFile())
            event.acceptProposedAction()


class _wait_cursor:
    def __enter__(self) -> None:
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        QtWidgets.QApplication.restoreOverrideCursor()
