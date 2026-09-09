from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from .geometry import Plane, plane_shape, project_position
from .interaction import NODE_EMPHASIS_STYLES, PickResult, PointerButton
from .model import Edge, InstanceGraph

pg.setConfigOption("imageAxisOrder", "row-major")
pg.setConfigOption("background", "#11151d")
pg.setConfigOption("foreground", "#d8dee9")

DEFAULT_NEAR_SLICE_FAR_DISTANCE = 4.0


@dataclass(slots=True)
class _ProjectedNode:
    instance_id: str
    node_id: str
    horizontal: float
    vertical: float
    depth: float
    active: bool


@dataclass(slots=True)
class _ProjectedEdge:
    instance_id: str
    edge: Edge
    first: tuple[float, float]
    second: tuple[float, float]
    active: bool


class _BatchedEdgesItem(pg.GraphicsObject):
    """Replay many independently styled edges as one graphics item."""

    def __init__(
        self,
        segments: list[tuple[tuple[float, float], tuple[float, float], QtGui.QPen]],
    ) -> None:
        super().__init__()
        self._picture = QtGui.QPicture()
        painter = QtGui.QPainter(self._picture)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        for first, second, pen in segments:
            painter.setPen(pen)
            painter.drawLine(QtCore.QPointF(*first), QtCore.QPointF(*second))
        painter.end()
        self._bounds = QtCore.QRectF(self._picture.boundingRect())

    def paint(
        self,
        painter: QtGui.QPainter,
        _option: QtWidgets.QStyleOptionGraphicsItem,
        _widget: QtWidgets.QWidget | None = None,
    ) -> None:
        painter.drawPicture(0, 0, self._picture)

    def boundingRect(self) -> QtCore.QRectF:
        return self._bounds


class InteractiveViewBox(pg.ViewBox):
    pointer_clicked = QtCore.Signal(object, object, float, float, object)
    pointer_dragged = QtCore.Signal(object, object, str, float, float, object)
    slice_stepped = QtCore.Signal(object, int)

    def __init__(self, plane: Plane) -> None:
        super().__init__(enableMenu=False)
        self.plane = plane
        self.setMouseMode(self.PanMode)
        self.rbScaleBox.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)
        self.borderRect.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)

    def mouseClickEvent(self, event: object) -> None:
        buttons = {
            QtCore.Qt.MouseButton.LeftButton: PointerButton.LEFT,
            QtCore.Qt.MouseButton.RightButton: PointerButton.RIGHT,
        }
        button = buttons.get(event.button())
        if button is not None:
            point = self.mapSceneToView(event.scenePos())
            self.pointer_clicked.emit(
                self.plane,
                button,
                float(point.x()),
                float(point.y()),
                event.modifiers(),
            )
            event.accept()
            return
        super().mouseClickEvent(event)

    def mouseDragEvent(self, event: object, axis: int | None = None) -> None:
        buttons = {
            QtCore.Qt.MouseButton.LeftButton: PointerButton.LEFT,
            QtCore.Qt.MouseButton.RightButton: PointerButton.RIGHT,
        }
        button = buttons.get(event.button())
        if button is not None:
            point = self.mapSceneToView(event.scenePos())
            phase = "start" if event.isStart() else "finish" if event.isFinish() else "move"
            self.pointer_dragged.emit(
                self.plane,
                button,
                phase,
                float(point.x()),
                float(point.y()),
                event.modifiers(),
            )
            event.accept()
            return
        super().mouseDragEvent(event, axis=axis)

    def wheelEvent(self, event: object, axis: int | None = None) -> None:
        if event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier:
            delta = event.delta()
            if delta:
                self.slice_stepped.emit(self.plane, 1 if delta > 0 else -1)
            event.accept()
            return
        super().wheelEvent(event, axis=axis)


class OrthoView(QtWidgets.QWidget):
    pointer_clicked = QtCore.Signal(object, object, float, float, object)
    pointer_dragged = QtCore.Signal(object, object, str, float, float, object)
    slice_requested = QtCore.Signal(object, int)
    hover_changed = QtCore.Signal(object, object)

    def __init__(self, plane: Plane, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.plane = plane
        self._slice_index = 0
        self._shape_zyx = (1, 1, 1)
        self._needs_initial_fit = True
        self._fit_scheduled = False
        self._graph_items: list[pg.GraphicsObject] = []
        self._highlight_items: list[pg.GraphicsObject] = []
        self._radius_sphere_items: list[QtWidgets.QGraphicsEllipseItem] = []
        self._projected_nodes: list[_ProjectedNode] = []
        self._projected_edges: list[_ProjectedEdge] = []
        self._projected_node_lookup: dict[tuple[str, str], _ProjectedNode] = {}
        self._projected_edge_lookup: dict[tuple[str, Edge], _ProjectedEdge] = {}
        self._node_pick_positions = np.empty((0, 2), dtype=float)
        self._node_pick_active = np.empty(0, dtype=bool)
        self._edge_pick_first = np.empty((0, 2), dtype=float)
        self._edge_pick_second = np.empty((0, 2), dtype=float)
        self._edge_pick_active = np.empty(0, dtype=bool)
        self._graph_key: tuple[object, ...] | None = None
        self._highlight_key: tuple[object, ...] | None = None
        self._radius_sphere_key: tuple[object, ...] | None = None
        self._hover_pick: PickResult | None = None
        self._press_button = QtCore.Qt.MouseButton.NoButton
        self._press_position = QtCore.QPoint()
        self._press_modifiers = QtCore.Qt.KeyboardModifier.NoModifier
        self._pointer_dragging = False

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        title_row = QtWidgets.QHBoxLayout()
        self.title_label = QtWidgets.QLabel(f"{plane.title} slice")
        title_font = self.title_label.font()
        title_font.setBold(True)
        self.title_label.setFont(title_font)
        self.coordinate_label = QtWidgets.QLabel()
        self.coordinate_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        title_row.addWidget(self.title_label)
        title_row.addStretch(1)
        title_row.addWidget(self.coordinate_label)
        layout.addLayout(title_row)

        self.view_box = InteractiveViewBox(plane)
        self.plot_widget = pg.PlotWidget(viewBox=self.view_box)
        self.plot_widget.setMinimumSize(260, 220)
        self.plot_widget.setMouseTracking(True)
        self.plot_widget.viewport().setMouseTracking(True)
        self.plot_widget.viewport().installEventFilter(self)
        self.plot_item = self.plot_widget.getPlotItem()
        self.plot_item.hideButtons()
        self.plot_item.setLabel("bottom", plane.horizontal_label)
        self.plot_item.setLabel("left", plane.vertical_label)
        self.plot_item.getViewBox().invertY(True)
        self.plot_item.getViewBox().setAspectLocked(True)
        self.image_item = pg.ImageItem()
        self.image_item.setZValue(-100)
        self.image_item.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)
        self.plot_item.addItem(self.image_item)
        self.mask_item = pg.ImageItem()
        self.mask_item.setZValue(-90)
        self.mask_item.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)
        self.mask_item.setCompositionMode(QtGui.QPainter.CompositionMode.CompositionMode_SourceOver)
        self.mask_item.setVisible(False)
        self.plot_item.addItem(self.mask_item)
        self.radius_item = pg.ImageItem()
        self.radius_item.setZValue(-80)
        self.radius_item.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)
        self.radius_item.setCompositionMode(
            QtGui.QPainter.CompositionMode.CompositionMode_SourceOver
        )
        self.radius_item.setVisible(False)
        self.plot_item.addItem(self.radius_item)

        cursor_pen = pg.mkPen("#ffffff", width=1)
        self.horizontal_cursor = pg.InfiniteLine(angle=0, movable=False, pen=cursor_pen)
        self.vertical_cursor = pg.InfiniteLine(angle=90, movable=False, pen=cursor_pen)
        self.horizontal_cursor.setZValue(50)
        self.vertical_cursor.setZValue(50)
        self.horizontal_cursor.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)
        self.vertical_cursor.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)
        self.plot_item.addItem(self.horizontal_cursor)
        self.plot_item.addItem(self.vertical_cursor)
        layout.addWidget(self.plot_widget, 1)

        slider_row = QtWidgets.QHBoxLayout()
        slider_row.addWidget(QtWidgets.QLabel(plane.depth_label))
        self.slice_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slice_slider.setRange(0, 0)
        slider_row.addWidget(self.slice_slider, 1)
        self.slice_value = QtWidgets.QSpinBox()
        self.slice_value.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.slice_value.setRange(0, 0)
        slider_row.addWidget(self.slice_value)
        layout.addLayout(slider_row)

        self.view_box.pointer_clicked.connect(self.pointer_clicked)
        self.view_box.pointer_dragged.connect(self.pointer_dragged)
        self.view_box.slice_stepped.connect(self._step_slice)
        self.slice_slider.valueChanged.connect(self._slider_changed)
        self.slice_value.valueChanged.connect(self._spin_changed)
        self.plot_widget.scene().sigMouseMoved.connect(self._scene_mouse_moved)

    @property
    def slice_index(self) -> int:
        return self._slice_index

    def configure_shape(self, shape_zyx: tuple[int, int, int]) -> None:
        self._shape_zyx = shape_zyx
        self._needs_initial_fit = True
        maximum = shape_zyx[self.plane.depth_axis] - 1
        self.slice_slider.setRange(0, maximum)
        self.slice_value.setRange(0, maximum)
        height, width = plane_shape(shape_zyx, self.plane)
        self.plot_item.setLimits(
            minXRange=1,
            minYRange=1,
            maxXRange=max(4, width * 4),
            maxYRange=max(4, height * 4),
        )

    def set_slice_index(self, index: int) -> None:
        maximum = self.slice_slider.maximum()
        index = min(max(int(index), 0), maximum)
        self._slice_index = index
        blockers = (
            QtCore.QSignalBlocker(self.slice_slider),
            QtCore.QSignalBlocker(self.slice_value),
        )
        self.slice_slider.setValue(index)
        self.slice_value.setValue(index)
        del blockers
        self.coordinate_label.setText(f"{self.plane.depth_label} = {index}")

    def set_image(self, image: np.ndarray, levels: tuple[float, float]) -> None:
        self.image_item.setImage(np.asarray(image), autoLevels=False, levels=levels)
        height, width = plane_shape(self._shape_zyx, self.plane)
        rectangle = QtCore.QRectF(-0.5, -0.5, width, height)
        self.image_item.setRect(rectangle)
        if self._needs_initial_fit and not self._fit_scheduled:
            self._fit_scheduled = True
            QtCore.QTimer.singleShot(0, self.fit_to_data)

    def set_mask_overlay(
        self,
        image_rgba: np.ndarray,
        *,
        opacity: float,
        visible: bool,
    ) -> None:
        self.mask_item.setImage(
            np.asarray(image_rgba, dtype=np.uint8),
            autoLevels=False,
        )
        height, width = plane_shape(self._shape_zyx, self.plane)
        self.mask_item.setRect(QtCore.QRectF(-0.5, -0.5, width, height))
        self.mask_item.setOpacity(min(max(float(opacity), 0.0), 1.0))
        self.mask_item.setVisible(bool(visible))

    def set_mask_opacity(self, opacity: float) -> None:
        self.mask_item.setOpacity(min(max(float(opacity), 0.0), 1.0))

    def set_mask_visible(self, visible: bool) -> None:
        self.mask_item.setVisible(bool(visible))

    def clear_mask_overlay(self) -> None:
        self.mask_item.clear()
        self.mask_item.setVisible(False)

    def set_radius_overlay(
        self,
        image_rgba: np.ndarray,
        *,
        opacity: float,
        visible: bool,
    ) -> None:
        self.radius_item.setImage(
            np.asarray(image_rgba, dtype=np.uint8),
            autoLevels=False,
        )
        height, width = plane_shape(self._shape_zyx, self.plane)
        self.radius_item.setRect(QtCore.QRectF(-0.5, -0.5, width, height))
        self.radius_item.setOpacity(min(max(float(opacity), 0.0), 1.0))
        self.radius_item.setVisible(bool(visible))

    def set_radius_opacity(self, opacity: float) -> None:
        self.radius_item.setOpacity(min(max(float(opacity), 0.0), 1.0))

    def set_radius_visible(self, visible: bool) -> None:
        self.radius_item.setVisible(bool(visible))

    def clear_radius_overlay(self) -> None:
        self.radius_item.clear()
        self.radius_item.setVisible(False)

    def set_radius_spheres(
        self,
        graphs: dict[str, InstanceGraph],
        *,
        opacity: float,
        visible: bool,
    ) -> None:
        geometry_key = (
            self._slice_index,
            tuple(
                (
                    instance_id,
                    tuple(
                        (node.id, node.position, node.radius)
                        for node in graph.nodes.values()
                        if node.radius > 0
                    ),
                )
                for instance_id, graph in graphs.items()
            ),
        )
        if geometry_key != self._radius_sphere_key:
            self._remove_items(self._radius_sphere_items)
            for graph in graphs.values():
                for node in graph.nodes.values():
                    if node.radius <= 0:
                        continue
                    horizontal, vertical, depth = project_position(
                        self.plane,
                        node.position,
                    )
                    squared_radius = node.radius**2 - (depth - self._slice_index) ** 2
                    if squared_radius <= 0:
                        continue
                    radius = float(np.sqrt(squared_radius))
                    self._add_black_white_circle(
                        self._radius_sphere_items,
                        horizontal,
                        vertical,
                        radius,
                        style=QtCore.Qt.PenStyle.SolidLine,
                        z_value=20.0,
                    )
            self._radius_sphere_key = geometry_key
        opacity = min(max(float(opacity), 0.0), 1.0)
        for item in self._radius_sphere_items:
            item.setOpacity(opacity)
            item.setVisible(bool(visible))

    def set_radius_sphere_opacity(self, opacity: float) -> None:
        opacity = min(max(float(opacity), 0.0), 1.0)
        for item in self._radius_sphere_items:
            item.setOpacity(opacity)

    def set_radius_spheres_visible(self, visible: bool) -> None:
        for item in self._radius_sphere_items:
            item.setVisible(bool(visible))

    def clear_radius_spheres(self) -> None:
        self._remove_items(self._radius_sphere_items)
        self._radius_sphere_key = None

    def fit_to_data(self) -> None:
        height, width = plane_shape(self._shape_zyx, self.plane)
        viewport_width = max(1.0, float(self.view_box.width()))
        viewport_height = max(1.0, float(self.view_box.height()))
        viewport_aspect = viewport_width / viewport_height
        image_aspect = width / height
        if image_aspect >= viewport_aspect:
            target_width = width * 1.04
            target_height = target_width / viewport_aspect
        else:
            target_height = height * 1.04
            target_width = target_height * viewport_aspect
        centre_x = (width - 1) / 2
        centre_y = (height - 1) / 2
        self.view_box.setRange(
            xRange=(centre_x - target_width / 2, centre_x + target_width / 2),
            yRange=(centre_y - target_height / 2, centre_y + target_height / 2),
            padding=0.0,
        )
        self._needs_initial_fit = False
        self._fit_scheduled = False

    def set_cursor(self, horizontal: float, vertical: float) -> None:
        self.vertical_cursor.setPos(horizontal)
        self.horizontal_cursor.setPos(vertical)

    def set_graphs(
        self,
        graphs: dict[str, InstanceGraph],
        *,
        active_instance: str | None,
        selected_node: tuple[str, str] | None,
        selected_edge: tuple[str, Edge] | None,
        action_source: tuple[str, str] | None = None,
        hovered: PickResult | None = None,
        radius_preview: tuple[str, str, float] | None = None,
        emphasize_near_slice_nodes: bool = False,
        near_slice_minimum_opacity: float = 0.0,
        near_slice_maximum_opacity: float = 1.0,
        near_slice_near_distance: float = 0.0,
        near_slice_far_distance: float = DEFAULT_NEAR_SLICE_FAR_DISTANCE,
    ) -> None:
        graph_key = (
            self._slice_index,
            emphasize_near_slice_nodes,
            near_slice_minimum_opacity,
            near_slice_maximum_opacity,
            near_slice_near_distance,
            near_slice_far_distance,
            active_instance,
            tuple(
                (
                    instance_id,
                    graph.color,
                    tuple((node_id, node.position) for node_id, node in graph.nodes.items()),
                    frozenset(graph.edges),
                )
                for instance_id, graph in graphs.items()
            ),
        )
        if graph_key == self._graph_key:
            self.set_highlights(
                selected_node=selected_node,
                selected_edge=selected_edge,
                action_source=action_source,
                hovered=hovered,
                radius_preview=radius_preview,
            )
            return

        self._remove_items(self._highlight_items)
        self._remove_items(self._graph_items)
        self._projected_nodes.clear()
        self._projected_edges.clear()
        self._projected_node_lookup.clear()
        self._projected_edge_lookup.clear()
        self._highlight_key = None

        edge_segments: dict[
            bool,
            list[tuple[tuple[float, float], tuple[float, float], QtGui.QPen]],
        ] = {False: [], True: []}
        node_positions: dict[bool, list[tuple[float, float]]] = {
            False: [],
            True: [],
        }
        node_sizes: dict[bool, list[float]] = {False: [], True: []}
        node_brushes: dict[bool, list[QtGui.QBrush]] = {False: [], True: []}
        node_pens: dict[bool, list[QtGui.QPen]] = {False: [], True: []}

        for instance_id, graph in graphs.items():
            active = instance_id == active_instance
            base_color = QtGui.QColor(graph.color)
            if not base_color.isValid():
                base_color = QtGui.QColor("#4cc9f0")

            for edge in sorted(graph.edges):
                first_node = graph.nodes[edge[0]]
                second_node = graph.nodes[edge[1]]
                first = project_position(self.plane, first_node.position)
                second = project_position(self.plane, second_node.position)
                depth_distance = abs(((first[2] + second[2]) / 2) - self._slice_index)
                alpha = self._depth_alpha(depth_distance, active)
                color = QtGui.QColor(base_color)
                color.setAlpha(alpha)
                first_2d = (first[0], first[1])
                second_2d = (second[0], second[1])
                edge_segments[active].append(
                    (
                        first_2d,
                        second_2d,
                        pg.mkPen(color, width=2.5 if active else 1.5),
                    )
                )
                projected_edge = _ProjectedEdge(
                    instance_id=instance_id,
                    edge=edge,
                    first=first_2d,
                    second=second_2d,
                    active=active,
                )
                self._projected_edges.append(projected_edge)
                self._projected_edge_lookup[(instance_id, edge)] = projected_edge

            for node_id, node in graph.nodes.items():
                horizontal, vertical, depth = project_position(self.plane, node.position)
                depth_distance = abs(depth - self._slice_index)
                alpha = self._node_depth_alpha(
                    depth_distance,
                    active,
                    emphasize_near_slice_nodes,
                    minimum_opacity=near_slice_minimum_opacity,
                    maximum_opacity=near_slice_maximum_opacity,
                    near_distance=near_slice_near_distance,
                    far_distance=near_slice_far_distance,
                )
                color = QtGui.QColor(base_color)
                color.setAlpha(alpha)
                node_positions[active].append((horizontal, vertical))
                node_sizes[active].append(10 if active else 8)
                node_brushes[active].append(pg.mkBrush(color))
                node_pens[active].append(pg.mkPen(color.lighter(145), width=2))
                projected_node = _ProjectedNode(
                    instance_id=instance_id,
                    node_id=node_id,
                    horizontal=horizontal,
                    vertical=vertical,
                    depth=depth,
                    active=active,
                )
                self._projected_nodes.append(projected_node)
                self._projected_node_lookup[(instance_id, node_id)] = projected_node

        for active in (False, True):
            segments = edge_segments[active]
            if segments:
                edges_item = _BatchedEdgesItem(segments)
                edges_item.setZValue(5 if active else 2)
                edges_item.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)
                self.plot_item.addItem(edges_item)
                self._graph_items.append(edges_item)
            positions = node_positions[active]
            if positions:
                scatter = pg.ScatterPlotItem(pxMode=True)
                scatter.setData(
                    pos=np.asarray(positions, dtype=float),
                    size=np.asarray(node_sizes[active], dtype=float),
                    brush=node_brushes[active],
                    pen=node_pens[active],
                )
                scatter.setZValue(15 if active else 10)
                scatter.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)
                self.plot_item.addItem(scatter)
                self._graph_items.append(scatter)

        self._node_pick_positions = np.asarray(
            [(node.horizontal, node.vertical) for node in self._projected_nodes],
            dtype=float,
        ).reshape((-1, 2))
        self._node_pick_active = np.asarray(
            [node.active for node in self._projected_nodes],
            dtype=bool,
        )
        self._edge_pick_first = np.asarray(
            [edge.first for edge in self._projected_edges],
            dtype=float,
        ).reshape((-1, 2))
        self._edge_pick_second = np.asarray(
            [edge.second for edge in self._projected_edges],
            dtype=float,
        ).reshape((-1, 2))
        self._edge_pick_active = np.asarray(
            [edge.active for edge in self._projected_edges],
            dtype=bool,
        )
        self._graph_key = graph_key
        self.set_highlights(
            selected_node=selected_node,
            selected_edge=selected_edge,
            action_source=action_source,
            hovered=hovered,
            radius_preview=radius_preview,
        )

    def set_highlights(
        self,
        *,
        selected_node: tuple[str, str] | None,
        selected_edge: tuple[str, Edge] | None,
        action_source: tuple[str, str] | None = None,
        hovered: PickResult | None = None,
        radius_preview: tuple[str, str, float] | None = None,
    ) -> None:
        hovered_identity = None if hovered is None else hovered.identity
        highlight_key = (
            selected_node,
            selected_edge,
            action_source,
            hovered_identity,
            radius_preview,
        )
        if highlight_key == self._highlight_key:
            return
        self._remove_items(self._highlight_items)

        edge_highlights = (
            (
                None
                if hovered is None or hovered.kind != "edge"
                else (hovered.instance_id, hovered.edge),
                "#76ff03",
                8,
                6,
            ),
            (selected_edge, "#00e5ff", 4, 7),
        )
        for identity, color, width, order in edge_highlights:
            if identity is None or identity[1] is None:
                continue
            projected = self._projected_edge_lookup.get((identity[0], identity[1]))
            if projected is None:
                continue
            curve = pg.PlotCurveItem(
                x=np.asarray((projected.first[0], projected.second[0])),
                y=np.asarray((projected.first[1], projected.second[1])),
                pen=pg.mkPen(color, width=width),
                antialias=True,
            )
            curve.setZValue(order)
            curve.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)
            self.plot_item.addItem(curve)
            self._highlight_items.append(curve)

        identities = {
            "selected": selected_node,
            "action_source": action_source,
            "hovered": (
                None
                if hovered is None or hovered.kind != "node" or hovered.node_id is None
                else (hovered.instance_id, hovered.node_id)
            ),
        }
        emphasis_spots: list[dict[str, object]] = []
        for style in NODE_EMPHASIS_STYLES:
            identity = identities[style.role]
            if identity is None:
                continue
            projected = self._projected_node_lookup.get(identity)
            if projected is None:
                continue
            emphasis_spots.append(
                {
                    "pos": (projected.horizontal, projected.vertical),
                    "size": style.size_2d,
                    "brush": pg.mkBrush(0, 0, 0, 0),
                    "pen": pg.mkPen(style.color, width=style.width),
                }
            )
        if emphasis_spots:
            emphasis = pg.ScatterPlotItem(pxMode=True)
            emphasis.addPoints(emphasis_spots)
            emphasis.setZValue(25)
            emphasis.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)
            self.plot_item.addItem(emphasis)
            self._highlight_items.append(emphasis)
        if radius_preview is not None and radius_preview[2] > 0:
            projected = self._projected_node_lookup.get(radius_preview[:2])
            if projected is not None:
                depth_delta = projected.depth - self._slice_index
                squared_radius = radius_preview[2] ** 2 - depth_delta**2
                if squared_radius >= 0:
                    radius = float(np.sqrt(squared_radius))
                    self._add_black_white_circle(
                        self._highlight_items,
                        projected.horizontal,
                        projected.vertical,
                        radius,
                        style=QtCore.Qt.PenStyle.DashLine,
                        z_value=24.0,
                    )
        self._highlight_key = highlight_key

    def _add_black_white_circle(
        self,
        target: list,
        horizontal: float,
        vertical: float,
        radius: float,
        *,
        style: QtCore.Qt.PenStyle,
        z_value: float,
    ) -> None:
        rectangle = QtCore.QRectF(
            horizontal - radius,
            vertical - radius,
            radius * 2,
            radius * 2,
        )
        for color, width, offset in (
            ("#000000", 4.5, 0.0),
            ("#ffffff", 1.5, 0.1),
        ):
            circle = QtWidgets.QGraphicsEllipseItem(rectangle)
            pen = pg.mkPen(color, width=width, style=style)
            pen.setCosmetic(True)
            circle.setPen(pen)
            circle.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            circle.setZValue(z_value + offset)
            circle.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)
            self.plot_item.addItem(circle)
            target.append(circle)

    def _remove_items(self, items: list[pg.GraphicsObject]) -> None:
        for item in items:
            self.plot_item.removeItem(item)
        items.clear()

    def pick(self, horizontal: float, vertical: float) -> PickResult | None:
        pixel_x, pixel_y = (abs(value) for value in self.view_box.viewPixelSize())
        pixel_x = max(pixel_x, 1e-9)
        pixel_y = max(pixel_y, 1e-9)

        if self._node_pick_positions.size:
            node_offsets = (
                np.asarray((horizontal, vertical)) - self._node_pick_positions
            ) / np.asarray((pixel_x, pixel_y))
            node_distances = np.hypot(node_offsets[:, 0], node_offsets[:, 1])
            candidates = np.flatnonzero(node_distances <= 15)
        else:
            candidates = np.empty(0, dtype=np.intp)
        if candidates.size:
            order = np.lexsort(
                (
                    ~self._node_pick_active[candidates],
                    node_distances[candidates],
                )
            )
            node = self._projected_nodes[int(candidates[order[0]])]
            return PickResult("node", node.instance_id, node_id=node.node_id)

        if self._edge_pick_first.size:
            scale = np.asarray((pixel_x, pixel_y))
            point = np.asarray((horizontal, vertical)) / scale
            first = self._edge_pick_first / scale
            direction = (self._edge_pick_second / scale) - first
            denominator = np.einsum("ij,ij->i", direction, direction)
            numerator = np.einsum("ij,ij->i", point - first, direction)
            fractions = np.divide(
                numerator,
                denominator,
                out=np.zeros_like(numerator),
                where=denominator != 0,
            )
            np.clip(fractions, 0.0, 1.0, out=fractions)
            closest = first + fractions[:, np.newaxis] * direction
            edge_distances = np.hypot(
                point[0] - closest[:, 0],
                point[1] - closest[:, 1],
            )
            candidates = np.flatnonzero(edge_distances <= 8)
        else:
            candidates = np.empty(0, dtype=np.intp)
            fractions = np.empty(0, dtype=float)
        if candidates.size:
            order = np.lexsort(
                (
                    ~self._edge_pick_active[candidates],
                    edge_distances[candidates],
                )
            )
            index = int(candidates[order[0]])
            edge = self._projected_edges[index]
            return PickResult(
                "edge",
                edge.instance_id,
                edge=edge.edge,
                fraction=float(fractions[index]),
            )
        return None

    @staticmethod
    def _depth_alpha(distance: float, active: bool) -> int:
        floor = 70 if active else 38
        peak = 255 if active else 180
        return int(max(floor, peak / (1.0 + distance / 4.0)))

    @classmethod
    def _node_depth_alpha(
        cls,
        distance: float,
        active: bool,
        emphasize_near_slice: bool,
        *,
        minimum_opacity: float = 0.0,
        maximum_opacity: float = 1.0,
        near_distance: float = 0.0,
        far_distance: float = DEFAULT_NEAR_SLICE_FAR_DISTANCE,
    ) -> int:
        baseline = cls._depth_alpha(distance, active)
        if not emphasize_near_slice:
            return baseline
        lower_opacity = min(max(float(minimum_opacity), 0.0), 1.0)
        upper_opacity = min(max(float(maximum_opacity), 0.0), 1.0)
        if lower_opacity > upper_opacity:
            lower_opacity, upper_opacity = upper_opacity, lower_opacity
        near_distance = max(0.0, float(near_distance))
        far_distance = max(near_distance, float(far_distance))
        distance = abs(float(distance))
        if distance <= near_distance:
            opacity = upper_opacity
        elif distance >= far_distance or far_distance == near_distance:
            opacity = lower_opacity
        else:
            fraction = (distance - near_distance) / (far_distance - near_distance)
            opacity = upper_opacity + (lower_opacity - upper_opacity) * fraction
        return round(255 * opacity)

    def _slider_changed(self, value: int) -> None:
        blocker = QtCore.QSignalBlocker(self.slice_value)
        self.slice_value.setValue(value)
        del blocker
        self.slice_requested.emit(self.plane, value)

    def _spin_changed(self, value: int) -> None:
        blocker = QtCore.QSignalBlocker(self.slice_slider)
        self.slice_slider.setValue(value)
        del blocker
        self.slice_requested.emit(self.plane, value)

    def _step_slice(self, plane: Plane, step: int) -> None:
        value = min(max(self._slice_index + step, 0), self.slice_slider.maximum())
        if value != self._slice_index:
            self.slice_requested.emit(plane, value)

    def _scene_mouse_moved(self, scene_position: QtCore.QPointF) -> None:
        if not self.plot_item.sceneBoundingRect().contains(scene_position):
            self._set_hover(None)
            return
        point = self.view_box.mapSceneToView(scene_position)
        self._set_hover(self.pick(float(point.x()), float(point.y())))

    def _set_hover(self, pick: PickResult | None) -> None:
        old_identity = None if self._hover_pick is None else self._hover_pick.identity
        new_identity = None if pick is None else pick.identity
        if old_identity == new_identity:
            return
        self._hover_pick = pick
        self.hover_changed.emit(self.plane, pick)

    def reset_hover_tracking(self) -> None:
        self._hover_pick = None

    def leaveEvent(self, event: QtCore.QEvent) -> None:
        self._set_hover(None)
        super().leaveEvent(event)

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if watched is not self.plot_widget.viewport():
            return super().eventFilter(watched, event)
        event_type = event.type()
        if event_type == QtCore.QEvent.Type.MouseButtonPress:
            button = event.button()
            if button in (
                QtCore.Qt.MouseButton.LeftButton,
                QtCore.Qt.MouseButton.RightButton,
            ):
                self._press_button = button
                self._press_position = event.position().toPoint()
                self._press_modifiers = event.modifiers()
                self._pointer_dragging = False
                event.accept()
                return True
        elif event_type == QtCore.QEvent.Type.MouseMove:
            if self._press_button in (
                QtCore.Qt.MouseButton.LeftButton,
                QtCore.Qt.MouseButton.RightButton,
            ):
                pointer_button = (
                    PointerButton.LEFT
                    if self._press_button == QtCore.Qt.MouseButton.LeftButton
                    else PointerButton.RIGHT
                )
                position = event.position().toPoint()
                if not self._pointer_dragging:
                    distance = (position - self._press_position).manhattanLength()
                    if distance >= QtWidgets.QApplication.startDragDistance():
                        self._pointer_dragging = True
                        self._emit_drag(
                            pointer_button,
                            "start",
                            self._press_position,
                            self._press_modifiers | event.modifiers(),
                        )
                if self._pointer_dragging:
                    self._emit_drag(
                        pointer_button,
                        "move",
                        position,
                        self._press_modifiers | event.modifiers(),
                    )
                event.accept()
                return True
        elif event_type == QtCore.QEvent.Type.MouseButtonRelease:
            button = event.button()
            if button == self._press_button and button in (
                QtCore.Qt.MouseButton.LeftButton,
                QtCore.Qt.MouseButton.RightButton,
            ):
                position = event.position().toPoint()
                if self._pointer_dragging:
                    pointer_button = (
                        PointerButton.LEFT
                        if button == QtCore.Qt.MouseButton.LeftButton
                        else PointerButton.RIGHT
                    )
                    self._emit_drag(
                        pointer_button,
                        "finish",
                        position,
                        self._press_modifiers | event.modifiers(),
                    )
                elif (
                    position - self._press_position
                ).manhattanLength() < QtWidgets.QApplication.startDragDistance():
                    self._emit_click(button, position, event.modifiers())
                self._press_button = QtCore.Qt.MouseButton.NoButton
                self._press_modifiers = QtCore.Qt.KeyboardModifier.NoModifier
                self._pointer_dragging = False
                event.accept()
                return True
        return super().eventFilter(watched, event)

    def _emit_click(
        self,
        button: QtCore.Qt.MouseButton,
        viewport_position: QtCore.QPoint,
        modifiers: QtCore.Qt.KeyboardModifier,
    ) -> None:
        pointer_button = (
            PointerButton.LEFT
            if button == QtCore.Qt.MouseButton.LeftButton
            else PointerButton.RIGHT
        )
        point = self._viewport_to_view(viewport_position)
        self.pointer_clicked.emit(
            self.plane,
            pointer_button,
            float(point.x()),
            float(point.y()),
            modifiers,
        )

    def _emit_drag(
        self,
        button: PointerButton,
        phase: str,
        viewport_position: QtCore.QPoint,
        modifiers: QtCore.Qt.KeyboardModifier,
    ) -> None:
        point = self._viewport_to_view(viewport_position)
        self.pointer_dragged.emit(
            self.plane,
            button,
            phase,
            float(point.x()),
            float(point.y()),
            modifiers,
        )

    def _viewport_to_view(self, position: QtCore.QPoint) -> QtCore.QPointF:
        scene_position = self.plot_widget.mapToScene(position)
        return self.view_box.mapSceneToView(scene_position)
