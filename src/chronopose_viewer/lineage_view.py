from __future__ import annotations

from collections.abc import Iterable

from PySide6 import QtCore, QtGui, QtWidgets

from .model import EventKind, GraphProject, InstanceGraph, LineageEvent


class LineageInstanceItem(QtWidgets.QGraphicsObject):
    clicked = QtCore.Signal(int, str)
    activated = QtCore.Signal(int, str)

    WIDTH = 126.0
    HEIGHT = 44.0

    def __init__(self, time: int, graph: InstanceGraph) -> None:
        super().__init__()
        self.time = time
        self.graph = graph
        self._active = False
        self._selected = False
        self._hovered = False
        self.setAcceptHoverEvents(True)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.setToolTip(
            f"t={time} · {graph.name}\n"
            "Cyan outline = active instance. Yellow outline = lineage connection source.\n"
            "Left click selects/connects; right click opens this instance's frame."
        )

    def boundingRect(self) -> QtCore.QRectF:
        return QtCore.QRectF(
            -self.WIDTH / 2,
            -self.HEIGHT / 2,
            self.WIDTH,
            self.HEIGHT,
        )

    def set_selected(self, selected: bool) -> None:
        if selected != self._selected:
            self._selected = selected
            self.update()

    def set_active(self, active: bool) -> None:
        if active != self._active:
            self._active = active
            self.update()

    def paint(
        self,
        painter: QtGui.QPainter,
        option: QtWidgets.QStyleOptionGraphicsItem,
        widget: QtWidgets.QWidget | None = None,
    ) -> None:
        del option, widget
        color = QtGui.QColor(self.graph.color)
        if not color.isValid():
            color = QtGui.QColor("#4cc9f0")
        fill = QtGui.QColor(color)
        fill.setAlpha(220 if self._active or self._hovered or self._selected else 165)
        if self._selected:
            outline = QtGui.QColor("#ffe66d")
            width = 4.0
        elif self._active:
            outline = QtGui.QColor("#00e5ff")
            width = 4.0
        elif self._hovered:
            outline = QtGui.QColor("#ffffff")
            width = 3.0
        else:
            outline = color.lighter(145)
            width = 2.0
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setPen(QtGui.QPen(outline, width))
        painter.setBrush(fill)
        painter.drawRoundedRect(self.boundingRect(), 8, 8)
        if self._active and self._selected:
            active_pen = QtGui.QPen(QtGui.QColor("#00e5ff"), 2.0)
            painter.setPen(active_pen)
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(self.boundingRect().adjusted(4, 4, -4, -4), 5, 5)
        luminance = 0.299 * fill.red() + 0.587 * fill.green() + 0.114 * fill.blue()
        painter.setPen(QtGui.QColor("#10141c" if luminance > 150 else "#ffffff"))
        painter.drawText(
            self.boundingRect().adjusted(6, 3, -6, -3),
            QtCore.Qt.AlignmentFlag.AlignCenter,
            self.graph.name,
        )

    def hoverEnterEvent(self, event: QtWidgets.QGraphicsSceneHoverEvent) -> None:
        self._hovered = True
        self.update()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event: QtWidgets.QGraphicsSceneHoverEvent) -> None:
        self._hovered = False
        self.update()
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.clicked.emit(self.time, self.graph.id)
            event.accept()
            return
        if event.button() == QtCore.Qt.MouseButton.RightButton:
            self.activated.emit(self.time, self.graph.id)
            event.accept()
            return
        super().mousePressEvent(event)


class LineageEdgeItem(QtWidgets.QGraphicsObject):
    clicked = QtCore.Signal(str)

    def __init__(
        self,
        event_id: str,
        path: QtGui.QPainterPath,
        color: QtGui.QColor,
        tooltip: str,
        *,
        dashed: bool = False,
    ) -> None:
        super().__init__()
        self.event_id = event_id
        self.path = path
        self.color = color
        self.dashed = dashed
        self._hovered = False
        stroker = QtGui.QPainterPathStroker()
        stroker.setWidth(14)
        self._shape = stroker.createStroke(path)
        self.setAcceptHoverEvents(True)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.setToolTip(f"{tooltip}\nLeft click removes this lineage event.")
        self.setZValue(-2)

    def boundingRect(self) -> QtCore.QRectF:
        return self._shape.boundingRect()

    def shape(self) -> QtGui.QPainterPath:
        return self._shape

    def paint(
        self,
        painter: QtGui.QPainter,
        option: QtWidgets.QStyleOptionGraphicsItem,
        widget: QtWidgets.QWidget | None = None,
    ) -> None:
        del option, widget
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        halo = QtGui.QPen(QtGui.QColor("#080b10"), 8 if self._hovered else 6)
        halo.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
        painter.setPen(halo)
        painter.drawPath(self.path)
        pen = QtGui.QPen(
            QtGui.QColor("#ffffff") if self._hovered else self.color,
            4 if self._hovered else 2.5,
        )
        pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
        if self.dashed:
            pen.setStyle(QtCore.Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.drawPath(self.path)

    def hoverEnterEvent(self, event: QtWidgets.QGraphicsSceneHoverEvent) -> None:
        self._hovered = True
        self.update()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event: QtWidgets.QGraphicsSceneHoverEvent) -> None:
        self._hovered = False
        self.update()
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.clicked.emit(self.event_id)
            event.accept()
            return
        super().mousePressEvent(event)


class LineageView(QtWidgets.QGraphicsView):
    connection_requested = QtCore.Signal(int, str, int, str)
    event_remove_requested = QtCore.Signal(str)
    instance_activated = QtCore.Signal(int, str)
    selection_changed = QtCore.Signal(object)
    message = QtCore.Signal(str)

    ROW_SPACING = 150.0
    COLUMN_SPACING = 170.0
    LEFT_MARGIN = 105.0
    TOP_MARGIN = 70.0

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setScene(QtWidgets.QGraphicsScene(self))
        self.setBackgroundBrush(QtGui.QColor("#11151d"))
        self.setRenderHints(
            QtGui.QPainter.RenderHint.Antialiasing | QtGui.QPainter.RenderHint.TextAntialiasing
        )
        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setMinimumHeight(320)
        self._project: GraphProject | None = None
        self._current_time = 0
        self._active: tuple[int, str] | None = None
        self._selected: tuple[int, str] | None = None
        self._instance_order: dict[int, list[str]] | None = None
        self._instance_items: dict[tuple[int, str], LineageInstanceItem] = {}
        self._panning = False
        self._pan_position = QtCore.QPoint()
        self._has_been_fitted = False

    @property
    def selected_instance(self) -> tuple[int, str] | None:
        return self._selected

    def set_project(
        self,
        project: GraphProject,
        current_time: int,
        *,
        active_instance_id: str | None = None,
        preserve_view: bool = False,
    ) -> None:
        if project is not self._project and not preserve_view:
            self._has_been_fitted = False
            self._instance_order = None
        self._project = project
        self._current_time = current_time
        self._active = (
            (current_time, active_instance_id)
            if active_instance_id is not None
            and active_instance_id in project.instances_at(current_time)
            else None
        )
        if self._selected is not None:
            time, instance_id = self._selected
            if instance_id not in project.instances_at(time):
                self._selected = None
        self.rebuild()

    def rebuild(self) -> None:
        scene = self.scene()
        scene.clear()
        self._instance_items.clear()
        if self._project is None:
            return

        maximum_instances = max(
            (len(self._project.instances_at(time)) for time in range(self._project.timepoints)),
            default=1,
        )
        scene_width = max(440.0, self.LEFT_MARGIN + (maximum_instances + 1) * self.COLUMN_SPACING)
        row_right = scene_width - 35.0
        positions: dict[tuple[int, str], QtCore.QPointF] = {}

        for time in range(self._project.timepoints):
            y = self.TOP_MARGIN + time * self.ROW_SPACING
            if time == self._current_time:
                highlight = scene.addRect(
                    18,
                    y - 53,
                    scene_width - 36,
                    106,
                    QtGui.QPen(QtGui.QColor("#52617a"), 1.5),
                    QtGui.QBrush(QtGui.QColor(42, 52, 70, 95)),
                )
                highlight.setZValue(-10)
            line = scene.addLine(
                70,
                y,
                row_right,
                y,
                QtGui.QPen(QtGui.QColor("#394252"), 1, QtCore.Qt.PenStyle.DashLine),
            )
            line.setZValue(-9)
            label = scene.addText(f"t = {time}")
            label.setDefaultTextColor(
                QtGui.QColor("#ffffff" if time == self._current_time else "#aab2c0")
            )
            label.setPos(25, y - 15)

            graphs = self._ordered_graphs(time)
            for index, graph in enumerate(graphs):
                x = self.LEFT_MARGIN + 70 + index * self.COLUMN_SPACING
                item = LineageInstanceItem(time, graph)
                item.setPos(x, y)
                item.clicked.connect(self._instance_clicked)
                item.activated.connect(self.instance_activated)
                item.set_active(self._active == (time, graph.id))
                item.set_selected(self._selected == (time, graph.id))
                scene.addItem(item)
                self._instance_items[(time, graph.id)] = item
                positions[(time, graph.id)] = QtCore.QPointF(x, y)

        for event in self._project.lineage_events:
            self._draw_event(event, positions)

        height = self.TOP_MARGIN * 2 + max(1, self._project.timepoints - 1) * self.ROW_SPACING
        scene.setSceneRect(0, 0, scene_width, height)

    def set_active_instance(self, time: int, instance_id: str | None) -> None:
        active = None
        if (
            self._project is not None
            and instance_id is not None
            and 0 <= time < self._project.timepoints
            and instance_id in self._project.instances_at(time)
        ):
            active = (time, instance_id)
        if active == self._active:
            return
        previous = self._active
        self._active = active
        if previous in self._instance_items:
            self._instance_items[previous].set_active(False)
        if active in self._instance_items:
            self._instance_items[active].set_active(True)

    def arrange_by_connections(self) -> None:
        if self._project is None:
            return
        self._instance_order = self._connection_aware_order()
        self.rebuild()
        self.message.emit("Arranged lineage rows to follow their connections.")

    def reset_instance_order(self) -> None:
        if self._project is None:
            return
        self._instance_order = None
        self.rebuild()
        self.message.emit("Restored the default lineage instance order.")

    def _ordered_graphs(self, time: int) -> list[InstanceGraph]:
        assert self._project is not None
        instances = self._project.instances_at(time)
        if self._instance_order is None:
            return list(instances.values())
        ordered_ids = [
            instance_id
            for instance_id in self._instance_order.get(time, ())
            if instance_id in instances
        ]
        seen = set(ordered_ids)
        ordered_ids.extend(instance_id for instance_id in instances if instance_id not in seen)
        return [instances[instance_id] for instance_id in ordered_ids]

    def _connection_aware_order(self) -> dict[int, list[str]]:
        assert self._project is not None
        order = {
            time: list(self._project.instances_at(time)) for time in range(self._project.timepoints)
        }
        predecessors: dict[tuple[int, str], list[tuple[int, str]]] = {}
        successors: dict[tuple[int, str], list[tuple[int, str]]] = {}
        for event in self._project.lineage_events:
            if event.source_time is None or event.target_time is None:
                continue
            for source_id in event.sources:
                source = (event.source_time, source_id)
                for target_id in event.targets:
                    target = (event.target_time, target_id)
                    successors.setdefault(source, []).append(target)
                    predecessors.setdefault(target, []).append(source)

        def positions() -> dict[tuple[int, str], int]:
            return {
                (time, instance_id): index
                for time, instance_ids in order.items()
                for index, instance_id in enumerate(instance_ids)
            }

        def sweep(
            times: Iterable[int], neighbours: dict[tuple[int, str], list[tuple[int, str]]]
        ) -> None:
            locations = positions()
            for time in times:
                current = order[time]
                current_locations = {
                    instance_id: index for index, instance_id in enumerate(current)
                }

                def score(instance_id: str) -> tuple[float, int]:
                    connected_positions = [
                        locations[connected]
                        for connected in neighbours.get((time, instance_id), ())
                        if connected in locations
                    ]
                    if connected_positions:
                        return (
                            sum(connected_positions) / len(connected_positions),
                            current_locations[instance_id],
                        )
                    return (float(current_locations[instance_id]), current_locations[instance_id])

                current.sort(key=score)
                for index, instance_id in enumerate(current):
                    locations[(time, instance_id)] = index

        # Alternating barycentric sweeps reduce crossings while keeping ties and
        # unconnected instances deterministic in their existing/default order.
        for _ in range(6):
            sweep(range(1, self._project.timepoints), predecessors)
            sweep(range(self._project.timepoints - 2, -1, -1), successors)
        return order

    def reset_view(self) -> None:
        if not self.scene().items():
            return
        self.resetTransform()
        self.fitInView(
            self.scene().sceneRect().adjusted(-20, -20, 20, 20),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
        )
        self._has_been_fitted = True

    def ensure_view_fitted(self) -> None:
        if not self._has_been_fitted:
            self.reset_view()

    def clear_selection(self) -> None:
        previous = self._selected
        self._selected = None
        if previous in self._instance_items:
            self._instance_items[previous].set_selected(False)
        self.selection_changed.emit(None)

    def _instance_clicked(self, time: int, instance_id: str) -> None:
        clicked = (time, instance_id)
        if self._selected is None:
            self._selected = clicked
            self._instance_items[clicked].set_selected(True)
            self.selection_changed.emit(clicked)
            self.message.emit(
                "Lineage source selected. Click an instance on another time row, or press Escape."
            )
            return
        if self._selected == clicked:
            self.clear_selection()
            return
        first = self._selected
        if first[0] == time:
            self.message.emit("Choose an instance on a different time row.")
            return
        self.connection_requested.emit(first[0], first[1], time, instance_id)

    def _draw_event(
        self,
        event: LineageEvent,
        positions: dict[tuple[int, str], QtCore.QPointF],
    ) -> None:
        source_points = [
            positions[(event.source_time, instance_id)]
            for instance_id in event.sources
            if event.source_time is not None and (event.source_time, instance_id) in positions
        ]
        target_points = [
            positions[(event.target_time, instance_id)]
            for instance_id in event.targets
            if event.target_time is not None and (event.target_time, instance_id) in positions
        ]
        color = self._event_color(event.kind)
        tooltip = self._event_tooltip(event)

        if source_points and target_points:
            if len(source_points) == 1 and len(target_points) == 1:
                self._add_edge(
                    event.id,
                    self._connection_path(source_points[0], target_points[0]),
                    color,
                    tooltip,
                )
                return
            all_points = source_points + target_points
            hub = QtCore.QPointF(
                sum(point.x() for point in all_points) / len(all_points),
                (
                    max(point.y() for point in source_points)
                    + min(point.y() for point in target_points)
                )
                / 2,
            )
            for point in source_points:
                self._add_edge(
                    event.id,
                    self._connection_path(point, hub),
                    color,
                    tooltip,
                )
            for point in target_points:
                self._add_edge(
                    event.id,
                    self._connection_path(hub, point),
                    color,
                    tooltip,
                )
            diamond = QtGui.QPolygonF(
                [
                    hub + QtCore.QPointF(0, -8),
                    hub + QtCore.QPointF(8, 0),
                    hub + QtCore.QPointF(0, 8),
                    hub + QtCore.QPointF(-8, 0),
                ]
            )
            item = self.scene().addPolygon(
                diamond,
                QtGui.QPen(QtGui.QColor("#ffffff"), 1.5),
                QtGui.QBrush(color),
            )
            item.setZValue(1)
            return

        if target_points:
            for point in target_points:
                start = point - QtCore.QPointF(0, 52)
                self._add_edge(
                    event.id,
                    self._connection_path(start, point),
                    color,
                    tooltip,
                    dashed=True,
                )
                self._add_endpoint_label("START", start + QtCore.QPointF(-24, -22), color)
        elif source_points:
            for point in source_points:
                end = point + QtCore.QPointF(0, 52)
                self._add_edge(
                    event.id,
                    self._connection_path(point, end),
                    color,
                    tooltip,
                    dashed=True,
                )
                self._add_endpoint_label("END", end + QtCore.QPointF(-17, 3), color)

    def _add_edge(
        self,
        event_id: str,
        path: QtGui.QPainterPath,
        color: QtGui.QColor,
        tooltip: str,
        *,
        dashed: bool = False,
    ) -> None:
        edge = LineageEdgeItem(event_id, path, color, tooltip, dashed=dashed)
        edge.clicked.connect(self.event_remove_requested)
        self.scene().addItem(edge)

    def _add_endpoint_label(self, text: str, position: QtCore.QPointF, color: QtGui.QColor) -> None:
        label = self.scene().addText(text)
        label.setDefaultTextColor(color)
        label.setScale(0.75)
        label.setPos(position)

    @staticmethod
    def _connection_path(first: QtCore.QPointF, second: QtCore.QPointF) -> QtGui.QPainterPath:
        path = QtGui.QPainterPath(first)
        delta_y = second.y() - first.y()
        control = max(22.0, abs(delta_y) * 0.42)
        direction = 1 if delta_y >= 0 else -1
        path.cubicTo(
            first + QtCore.QPointF(0, direction * control),
            second - QtCore.QPointF(0, direction * control),
            second,
        )
        return path

    @staticmethod
    def _event_color(kind: EventKind) -> QtGui.QColor:
        colors = {
            EventKind.START: "#80ed99",
            EventKind.END: "#ff8fa3",
            EventKind.ONE_TO_ONE: "#90e0ef",
            EventKind.FISSION: "#ffd166",
            EventKind.FUSION: "#c77dff",
            EventKind.FISSION_FUSION: "#f72585",
        }
        return QtGui.QColor(colors[kind])

    def _event_tooltip(self, event: LineageEvent) -> str:
        if self._project is None:
            return event.label
        sources = self._names(event.source_time, event.sources)
        targets = self._names(event.target_time, event.targets)
        if event.source_time is None:
            return f"start → t{event.target_time}: {targets}"
        if event.target_time is None:
            return f"t{event.source_time}: {sources} → end"
        return f"t{event.source_time}: {sources} — {event.label} → t{event.target_time}: {targets}"

    def _names(self, time: int | None, instance_ids: Iterable[str]) -> str:
        if self._project is None or time is None:
            return "—"
        graphs = self._project.instances_at(time)
        return ", ".join(
            graphs[instance_id].name if instance_id in graphs else instance_id[:8]
            for instance_id in instance_ids
        )

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        factor = 1.18 if event.angleDelta().y() > 0 else 1 / 1.18
        current = self.transform().m11()
        target = current * factor
        if 0.15 <= target <= 5.0:
            self.scale(factor, factor)
        event.accept()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.MiddleButton:
            self._panning = True
            self._pan_position = event.pos()
            self.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._panning:
            delta = event.pos() - self._pan_position
            self._pan_position = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.MiddleButton and self._panning:
            self._panning = False
            self.unsetCursor()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == QtCore.Qt.Key.Key_Escape:
            self.clear_selection()
            event.accept()
            return
        super().keyPressEvent(event)
