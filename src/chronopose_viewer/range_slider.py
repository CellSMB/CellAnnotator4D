from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets


class RangeSlider(QtWidgets.QWidget):
    """A compact horizontal integer slider with independently draggable endpoints."""

    valuesChanged = QtCore.Signal(int, int)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._minimum = 0
        self._maximum = 100
        self._lower_value = 0
        self._upper_value = 100
        self._active_handle: str | None = None
        self._last_active_handle = "upper"
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setMinimumWidth(100)

    def minimum(self) -> int:
        return self._minimum

    def maximum(self) -> int:
        return self._maximum

    def lowerValue(self) -> int:
        return self._lower_value

    def upperValue(self) -> int:
        return self._upper_value

    def setRange(self, minimum: int, maximum: int) -> None:
        minimum = int(minimum)
        maximum = int(maximum)
        if minimum > maximum:
            minimum, maximum = maximum, minimum
        self._minimum = minimum
        self._maximum = maximum
        self.setValues(self._lower_value, self._upper_value)
        self.update()

    def setValues(self, lower: int, upper: int) -> None:
        lower = min(max(int(lower), self._minimum), self._maximum)
        upper = min(max(int(upper), self._minimum), self._maximum)
        if lower > upper:
            lower, upper = upper, lower
        if (lower, upper) == (self._lower_value, self._upper_value):
            return
        self._lower_value = lower
        self._upper_value = upper
        self.valuesChanged.emit(lower, upper)
        self.update()

    def sizeHint(self) -> QtCore.QSize:
        option = self._style_option(self._lower_value)
        thickness = self.style().pixelMetric(
            QtWidgets.QStyle.PixelMetric.PM_SliderThickness,
            option,
            self,
        )
        return QtCore.QSize(170, max(24, thickness))

    def minimumSizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(100, self.sizeHint().height())

    def paintEvent(self, _event: QtGui.QPaintEvent) -> None:
        painter = QtWidgets.QStylePainter(self)
        groove_option = self._style_option(self._lower_value)
        groove_option.subControls = QtWidgets.QStyle.SubControl.SC_SliderGroove
        painter.drawComplexControl(
            QtWidgets.QStyle.ComplexControl.CC_Slider,
            groove_option,
        )

        lower_rect = self._handle_rect(self._lower_value)
        upper_rect = self._handle_rect(self._upper_value)
        groove_rect = self.style().subControlRect(
            QtWidgets.QStyle.ComplexControl.CC_Slider,
            groove_option,
            QtWidgets.QStyle.SubControl.SC_SliderGroove,
            self,
        )
        selection = QtCore.QRect(
            lower_rect.center().x(),
            groove_rect.center().y() - 2,
            max(1, upper_rect.center().x() - lower_rect.center().x()),
            4,
        )
        painter.fillRect(
            selection,
            self.palette().color(
                QtGui.QPalette.ColorGroup.Active
                if self.isEnabled()
                else QtGui.QPalette.ColorGroup.Disabled,
                QtGui.QPalette.ColorRole.Highlight,
            ),
        )

        handles = (("lower", self._lower_value), ("upper", self._upper_value))
        if self._active_handle == "lower":
            handles = (handles[1], handles[0])
        for name, value in handles:
            option = self._style_option(value)
            option.subControls = QtWidgets.QStyle.SubControl.SC_SliderHandle
            if name == self._active_handle:
                option.activeSubControls = QtWidgets.QStyle.SubControl.SC_SliderHandle
                option.state |= QtWidgets.QStyle.StateFlag.State_Sunken
            painter.drawComplexControl(
                QtWidgets.QStyle.ComplexControl.CC_Slider,
                option,
            )

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() != QtCore.Qt.MouseButton.LeftButton or not self.isEnabled():
            event.ignore()
            return
        self.setFocus(QtCore.Qt.FocusReason.MouseFocusReason)
        point = event.position().toPoint()
        lower_rect = self._handle_rect(self._lower_value)
        upper_rect = self._handle_rect(self._upper_value)
        if lower_rect.contains(point) and upper_rect.contains(point):
            handle = "lower" if self._last_active_handle == "upper" else "upper"
        elif lower_rect.contains(point):
            handle = "lower"
        elif upper_rect.contains(point):
            handle = "upper"
        else:
            handle = (
                "lower"
                if abs(point.x() - lower_rect.center().x())
                <= abs(point.x() - upper_rect.center().x())
                else "upper"
            )
        self._active_handle = handle
        self._last_active_handle = handle
        self._move_active_handle(self._value_at(point.x()))
        self.update()
        event.accept()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._active_handle is None:
            event.ignore()
            return
        self._move_active_handle(self._value_at(round(event.position().x())))
        event.accept()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton and self._active_handle is not None:
            self._active_handle = None
            self.update()
            event.accept()
            return
        event.ignore()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        key = event.key()
        if key in (QtCore.Qt.Key.Key_Left, QtCore.Qt.Key.Key_Down):
            delta = -1
        elif key in (QtCore.Qt.Key.Key_Right, QtCore.Qt.Key.Key_Up):
            delta = 1
        else:
            super().keyPressEvent(event)
            return
        handle = self._last_active_handle
        self._active_handle = handle
        value = self._lower_value if handle == "lower" else self._upper_value
        self._move_active_handle(value + delta)
        self._active_handle = None
        event.accept()

    def _move_active_handle(self, value: int) -> None:
        if self._active_handle == "lower":
            self.setValues(min(value, self._upper_value), self._upper_value)
        elif self._active_handle == "upper":
            self.setValues(self._lower_value, max(value, self._lower_value))

    def _style_option(self, value: int) -> QtWidgets.QStyleOptionSlider:
        option = QtWidgets.QStyleOptionSlider()
        option.initFrom(self)
        option.orientation = QtCore.Qt.Orientation.Horizontal
        option.minimum = self._minimum
        option.maximum = self._maximum
        option.sliderPosition = value
        option.sliderValue = value
        option.singleStep = 1
        option.pageStep = 10
        option.upsideDown = False
        return option

    def _handle_rect(self, value: int) -> QtCore.QRect:
        option = self._style_option(value)
        return self.style().subControlRect(
            QtWidgets.QStyle.ComplexControl.CC_Slider,
            option,
            QtWidgets.QStyle.SubControl.SC_SliderHandle,
            self,
        )

    def _value_at(self, horizontal: int) -> int:
        option = self._style_option(self._lower_value)
        groove = self.style().subControlRect(
            QtWidgets.QStyle.ComplexControl.CC_Slider,
            option,
            QtWidgets.QStyle.SubControl.SC_SliderGroove,
            self,
        )
        handle = self.style().subControlRect(
            QtWidgets.QStyle.ComplexControl.CC_Slider,
            option,
            QtWidgets.QStyle.SubControl.SC_SliderHandle,
            self,
        )
        slider_minimum = groove.x()
        slider_maximum = groove.right() - handle.width() + 1
        span = max(1, slider_maximum - slider_minimum)
        position = round(horizontal - handle.width() / 2) - slider_minimum
        return QtWidgets.QStyle.sliderValueFromPosition(
            self._minimum,
            self._maximum,
            min(max(position, 0), span),
            span,
        )
