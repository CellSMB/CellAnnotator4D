from __future__ import annotations

from PySide6 import QtCore, QtTest

from chronopose_viewer.range_slider import RangeSlider


def test_range_slider_orders_clamps_and_emits_both_values(qtbot) -> None:
    slider = RangeSlider()
    qtbot.addWidget(slider)
    changes: list[tuple[int, int]] = []
    slider.valuesChanged.connect(lambda lower, upper: changes.append((lower, upper)))

    slider.setValues(80, 20)
    slider.setRange(30, 70)

    assert (slider.minimum(), slider.maximum()) == (30, 70)
    assert (slider.lowerValue(), slider.upperValue()) == (30, 70)
    assert changes == [(20, 80), (30, 70)]


def test_range_slider_handles_respond_independently_to_mouse_and_keyboard(qtbot) -> None:
    slider = RangeSlider()
    qtbot.addWidget(slider)
    slider.resize(240, 28)
    slider.setValues(20, 80)
    slider.show()
    qtbot.waitExposed(slider)

    lower_handle = slider._handle_rect(slider.lowerValue()).center()
    QtTest.QTest.mouseClick(
        slider,
        QtCore.Qt.MouseButton.LeftButton,
        pos=lower_handle,
    )
    QtTest.QTest.keyClick(slider, QtCore.Qt.Key.Key_Right)

    assert (slider.lowerValue(), slider.upperValue()) == (21, 80)

    upper_handle = slider._handle_rect(slider.upperValue()).center()
    QtTest.QTest.mouseClick(
        slider,
        QtCore.Qt.MouseButton.LeftButton,
        pos=upper_handle,
    )
    QtTest.QTest.keyClick(slider, QtCore.Qt.Key.Key_Left)

    assert (slider.lowerValue(), slider.upperValue()) == (21, 79)
