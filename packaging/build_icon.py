"""Render the app's vector icon to native desktop icon formats."""

from pathlib import Path

from PIL import Image
from PySide6 import QtCore, QtGui, QtSvg


def build_icon(root: Path, platform: str) -> Path:
    directory = root / "build/icons"
    directory.mkdir(parents=True, exist_ok=True)
    renderer = QtSvg.QSvgRenderer(str(root / "src/chronopose_viewer/assets/icon.svg"))
    if not renderer.isValid():
        raise RuntimeError("Invalid app icon")
    raster = QtGui.QImage(1024, 1024, QtGui.QImage.Format.Format_ARGB32)
    raster.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(raster)
    renderer.render(painter)
    painter.end()
    png = directory / "icon.png"
    if not raster.save(str(png)):
        raise RuntimeError("Could not render app icon")
    suffix = ".icns" if platform == "darwin" else ".ico"
    icon = directory / f"CellAnnotator4D{suffix}"
    with Image.open(png) as image:
        image.save(icon)
    return icon
