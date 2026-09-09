"""Exercise the frozen runtime; write a report even without a console."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import traceback


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="JSON report to create")
    parser.add_argument("--with-3d", action="store_true", help="Require an OpenGL display")
    args = parser.parse_args(argv)
    report = {"ok": False, "frozen": bool(getattr(sys, "frozen", False)), "checks": []}
    try:
        if not args.with_3d:
            os.environ["QT_QPA_PLATFORM"] = "offscreen"
            os.environ["CHRONOPOSE_VIEWER_DISABLE_3D"] = "1"
        else:
            os.environ.pop("CHRONOPOSE_VIEWER_DISABLE_3D", None)

        import numpy as np
        from OpenGL import GL
        from PySide6 import QtWidgets
        from scipy.ndimage import distance_transform_edt
        from skimage.morphology import skeletonize
        import tifffile
        from vispy import app, scene

        from chronopose_viewer.io import load_project, load_tiff, save_project
        from chronopose_viewer.main_window import MainWindow
        from chronopose_viewer.model import GraphProject

        assert callable(GL.glGetString)
        assert app.use_app().backend_name.lower() == "pyside6"
        report["checks"].append("Qt/VisPy/PyOpenGL imports")
        volume = np.zeros((2, 9, 16, 16), dtype=np.uint16)
        volume[:, 2:7, 5:11, 5:11] = 1000
        assert distance_transform_edt(volume[0] > 0).max() > 0
        assert skeletonize(volume[0] > 0).any()
        report["checks"].append("SciPy distance transform / scikit-image skeletonization")
        application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        with tempfile.TemporaryDirectory(prefix="cellannotator-bundle-") as directory:
            directory = Path(directory)
            for compression in ("deflate", "lzw", "zstd"):
                path = directory / f"{compression}.tif"
                tifffile.imwrite(path, volume, metadata={"axes": "TZYX"}, compression=compression)
                np.testing.assert_array_equal(load_tiff(path).data, volume)
            report["checks"].append("Deflate/LZW/Zstd TIFF round trips")

            window = MainWindow()
            window.show()
            application.processEvents()
            window.open_path(path)
            application.processEvents()
            assert window.project is not None, "Viewer did not load the test TIFF"
            # Use the same model and I/O as the editor, including relative paths.
            project_path = directory / "smoke.cpv.json"
            graph = window.project.add_instance(0, name="Bundle check")
            first = graph.add_node((3, 6, 6), radius=1.5)
            second = graph.add_node((5, 8, 8), radius=2.0)
            graph.add_edge(first, second)
            save_project(window.project, project_path)
            project, restored = load_project(project_path)
            assert isinstance(project, GraphProject)
            assert project.to_dict() == window.project.to_dict()
            np.testing.assert_array_equal(restored.data, volume)
            report["checks"].append("Viewer startup / project save and reopen")
            if args.with_3d:
                assert window.mip_view.rendering_enabled, "Viewer fell back to 2D"
                # Rendering forces shader loading and a real GL context; merely
                # constructing the main window can silently fall back to 2D.
                canvas = scene.SceneCanvas(size=(128, 128), show=True)
                view = canvas.central_widget.add_view()
                scene.visuals.Volume(volume[0].astype(np.float32), parent=view.scene)
                view.camera = "turntable"
                view.camera.set_range()
                pixels = canvas.render()
                assert pixels.shape[:2] == (128, 128)
                canvas.close()
                report["checks"].append("OpenGL volume shader rendering")
            window.close()
            application.processEvents()
        report["ok"] = True
    except Exception:
        report["error"] = traceback.format_exc()
    args.report.resolve().write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if report["ok"] else 1
