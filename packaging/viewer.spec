# Build on each target OS/architecture with: python -m PyInstaller packaging/viewer.spec
from pathlib import Path
import sys
import tomllib

from PyInstaller.utils.hooks import (
    collect_all,
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
    copy_metadata,
)

root = Path(SPECPATH).parent
version = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
datas = [(str(root / "LICENSE"), "licenses/chronopose-viewer")]
binaries = []
hiddenimports = [
    "PySide6.QtCore", "PySide6.QtGui", "PySide6.QtWidgets",
    "PySide6.QtOpenGL", "PySide6.QtOpenGLWidgets", "PySide6.QtSvg",
    "vispy.app.backends._pyside6", "bundle_smoke_test",
]


def runtime_module(name):
    # Keep runtime modules, including lazy codecs and compiled extensions, while
    # avoiding test suites that import unrelated development dependencies.
    return (
        not any(part in {"tests", "test", "testing", "_testing"} for part in name.split("."))
        and not name.startswith(("OpenGL.Tk", "OpenGL.osmesa"))
    )


# Prefer completeness over size for the scientific stack and its native codecs.
# PyInstaller's standard/contributed hooks additionally handle wheel .libs
# directories (BLAS, Fortran, image codecs) and platform-specific dependencies.
for package in ("numpy", "scipy", "skimage", "tifffile", "imagecodecs", "OpenGL"):
    package_datas, package_binaries, package_imports = collect_all(
        package, filter_submodules=runtime_module,
    )
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_imports

# VisPy backends and pyqtgraph Qt bindings are optional alternatives. Collect
# their resources/native libraries, and let analysis follow the PySide6 backend
# rather than importing every alternative GUI toolkit.
for package in ("vispy", "pyqtgraph"):
    datas += collect_data_files(package)
    binaries += collect_dynamic_libs(package)
hiddenimports += collect_submodules("vispy.visuals", filter=runtime_module)
hiddenimports += collect_submodules("vispy.glsl", filter=runtime_module)
datas += copy_metadata("chronopose-viewer", recursive=True)

a = Analysis(
    [str(root / "packaging" / "launcher.py")],
    pathex=[str(root / "src"), str(root / "packaging")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    # Only PySide6 is supported by this app. Qt's own hooks collect its platform
    # plugins, image plugins, translations, and dependent native libraries.
    excludes=["PyQt5", "PyQt6", "PySide2", "tkinter"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="CellAnnotator4D",
    console=False,
    strip=False,
    upx=False,
    argv_emulation=False,
)
bundle = COLLECT(
    exe, a.binaries, a.datas,
    strip=False,
    upx=False,
    name="CellAnnotator4D",
)
if sys.platform == "darwin":
    app = BUNDLE(
        bundle,
        name="CellAnnotator4D.app",
        bundle_identifier="org.cellannotator4d.viewer",
        version=version,
        info_plist={
            "CFBundleDisplayName": "CellAnnotator4D",
            "CFBundleShortVersionString": version,
            "NSHighResolutionCapable": True,
        },
    )
