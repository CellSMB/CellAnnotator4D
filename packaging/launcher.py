"""Entry point for the self-contained desktop bundle."""

from __future__ import annotations

import multiprocessing
import os
import sys


def main() -> int:
    multiprocessing.freeze_support()
    # Windowed Windows executables have no standard streams. Some dependencies
    # and the existing CLI commands still write to them.
    for name in ("stdin", "stdout", "stderr"):
        if getattr(sys, name) is None:
            setattr(sys, name, open(os.devnull, "r" if name == "stdin" else "w"))

    # Both libraries discover Qt bindings dynamically; choose the bundled one.
    os.environ["PYQTGRAPH_QT_LIB"] = "PySide6"
    from vispy import app

    app.use_app("pyside6")

    if len(sys.argv) > 1 and sys.argv[1] == "--bundle-smoke-test":
        from bundle_smoke_test import main as smoke_test

        return smoke_test(sys.argv[2:])

    from chronopose_viewer.app import main as run_viewer

    return run_viewer()


if __name__ == "__main__":
    raise SystemExit(main())
