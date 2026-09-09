from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from . import __version__
from .main_window import MainWindow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chronopose-viewer",
        description="Edit 3D instance graphs and lineage events in TZYX TIFF data.",
        epilog=(
            "Convert Chronopose inference outputs without opening the GUI:\n"
            "  chronopose-viewer convert PATH\n"
            "Then populate node radii from the companion masks:\n"
            "  chronopose-viewer radii-from-masks PATH"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        help="Optional TZYX TIFF or .cpv.json annotation project to open",
    )
    parser.add_argument(
        "--disable-3d",
        action="store_true",
        help="Disable the OpenGL MIP pane (useful for remote/headless sessions)",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def build_convert_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chronopose-viewer convert",
        description=(
            "Extract 3D centrelines and lineage events from finalized Chronopose tracked masks."
        ),
    )
    parser.add_argument(
        "path",
        type=Path,
        help=(
            "Inference directory, final directory, output TIFF, *_cp_masks TIFF, or *_lineage.json"
        ),
    )
    parser.add_argument(
        "--voxel-size",
        nargs=3,
        type=float,
        default=(1.0, 1.0, 1.0),
        metavar=("Z", "Y", "X"),
        help="Physical Z Y X voxel size stored in the project (default: 1 1 1)",
    )
    parser.add_argument(
        "--node-spacing",
        "--spacing",
        type=float,
        default=None,
        metavar="VOXELS",
        help=(
            "Target arc-length spacing along each branch; fixed junction and "
            "leaf nodes are included when distributing intermediate nodes"
        ),
    )
    parser.add_argument(
        "--min-terminal-branch-length",
        "--min-branch-length",
        type=float,
        default=5.0,
        metavar="VOXELS",
        help=(
            "Remove leaf-to-junction branches shorter than this arc length "
            "(default: 5; use 0 to disable)"
        ),
    )
    return parser


def build_mask_radii_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chronopose-viewer radii-from-masks",
        description=(
            "Populate node radii in existing .cpv.json projects from companion mask "
            "distance transforms."
        ),
    )
    parser.add_argument(
        "path",
        type=Path,
        help=("Project, directory containing projects, source TIFF, mask TIFF, or lineage JSON"),
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        metavar="FACTOR",
        help="Multiply every measured radius by this factor (default: 1)",
    )
    parser.add_argument(
        "--offset",
        type=float,
        default=0.0,
        metavar="VOXELS",
        help="Add this value to each distance-transform radius before scaling (default: 0)",
    )
    parser.add_argument(
        "--max-search-distance",
        type=float,
        default=3.0,
        metavar="VOXELS",
        help=(
            "Associate a graph node just outside its mask with a crossed component this "
            "distance away (default: 3)"
        ),
    )
    parser.add_argument(
        "--only-zero",
        action="store_true",
        help="Preserve every existing non-zero manual or automatic radius",
    )
    return parser


def _run_conversion(argv: list[str]) -> int:
    from .converter import ConversionError, convert_inference_path

    parser = build_convert_parser()
    arguments = parser.parse_args(argv)
    try:
        results = convert_inference_path(
            arguments.path,
            voxel_size_zyx=tuple(arguments.voxel_size),
            node_spacing=arguments.node_spacing,
            min_terminal_branch_length=arguments.min_terminal_branch_length,
        )
    except (ConversionError, OSError, ValueError) as error:
        print(f"{parser.prog}: error: {error}", file=sys.stderr)
        return 2

    for result in results:
        if result.status == "skipped":
            print(f"Skipped existing project: {result.bundle.project_path}")
        else:
            print(
                f"Created {result.bundle.project_path} "
                f"({result.instances} instances, {result.nodes} nodes, "
                f"{result.edges} edges, {result.lineage_events} lineage events)"
            )
    return 0


def _run_mask_radii(argv: list[str]) -> int:
    from .radius_estimation import update_project_radii_from_masks

    parser = build_mask_radii_parser()
    arguments = parser.parse_args(argv)
    try:
        results = update_project_radii_from_masks(
            arguments.path,
            scale=arguments.scale,
            offset=arguments.offset,
            max_search_distance=arguments.max_search_distance,
            only_zero=arguments.only_zero,
        )
    except (KeyError, OSError, TypeError, ValueError) as error:
        print(f"{parser.prog}: error: {error}", file=sys.stderr)
        return 2

    for result in results:
        details = (
            f"{result.nodes_updated} updated, {result.nodes_unmatched} unmatched"
            f", {result.nodes_skipped} preserved"
        )
        print(f"Updated {result.project_path} ({details})")
    return 0


def main(argv: list[str] | None = None) -> int:
    command_line = list(sys.argv[1:] if argv is None else argv)
    if command_line and command_line[0] in {"convert", "convert-inference"}:
        return _run_conversion(command_line[1:])
    if command_line and command_line[0] in {
        "radii-from-masks",
        "radius-from-masks",
        "mask-radii",
    }:
        return _run_mask_radii(command_line[1:])

    arguments = build_parser().parse_args(command_line)
    if arguments.disable_3d:
        os.environ["CHRONOPOSE_VIEWER_DISABLE_3D"] = "1"
    QtCore.QCoreApplication.setOrganizationName("Chronopose")
    QtCore.QCoreApplication.setApplicationName("CellAnnotator4D")
    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    application.setWindowIcon(QtGui.QIcon(str(Path(__file__).with_name("assets") / "icon.svg")))
    application.setStyle("Fusion")
    window = MainWindow(arguments.path)
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
