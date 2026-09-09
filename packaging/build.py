"""Build, package, and check unsigned desktop distributions on the target OS."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import plistlib
import shutil
import subprocess
import sys
import tempfile
import tomllib

ROOT = Path(__file__).resolve().parents[1]
VERSION = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
OUTPUT = ROOT / "dist" / "installers"


def run(*args: str | Path, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run([str(arg) for arg in args], check=True, **kwargs)


def smoke(executable: Path, report: Path) -> None:
    report.unlink(missing_ok=True)
    env = os.environ.copy()
    for key in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV", "CONDA_PREFIX"):
        env.pop(key, None)
    with tempfile.TemporaryDirectory(prefix="CellAnnotator check µ ") as directory:
        run(executable, "--bundle-smoke-test", report, cwd=directory, env=env, timeout=120)
    result = json.loads(report.read_text())
    if not result.get("ok") or not result.get("frozen"):
        raise RuntimeError(f"Bundle check failed: {result}")


def package_windows() -> Path:
    compiler = shutil.which("ISCC") or shutil.which("ISCC.exe")
    if not compiler:
        candidate = Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)"))
        candidate /= "Inno Setup 6/ISCC.exe"
        if candidate.is_file():
            compiler = str(candidate)
    if not compiler:
        raise RuntimeError("Install Inno Setup 6 and add ISCC.exe to PATH")
    run(compiler, f"/DAppVersion={VERSION}", f"/DRepoRoot={ROOT}", ROOT / "packaging/windows.iss")
    installer = OUTPUT / f"CellAnnotator4D-{VERSION}-windows-x64-setup.exe"
    # Check the installed payload, not only PyInstaller's output folder.
    with tempfile.TemporaryDirectory(prefix="CellAnnotator install µ ") as directory:
        destination = Path(directory) / "App"
        run(
            installer,
            "/VERYSILENT",
            "/SUPPRESSMSGBOXES",
            "/NORESTART",
            "/SP-",
            "/NOICONS",
            "/TASKS=",
            f"/DIR={destination}",
            f"/LOG={ROOT / 'build' / 'installer-test.log'}",
            timeout=180,
        )
        try:
            smoke(destination / "CellAnnotator4D.exe", ROOT / "build/installed-smoke.json")
        finally:
            run(
                destination / "unins000.exe",
                "/VERYSILENT",
                "/SUPPRESSMSGBOXES",
                "/NORESTART",
                timeout=120,
            )
    return installer


def package_macos() -> Path:
    architecture = "arm64" if platform.machine() == "arm64" else "x64"
    dmg = OUTPUT / f"CellAnnotator4D-{VERSION}-macos-{architecture}.dmg"
    with tempfile.TemporaryDirectory(prefix="cellannotator-dmg-") as directory:
        staging = Path(directory) / "image"
        staging.mkdir()
        run("ditto", ROOT / "dist/CellAnnotator4D.app", staging / "CellAnnotator4D.app")
        (staging / "Applications").symlink_to("/Applications", target_is_directory=True)
        shutil.copy2(ROOT / "packaging/INSTALL.md", staging / "INSTALL.txt")
        run(
            "hdiutil",
            "create",
            "-volname",
            "CellAnnotator4D",
            "-srcfolder",
            staging,
            "-format",
            "UDZO",
            "-ov",
            dmg,
        )
        run("hdiutil", "verify", dmg)
        mounted = run(
            "hdiutil", "attach", "-readonly", "-nobrowse", "-plist", dmg, capture_output=True
        )
        entities = plistlib.loads(mounted.stdout)["system-entities"]
        mount = next(Path(item["mount-point"]) for item in entities if "mount-point" in item)
        try:
            # Copy as a user would, retaining executable permissions and symlinks.
            installed = Path(directory) / "Installed App µ.app"
            run("ditto", mount / "CellAnnotator4D.app", installed)
            smoke(installed / "Contents/MacOS/CellAnnotator4D", ROOT / "build/installed-smoke.json")
        finally:
            run("hdiutil", "detach", mount)
    return dmg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package-only", action="store_true", help="Use the existing PyInstaller bundle"
    )
    args = parser.parse_args()
    if sys.platform not in {"win32", "darwin"}:
        parser.error("Installers must be built on Windows or macOS")
    if platform.machine().lower() not in {"amd64", "x86_64", "arm64"}:
        parser.error("Unsupported architecture")
    if sys.platform == "win32" and platform.machine().lower() == "arm64":
        parser.error("The Windows installer currently targets x64")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (ROOT / "build").mkdir(exist_ok=True)
    if not args.package_only:
        run(
            sys.executable,
            "-m",
            "PyInstaller",
            "--noconfirm",
            "--clean",
            ROOT / "packaging/viewer.spec",
            cwd=ROOT,
        )
    artifact = package_windows() if sys.platform == "win32" else package_macos()
    with artifact.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    artifact.with_suffix(artifact.suffix + ".sha256").write_text(f"{digest}  {artifact.name}\n")
    dependencies = run(sys.executable, "-m", "pip", "freeze", capture_output=True, text=True).stdout
    (OUTPUT / f"{artifact.stem}-dependencies.txt").write_text(dependencies, encoding="utf-8")
    shutil.copy2(ROOT / "packaging/INSTALL.md", OUTPUT / "INSTALL.md")
    print(f"Built and checked: {artifact}")


if __name__ == "__main__":
    main()
