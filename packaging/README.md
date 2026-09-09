# Desktop bundles

This builds a self-contained, folder-based CellAnnotator4D executable with
PyInstaller. End users do not install Python or micromamba. Build separately
on Windows and macOS, using Python 3.11 or newer **of the target architecture**.
For macOS, build separately for Apple Silicon and Intel; a build is not
automatically universal. The minimum supported OS also depends on the Python,
Qt, and native-library versions used to build it.

## Build

Run from the repository root. Use a fresh build environment so unrelated GUI
toolkits and optional packages in a development environment cannot leak into
the bundle. These commands use a local venv; the existing micromamba development
environment can remain unchanged.

Use a native Python installation (for example, python.org) for these venv
commands. Do not layer a venv over conda Python: PyInstaller may then select
incompatible system libraries. If using micromamba instead, create a dedicated
conda environment with Python and pip, install `'.[bundle]'` there, and run
PyInstaller directly inside that environment without an additional venv.

For example, from the repository root:

```bash
micromamba create -y -n cellannotator-bundle -c conda-forge python=3.11 pip
micromamba run -n cellannotator-bundle python -m pip install '.[bundle]'
micromamba run -n cellannotator-bundle python -m PyInstaller --noconfirm --clean packaging/viewer.spec
micromamba run -n cellannotator-bundle python -m pip freeze > build/bundle-dependencies.txt
```

Windows (PowerShell, with Python 3.11 installed):

```powershell
py -3.11 -m venv build/packaging-venv
build/packaging-venv/Scripts/python.exe -m pip install --upgrade pip
build/packaging-venv/Scripts/python.exe -m pip install ".[bundle]"
build/packaging-venv/Scripts/python.exe -m PyInstaller --noconfirm --clean packaging/viewer.spec
build/packaging-venv/Scripts/python.exe -m pip freeze > build/bundle-dependencies.txt
```

macOS (or Linux, for development checks):

```bash
python3.11 -m venv build/packaging-venv
build/packaging-venv/bin/python -m pip install --upgrade pip
build/packaging-venv/bin/python -m pip install '.[bundle]'
build/packaging-venv/bin/python -m PyInstaller --noconfirm --clean packaging/viewer.spec
build/packaging-venv/bin/python -m pip freeze > build/bundle-dependencies.txt
```

The dependency ranges resolve when installing; keep `bundle-dependencies.txt`
with build records. A release pipeline should pin the tested versions for each
platform. Rebuild after source changes; the spec reads this checkout's `src/`.

Outputs:

- Windows: `dist/CellAnnotator4D/CellAnnotator4D.exe`. Distribute the **entire
  CellAnnotator4D directory**, including `_internal`, not just the executable.
- macOS: `dist/CellAnnotator4D.app`. Distribute the complete app bundle with its
  symlinks preserved.
- Linux: `dist/CellAnnotator4D/CellAnnotator4D`, for local bundle verification.

The spec disables console windows, binary stripping, and UPX compression.
It collects NumPy, SciPy, scikit-image, tifffile, imagecodecs, and PyOpenGL runtime
submodules, data, and native libraries broadly. VisPy shaders/resources and the
PySide6 backend are included explicitly. PyInstaller's Qt hooks collect the
Qt libraries and plugins needed by Widgets, OpenGL, and SVG, while the scientific
hooks handle native dependencies such as BLAS and wheel codec libraries.
Package metadata (including packaged license files) and the project's LICENSE
are included. Tests and alternative Qt bindings are not required at runtime.

## Check the built executable

The bundle has a diagnostic mode that writes JSON and exits, including on
Windows where there is no console. Use an absolute report path in a writable
directory; the parent directory must already exist.

Windows (PowerShell; `Start-Process -Wait` waits for the windowed executable):

```powershell
$bundleReport = Join-Path $PWD "build/bundle-smoke.json"
$bundleProcess = Start-Process -FilePath .\dist\CellAnnotator4D\CellAnnotator4D.exe -ArgumentList "--bundle-smoke-test `"$bundleReport`"" -Wait -PassThru
Get-Content $bundleReport
if ($bundleProcess.ExitCode -ne 0) { throw "Bundle smoke test failed" }
```

macOS:

```bash
dist/CellAnnotator4D.app/Contents/MacOS/CellAnnotator4D --bundle-smoke-test "$PWD/build/bundle-smoke.json"
cat build/bundle-smoke.json
```

On Linux, use `dist/CellAnnotator4D/CellAnnotator4D` instead.
The report must contain `"ok": true` and `"frozen": true`. The check creates
temporary synthetic TIFFs and exercises Deflate/LZW/Zstd codecs, SciPy distance
transforms, scikit-image skeletonization, viewer startup, and project save/open.
It defaults to Qt's offscreen platform with 3D disabled, for headless CI.

Repeat with `--with-3d` appended on a machine with a working graphics display.
This additionally renders a VisPy volume to exercise shader resources and the
OpenGL context. Also launch normally and manually check opening a real TIFF,
editing, saving/reopening, and navigating 2D/3D views. Test on clean target
machines without Python installed, including paths containing spaces and
non-ASCII characters. Passing Linux checks does not validate Windows or macOS.

## Installers and GitHub Releases

After installing `'.[bundle]'`, run `python packaging/build.py` on Windows or
macOS to build the bundle, create the installer/DMG, and smoke-test the installed
payload. Windows needs Inno Setup 6 (`ISCC.exe` on PATH, or its standard install
location). macOS uses the built-in `hdiutil` and `ditto` tools. To repackage an
existing bundle, add `--package-only`. Output is in `dist/installers/`, including
SHA-256 checksums, a dependency manifest, and installation instructions.

The GitHub workflow `.github/workflows/desktop.yml` runs on main,
pull requests, version tags, and manual dispatch. It uses Windows x64,
macOS ARM64, and macOS Intel runners. It installs the dependency baseline in
`packaging/constraints.txt`, builds and checks installers, and runs the test
suite. Download test installers from the workflow run's `unsigned-*` artifacts;
extract the downloaded ZIP to get the installer. These artifacts expire after
30 days. Diagnostics are uploaded even when a build fails.

To prepare a release after merging the changes:

1. Set the same version in `pyproject.toml` and
   `src/chronopose_viewer/__init__.py`, then commit and push it.
2. Tag that commit, for example `git tag v0.5.3`, and push it with
   `git push origin v0.5.3`. The workflow checks the tag against both versions.
3. Wait for all three builds. The workflow creates a **draft** GitHub Release
   containing the installers, checksums, and dependency manifests.
4. Test the downloads on actual machines, then publish the draft through GitHub
   Releases (or `gh release edit v0.5.3 --draft=false`). Draft release downloads
   are not publicly available until it is published.

No signing secrets are required. The release job uses the repository's automatic
`GITHUB_TOKEN` with `contents: write`. Actions must be enabled; organisation
policies may restrict the workflow. Rerunning a tag build can update its draft,
but the workflow refuses to overwrite an already published release. Updates are
manual downloads; there is no automatic updater.

## Scope and compatibility

These installers are unsigned. macOS apps have PyInstaller's local ad-hoc
signature where required by the OS, but no trusted Developer ID signature or
Apple notarization. See [user installation instructions](INSTALL.md).

Python dependencies are bundled, but a compatible OS and working system GPU
driver/OpenGL implementation are still required for 3D. Increasing bundle size
cannot supply a missing GPU driver. The existing `--disable-3d` launch option
remains available. Linux additionally relies on system display/graphics
libraries and a compatible glibc; it is not the distribution target here.

The existing conversion CLI commands remain in the application, but this is a
windowed desktop build and does not provide normal console output on Windows.
Use the Python CLI for interactive terminal conversions until a dedicated
console build or GUI conversion controls are added.

References: [PyInstaller spec files](https://pyinstaller.org/en/stable/spec-files.html)
and [dependency collection hooks](https://pyinstaller.org/en/stable/hooks.html).
