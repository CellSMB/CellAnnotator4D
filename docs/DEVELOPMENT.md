# Source setup and conversion

For the desktop app, use the [release download and user guide](../README.md).
These instructions are for running from source and converting inference outputs.
Run commands from the repository root.

## Environment

All installation and development commands run inside the dedicated micromamba
environment:

```bash
micromamba create -f environment.yml
micromamba run -n chronopose-viewer python -m pip install -e . --no-deps --no-build-isolation
```

To update an existing environment after changing `environment.yml`:

```bash
micromamba update -n chronopose-viewer -f environment.yml --prune
micromamba run -n chronopose-viewer python -m pip install -e . --no-deps --no-build-isolation
```

No system packages or global Python packages are required.

The commands below use `micromamba run` explicitly. Alternatively, activate the
environment with `micromamba activate chronopose-viewer` before running the
shorter commands that start with `chronopose-viewer`.

## Run

```bash
micromamba run -n chronopose-viewer chronopose-viewer
micromamba run -n chronopose-viewer chronopose-viewer path/to/volume.tif
micromamba run -n chronopose-viewer chronopose-viewer path/to/annotations.cpv.json
```

To run without the OpenGL 3D pane, use:

```bash
micromamba run -n chronopose-viewer chronopose-viewer --disable-3d path/to/volume.tif
```

This still opens the Qt application with its 2D panes. `--help` lists launch
options, and `--version` prints the installed version. Each conversion command
also accepts `--help`.

TIFF data are expected to be `TZYX`; a single `ZYX` volume is accepted as a
one-frame series. When TIFF metadata contains extra singleton axes they are
removed. Ambiguous four- and three-dimensional TIFFs are interpreted as
`TZYX` and `ZYX`, respectively, and the interpretation is shown in the status
bar.

Annotation projects are small JSON files. The source TIFF path is stored
relative to the project where possible, so the TIFF and project can be moved
together.

Use **File → Save** (`Ctrl+S`) to save annotations, or **File → Save as…** to
choose another project path. The first save prompts for a `.cpv.json` file,
defaulting to the source image's name and directory. Reopen it with
**File → Open project…** (`Ctrl+Shift+O`). Projects reference the source TIFF;
they do not embed its image data.

## Convert inference outputs

Finalized inference directories can be converted directly into
editable centreline projects:

```bash
micromamba run -n chronopose-viewer chronopose-viewer convert \
  /path/to/infer_work/2x
```

The command finds each matching output image, `_cp_masks.tif`, and
`_lineage.json` set (including sets inside a `final` subdirectory). Each
tracked 3D mask is thinned to a centreline graph, track IDs and colours remain
stable between frames, and the JSON observation links become 1–1, fission,
fusion, start, and end events. The project is written beside the output image
as `<image-name>.cpv.json`.

Directory discovery is recursive. The input may also be an individual output
TIFF, `_cp_masks` TIFF, `_lineage.json`, or `finalize_manifest.json`. The files
must follow the converter's expected tracked-mask and observation-link schema;
arbitrary segmentation TIFFs alone are not sufficient.

Existing project files are always skipped, so rerunning the command does not
replace prior conversion or editing work. Pass physical voxel spacing when it
is known:

```bash
chronopose-viewer convert /path/to/final --voxel-size 2.0 0.5 0.5
```

To control centreline graph density, pass the approximate branch spacing in
voxel coordinates. Junction and leaf positions remain fixed and are included
when distributing the intermediate nodes, avoiding a short leftover segment
at the end of a branch where topology permits:

```bash
chronopose-viewer convert /path/to/final --node-spacing 5
```

`--spacing` is a shorter alias for `--node-spacing`. Omitting the option keeps
the converter's full centreline detail.

Short terminal ticks introduced by voxel thinning are pruned by default. Only
leaf-to-junction paths shorter than 5 voxels are removed; leaf-to-leaf paths
and loops are retained. Change the cutoff or disable pruning with:

```bash
chronopose-viewer convert /path/to/final --min-terminal-branch-length 3
chronopose-viewer convert /path/to/final --min-terminal-branch-length 0
```

Every connected mask component larger than one voxel receives at least two
centreline nodes joined by an edge, including when every arm of a tiny
branched skeleton is shorter than the pruning cutoff. One-voxel components
remain single nodes. Disconnected components belonging to the same tracked
observation are skeletonized separately and retained as disconnected
components of the same instance graph and track ID.

After conversion, populate a radius at every centreline node from the companion
mask's Euclidean distance transform:

```bash
micromamba run -n chronopose-viewer chronopose-viewer radii-from-masks \
  /path/to/infer_work/2x
```

The command accepts a `.cpv.json` project, a directory of projects, the source
or mask TIFF, or its lineage JSON. It updates each project in place. It is
rerunnable and replaces existing radii by default; use `--only-zero` to retain
manual or previously generated non-zero values. `--scale`, `--offset`, and
`--max-search-distance` tune the mask-derived result:

```bash
chronopose-viewer radii-from-masks sample.cpv.json --scale 1.1 --offset -0.25
chronopose-viewer radii-from-masks sample.cpv.json --only-zero
```

The defaults are scale 1, offset 0, and maximum search distance 3 voxels.
The resulting radius is `max(0, (measured_radius + offset) * scale)`.
Nodes that cannot be associated with a mask component are reset to zero and
reported as unmatched. With `--only-zero`, existing non-zero radii are preserved
before mask matching is attempted.


## Development

For self-contained Windows/macOS desktop builds, see
[Desktop bundles](../packaging/README.md). The PyInstaller configuration bundles
Python, Qt, and the scientific dependencies. GitHub Actions builds unsigned
Windows installers and macOS DMGs, and creates draft releases for version tags.

```bash
micromamba run -n chronopose-viewer pytest
micromamba run -n chronopose-viewer ruff check .
micromamba run -n chronopose-viewer ruff format --check src tests
```
