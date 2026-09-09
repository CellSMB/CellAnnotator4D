# Chronopose Viewer

Chronopose Viewer is a standalone Python/Qt application for annotating arbitrary
3D centreline instance graphs in `TZYX` TIFF time series and recording
many-to-many lineage relationships between instances. It is deliberately not a
napari plugin and does not import or modify napari.

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

## Run

```bash
micromamba run -n chronopose-viewer chronopose-viewer
micromamba run -n chronopose-viewer chronopose-viewer path/to/volume.tif
micromamba run -n chronopose-viewer chronopose-viewer path/to/annotations.cpv.json
```

TIFF data are expected to be `TZYX`; a single `ZYX` volume is accepted as a
one-frame series. When TIFF metadata contains extra singleton axes they are
removed. Ambiguous four- and three-dimensional TIFFs are interpreted as
`TZYX` and `ZYX`, respectively, and the interpretation is shown in the status
bar.

Annotation projects are small JSON files. The source TIFF path is stored
relative to the project where possible, so the TIFF and project can be moved
together.

## Convert inference outputs

Finalized Chronopose inference directories can be converted directly into
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

## Annotation workflow

1. Open a `TZYX` TIFF with **File → Open TIFF**.
2. Move to the required time with the `T` slider.
3. Use the `Z`, `Y`, and `X` sliders below the XY, XZ, and YZ panes to choose
   the three intersecting slices.
4. In **Instances**, click **Create new**, then click the first node location
   in any 2D pane. The two visible coordinates come from the click and the
   hidden coordinate comes from that pane's current slice.
5. Continue with the numbered modes in **Edit graph**. Keys `1`–`9` switch to
   those tools, `0` selects tool 10, and `R` selects tool 11:

   - **1. Drag node**: left click selects a node of the active instance; left
     drag selects and moves it in any 2D pane. Right click an instance first if
     it is not already active.
   - **2. Add connected node**: left click a node to mark the magenta source,
     then left click empty 2D space to extend it. The new node becomes the next
     source for continuous tracing. Without a source, empty-space clicks are
     ignored; use tool 3 for a disconnected node.
   - **3. Add isolated node** always creates a disconnected node.
   - **4. Connect two nodes** marks its first node as a magenta source, then
     adds an edge to the second node within the same instance. Cycles are
     allowed.
   - **5. Split edge with node**, **6. Remove edge**,
     **7. Remove node + edges**, and **8. Dissolve degree-2 node** perform the
     corresponding topology edits. Removing an edge does not move the
     crosshair or change the view position.
   - **9. Connect two instances**: click a node to mark the first instance's
     magenta source, then click a node in a different instance. The graphs are
     joined by an edge; the first instance keeps its name, colour, ID, and
     reconciled lineage endpoints.
   - **10. Split instance at edge**: click a bridge whose removal produces
     exactly two connected components. The second component becomes a new
     instance. Loop edges and ambiguous splits are refused.
   - **11. Edit node radius**: left or right click a node to select it. In a
     2D pane, Ctrl+left drag away from the node to preview its radius as a
     dashed circle. The saved radius and filled volume change only when the
     mouse button is released; Escape cancels the preview.

The edit tool only acts while the **Edit graph** tab is open. Navigation is
available from every tab:

- right click a node or edge in 2D or 3D to select it and recenter all slice
  panes;
- right click empty space in a 2D pane to move the crosshair there while
  retaining that pane's hidden slice coordinate;
- right drag from empty space in a 2D pane to move the crosshair continuously.
  The drag remains cursor navigation when it crosses nodes or edges, so it
  never snaps or changes the selection;
- middle drag pans a 2D or 3D view;
- the mouse wheel zooms;
- Shift + mouse wheel steps through the hovered 2D pane's slice axis.

Items compatible with the current edit tool glow on hover, previewing what a
left click will affect. Node state remains readable when states overlap:
selection is a cyan ring, the tool 2, tool 4, or tool 9 source is magenta, and
hover is a larger lime-green ring. Drag sources do not use the magenta ring.

Escape cancels pending new-instance placement, stops an active drag, discards
an uncommitted radius preview, clears a tool 2, 4, or 9 source, clears hover,
and cancels the selected lineage source. It keeps the active tool and
right-click selection unchanged. In tool 2, empty-space clicks remain disabled
after Escape until another source node is chosen. The dark tool-help panel
describes the applicable Escape behavior for every tool.

`Ctrl+Z` undoes annotation edits and `Ctrl+Y` or `Ctrl+Shift+Z` redoes them.
History covers node, edge, instance, node-position, node-radius, automatic
radius, and lineage changes; one continuous node or radius drag is one undo
step. Cursor and slice navigation, 2D and 3D camera movement, contrast,
overlay opacity/visibility, active tool, and hover are not added to history.
Undoing back to the last save point also clears the unsaved-change marker.

In **Instances**, **Copy to previous** and **Copy to next** create a new
instance on the adjacent frame with identical positions and topology but fresh
instance and node IDs. Copying does not create a lineage event automatically.
When instances are joined, compatible lineage events are consolidated and an
operation with contradictory source/target frames is refused without changing
either graph. Splitting an instance adds both resulting instances to its
existing incoming and outgoing lineage events.

All current-frame instances are drawn in every view. The active instance is
brighter. Nodes and edges are depth-faded in 2D but remain pickable; right
clicking either recentres the three slice axes. In **Opacity**, enable
**2D graph display → Adjust node opacity by slice distance** to apply
configurable absolute opacity.
The two-ended percentage slider sets the lower and upper opacity: 0% is
invisible and 100% is fully opaque. The near distance applies the upper opacity
to nodes at or below its threshold; the far distance applies the lower opacity
at or beyond its threshold, with linear interpolation between them. By default,
nodes on the current slice are fully opaque and fade to invisible at four
slices away. Disabling the option restores normal depth-faded opacity. Edge
appearance is unchanged. Nodes and edges can also be selected or edited in 3D,
while node dragging is intentionally limited to the 2D panes. The 3D graph is
rendered as an x-ray overlay so bright MIP voxels cannot make graph elements
flicker or disappear; perspective and spatial depth are retained.

The 3D pane uses nearest-sampled maximum-intensity projection:

- left drag rotates;
- middle drag pans;
- the mouse wheel zooms;
- right click selects and recentres a node or edge;
- a stationary left click performs the current non-drag edit tool;
- `Ctrl+R` resets the 3D camera;
- `Ctrl+0` fits all 2D views to their complete image extents.

The two intensity sliders apply the same minimum and maximum display limits to
all slice panes and the 3D MIP.

## Instance mask overlays

When an image or project is opened, the viewer looks beside its source TIFF for
a companion mask named with the `_cp_masks`, `_masks`, or `_mask` suffix (in
that priority order, accepting both `.tif` and `.tiff`). For example,
`10000.tiff` automatically loads `10000_cp_masks.tif`. A missing mask has no
effect on graph editing, and an invalid optional mask is reported without
preventing the source image or project from opening.

Use **Opacity → Instance mask overlay → Show masks** to toggle the overlay and
its opacity slider to control translucent blending. Masks are shown in all
three slice panes and as a categorical translucent volume in the 3D pane. They
are visual guidance only and do not participate in picking or editing.

Each disconnected 3D mask component takes the colour of the graph with the
most centreline voxels inside it. Components without an overlapping graph
receive a stable pseudo-random colour. Component identification is cached per
timepoint, while graph-to-colour matches are recomputed only when that
timepoint's graph geometry or colours change.

## Node-radius volumes

Every graph node stores one non-negative radius in voxel-coordinate units.
Older projects load with radius zero, and new nodes also start at zero, which
means off. Connected non-zero nodes are rendered as the filled union of their
balls and a linearly changing swept-ball envelope along each edge. This forms
a continuous tapered worm while retaining arbitrary branches, loops, and
disconnected components. The true cross-section is shown in each 2D slice and
a downsampled categorical volume is shown in 3D.

In **Opacity → Node radii**, the connected fill and independent node circles
each have their own **All**, **Selected**, and **None** control. All displays
every current-frame instance; Selected restricts that layer to the active
mitochondrion; None hides it. The node-circle layer draws a black-backed thin
white circle for every individual node sphere intersecting each slice, without
connecting neighbouring nodes. In 3D, those radii remain transparent,
camera-facing 2D circles centred on their nodes instead of becoming white
voxel shells. Its opacity is independent of the filled volume. Each fill uses
its instance graph colour and never participates in picking. Slice rasters and
the bounded 3D fill are cached; a manual Ctrl-drag renders only a dashed
black-and-white circle until release so it does not repeatedly rebuild the
filled volume.

All mask, radius-fill, node-sphere, and distance-based node-opacity controls
are grouped in the **Opacity** tab. These settings affect display only and do
not mark the annotation project as modified.

The **Estimate radii from intensity** controls in **Edit graph** provide a
rerunnable in-app alternative for bright tubular signal. Choose a boundary
level between local background and centre intensity, a maximum radius, the
radial-ray percentile, and radial smoothing, then apply the estimate to the
active instance or every graph at the current timepoint. Each run is one
undoable edit. The conservative default 10th ray percentile was calibrated
against the mask-derived radii in the supplied `10000` example; different
imaging conditions can be tuned and rerun without restarting the viewer.

## Lineage events

The **Lineage** tab displays every instance as a node on horizontal time rows,
with positive time running downward. Left click an instance to highlight it,
then click an instance on any other time row to connect them. Escape cancels
the highlighted node. Middle drag pans the scene and the mouse wheel zooms it.
Right clicking an instance opens that frame and makes the instance active. The
active instance in the current frame has a cyan outline; the yellow outline is
reserved for the temporary lineage connection source.

Repeated pairwise connections merge automatically, inferring the event from
its cardinality:

- one source to one target: 1–1;
- one source to many targets: fission;
- many sources to one target: fusion;
- many sources to many targets: fission + fusion.

Use **Mark START** or **Mark END** on the highlighted instance when one side of
its lineage is absent. Each instance can have at most one incoming and one
outgoing event, preventing contradictory assignments. Hovering a lineage edge
highlights it; left clicking it removes the represented event.

**Arrange** reorders each time row to align connected instances and reduce
crossings. It changes only the view; **Default order** restores the order stored
in the project. **Auto 1→1** makes a maximal one-to-one assignment between free
endpoints in each pair of adjacent frames. It minimizes a global centerline
matching cost based on physical position, length, extent, and topology, while
leaving all existing connections and START/END markers untouched.

## Development

```bash
micromamba run -n chronopose-viewer pytest
micromamba run -n chronopose-viewer ruff check .
micromamba run -n chronopose-viewer ruff format --check src tests
```
