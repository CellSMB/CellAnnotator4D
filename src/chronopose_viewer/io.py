from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

from .model import GraphProject


TIFF_SUFFIXES = (".tif", ".tiff")
MASK_NAME_SUFFIXES = ("_cp_masks", "_masks", "_mask")


@dataclass(slots=True)
class MaskData:
    path: Path
    data: np.ndarray
    source_axes: str
    axes_inferred: bool

    @property
    def shape_tzyx(self) -> tuple[int, int, int, int]:
        return tuple(int(value) for value in self.data.shape)  # type: ignore[return-value]


@dataclass(slots=True)
class VolumeData:
    path: Path
    data: np.ndarray
    source_axes: str
    axes_inferred: bool
    masks: MaskData | None = None
    mask_error: str | None = None

    @property
    def shape_tzyx(self) -> tuple[int, int, int, int]:
        return tuple(int(value) for value in self.data.shape)  # type: ignore[return-value]


def _normalise_tiff_array(array: np.ndarray, axes: str | None) -> tuple[np.ndarray, str, bool]:
    source_axes = (axes or "").upper()
    inferred = False

    if source_axes and len(source_axes) == array.ndim:
        squeeze_axes = tuple(
            index
            for index, (axis, size) in enumerate(zip(source_axes, array.shape, strict=True))
            if axis not in "TZYX" and size == 1
        )
        if squeeze_axes:
            array = np.squeeze(array, axis=squeeze_axes)
            source_axes = "".join(
                axis for index, axis in enumerate(source_axes) if index not in squeeze_axes
            )

    if set(source_axes) == set("TZYX") and len(source_axes) == 4:
        order = tuple(source_axes.index(axis) for axis in "TZYX")
        return np.transpose(array, order), source_axes, inferred
    if set(source_axes) == set("ZYX") and len(source_axes) == 3:
        order = tuple(source_axes.index(axis) for axis in "ZYX")
        return np.transpose(array, order)[np.newaxis, ...], source_axes, inferred

    if array.ndim == 4:
        inferred = True
        return array, source_axes or "TZYX", inferred
    if array.ndim == 3:
        inferred = True
        return array[np.newaxis, ...], source_axes or "ZYX", inferred

    raise ValueError(
        "Expected a TZYX or ZYX TIFF. "
        f"The selected series has shape {array.shape} and axes {source_axes or 'unknown'}."
    )


def _read_tiff(path: str | Path) -> tuple[Path, np.ndarray, str, bool]:
    resolved = Path(path).expanduser().resolve()
    with tifffile.TiffFile(resolved) as tif:
        if not tif.series:
            raise ValueError("The TIFF contains no image series")
        series = tif.series[0]
        array = series.asarray()
        axes = getattr(series, "axes", "")
    normalised, source_axes, inferred = _normalise_tiff_array(np.asarray(array), axes)
    if not np.issubdtype(normalised.dtype, np.number):
        raise ValueError(f"Unsupported TIFF dtype {normalised.dtype}")
    return resolved, normalised, source_axes, inferred


def discover_companion_mask(path: str | Path) -> Path | None:
    """Return the preferred same-directory mask TIFF for an image, if present."""

    source = Path(path).expanduser().resolve()
    if source.suffix.casefold() not in TIFF_SUFFIXES:
        return None
    source_stem = source.stem
    if source_stem.casefold().endswith(MASK_NAME_SUFFIXES):
        return None

    files_by_name = {
        candidate.name.casefold(): candidate
        for candidate in source.parent.iterdir()
        if candidate.is_file()
    }
    extension_order = tuple(dict.fromkeys((source.suffix.casefold(), *TIFF_SUFFIXES)))
    for name_suffix in MASK_NAME_SUFFIXES:
        for extension in extension_order:
            name = f"{source_stem}{name_suffix}{extension}".casefold()
            candidate = files_by_name.get(name)
            if candidate is not None:
                return candidate.resolve()
    return None


def load_mask_tiff(
    path: str | Path,
    *,
    expected_shape: tuple[int, int, int, int] | None = None,
) -> MaskData:
    resolved, data, source_axes, inferred = _read_tiff(path)
    if not np.issubdtype(data.dtype, np.integer):
        raise ValueError(f"Instance masks must use an integer dtype, got {data.dtype}")
    if np.any(data < 0):
        raise ValueError("Instance masks cannot contain negative labels")
    shape = tuple(int(value) for value in data.shape)
    if expected_shape is not None and shape != expected_shape:
        raise ValueError(
            f"Instance mask shape does not match the source TIFF: {shape} != {expected_shape}"
        )
    return MaskData(
        path=resolved,
        data=data,
        source_axes=source_axes,
        axes_inferred=inferred,
    )


def load_tiff(
    path: str | Path,
    *,
    load_companion_masks: bool = True,
) -> VolumeData:
    resolved, normalised, source_axes, inferred = _read_tiff(path)
    volume = VolumeData(
        path=resolved,
        data=normalised,
        source_axes=source_axes,
        axes_inferred=inferred,
    )
    if not load_companion_masks:
        return volume

    try:
        mask_path = discover_companion_mask(resolved)
    except OSError as error:
        volume.mask_error = f"Could not search for a companion mask: {error}"
        return volume
    if mask_path is None:
        return volume
    try:
        volume.masks = load_mask_tiff(mask_path, expected_shape=volume.shape_tzyx)
    except (OSError, ValueError) as error:
        # Masks are optional visual guidance. A bad companion must not make the
        # source image or an existing annotation project inaccessible.
        volume.mask_error = f"{mask_path.name}: {error}"
    return volume


def new_project_for_volume(volume: VolumeData) -> GraphProject:
    return GraphProject(
        shape_tzyx=volume.shape_tzyx,
        source_path=str(volume.path),
        source_axes=volume.source_axes,
    )


def save_project(
    project: GraphProject,
    path: str | Path,
    *,
    overwrite: bool = True,
) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = project.to_dict()
    source = Path(project.source_path).expanduser()
    if source.is_absolute():
        try:
            payload["source"]["path"] = os.path.relpath(source, start=target.parent)
        except ValueError:
            payload["source"]["path"] = str(source)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(json.dumps(payload, indent=2) + "\n")
            output.flush()
            os.fsync(output.fileno())
        if overwrite:
            temporary.replace(target)
        else:
            os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def load_project(path: str | Path) -> tuple[GraphProject, VolumeData]:
    project_path = Path(path).expanduser().resolve()
    payload: dict[str, Any] = json.loads(project_path.read_text(encoding="utf-8"))
    project = GraphProject.from_dict(payload)
    source = Path(project.source_path).expanduser()
    if not source.is_absolute():
        source = (project_path.parent / source).resolve()
    volume = load_tiff(source)
    if volume.shape_tzyx != project.shape_tzyx:
        raise ValueError(
            "The source TIFF shape no longer matches the annotation project: "
            f"{volume.shape_tzyx} != {project.shape_tzyx}"
        )
    project.source_path = str(volume.path)
    return project, volume


def robust_intensity_limits(data: np.ndarray) -> tuple[float, float, float, float]:
    array = np.asarray(data)
    flat = array.reshape(-1)
    if flat.size > 1_000_000:
        step = max(1, flat.size // 1_000_000)
        sample = flat[::step]
    else:
        sample = flat
    finite = sample[np.isfinite(sample)]
    if finite.size == 0:
        return 0.0, 1.0, 0.0, 1.0
    full_min = float(np.nanmin(array))
    full_max = float(np.nanmax(array))
    if not np.isfinite(full_min):
        full_min = float(np.min(finite))
    if not np.isfinite(full_max):
        full_max = float(np.max(finite))
    if full_min == full_max:
        full_max = full_min + 1.0
    low, high = (float(value) for value in np.percentile(finite, (1.0, 99.0)))
    if low >= high:
        low, high = full_min, full_max
    return full_min, full_max, low, high
