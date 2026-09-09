from __future__ import annotations

import numpy as np
import pytest
import tifffile

from chronopose_viewer.io import (
    _normalise_tiff_array,
    discover_companion_mask,
    load_mask_tiff,
    load_project,
    load_tiff,
    new_project_for_volume,
    robust_intensity_limits,
    save_project,
)
from chronopose_viewer.model import GraphProject


def test_normalises_declared_axis_order() -> None:
    data = np.arange(2 * 3 * 4 * 5).reshape(5, 4, 3, 2)
    normalised, source_axes, inferred = _normalise_tiff_array(data, "XYZT")
    assert normalised.shape == (2, 3, 4, 5)
    assert source_axes == "XYZT"
    assert not inferred


def test_three_dimensional_tiff_becomes_one_timepoint(tmp_path) -> None:
    path = tmp_path / "single.tif"
    tifffile.imwrite(
        path,
        np.zeros((3, 4, 5), dtype=np.uint16),
        metadata={"axes": "ZYX"},
        photometric="minisblack",
    )
    volume = load_tiff(path)
    assert volume.shape_tzyx == (1, 3, 4, 5)


def test_project_path_is_portable_and_loads_source(tmp_path) -> None:
    source = tmp_path / "images" / "series.tif"
    source.parent.mkdir()
    tifffile.imwrite(
        source,
        np.zeros((2, 3, 4, 5), dtype=np.uint8),
        metadata={"axes": "TZYX"},
        photometric="minisblack",
    )
    tifffile.imwrite(
        source.with_name("series_masks.tiff"),
        np.ones((2, 3, 4, 5), dtype=np.uint8),
        metadata={"axes": "TZYX"},
        photometric="minisblack",
    )
    volume = load_tiff(source)
    project = new_project_for_volume(volume)
    graph = project.add_instance(0)
    graph.add_node((1, 2, 3))
    project_path = save_project(project, tmp_path / "labels" / "series.cpv.json")

    text = project_path.read_text()
    assert str(tmp_path) not in text
    restored, restored_volume = load_project(project_path)
    assert restored.to_dict()["frames"] == project.to_dict()["frames"]
    assert restored_volume.shape_tzyx == (2, 3, 4, 5)
    assert restored_volume.masks is not None
    assert restored_volume.masks.path.name == "series_masks.tiff"


def test_intensity_limits_ignore_non_finite_values() -> None:
    limits = robust_intensity_limits(
        np.array([np.nan, -np.inf, 2.0, 3.0, np.inf], dtype=np.float32)
    )
    assert limits[:2] == (2.0, 3.0)
    assert 2.0 <= limits[2] < limits[3] <= 3.0


def test_project_save_can_refuse_overwrite(tmp_path) -> None:
    target = tmp_path / "existing.cpv.json"
    target.write_text("existing work\n", encoding="utf-8")
    project = GraphProject(shape_tzyx=(1, 2, 2, 2), source_path="source.tif")

    with pytest.raises(FileExistsError):
        save_project(project, target, overwrite=False)

    assert target.read_text(encoding="utf-8") == "existing work\n"


def test_load_tiff_discovers_preferred_companion_mask_suffix(tmp_path) -> None:
    source = tmp_path / "sample.tiff"
    shape = (2, 3, 4, 5)
    tifffile.imwrite(
        source,
        np.zeros(shape, dtype=np.uint8),
        metadata={"axes": "TZYX"},
        photometric="minisblack",
    )
    for suffix, value in (("_mask", 1), ("_masks", 2), ("_cp_masks", 3)):
        tifffile.imwrite(
            tmp_path / f"sample{suffix}.tif",
            np.full(shape, value, dtype=np.uint16),
            metadata={"axes": "TZYX"},
            photometric="minisblack",
        )

    assert discover_companion_mask(source) == (tmp_path / "sample_cp_masks.tif").resolve()
    volume = load_tiff(source)

    assert volume.masks is not None
    assert volume.masks.path.name == "sample_cp_masks.tif"
    assert volume.masks.shape_tzyx == shape
    assert np.all(volume.masks.data == 3)
    assert volume.mask_error is None

    (tmp_path / "sample_cp_masks.tif").unlink()
    assert discover_companion_mask(source) == (tmp_path / "sample_masks.tif").resolve()
    (tmp_path / "sample_masks.tif").unlink()
    assert discover_companion_mask(source) == (tmp_path / "sample_mask.tif").resolve()


def test_bad_optional_companion_mask_does_not_block_source_loading(tmp_path) -> None:
    source = tmp_path / "sample.tif"
    tifffile.imwrite(
        source,
        np.zeros((1, 3, 4, 5), dtype=np.uint8),
        metadata={"axes": "TZYX"},
        photometric="minisblack",
    )
    mask_path = tmp_path / "sample_mask.tif"
    tifffile.imwrite(
        mask_path,
        np.zeros((1, 2, 4, 5), dtype=np.uint8),
        metadata={"axes": "TZYX"},
        photometric="minisblack",
    )

    volume = load_tiff(source)

    assert volume.masks is None
    assert volume.mask_error is not None
    assert "does not match" in volume.mask_error
    with pytest.raises(ValueError, match="does not match"):
        load_mask_tiff(mask_path, expected_shape=volume.shape_tzyx)
