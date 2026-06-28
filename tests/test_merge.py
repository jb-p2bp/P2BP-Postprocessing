from pathlib import Path

import laspy
import numpy as np
import pytest

from scanproject_merger import ScanProjectError, merge_scan_projects
from scanproject_merger.merge import discover
from test_format import make_project


def test_discover_finds_packages_under_a_parent_directory(tmp_path: Path):
    make_project(tmp_path / "a.scanproject", np.zeros((3, 3)))
    make_project(tmp_path / "b.scanproject", np.zeros((3, 3)))
    found = discover([tmp_path])
    assert found == sorted(p.resolve() for p in [tmp_path / "a.scanproject", tmp_path / "b.scanproject"])


def test_discover_rejects_inputs_that_are_not_packages(tmp_path: Path):
    with pytest.raises(ScanProjectError):
        discover([tmp_path / "not-a-scan.txt"])


def test_discover_rejects_empty_input(tmp_path: Path):
    with pytest.raises(ScanProjectError, match="no .scanproject packages found"):
        discover([tmp_path])


def test_merge_single_scan_writes_cloud_and_report(tmp_path: Path):
    make_project(tmp_path / "one.scanproject", np.array([[0, 0, 0], [1, 2, 3], [2, 4, 6]]))
    output = tmp_path / "out" / "merged.laz"

    outputs = merge_scan_projects(
        [tmp_path],
        output,
        deduplicate_voxel=0,
        registration_minimum_confidence=0,
        export_minimum_confidence=0,
    )

    assert outputs.output == output
    assert outputs.report == output.with_suffix(".registration.json")
    assert outputs.point_count == 3
    assert outputs.report.is_file()
    cloud = laspy.read(output)
    assert cloud.header.parse_crs().to_epsg() == 32618
    assert len(cloud.points) == 3


def test_merge_emits_transformed_and_original_sources(tmp_path: Path):
    make_project(tmp_path / "one.scanproject", np.array([[0, 0, 0], [1, 2, 3], [2, 4, 6]]))
    output = tmp_path / "merged.laz"

    outputs = merge_scan_projects(
        [tmp_path],
        output,
        deduplicate_voxel=0,
        registration_minimum_confidence=0,
        export_minimum_confidence=0,
        transformed_scans_dir=tmp_path / "aligned",
        original_scans_dir=tmp_path / "original",
    )

    assert outputs.transformed_scans == [tmp_path / "aligned" / "one.laz"]
    assert outputs.original_scans == [tmp_path / "original" / "one.laz"]
    assert all(path.is_file() for path in outputs.transformed_scans + outputs.original_scans)
