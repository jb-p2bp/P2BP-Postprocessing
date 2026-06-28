from pathlib import Path

import laspy
import numpy as np

from scanproject_merger.export import export_merged_cloud, export_original_scans, export_transformed_scans
from scanproject_merger.format import ScanProject
from scanproject_merger.registration import RegistrationResult, prepare_scans
from test_format import make_project


def test_exports_laz_with_crs_and_provenance(tmp_path: Path):
    path = make_project(tmp_path / "one.scanproject", np.array([[0, 0, 0], [1, 2, 3], [2, 4, 6]]))
    scan = ScanProject.open(path)
    prepared = prepare_scans([scan], voxel_size=0.01, minimum_confidence=0)
    result = RegistrationResult(prepared, [np.eye(4)], [])
    output = tmp_path / "merged.laz"
    assert export_merged_cloud(result, output, minimum_confidence=0, deduplicate_voxel=0) == 3
    cloud = laspy.read(output)
    assert cloud.header.parse_crs().to_epsg() == 32618
    assert set(("confidence", "source_id", "scan_time")) <= set(cloud.point_format.extra_dimension_names)
    np.testing.assert_array_equal(cloud.source_id, [0, 0, 0])


def test_exports_each_transformed_source_without_deduplication(tmp_path: Path):
    path = make_project(
        tmp_path / "B1-one.scanproject",
        np.array([[0, 0, 0], [1, 2, 3], [1, 2, 3], [2, 4, 6]]),
    )
    scan = ScanProject.open(path)
    prepared = prepare_scans([scan], voxel_size=0.01, minimum_confidence=0)
    result = RegistrationResult(prepared, [np.eye(4)], [])
    outputs = export_transformed_scans(result, tmp_path / "aligned", minimum_confidence=0)
    assert outputs == [tmp_path / "aligned" / "B1-one.laz"]
    cloud = laspy.read(outputs[0])
    assert len(cloud.points) == 4
    np.testing.assert_allclose(cloud.x, [500_000, 500_001, 500_001, 500_002])


def test_exports_empty_cloud_when_confidence_filter_removes_all_points(tmp_path: Path):
    path = make_project(tmp_path / "one.scanproject", np.array([[0, 0, 0], [1, 2, 3], [2, 4, 6]]))
    scan = ScanProject.open(path)
    prepared = prepare_scans([scan], voxel_size=0.01, minimum_confidence=0)
    result = RegistrationResult(prepared, [np.eye(4)], [])
    output = tmp_path / "empty.laz"

    # A confidence threshold above every point's confidence filters them all out.
    assert export_merged_cloud(result, output, minimum_confidence=5, deduplicate_voxel=0) == 0

    cloud = laspy.read(output)
    assert len(cloud.points) == 0


def test_exports_each_original_source_as_laz(tmp_path: Path):
    path = make_project(
        tmp_path / "B1-one.scanproject",
        np.array([[0, 0, 0], [1, 2, 3], [2, 4, 6]]),
    )
    scan = ScanProject.open(path)
    prepared = prepare_scans([scan], voxel_size=0.01, minimum_confidence=0)
    result = RegistrationResult(prepared, [np.eye(4)], [])

    outputs = export_original_scans(result, tmp_path / "original", minimum_confidence=0)

    assert outputs == [tmp_path / "original" / "B1-one.laz"]
    cloud = laspy.read(outputs[0])
    assert len(cloud.points) == 3
    assert cloud.header.parse_crs().to_epsg() == 32618
    np.testing.assert_allclose(cloud.x, [500_000, 500_001, 500_002])
