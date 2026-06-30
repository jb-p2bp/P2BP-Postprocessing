from pathlib import Path

import laspy
import numpy as np

from scanproject_merger.export import (
    export_merged_cloud,
    export_merged_cloud_bin,
    export_merged_cloud_outputs,
    export_original_scans,
    export_transformed_scans,
)
from scanproject_merger.format import POINT_DTYPE, ScanProject
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


def test_exports_bin_with_scanner_consolidator_record_layout(tmp_path: Path):
    path = make_project(tmp_path / "one.scanproject", np.array([[0, 0, 0], [1, 2, 3], [2, 4, 6]]))
    scan = ScanProject.open(path)
    prepared = prepare_scans([scan], voxel_size=0.01, minimum_confidence=0)
    result = RegistrationResult(prepared, [np.eye(4)], [])
    output = tmp_path / "merged.bin"

    assert export_merged_cloud_bin(result, output, minimum_confidence=0, deduplicate_voxel=0) == 3

    records = np.fromfile(output, dtype=POINT_DTYPE)
    assert output.stat().st_size == 3 * POINT_DTYPE.itemsize
    np.testing.assert_allclose(
        records["position"],
        [[500_000, 4_500_000, 100], [500_001, 4_499_997, 102], [500_002, 4_499_994, 104]],
    )
    np.testing.assert_array_equal(records["color"], [[10, 20, 30], [10, 20, 30], [10, 20, 30]])
    np.testing.assert_array_equal(records["confidence"], [0, 1, 2])


def test_exports_laz_and_bin_from_same_filtered_points(tmp_path: Path):
    path = make_project(
        tmp_path / "one.scanproject",
        np.array([[0, 0, 0], [0.001, 0.001, 0], [2, 4, 6]]),
    )
    scan = ScanProject.open(path)
    prepared = prepare_scans([scan], voxel_size=0.0001, minimum_confidence=0)
    result = RegistrationResult(prepared, [np.eye(4)], [])
    laz_output = tmp_path / "merged.laz"
    bin_output = tmp_path / "merged.bin"

    count = export_merged_cloud_outputs(
        result,
        laz_output=laz_output,
        bin_output=bin_output,
        minimum_confidence=0,
        deduplicate_voxel=0.02,
    )

    assert count == 2
    cloud = laspy.read(laz_output)
    records = np.fromfile(bin_output, dtype=POINT_DTYPE)
    assert len(cloud.points) == len(records) == count
    np.testing.assert_allclose(records["position"], np.column_stack((cloud.x, cloud.y, cloud.z)))


def test_exports_each_transformed_source_without_deduplication(tmp_path: Path):
    path = make_project(
        tmp_path / "B1-one.scanproject",
        np.array([[0, 0, 0], [1, 2, 3], [1, 2, 3], [2, 4, 6]]),
    )
    scan = ScanProject.open(path)
    prepared = prepare_scans([scan], voxel_size=0.01, minimum_confidence=0)
    result = RegistrationResult(prepared, [np.eye(4)], [])
    outputs = export_transformed_scans(result, tmp_path / "aligned", minimum_confidence=0)
    assert outputs == [tmp_path / "aligned" / "000-B1-one.laz"]
    cloud = laspy.read(outputs[0])
    assert len(cloud.points) == 4
    np.testing.assert_allclose(cloud.x, [500_000, 500_001, 500_001, 500_002])


def test_transformed_filenames_disambiguate_shared_stems(tmp_path: Path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    points = np.array([[0, 0, 0], [1, 2, 3], [2, 4, 6]])
    first = ScanProject.open(make_project(tmp_path / "a" / "dup.scanproject", points))
    second = ScanProject.open(make_project(tmp_path / "b" / "dup.scanproject", points))
    prepared = prepare_scans([first, second], voxel_size=0.01, minimum_confidence=0)
    result = RegistrationResult(prepared, [np.eye(4), np.eye(4)], [])

    outputs = export_transformed_scans(result, tmp_path / "aligned", minimum_confidence=0)

    # Both packages share the stem "dup"; the source index keeps them distinct.
    assert outputs == [tmp_path / "aligned" / "000-dup.laz", tmp_path / "aligned" / "001-dup.laz"]
    assert all(path.is_file() for path in outputs)


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

    assert outputs == [tmp_path / "original" / "000-B1-one.laz"]
    cloud = laspy.read(outputs[0])
    assert len(cloud.points) == 3
    assert cloud.header.parse_crs().to_epsg() == 32618
    np.testing.assert_allclose(cloud.x, [500_000, 500_001, 500_002])
