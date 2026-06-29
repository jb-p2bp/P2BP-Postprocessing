import json
from pathlib import Path

import numpy as np
import pytest

from scanproject_merger.format import POINT_DTYPE, ScanProject, ScanProjectError, transform_points


def make_project(path: Path, points: np.ndarray, *, easting: float = 500_000.0) -> Path:
    path.mkdir()
    records = np.zeros(len(points), dtype=POINT_DTYPE)
    records["position"] = points
    records["color"] = [10, 20, 30]
    records["confidence"] = np.arange(len(points)) % 3
    records["timestamp"] = np.arange(len(points)) * 0.1
    records.tofile(path / "chunk-00001.bin")
    manifest = {
        "id": path.stem,
        "pointCount": len(points),
        "chunkCount": 1,
        "voxelSizeMeters": 0.02,
        "geoReference": {
            "epsgCode": 32618,
            "originEasting": easting,
            "originNorthing": 4_500_000.0,
            "originAltitude": 100.0,
            "originHorizontalAccuracy": 3.0,
            "originVerticalAccuracy": 5.0,
            "headingAccuracy": 8.0,
            "worldToENURotationRadians": 0.0,
        },
    }
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_reads_binary_records_and_filters_confidence(tmp_path: Path):
    path = make_project(tmp_path / "one.scanproject", np.arange(18).reshape(6, 3))
    project = ScanProject.open(path)
    points = project.points(minimum_confidence=2)
    assert POINT_DTYPE.itemsize == 24
    np.testing.assert_array_equal(points.confidence, [2, 2])
    np.testing.assert_array_equal(points.colors, [[10, 20, 30], [10, 20, 30]])


def test_georeference_matches_scanner_axis_mapping(tmp_path: Path):
    project = ScanProject.open(make_project(tmp_path / "one.scanproject", np.zeros((3, 3))))
    actual = transform_points(np.array([[2.0, 3.0, 4.0]]), project.georeference.local_to_projected())
    np.testing.assert_allclose(actual, [[500_002.0, 4_499_996.0, 103.0]])


def test_rejects_inconsistent_point_count(tmp_path: Path):
    path = make_project(tmp_path / "bad.scanproject", np.zeros((3, 3)))
    manifest = json.loads((path / "manifest.json").read_text())
    manifest["pointCount"] = 4
    (path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ScanProjectError, match="chunks contain 3"):
        ScanProject.open(path)


def test_rejects_point_count_above_the_limit(tmp_path: Path):
    path = make_project(tmp_path / "huge.scanproject", np.zeros((3, 3)))
    manifest = json.loads((path / "manifest.json").read_text())
    manifest["pointCount"] = 10_000_000_000  # Absurd count that should be refused before reading chunks.
    (path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ScanProjectError, match="exceeding the"):
        ScanProject.open(path, maximum_points=1_000_000)


def test_rejects_negative_counts(tmp_path: Path):
    path = make_project(tmp_path / "neg.scanproject", np.zeros((3, 3)))
    manifest = json.loads((path / "manifest.json").read_text())
    manifest["pointCount"] = -1
    (path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ScanProjectError, match="negative counts"):
        ScanProject.open(path)
