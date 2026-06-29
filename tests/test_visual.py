import types
from pathlib import Path

import numpy as np

from scanproject_merger.format import transform_points
from scanproject_merger.registration import register_pair, rigid_transform
from scanproject_merger.visual import _Features, _fit_yaw, _keyframe_file, _matrix, _world_point


def _depth_frame(filename: str, width: int, height: int) -> dict:
    return {
        "sceneDepthPayload": {"width": width, "height": height, "depthMapFilename": filename},
        "imageWidth": width,
        "imageHeight": height,
        "cameraIntrinsics": [1, 0, 0, 0, 1, 0, 0, 0, 1],
        "cameraTransform": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    }


def _single_feature(frame: dict) -> _Features:
    return _Features(
        frames=[frame],
        descriptors=np.zeros((1, 128), dtype=np.float32),
        frame_indices=np.array([0], dtype=np.int32),
        image_points=np.array([[0.0, 0.0]], dtype=np.float32),
    )


def test_world_point_reads_a_valid_depth_sample(tmp_path: Path):
    (tmp_path / "keyframes").mkdir()
    np.full(4, 2.0, dtype="<f4").tofile(tmp_path / "keyframes" / "d.bin")  # 2x2 grid, all 2.0 m
    project = types.SimpleNamespace(path=tmp_path)
    result = _world_point(project, _single_feature(_depth_frame("d.bin", 2, 2)), 0, {})
    assert result is not None
    np.testing.assert_allclose(result, [0.0, 0.0, -2.0])


def test_world_point_rejects_truncated_depth_map(tmp_path: Path):
    (tmp_path / "keyframes").mkdir()
    np.full(1, 2.0, dtype="<f4").tofile(tmp_path / "keyframes" / "d.bin")  # only 1 of the declared 4 samples
    project = types.SimpleNamespace(path=tmp_path)
    assert _world_point(project, _single_feature(_depth_frame("d.bin", 2, 2)), 0, {}) is None


def test_world_point_rejects_nonpositive_dimensions(tmp_path: Path):
    project = types.SimpleNamespace(path=tmp_path)
    assert _world_point(project, _single_feature(_depth_frame("d.bin", 0, 2)), 0, {}) is None


def test_world_point_reuses_one_memmap_per_depth_file(tmp_path: Path):
    (tmp_path / "keyframes").mkdir()
    np.full(4, 2.0, dtype="<f4").tofile(tmp_path / "keyframes" / "d.bin")
    project = types.SimpleNamespace(path=tmp_path)
    features = _single_feature(_depth_frame("d.bin", 2, 2))
    grids: dict = {}

    _world_point(project, features, 0, grids)
    _world_point(project, features, 0, grids)

    # Both calls reference the same sidecar, so it is mapped exactly once.
    assert len(grids) == 1


def test_keyframe_file_allows_names_inside_the_keyframes_directory(tmp_path: Path):
    resolved = _keyframe_file(tmp_path, "frame-001.jpg")
    assert resolved == (tmp_path / "keyframes" / "frame-001.jpg").resolve()


def test_keyframe_file_rejects_traversal_and_absolute_paths(tmp_path: Path):
    assert _keyframe_file(tmp_path, "../../secret.txt") is None
    assert _keyframe_file(tmp_path, "../keyframes-sibling/x") is None
    assert _keyframe_file(tmp_path, str(Path(tmp_path).anchor or "/") + "etc/passwd") is None


def test_decodes_swift_column_major_camera_matrix():
    values = [2, 0, 0, 0, 3, 0, 4, 5, 1]
    np.testing.assert_array_equal(_matrix(values, 3), [[2, 0, 4], [0, 3, 5], [0, 0, 1]])


def test_visual_yaw_fit_recovers_metric_transform():
    rng = np.random.default_rng(8)
    moving = rng.normal(size=(50, 3))
    expected = rigid_transform(-0.2, [0.4, -0.3, 0.1])
    fixed = transform_points(moving, expected)
    np.testing.assert_allclose(_fit_yaw(moving, fixed), expected, atol=1e-10)


def test_icp_refines_from_visual_seed_without_returning_to_identity():
    rng = np.random.default_rng(9)
    fixed = rng.uniform([-4, -2, -1], [4, 2, 2], size=(2_000, 3))
    expected = rigid_transform(0.25, [1.5, -0.8, 0.2])
    moving = transform_points(fixed, np.linalg.inv(expected))
    seed = rigid_transform(0.24, [1.45, -0.75, 0.18])
    result = register_pair(fixed, moving, 0.75, initial_transform=seed)
    assert result is not None
    np.testing.assert_allclose(result[0], expected, atol=1e-3)
