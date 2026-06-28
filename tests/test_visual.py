import numpy as np

from scanproject_merger.format import transform_points
from scanproject_merger.registration import register_pair, rigid_transform
from scanproject_merger.visual import _fit_yaw, _matrix


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
