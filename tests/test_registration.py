import numpy as np

from scanproject_merger.format import transform_points
from scanproject_merger.registration import (
    RegistrationEdge,
    optimize_pose_graph,
    rebase_transform,
    select_consistent_edges,
    register_pair,
    rigid_transform,
)


def test_register_pair_recovers_yaw_and_translation():
    rng = np.random.default_rng(42)
    fixed = rng.uniform([-4, -2, -1], [4, 2, 2], size=(2_000, 3))
    expected = rigid_transform(0.07, [0.35, -0.22, 0.12])
    moving = transform_points(fixed, np.linalg.inv(expected))
    outcome = register_pair(fixed, moving, maximum_distance=1.0)
    assert outcome is not None
    actual, count, rmse, overlap = outcome
    np.testing.assert_allclose(actual, expected, atol=1e-3)
    assert count == len(fixed)
    assert rmse < 1e-3
    assert overlap == 1.0


def test_pose_graph_uses_pair_transform_as_moving_correction():
    transform = rigid_transform(0.05, [0.4, -0.1, 0.2])
    edge = RegistrationEdge(0, 1, transform, 100, 0.01, 0.5)
    corrections = optimize_pose_graph(2, [edge])
    np.testing.assert_allclose(corrections[0], np.eye(4), atol=1e-7)
    np.testing.assert_allclose(corrections[1], transform, atol=1e-6)


def test_pose_graph_handles_rotation_in_large_projected_coordinates():
    transform = rigid_transform(0.13, [-399_694.0, 85_842.0, -2.1])
    edge = RegistrationEdge(0, 1, transform, 100, 0.01, 0.5)
    corrections = optimize_pose_graph(2, [edge])
    np.testing.assert_allclose(corrections[1], transform, atol=1e-6)


def test_rebased_pose_graph_limits_multi_scan_loop_error():
    origin = np.array([465_182.0, 3_139_419.0, 27.0])
    centered_corrections = [
        np.eye(4),
        rigid_transform(0.075, [0.3, -0.2, 1.4]),
        rigid_transform(-0.055, [-0.4, 0.1, -0.6]),
    ]
    edges = []
    for fixed, moving, yaw_noise in [(0, 1, 0.001), (0, 2, -0.001), (1, 2, 0.0008)]:
        measured_centered = np.linalg.inv(centered_corrections[fixed]) @ centered_corrections[moving]
        measured_centered = rigid_transform(yaw_noise, [0.02, -0.01, 0.01]) @ measured_centered
        measured = rebase_transform(measured_centered, origin, to_centered=False)
        edges.append(RegistrationEdge(fixed, moving, measured, 100, 0.1, 0.4))
    centered = [
        RegistrationEdge(
            edge.fixed,
            edge.moving,
            rebase_transform(edge.moving_to_fixed, origin, to_centered=True),
            edge.correspondence_count,
            edge.rmse,
            edge.overlap_ratio,
        )
        for edge in edges
    ]
    optimized = optimize_pose_graph(3, centered)
    # A sub-degree inconsistent loop should stay local instead of becoming kilometre-scale translation.
    assert max(np.linalg.norm(transform[:2, 3]) for transform in optimized) < 2.0


def test_rejects_weaker_edge_with_contradictory_loop():
    strongest = RegistrationEdge(1, 2, rigid_transform(-0.05, [0.2, 0.1, -0.2]), 300, 0.15, 0.6)
    second = RegistrationEdge(0, 2, rigid_transform(-0.02, [0.1, 0.2, -0.1]), 200, 0.18, 0.4)
    contradictory = RegistrationEdge(0, 1, rigid_transform(-0.15, [-0.5, 0.3, 1.5]), 150, 0.27, 0.35)
    selected, rejected = select_consistent_edges(
        3,
        [contradictory, second, strongest],
        maximum_loop_yaw_degrees=3.0,
        maximum_loop_horizontal_error=1.0,
        maximum_loop_vertical_error=1.0,
    )
    assert selected[0] is strongest
    assert selected[1] is second
    assert len(rejected) == 1
    assert rejected[0].edge is contradictory
    assert rejected[0].loop_yaw_error_degrees > 3.0
