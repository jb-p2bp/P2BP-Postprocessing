"""Overlap-aware rigid registration for independently georeferenced scans."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Iterable, NamedTuple

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from .format import ScanProject, transform_points
from .visual import FeatureCache, register_keyframes


@dataclass
class PreparedScan:
    project: ScanProject
    points: NDArray[np.float64]
    initial_transform: NDArray[np.float64]
    projected_points: NDArray[np.float64]


@dataclass(frozen=True)
class RegistrationEdge:
    fixed: int
    moving: int
    moving_to_fixed: NDArray[np.float64]
    correspondence_count: int
    rmse: float
    overlap_ratio: float
    initialization: str = "georeference"
    visual_match_count: int = 0
    visual_inlier_count: int = 0
    visual_rmse: float | None = None

    @property
    def quality_score(self) -> float:
        """Rank useful overlap against geometric residual error."""
        return self.overlap_ratio / max(self.rmse, 0.001)


@dataclass(frozen=True)
class RejectedRegistrationEdge:
    edge: RegistrationEdge
    reason: str
    loop_yaw_error_degrees: float
    loop_horizontal_error: float
    loop_vertical_error: float


@dataclass(frozen=True)
class RegistrationResult:
    scans: list[PreparedScan]
    correction_transforms: list[NDArray[np.float64]]
    edges: list[RegistrationEdge]
    rejected_edges: list[RejectedRegistrationEdge] = field(default_factory=list)

    @property
    def final_transforms(self) -> list[NDArray[np.float64]]:
        return [correction @ scan.initial_transform for correction, scan in zip(self.correction_transforms, self.scans)]


def rigid_transform(yaw: float, translation: Iterable[float]) -> NDArray[np.float64]:
    cosine, sine = np.cos(yaw), np.sin(yaw)
    tx, ty, tz = translation
    return np.array(
        [[cosine, -sine, 0.0, tx], [sine, cosine, 0.0, ty], [0.0, 0.0, 1.0, tz], [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def parameters(transform: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.array([np.arctan2(transform[1, 0], transform[0, 0]), *transform[:3, 3]], dtype=np.float64)


def rebase_transform(
    transform: NDArray[np.float64], origin: NDArray[np.float64], *, to_centered: bool
) -> NDArray[np.float64]:
    """Change the coordinate origin used to express a rigid transform."""
    global_to_centered = rigid_transform(0.0, -origin)
    centered_to_global = rigid_transform(0.0, origin)
    if to_centered:
        return global_to_centered @ transform @ centered_to_global
    return centered_to_global @ transform @ global_to_centered


def voxel_downsample(points: NDArray[np.float64], voxel_size: float) -> NDArray[np.float64]:
    if len(points) == 0:
        return points
    keys = np.floor(points / voxel_size).astype(np.int64)
    _, indices = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(indices)]


def prepare_scans(
    projects: list[ScanProject], voxel_size: float, minimum_confidence: int
) -> list[PreparedScan]:
    if not projects:
        raise ValueError("at least one scan is required")
    epsg_codes = {project.georeference.epsg_code for project in projects if project.georeference}
    if len(epsg_codes) > 1:
        raise ValueError(f"scans use different projected coordinate systems: {sorted(epsg_codes)}")
    prepared: list[PreparedScan] = []
    for project in projects:
        if project.georeference is None:
            raise ValueError(f"scan has no georeference: {project.path}")
        local = voxel_downsample(project.points(minimum_confidence).positions, voxel_size)
        if len(local) < 3:  # Minimum points needed to constrain a rigid transform.
            raise ValueError(f"scan has too few usable points: {project.path}")
        initial = project.georeference.local_to_projected()
        prepared.append(PreparedScan(project, local, initial, transform_points(local, initial)))
    return prepared


def candidate_pairs(scans: list[PreparedScan], padding: float) -> list[tuple[int, int]]:
    bounds = [(scan.projected_points.min(axis=0), scan.projected_points.max(axis=0)) for scan in scans]
    candidates = []
    for left, right in combinations(range(len(scans)), 2):
        left_min, left_max = bounds[left]
        right_min, right_max = bounds[right]
        # Candidate discovery is horizontal; phone altitude is too unreliable to gate adjacency.
        if np.all(left_min[:2] - padding <= right_max[:2]) and np.all(right_min[:2] - padding <= left_max[:2]):
            candidates.append((left, right))
    return candidates


def _fit_yaw_translation(source: NDArray[np.float64], target: NDArray[np.float64]) -> NDArray[np.float64]:
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    source_xy = source[:, :2] - source_center[:2]
    target_xy = target[:, :2] - target_center[:2]
    covariance = source_xy.T @ target_xy
    yaw = np.arctan2(covariance[0, 1] - covariance[1, 0], covariance[0, 0] + covariance[1, 1])
    rotation = rigid_transform(yaw, (0.0, 0.0, 0.0))
    translation = target_center - transform_points(source_center[None, :], rotation)[0]
    return rigid_transform(yaw, translation)


class PairRegistration(NamedTuple):
    """Result of aligning one moving cloud to a fixed cloud with ICP."""

    transform: NDArray[np.float64]
    correspondence_count: int
    rmse: float
    overlap_ratio: float


def register_pair(
    fixed: NDArray[np.float64],
    moving: NDArray[np.float64],
    maximum_distance: float,
    iterations: int = 40,  # Coarse-to-fine ICP correspondence gates.
    initial_transform: NDArray[np.float64] | None = None,
) -> PairRegistration | None:
    tree = cKDTree(fixed)
    transform = np.eye(4) if initial_transform is None else initial_transform.copy()
    minimum_gate = min(0.50, maximum_distance)  # Finish with a 50 cm correspondence gate.
    gates = np.geomspace(maximum_distance, minimum_gate, num=max(2, iterations))  # At least coarse and fine gates.
    for gate in gates:
        transformed = transform_points(moving, transform)
        distances, indices = tree.query(transformed, distance_upper_bound=gate)
        keep = np.isfinite(distances)
        if keep.sum() < 12:  # Reject unstable fits with too few correspondences.
            return None
        # Trim the worst correspondences to reduce edge clutter and partial-overlap bias.
        cutoff = np.quantile(distances[keep], 0.8)  # Trim the worst 20% of matches.
        keep &= distances <= cutoff
        increment = _fit_yaw_translation(transformed[keep], fixed[indices[keep]])
        transform = increment @ transform
        rmse = float(np.sqrt(np.mean(np.square(distances[keep]))))
    transformed = transform_points(moving, transform)
    distances, _ = tree.query(transformed, distance_upper_bound=minimum_gate)
    keep = np.isfinite(distances)
    count = int(keep.sum())
    if count < 12:  # Apply the same minimum to the final correspondence set.
        return None
    return PairRegistration(
        transform, count, float(np.sqrt(np.mean(np.square(distances[keep])))), count / len(moving)
    )


def _reachable(scan_count: int, edges: list[RegistrationEdge]) -> bool:
    seen, pending = {0}, [0]
    while pending:
        current = pending.pop()
        for edge in edges:
            neighbor = edge.moving if edge.fixed == current else edge.fixed if edge.moving == current else None
            if neighbor is not None and neighbor not in seen:
                seen.add(neighbor)
                pending.append(neighbor)
    return len(seen) == scan_count


def _tree_poses(scan_count: int, edges: list[RegistrationEdge]) -> list[NDArray[np.float64] | None]:
    poses: list[NDArray[np.float64] | None] = [None] * scan_count
    poses[0] = np.eye(4)
    pending = [0]
    while pending:
        current = pending.pop()
        for edge in edges:
            if edge.fixed == current and poses[edge.moving] is None:
                poses[edge.moving] = poses[current] @ edge.moving_to_fixed
                pending.append(edge.moving)
            elif edge.moving == current and poses[edge.fixed] is None:
                poses[edge.fixed] = poses[current] @ np.linalg.inv(edge.moving_to_fixed)
                pending.append(edge.fixed)
    return poses


def select_consistent_edges(
    scan_count: int,
    edges: list[RegistrationEdge],
    maximum_loop_yaw_degrees: float,
    maximum_loop_horizontal_error: float,
    maximum_loop_vertical_error: float,
) -> tuple[list[RegistrationEdge], list[RejectedRegistrationEdge]]:
    """Build a maximum-quality tree, then admit only loop-consistent extra edges."""
    parent = list(range(scan_count))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    selected: list[RegistrationEdge] = []
    remaining: list[RegistrationEdge] = []
    for edge in sorted(edges, key=lambda item: item.quality_score, reverse=True):
        fixed_root, moving_root = find(edge.fixed), find(edge.moving)
        if fixed_root != moving_root:
            parent[moving_root] = fixed_root
            selected.append(edge)
        else:
            remaining.append(edge)

    if not _reachable(scan_count, selected):
        raise ValueError("accepted registration edges do not connect every scan")

    rejected: list[RejectedRegistrationEdge] = []
    for edge in remaining:
        poses = _tree_poses(scan_count, selected)
        fixed_pose, moving_pose = poses[edge.fixed], poses[edge.moving]
        assert fixed_pose is not None and moving_pose is not None
        predicted = np.linalg.inv(fixed_pose) @ moving_pose
        closure_error = np.linalg.inv(predicted) @ edge.moving_to_fixed
        values = parameters(closure_error)
        yaw_degrees = abs(float(np.degrees(values[0])))
        horizontal = float(np.linalg.norm(values[1:3]))
        vertical = abs(float(values[3]))
        if (
            yaw_degrees <= maximum_loop_yaw_degrees
            and horizontal <= maximum_loop_horizontal_error
            and vertical <= maximum_loop_vertical_error
        ):
            selected.append(edge)
        else:
            rejected.append(
                RejectedRegistrationEdge(
                    edge=edge,
                    reason="loop closure exceeds configured tolerance",
                    loop_yaw_error_degrees=yaw_degrees,
                    loop_horizontal_error=horizontal,
                    loop_vertical_error=vertical,
                )
            )
    return selected, rejected


def optimize_pose_graph(scan_count: int, edges: list[RegistrationEdge]) -> list[NDArray[np.float64]]:
    if scan_count == 1:
        return [np.eye(4)]
    if not _reachable(scan_count, edges):
        raise ValueError("accepted registration edges do not connect every scan")

    # Seed every correction by traversing measured edges from the anchor. This is
    # essential in projected CRSs: expressing a small yaw around coordinate zero
    # legitimately requires a translation of hundreds of kilometres.
    initial = _tree_poses(scan_count, edges)
    initial_transforms = [transform for transform in initial if transform is not None]

    def residuals(values: NDArray[np.float64]) -> NDArray[np.float64]:
        poses = [np.eye(4)] + [
            rigid_transform(values[index * 4], values[index * 4 + 1 : index * 4 + 4])
            for index in range(scan_count - 1)
        ]
        residual: list[float] = []
        for edge in edges:
            predicted = (
                np.linalg.inv(poses[edge.fixed])
                @ poses[edge.moving]
                @ np.linalg.inv(edge.moving_to_fixed)
            )
            error = parameters(predicted)
            # One radian of yaw is scaled to ten metres for meaningful robust weighting.
            residual.extend([error[0] * 10.0, error[1], error[2], error[3]])  # 1 radian weighs like 10 metres.
        return np.asarray(residual)

    initial_values = np.concatenate([parameters(transform) for transform in initial_transforms[1:]])
    # Downweight residuals beyond the 25 cm soft-L1 transition scale.
    solution = least_squares(residuals, initial_values, loss="soft_l1", f_scale=0.25)
    return [np.eye(4)] + [
        rigid_transform(values[0], values[1:]) for values in solution.x.reshape((-1, 4))
    ]


@dataclass(frozen=True)
class RegistrationParams:
    """Tuning parameters for :func:`register_scans`.

    Defined once here so callers (including ``merge_scan_projects``) share a single
    source of truth rather than restating these defaults.
    """

    voxel_size: float = 0.10  # 10 cm registration sampling grid.
    minimum_confidence: int = 1  # Exclude low-confidence (0) points from alignment.
    candidate_padding: float = 10.0  # Metres added to bounds when finding neighbors.
    maximum_distance: float = 5.0  # Initial ICP correspondence gate in metres.
    minimum_overlap: float = 0.03  # Require matches for 3% of the moving sample.
    maximum_rmse: float = 0.40  # Maximum accepted pair RMSE in metres.
    maximum_loop_yaw_degrees: float = 3.0
    maximum_loop_horizontal_error: float = 1.0
    maximum_loop_vertical_error: float = 1.0
    use_visual_registration: bool = True


def register_scans(
    projects: list[ScanProject],
    params: RegistrationParams = RegistrationParams(),  # Frozen, so a shared default is safe.
) -> RegistrationResult:
    scans = prepare_scans(projects, params.voxel_size, params.minimum_confidence)
    edges: list[RegistrationEdge] = []
    feature_cache: FeatureCache = {}  # Reuse keyframe features across pairs within this run only.
    for fixed, moving in candidate_pairs(scans, params.candidate_padding):
        visual = (
            register_keyframes(
                scans[fixed].project,
                scans[moving].project,
                scans[fixed].initial_transform,
                scans[moving].initial_transform,
                cache=feature_cache,
            )
            if params.use_visual_registration
            else None
        )
        outcome = register_pair(
            scans[fixed].projected_points,
            scans[moving].projected_points,
            min(params.maximum_distance, 0.75) if visual else params.maximum_distance,
            initial_transform=visual.moving_to_fixed if visual else None,
        )
        if outcome is None:
            continue
        transform, count, rmse, overlap = outcome
        if overlap >= params.minimum_overlap and rmse <= params.maximum_rmse:
            edges.append(
                RegistrationEdge(
                    fixed,
                    moving,
                    transform,
                    count,
                    rmse,
                    overlap,
                    initialization="rgbDepth" if visual else "georeference",
                    visual_match_count=visual.match_count if visual else 0,
                    visual_inlier_count=visual.inlier_count if visual else 0,
                    visual_rmse=visual.rmse if visual else None,
                )
            )
    graph_origin = scans[0].initial_transform[:3, 3]
    centered_edges = [
        RegistrationEdge(
            edge.fixed,
            edge.moving,
            rebase_transform(edge.moving_to_fixed, graph_origin, to_centered=True),
            edge.correspondence_count,
            edge.rmse,
            edge.overlap_ratio,
            edge.initialization,
            edge.visual_match_count,
            edge.visual_inlier_count,
            edge.visual_rmse,
        )
        for edge in edges
    ]
    selected_centered_edges, rejected_centered_edges = select_consistent_edges(
        len(scans),
        centered_edges,
        params.maximum_loop_yaw_degrees,
        params.maximum_loop_horizontal_error,
        params.maximum_loop_vertical_error,
    )
    centered_corrections = optimize_pose_graph(len(scans), selected_centered_edges)
    corrections = [
        rebase_transform(correction, graph_origin, to_centered=False)
        for correction in centered_corrections
    ]
    selected_keys = {(edge.fixed, edge.moving) for edge in selected_centered_edges}
    selected_edges = [edge for edge in edges if (edge.fixed, edge.moving) in selected_keys]
    rejected_by_key = {
        (rejection.edge.fixed, rejection.edge.moving): rejection for rejection in rejected_centered_edges
    }
    rejected_edges = [
        RejectedRegistrationEdge(
            edge=edge,
            reason=rejected_by_key[(edge.fixed, edge.moving)].reason,
            loop_yaw_error_degrees=rejected_by_key[(edge.fixed, edge.moving)].loop_yaw_error_degrees,
            loop_horizontal_error=rejected_by_key[(edge.fixed, edge.moving)].loop_horizontal_error,
            loop_vertical_error=rejected_by_key[(edge.fixed, edge.moving)].loop_vertical_error,
        )
        for edge in edges
        if (edge.fixed, edge.moving) in rejected_by_key
    ]
    return RegistrationResult(scans, corrections, selected_edges, rejected_edges)
