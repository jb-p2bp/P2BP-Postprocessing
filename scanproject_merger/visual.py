"""RGB/depth keyframe matching for cross-session registration constraints."""

from __future__ import annotations

from dataclasses import dataclass
import json

import cv2
import numpy as np
from numpy.typing import NDArray

from .format import ScanProject, transform_points


@dataclass(frozen=True)
class VisualRegistration:
    moving_to_fixed: NDArray[np.float64]
    match_count: int
    inlier_count: int
    rmse: float


@dataclass
class _Features:
    frames: list[dict]
    descriptors: NDArray[np.float32]
    frame_indices: NDArray[np.int32]
    image_points: NDArray[np.float32]


_FEATURE_CACHE: dict[str, _Features | None] = {}


def _matrix(values: list[float], size: int) -> NDArray[np.float64]:
    # Swift simd matrices are serialized column by column.
    return np.asarray(values, dtype=np.float64).reshape((size, size), order="F")


def _extract(project: ScanProject, maximum_keyframes: int = 120) -> _Features | None:
    cache_key = str(project.path.resolve())
    if cache_key in _FEATURE_CACHE:
        return _FEATURE_CACHE[cache_key]
    metadata_path = project.path / "keyframes.json"
    if not metadata_path.is_file():
        _FEATURE_CACHE[cache_key] = None
        return None
    frames = json.loads(metadata_path.read_text(encoding="utf-8"))
    frames = [frame for frame in frames if frame.get("sceneDepthPayload")]
    if not frames:
        _FEATURE_CACHE[cache_key] = None
        return None
    if len(frames) > maximum_keyframes:
        indices = np.linspace(0, len(frames) - 1, maximum_keyframes, dtype=int)
        frames = [frames[index] for index in indices]

    detector = cv2.SIFT_create(nfeatures=600, contrastThreshold=0.025)
    descriptors: list[NDArray[np.float32]] = []
    frame_indices: list[NDArray[np.int32]] = []
    image_points: list[NDArray[np.float32]] = []
    usable_frames: list[dict] = []
    for frame in frames:
        image = cv2.imread(str(project.path / "keyframes" / frame["imageFilename"]), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        scale = min(1.0, 960.0 / image.shape[1])
        resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else image
        keypoints, descriptor = detector.detectAndCompute(resized, None)
        if descriptor is None or len(keypoints) < 8:
            continue
        frame_index = len(usable_frames)
        usable_frames.append(frame)
        descriptors.append(descriptor.astype(np.float32))
        frame_indices.append(np.full(len(keypoints), frame_index, dtype=np.int32))
        image_points.append(np.asarray([point.pt for point in keypoints], dtype=np.float32) / scale)
    if not descriptors:
        _FEATURE_CACHE[cache_key] = None
        return None
    result = _Features(
        usable_frames,
        np.concatenate(descriptors),
        np.concatenate(frame_indices),
        np.concatenate(image_points),
    )
    _FEATURE_CACHE[cache_key] = result
    return result


def _world_point(project: ScanProject, features: _Features, feature_index: int) -> NDArray[np.float64] | None:
    frame = features.frames[int(features.frame_indices[feature_index])]
    payload = frame["sceneDepthPayload"]
    width, height = int(payload["width"]), int(payload["height"])
    point = features.image_points[feature_index]
    depth_x = int(np.clip(round(point[0] * width / frame["imageWidth"]), 0, width - 1))
    depth_y = int(np.clip(round(point[1] * height / frame["imageHeight"]), 0, height - 1))
    offset = depth_y * width + depth_x
    directory = project.path / "keyframes"
    depth = np.memmap(directory / payload["depthMapFilename"], dtype="<f4", mode="r", shape=(height * width,))
    value = float(depth[offset])
    if not np.isfinite(value) or value <= 0.05 or value > 8.0:
        return None
    confidence_name = payload.get("confidenceMapFilename")
    if confidence_name:
        confidence = np.memmap(directory / confidence_name, dtype="u1", mode="r", shape=(height * width,))
        if confidence[offset] < 1:
            return None
    intrinsics = _matrix(frame["cameraIntrinsics"], 3)
    x = (float(point[0]) - intrinsics[0, 2]) * value / intrinsics[0, 0]
    y = -((float(point[1]) - intrinsics[1, 2]) * value / intrinsics[1, 1])
    camera_point = np.array([[x, y, -value]], dtype=np.float64)
    return transform_points(camera_point, _matrix(frame["cameraTransform"], 4))[0]


def _fit_yaw(source: NDArray[np.float64], target: NDArray[np.float64]) -> NDArray[np.float64]:
    source_center, target_center = source.mean(axis=0), target.mean(axis=0)
    covariance = (source[:, :2] - source_center[:2]).T @ (target[:, :2] - target_center[:2])
    yaw = np.arctan2(covariance[0, 1] - covariance[1, 0], covariance[0, 0] + covariance[1, 1])
    cosine, sine = np.cos(yaw), np.sin(yaw)
    transform = np.array(
        [[cosine, -sine, 0, 0], [sine, cosine, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64
    )
    transform[:3, 3] = target_center - transform_points(source_center[None], transform)[0]
    return transform


def register_keyframes(
    fixed: ScanProject,
    moving: ScanProject,
    fixed_initial: NDArray[np.float64],
    moving_initial: NDArray[np.float64],
    minimum_inliers: int = 20,
    ransac_threshold: float = 0.20,
) -> VisualRegistration | None:
    """Estimate a projected moving-to-fixed transform from RGB matches with metric depth."""
    fixed_features, moving_features = _extract(fixed), _extract(moving)
    if fixed_features is None or moving_features is None:
        return None
    matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64))
    matches = matcher.knnMatch(moving_features.descriptors, fixed_features.descriptors, k=2)
    matches = [pair[0] for pair in matches if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance]
    fixed_points, moving_points = [], []
    for match in matches:
        moving_point = _world_point(moving, moving_features, match.queryIdx)
        fixed_point = _world_point(fixed, fixed_features, match.trainIdx)
        if moving_point is not None and fixed_point is not None:
            moving_points.append(moving_point)
            fixed_points.append(fixed_point)
    if len(fixed_points) < minimum_inliers:
        return None
    fixed_projected = transform_points(np.asarray(fixed_points), fixed_initial)
    moving_projected = transform_points(np.asarray(moving_points), moving_initial)
    rng = np.random.default_rng(0)
    best = np.zeros(len(fixed_projected), dtype=bool)
    for _ in range(2500):
        sample = rng.choice(len(fixed_projected), 3, replace=False)
        candidate = _fit_yaw(moving_projected[sample], fixed_projected[sample])
        errors = np.linalg.norm(transform_points(moving_projected, candidate) - fixed_projected, axis=1)
        inliers = errors <= ransac_threshold
        if inliers.sum() > best.sum():
            best = inliers
    if best.sum() < minimum_inliers:
        return None
    transform = _fit_yaw(moving_projected[best], fixed_projected[best])
    errors = np.linalg.norm(transform_points(moving_projected[best], transform) - fixed_projected[best], axis=1)
    return VisualRegistration(transform, len(matches), int(best.sum()), float(np.sqrt(np.mean(errors**2))))
