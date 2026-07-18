"""Floor-plan-style top-down rendering for merged LAS/LAZ point clouds."""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import laspy
import numpy as np


FLOOR_BAND_TOLERANCE_METERS = 0.10
AXIS_SUPPORT_ADVANTAGE = 1.25
PCA_FLATNESS_RATIO = 0.35
SHADOW_NORMALIZATION_RADIUS_METERS = 0.35
LOCAL_TERRAIN_NEIGHBORHOOD_METERS = 1.0
PLAN_SURFACE_MAX_HEIGHT_METERS = 2.0
OVERHEAD_MIN_HEIGHT_METERS = 2.15
VERTICAL_SURFACE_MIN_HEIGHT_METERS = 0.15
GROUND_LIKE_MAX_HEIGHT_METERS = 0.30
WALL_MIN_VERTICAL_SPAN_METERS = 0.75
FLOOR_GAP_CLOSE_METERS = 0.08
MAX_FLOOR_REPAIR_AREA_SQUARE_METERS = 1.0
WALL_GAP_CLOSE_METERS = 0.10
OVERHEAD_GAP_CLOSE_METERS = 0.35
MIN_OVERHEAD_FEATURE_METERS = 0.20


def _dominant_lower_level(values: np.ndarray) -> float:
    """Return the strongest elevation band in the lower 65% of samples."""
    low, high = float(values.min()), float(values.max())
    if high <= low:
        return low
    histogram, edges = np.histogram(values, bins=512, range=(low, high))
    cumulative = np.cumsum(histogram)
    cutoff = int(np.searchsorted(cumulative, len(values) * 0.65))
    candidates = np.arange(cutoff + 1)
    index = int(candidates[np.argmax(histogram[candidates])])
    return float((edges[index] + edges[index + 1]) / 2)


def _floor_frame(sample: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Estimate right, north, up, and floor elevation from point positions."""
    if len(sample) < 100:
        up = np.array([0.0, 0.0, 1.0])
    else:
        # Scanner exports commonly use either Y-up (AR/graphics coordinates)
        # or Z-up (LAS/geospatial coordinates). Prefer the axis with a clearly
        # stronger, broad floor-height band before consulting the cloud's
        # overall shape, which can be dominated by walls or trees.
        axis_support: list[int] = []
        for axis in (np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])):
            heights = sample @ axis
            level = _dominant_lower_level(heights)
            axis_support.append(
                int(
                    np.count_nonzero(
                        np.abs(heights - level) <= FLOOR_BAND_TOLERANCE_METERS
                    )
                )
            )

        centered = sample - sample.mean(axis=0)
        eigenvalues, eigenvectors = np.linalg.eigh(np.cov(centered, rowvar=False))
        if axis_support[0] > axis_support[1] * AXIS_SUPPORT_ADVANTAGE:
            up = np.array([0.0, 1.0, 0.0])
        elif axis_support[1] > axis_support[0] * AXIS_SUPPORT_ADVANTAGE:
            up = np.array([0.0, 0.0, 1.0])
        # Only infer a tilted/nonstandard plane from the overall shape when
        # neither standard axis has decisive floor support.
        elif eigenvalues[0] / max(eigenvalues[1], 1e-12) < PCA_FLATNESS_RATIO:
            up = eigenvectors[:, 0]
            dominant_axis = int(np.argmax(np.abs(up)))
            if up[dominant_axis] < 0:
                up = -up
        else:
            up = np.array([0.0, 0.0, 1.0])

    preliminary_height = sample @ up
    preliminary_floor = _dominant_lower_level(preliminary_height)
    floor_band = (
        np.abs(preliminary_height - preliminary_floor)
        <= FLOOR_BAND_TOLERANCE_METERS
    )
    if floor_band.sum() >= max(100, len(sample) // 200):
        floor_points = sample[floor_band]
        _, floor_vectors = np.linalg.eigh(
            np.cov(floor_points - floor_points.mean(axis=0), rowvar=False)
        )
        refined = floor_vectors[:, 0]
        if np.dot(refined, up) < 0:
            refined = -refined
        up = refined

    # Project a conventional X direction into the detected floor. For Y-up
    # scanner coordinates this yields X right and -Z north; for Z-up
    # georeferenced coordinates it yields X right and Y north.
    reference = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(reference, up))) > 0.90:
        reference = np.array([0.0, 1.0, 0.0])
    right = reference - np.dot(reference, up) * up
    right /= np.linalg.norm(right)
    north = np.cross(up, right)
    north /= np.linalg.norm(north)
    floor = _dominant_lower_level(sample @ up)
    return right, north, up, floor


def _odd_kernel(pixels: float, minimum: int = 3) -> int:
    size = max(minimum, int(round(pixels)))
    return size if size % 2 else size + 1


def _skeletonize_mask(mask: np.ndarray) -> np.ndarray:
    """Collapse thick detected regions to a single-pixel centerline."""
    remaining = np.where(mask > 0, 255, 0).astype(np.uint8)
    skeleton = np.zeros_like(remaining)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while cv2.countNonZero(remaining):
        opened = cv2.morphologyEx(remaining, cv2.MORPH_OPEN, element)
        skeleton = cv2.bitwise_or(
            skeleton,
            cv2.subtract(remaining, opened),
        )
        remaining = cv2.erode(remaining, element)

    return skeleton


def _enclosed_holes(mask: np.ndarray, maximum_area: float) -> np.ndarray:
    """Return enclosed empty components small enough to be floor gaps."""
    empty = (mask == 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(empty, connectivity=8)
    height, width = mask.shape
    x = stats[:, cv2.CC_STAT_LEFT]
    y = stats[:, cv2.CC_STAT_TOP]
    component_width = stats[:, cv2.CC_STAT_WIDTH]
    component_height = stats[:, cv2.CC_STAT_HEIGHT]
    area = stats[:, cv2.CC_STAT_AREA]
    touches_edge = (
        (x == 0)
        | (y == 0)
        | (x + component_width == width)
        | (y + component_height == height)
    )
    accepted = (~touches_edge) & (area <= maximum_area)
    accepted[0] = False
    return (accepted[labels].astype(np.uint8) * 255)


def _normalize_illumination(
    gray: np.ndarray, mask: np.ndarray, pixels_per_meter: float
) -> np.ndarray:
    """Reduce broad RGB lighting changes while retaining local surface detail."""
    if not np.any(mask):
        return gray

    values = gray.astype(np.float32)
    weights = mask.astype(np.float32)
    # The local Gaussian scale treats cast shadows as illumination while leaving
    # smaller features such as curb, bench, and material edges visible.
    sigma = max(1.0, pixels_per_meter * SHADOW_NORMALIZATION_RADIUS_METERS)
    local_sum = cv2.GaussianBlur(values * weights, (0, 0), sigma)
    local_weight = cv2.GaussianBlur(weights, (0, 0), sigma)
    illumination = local_sum / np.maximum(local_weight, 1e-4)
    target = max(160.0, float(np.percentile(values[mask], 65)))
    corrected = np.clip(values * target / np.maximum(illumination, 1.0), 0, 255)
    # Retain a small amount of original luminance so real fine-scale texture is
    # not flattened completely, but broad shadows no longer dominate the plan.
    corrected = corrected * 0.97 + values * 0.03
    corrected[~mask] = 255
    return corrected.astype(np.uint8)


def _draw_dotted_contour(
    image: np.ndarray,
    contour: np.ndarray,
    color: tuple[int, int, int],
    thickness: int,
    dash: int = 7,
    gap: int = 5,
) -> None:
    """Draw a closed OpenCV contour as evenly spaced line segments."""
    points = contour.reshape(-1, 2)
    if len(points) < 2:
        return
    points = np.vstack((points, points[0]))
    period = dash + gap
    distance = 0.0
    for start, end in zip(points[:-1], points[1:]):
        vector = end.astype(np.float64) - start
        length = float(np.linalg.norm(vector))
        if length == 0:
            continue
        direction = vector / length
        position = 0.0
        while position < length:
            phase = distance % period
            step = min(length - position, period - phase)
            if phase < dash:
                drawn = min(step, dash - phase)
                first = np.rint(start + direction * position).astype(int)
                last = np.rint(start + direction * (position + drawn)).astype(int)
                cv2.line(image, tuple(first), tuple(last), color, thickness, cv2.LINE_AA)
            position += step
            distance += step


def _image_shape(
    width_meters: float,
    height_meters: float,
    pixels_per_meter: float,
    maximum_dimension: int,
    padding: int,
) -> tuple[int, int, float]:
    usable_limit = maximum_dimension - 2 * padding
    scale = min(
        pixels_per_meter,
        max(1, usable_limit - 1)
        / max(width_meters, height_meters, 1 / pixels_per_meter),
    )
    width = max(1, int(np.ceil(width_meters * scale)) + 1) + 2 * padding
    height = max(1, int(np.ceil(height_meters * scale)) + 1) + 2 * padding
    return height, width, scale


def export_top_down_view(
    source: str | Path,
    output: str | Path,
    *,
    pixels_per_meter: float = 50.0,
    maximum_dimension: int = 4096,
    padding: int = 32,
    chunk_size: int = 1_000_000,
) -> int:
    """Render a floor-aligned outdoor site-plan PNG and return used point count.

    Low surfaces and objects form a light desaturated base, vertically
    continuous surfaces are outlined as solid walls, and elevated surfaces
    without vertical support are outlined with dots as canopy/overhangs. The
    lower scan beneath an overhead region remains visible. Processing is
    streamed in chunks so large scans are not loaded fully into memory.
    """
    source_path, output_path = Path(source), Path(output)
    if pixels_per_meter <= 0 or maximum_dimension <= 2 * padding or padding < 0:
        raise ValueError("invalid top-down image dimensions")

    with laspy.open(source_path) as reader:
        point_count = reader.header.point_count
        if point_count == 0:
            image = np.full((1 + 2 * padding, 1 + 2 * padding), 255, dtype=np.uint8)
            used_points = 0
        else:
            mins = np.asarray(reader.header.mins, dtype=np.float64)
            maxs = np.asarray(reader.header.maxs, dtype=np.float64)
            if not np.all(np.isfinite(np.r_[mins, maxs])):
                raise ValueError("point cloud bounds must be finite")

            sample_stride = max(1, point_count // 200_000)
            samples: list[np.ndarray] = []
            for points in reader.chunk_iterator(chunk_size):
                xyz = np.column_stack((points.x, points.y, points.z)).astype(np.float64)
                finite = np.all(np.isfinite(xyz), axis=1)
                samples.append(xyz[finite][::sample_stride])
            sample = np.concatenate(samples)
            if len(sample) > 200_000:
                indices = np.linspace(0, len(sample) - 1, 200_000, dtype=np.int64)
                sample = sample[indices]
            right_axis, north_axis, up_axis, _ = _floor_frame(sample)

            corners = np.array(
                [
                    [x, y, z]
                    for x in (mins[0], maxs[0])
                    for y in (mins[1], maxs[1])
                    for z in (mins[2], maxs[2])
                ],
                dtype=np.float64,
            )
            corner_right = corners @ right_axis
            corner_north = corners @ north_axis
            right_min, right_max = float(corner_right.min()), float(corner_right.max())
            north_min, north_max = float(corner_north.min()), float(corner_north.max())

            height, width, scale = _image_shape(
                right_max - right_min,
                north_max - north_min,
                pixels_per_meter,
                maximum_dimension,
                padding,
            )
            used_points = 0

            def pixel_indices(
                points: laspy.ScaleAwarePointRecord,
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
                xyz = np.column_stack((points.x, points.y, points.z)).astype(np.float64)
                keep = np.all(np.isfinite(xyz), axis=1)
                source_indices = np.flatnonzero(keep)
                positions = xyz[keep]
                right_values = positions @ right_axis
                north_values = positions @ north_axis
                elevations = positions @ up_axis
                columns = np.clip(
                    ((right_values - right_min) * scale).astype(np.int64) + padding,
                    0,
                    width - 1,
                )
                rows = np.clip(
                    ((north_max - north_values) * scale).astype(np.int64) + padding,
                    0,
                    height - 1,
                )
                return rows * width + columns, elevations, source_indices

            used_points = point_count

            pixel_count = height * width
            minimum_elevation = np.full(pixel_count, np.inf, dtype=np.float32)
            with laspy.open(source_path) as chunks:
                for points in chunks.chunk_iterator(chunk_size):
                    indices, elevations, _ = pixel_indices(points)
                    np.minimum.at(
                        minimum_elevation, indices, elevations.astype(np.float32)
                    )

            # Estimate terrain locally from the lowest observed surface in a
            # one-meter neighborhood. Unlike one global height cutoff, this
            # retains raised patios, terraces, and gradual outdoor slopes.
            terrain_kernel = _odd_kernel(scale * LOCAL_TERRAIN_NEIGHBORHOOD_METERS)
            local_ground = cv2.erode(
                minimum_elevation.reshape(height, width),
                np.ones((terrain_kernel, terrain_kernel), np.uint8),
            ).reshape(-1)

            visible_height = np.full(pixel_count, -np.inf, dtype=np.float32)
            visible_elevation = np.full(pixel_count, -np.inf, dtype=np.float32)
            vertical_min = np.full(pixel_count, np.inf, dtype=np.float32)
            high_seen = np.zeros(pixel_count, dtype=bool)

            with laspy.open(source_path) as chunks:
                for points in chunks.chunk_iterator(chunk_size):
                    indices, elevations, _ = pixel_indices(points)
                    relative_height = (
                        elevations - local_ground[indices]
                    ).astype(np.float32)
                    plan = relative_height <= PLAN_SURFACE_MAX_HEIGHT_METERS
                    np.maximum.at(
                        visible_height, indices[plan], relative_height[plan]
                    )
                    np.maximum.at(
                        visible_elevation,
                        indices[plan],
                        elevations[plan].astype(np.float32),
                    )
                    vertical = (
                        relative_height >= VERTICAL_SURFACE_MIN_HEIGHT_METERS
                    ) & (relative_height <= PLAN_SURFACE_MAX_HEIGHT_METERS)
                    np.minimum.at(
                        vertical_min, indices[vertical], relative_height[vertical]
                    )
                    high_seen[
                        indices[relative_height >= OVERHEAD_MIN_HEIGHT_METERS]
                    ] = True

            surface_height = visible_height.reshape(height, width)
            surface_elevation = visible_elevation.reshape(height, width)
            occupied = np.isfinite(surface_height)
            image = np.full((height, width, 3), 255, dtype=np.uint8)
            dimension_names = set(reader.header.point_format.dimension_names)
            has_rgb = {"red", "green", "blue"} <= dimension_names

            if has_rgb:
                # Make a second streaming pass and copy the RGB value belonging
                # to the highest point in each pixel.
                flat_image = image.reshape(-1, 3)
                with laspy.open(source_path) as chunks:
                    for points in chunks.chunk_iterator(chunk_size):
                        indices, elevations, finite_indices = pixel_indices(points)
                        relative_height = (
                            elevations - local_ground[indices]
                        ).astype(np.float32)
                        visible = (
                            relative_height <= PLAN_SURFACE_MAX_HEIGHT_METERS
                        ) & (
                            relative_height >= visible_height[indices] - 0.001
                        )
                        source_indices = finite_indices[visible]
                        # LAS stores RGB as 16-bit values; converting through
                        # 257 maps the common 0..65535 range exactly to 0..255.
                        def eight_bit(channel: np.ndarray) -> np.ndarray:
                            values = channel[source_indices].astype(np.uint32)
                            divisor = 1 if values.size and values.max() <= 255 else 257
                            return (values // divisor).astype(np.uint8)

                        red = eight_bit(np.asarray(points.red))
                        green = eight_bit(np.asarray(points.green))
                        blue = eight_bit(np.asarray(points.blue))
                        flat_image[indices[visible]] = np.column_stack((blue, green, red))
            elif occupied.any():
                # Aerial LiDAR often has no RGB. A perceptually uniform height
                # palette still makes terrain and structures easy to distinguish.
                visible_elevations = surface_elevation[occupied]
                low, high = np.percentile(visible_elevations, (2, 98))
                span = max(float(high - low), 0.001)
                normalized = np.zeros((height, width), dtype=np.uint8)
                normalized[occupied] = np.clip(
                    (surface_elevation[occupied] - low) / span * 255, 0, 255
                ).astype(np.uint8)
                colored = cv2.applyColorMap(normalized, cv2.COLORMAP_VIRIDIS)
                image[occupied] = colored[occupied]

            # Join point-sized sampling gaps and fill reasonably small enclosed
            # floor holes. Empty regions connected to the image edge remain
            # background, so this does not invent a complete rectangular site.
            mask = occupied.astype(np.uint8) * 255
            # Only ground-like points can authorize a repair. This prevents
            # walls, furniture, or canopy projections from enclosing and
            # fabricating unrelated empty regions.
            ground_mask = (
                occupied & (surface_height <= GROUND_LIKE_MAX_HEIGHT_METERS)
            ).astype(np.uint8) * 255
            floor_kernel = _odd_kernel(scale * FLOOR_GAP_CLOSE_METERS)
            closed = cv2.morphologyEx(
                ground_mask,
                cv2.MORPH_CLOSE,
                np.ones((floor_kernel, floor_kernel), np.uint8),
            )
            sampling_gaps = cv2.subtract(closed, ground_mask)
            enclosed_holes = _enclosed_holes(
                closed,
                max(64.0, scale**2 * MAX_FLOOR_REPAIR_AREA_SQUARE_METERS),
            )
            floor_repairs = cv2.bitwise_or(sampling_gaps, enclosed_holes)
            display_mask = cv2.bitwise_or(mask, floor_repairs).astype(bool)
            if np.any(floor_repairs):
                image = cv2.inpaint(image, floor_repairs, 3, cv2.INPAINT_TELEA)

            # Fade the repaired surface into a paper-like plan base. Benches,
            # sidewalks, and other low objects remain visible without
            # competing with the darker wall and overhead linework.
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if has_rgb:
                gray = _normalize_illumination(gray, display_mask, scale)
            plan_base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            plan_base = cv2.addWeighted(plan_base, 0.48, np.full_like(plan_base, 255), 0.52, 0)
            plan_base[~display_mask] = 255

            relative_min = vertical_min.reshape(height, width)
            wall_mask = occupied & np.isfinite(relative_min) & (
                surface_height - relative_min >= WALL_MIN_VERTICAL_SPAN_METERS
            )
            wall_kernel = _odd_kernel(scale * WALL_GAP_CLOSE_METERS)
            wall_bytes = cv2.morphologyEx(
                wall_mask.astype(np.uint8) * 255,
                cv2.MORPH_CLOSE,
                np.ones((wall_kernel, wall_kernel), np.uint8),
            )
            # A wall mask is a narrow band. Drawing its contour traces both
            # sides of that band and produces two or three parallel outlines.
            # Reduce the band to its centerline, then widen that centerline only
            # for visibility so every wall is represented by one solid stroke.
            wall_centerline = _skeletonize_mask(wall_bytes)
            wall_stroke = cv2.dilate(
                wall_centerline,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            )
            plan_base[wall_stroke > 0] = (45, 55, 48)

            # High points without a vertically continuous wall below them form
            # canopy/overhang footprints. Close scan-line gaps, discard specks,
            # and show only their boundary so lower objects stay untouched.
            overhead = high_seen.reshape(height, width) & ~wall_mask
            overhead_kernel = _odd_kernel(scale * OVERHEAD_GAP_CLOSE_METERS)
            overhead_bytes = cv2.morphologyEx(
                overhead.astype(np.uint8) * 255,
                cv2.MORPH_CLOSE,
                np.ones((overhead_kernel, overhead_kernel), np.uint8),
            )
            minimum_area = max(8.0, (scale * MIN_OVERHEAD_FEATURE_METERS) ** 2)
            overhead_contours, _ = cv2.findContours(
                overhead_bytes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for contour in overhead_contours:
                if cv2.contourArea(contour) >= minimum_area:
                    _draw_dotted_contour(plan_base, contour, (105, 110, 105), 2)
            image = plan_base

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp = output_path.with_suffix(f".{os.getpid()}.tmp{output_path.suffix}")
    try:
        if not cv2.imwrite(str(temp), image):
            raise OSError(f"could not write top-down image: {output_path}")
        os.replace(temp, output_path)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    return used_points
