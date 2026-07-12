"""Floor-plan-style top-down rendering for merged LAS/LAZ point clouds."""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import laspy
import numpy as np


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
    """Render a north-up outdoor surface PNG and return used point count.

    The highest point at each X/Y image location is treated as the visible
    surface, so terrain, paths, vegetation, furniture, and structures are all
    retained. RGB is copied from the cloud when available; otherwise the
    surface is colored by elevation. The input is streamed in chunks so large
    outdoor scans do not have to be loaded fully into memory.
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

            height, width, scale = _image_shape(
                maxs[0] - mins[0], maxs[1] - mins[1], pixels_per_meter, maximum_dimension, padding
            )
            maximum_z = np.full(height * width, -np.inf, dtype=np.float32)
            used_points = 0

            def pixel_indices(points: laspy.ScaleAwarePointRecord) -> tuple[np.ndarray, np.ndarray]:
                x, y, z = np.asarray(points.x), np.asarray(points.y), np.asarray(points.z)
                keep = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
                columns = np.clip(
                    ((x[keep] - mins[0]) * scale).astype(np.int64) + padding, 0, width - 1
                )
                rows = np.clip(
                    ((maxs[1] - y[keep]) * scale).astype(np.int64) + padding, 0, height - 1
                )
                return rows * width + columns, z[keep]

            for points in reader.chunk_iterator(chunk_size):
                indices, z = pixel_indices(points)
                np.maximum.at(maximum_z, indices, (z - mins[2]).astype(np.float32))
                used_points += len(indices)

            surface_z = maximum_z.reshape(height, width)
            occupied = np.isfinite(surface_z)
            image = np.full((height, width, 3), 255, dtype=np.uint8)
            dimension_names = set(reader.header.point_format.dimension_names)
            has_rgb = {"red", "green", "blue"} <= dimension_names

            if has_rgb:
                # Make a second streaming pass and copy the RGB value belonging
                # to the highest point in each pixel.
                flat_image = image.reshape(-1, 3)
                with laspy.open(source_path) as chunks:
                    for points in chunks.chunk_iterator(chunk_size):
                        indices, z = pixel_indices(points)
                        relative_z = (z - mins[2]).astype(np.float32)
                        visible = relative_z >= maximum_z[indices] - 0.001
                        source_indices = np.flatnonzero(
                            np.isfinite(np.asarray(points.x))
                            & np.isfinite(np.asarray(points.y))
                            & np.isfinite(np.asarray(points.z))
                        )[visible]
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
                elevations = surface_z[occupied]
                low, high = np.percentile(elevations, (2, 98))
                span = max(float(high - low), 0.001)
                normalized = np.zeros((height, width), dtype=np.uint8)
                normalized[occupied] = np.clip(
                    (surface_z[occupied] - low) / span * 255, 0, 255
                ).astype(np.uint8)
                colored = cv2.applyColorMap(normalized, cv2.COLORMAP_VIRIDIS)
                image[occupied] = colored[occupied]

            # Fill only tiny enclosed sampling holes; open unscanned areas stay
            # white and the renderer never invents large regions of terrain.
            mask = occupied.astype(np.uint8) * 255
            closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            holes = cv2.subtract(closed, mask)
            if np.any(holes):
                image = cv2.inpaint(image, holes, 2, cv2.INPAINT_TELEA)

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
