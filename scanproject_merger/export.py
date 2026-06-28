"""LAS/LAZ and audit-report output for a registration result."""

from __future__ import annotations

import json
from pathlib import Path

import laspy
import numpy as np
from pyproj import CRS

from .format import transform_points
from .registration import RegistrationResult


def _write_cloud(
    output: Path,
    xyz: np.ndarray,
    rgb: np.ndarray,
    confidence: np.ndarray,
    timestamps: np.ndarray,
    source_ids: np.ndarray,
    epsg: int,
) -> None:
    header = laspy.LasHeader(point_format=7, version="1.4")  # LAS 1.4 format 7 supports RGB.
    header.scales = np.array([0.001, 0.001, 0.001])  # Store coordinates at 1 mm resolution.
    # An all-filtered scan yields no points; fall back to a zero origin so the
    # empty cloud still writes instead of crashing on min() of an empty array.
    header.offsets = xyz.min(axis=0) if len(xyz) else np.zeros(3)
    header.add_crs(CRS.from_epsg(epsg))
    header.add_extra_dim(laspy.ExtraBytesParams(name="confidence", type=np.uint8))
    header.add_extra_dim(laspy.ExtraBytesParams(name="source_id", type=np.uint16))
    header.add_extra_dim(laspy.ExtraBytesParams(name="scan_time", type=np.float64))
    cloud = laspy.LasData(header)
    cloud.x, cloud.y, cloud.z = xyz.T
    # Expand 8-bit RGB to LAS 16-bit RGB; 255 * 257 maps exactly to 65535.
    cloud.red = rgb[:, 0].astype(np.uint16) * 257
    cloud.green = rgb[:, 1].astype(np.uint16) * 257
    cloud.blue = rgb[:, 2].astype(np.uint16) * 257
    cloud.confidence = confidence
    cloud.source_id = source_ids
    cloud.scan_time = timestamps
    cloud.write(output)


def export_merged_cloud(
    result: RegistrationResult,
    output: str | Path,
    minimum_confidence: int = 1,  # Export medium- and high-confidence points by default.
    deduplicate_voxel: float = 0.02,  # Keep one point per 2 cm output voxel.
) -> int:
    """Write merged points and return the number of exported records."""
    positions, colors, confidence, timestamps, source_ids = [], [], [], [], []
    for source_id, (scan, transform) in enumerate(zip(result.scans, result.final_transforms)):
        batch = scan.project.points(minimum_confidence)
        positions.append(transform_points(batch.positions, transform))
        colors.append(batch.colors)
        confidence.append(batch.confidence)
        timestamps.append(batch.timestamps)
        source_ids.append(np.full(len(batch.positions), source_id, dtype=np.uint16))
    xyz = np.concatenate(positions)
    rgb = np.concatenate(colors)
    confidence_values = np.concatenate(confidence)
    timestamp_values = np.concatenate(timestamps)
    source_values = np.concatenate(source_ids)
    if deduplicate_voxel > 0:  # A non-positive size disables deduplication.
        keys = np.floor(xyz / deduplicate_voxel).astype(np.int64)
        # Keep the highest-confidence sample in each output voxel.
        order = np.argsort(-confidence_values, kind="stable")
        _, first = np.unique(keys[order], axis=0, return_index=True)
        keep = np.sort(order[first])
        xyz, rgb = xyz[keep], rgb[keep]
        confidence_values, timestamp_values, source_values = (
            confidence_values[keep], timestamp_values[keep], source_values[keep]
        )

    epsg = result.scans[0].project.georeference.epsg_code  # validated during preparation
    _write_cloud(Path(output), xyz, rgb, confidence_values, timestamp_values, source_values, epsg)
    return len(xyz)


def export_transformed_scans(
    result: RegistrationResult,
    output_directory: str | Path,
    minimum_confidence: int = 1,
) -> list[Path]:
    """Write one aligned LAZ file per source scan without cross-scan deduplication."""
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    epsg = result.scans[0].project.georeference.epsg_code
    outputs: list[Path] = []
    for source_id, (scan, transform) in enumerate(zip(result.scans, result.final_transforms)):
        batch = scan.project.points(minimum_confidence)
        xyz = transform_points(batch.positions, transform)
        output = directory / f"{scan.project.path.stem}.laz"
        _write_cloud(
            output,
            xyz,
            batch.colors,
            batch.confidence,
            batch.timestamps,
            np.full(len(xyz), source_id, dtype=np.uint16),
            epsg,
        )
        outputs.append(output)
    return outputs


def export_original_scans(
    result: RegistrationResult,
    output_directory: str | Path,
    minimum_confidence: int = 1,
) -> list[Path]:
    """Convert each source scan to LAZ using its original georeference."""
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    epsg = result.scans[0].project.georeference.epsg_code
    outputs: list[Path] = []
    for source_id, scan in enumerate(result.scans):
        batch = scan.project.points(minimum_confidence)
        xyz = transform_points(batch.positions, scan.initial_transform)
        output = directory / f"{scan.project.path.stem}.laz"
        _write_cloud(
            output,
            xyz,
            batch.colors,
            batch.confidence,
            batch.timestamps,
            np.full(len(xyz), source_id, dtype=np.uint16),
            epsg,
        )
        outputs.append(output)
    return outputs


def write_registration_report(result: RegistrationResult, output: str | Path, exported_points: int) -> None:
    report = {
        "formatVersion": 1,  # Audit-report schema version.
        "exportedPointCount": exported_points,
        "anchorScan": result.scans[0].project.identifier,
        "scans": [
            {
                "sourceId": index,
                "id": scan.project.identifier,
                "path": str(scan.project.path),
                "pointCount": scan.project.point_count,
                "initialTransform": scan.initial_transform.tolist(),
                "correctionTransform": correction.tolist(),
                "finalTransform": final.tolist(),
            }
            for index, (scan, correction, final) in enumerate(
                zip(result.scans, result.correction_transforms, result.final_transforms)
            )
        ],
        "registrationEdges": [
            {
                "fixedSourceId": edge.fixed,
                "movingSourceId": edge.moving,
                "correspondenceCount": edge.correspondence_count,
                "rmseMeters": edge.rmse,
                "overlapRatio": edge.overlap_ratio,
                "initialization": edge.initialization,
                "visualMatchCount": edge.visual_match_count,
                "visualInlierCount": edge.visual_inlier_count,
                "visualRmseMeters": edge.visual_rmse,
                "movingToFixedTransform": edge.moving_to_fixed.tolist(),
            }
            for edge in result.edges
        ],
        "rejectedRegistrationEdges": [
            {
                "fixedSourceId": rejection.edge.fixed,
                "movingSourceId": rejection.edge.moving,
                "reason": rejection.reason,
                "correspondenceCount": rejection.edge.correspondence_count,
                "rmseMeters": rejection.edge.rmse,
                "overlapRatio": rejection.edge.overlap_ratio,
                "qualityScore": rejection.edge.quality_score,
                "loopYawErrorDegrees": rejection.loop_yaw_error_degrees,
                "loopHorizontalErrorMeters": rejection.loop_horizontal_error,
                "loopVerticalErrorMeters": rejection.loop_vertical_error,
                "movingToFixedTransform": rejection.edge.moving_to_fixed.tolist(),
            }
            for rejection in result.rejected_edges
        ],
    }
    Path(output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")  # Two-space JSON indent.
