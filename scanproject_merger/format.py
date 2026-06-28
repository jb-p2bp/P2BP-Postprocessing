"""Reader for ScannerConsolidator's directory-based ``.scanproject`` format."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from numpy.typing import NDArray


POINT_DTYPE = np.dtype(
    [
        ("position", "<f4", (3,)),
        ("color", "u1", (3,)),
        ("confidence", "u1"),
        ("timestamp", "<f8"),
    ],
    align=False,
)


class ScanProjectError(ValueError):
    """Raised when a scan package is missing or malformed."""


@dataclass(frozen=True)
class GeoReference:
    epsg_code: int
    origin_easting: float
    origin_northing: float
    origin_altitude: float
    rotation_radians: float
    horizontal_accuracy: float | None
    vertical_accuracy: float | None
    heading_accuracy: float | None

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> GeoReference:
        def optional_accuracy(*keys: str) -> float | None:
            current: Any = value
            for key in keys:
                if not isinstance(current, dict) or key not in current:
                    return None
                current = current[key]
            result = float(current)
            return result if result >= 0 else None  # Negative accuracy means unavailable.

        return cls(
            epsg_code=int(value["epsgCode"]),
            origin_easting=float(value["originEasting"]),
            origin_northing=float(value["originNorthing"]),
            origin_altitude=float(value["originAltitude"]),
            rotation_radians=float(value["worldToENURotationRadians"]),
            horizontal_accuracy=optional_accuracy("alignmentMetadata", "horizontalAccuracyMeters")
            or optional_accuracy("originHorizontalAccuracy"),
            vertical_accuracy=optional_accuracy("alignmentMetadata", "verticalAccuracyMeters")
            or optional_accuracy("originVerticalAccuracy"),
            heading_accuracy=optional_accuracy("alignmentMetadata", "headingAccuracyDegrees")
            or optional_accuracy("headingAccuracy"),
        )

    def local_to_projected(self) -> NDArray[np.float64]:
        """Return a homogeneous transform from ARKit coordinates to UTM."""
        cosine, sine = np.cos(self.rotation_radians), np.sin(self.rotation_radians)
        # ScannerConsolidator maps local horizontal (x, -z) into east/north.
        return np.array(
            [
                [cosine, 0.0, sine, self.origin_easting],
                [sine, 0.0, -cosine, self.origin_northing],
                [0.0, 1.0, 0.0, self.origin_altitude],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class PointBatch:
    positions: NDArray[np.float64]
    colors: NDArray[np.uint8]
    confidence: NDArray[np.uint8]
    timestamps: NDArray[np.float64]


@dataclass(frozen=True)
class ScanProject:
    path: Path
    identifier: str
    point_count: int
    chunk_count: int
    voxel_size: float
    georeference: GeoReference | None
    manifest: dict[str, Any]

    @classmethod
    def open(cls, path: str | Path) -> ScanProject:
        package = Path(path)
        manifest_path = package / "manifest.json"
        if not package.is_dir() or not manifest_path.is_file():
            raise ScanProjectError(f"not a .scanproject package: {package}")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            chunk_count = int(manifest["chunkCount"])
            point_count = int(manifest["pointCount"])
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            raise ScanProjectError(f"invalid manifest in {package}: {error}") from error
        geo_value = manifest.get("geoReference")
        try:
            georeference = GeoReference.from_json(geo_value) if geo_value else None
        except (KeyError, TypeError, ValueError) as error:
            raise ScanProjectError(f"invalid georeference in {package}: {error}") from error
        project = cls(
            path=package,
            identifier=str(manifest.get("id", package.stem)),
            point_count=point_count,
            chunk_count=chunk_count,
            voxel_size=float(manifest.get("voxelSizeMeters", 0.02)),  # 2 cm legacy/default voxel size.
            georeference=georeference,
            manifest=manifest,
        )
        project._validate_chunks()
        return project

    def _validate_chunks(self) -> None:
        actual_points = 0
        for index in range(1, self.chunk_count + 1):  # Chunk numbering is one-based.
            path = self.path / f"chunk-{index:05d}.bin"  # Five-digit names, e.g. chunk-00001.bin.
            if not path.is_file():
                raise ScanProjectError(f"missing point chunk: {path}")
            size = path.stat().st_size
            if size % POINT_DTYPE.itemsize:
                # Each packed record is 24 bytes: XYZ (12), RGB (3), confidence (1), timestamp (8).
                raise ScanProjectError(f"chunk size is not divisible by 24: {path}")
            actual_points += size // POINT_DTYPE.itemsize
        if actual_points != self.point_count:
            raise ScanProjectError(
                f"manifest says {self.point_count} points but chunks contain {actual_points}: {self.path}"
            )

    def batches(self) -> Iterator[PointBatch]:
        for index in range(1, self.chunk_count + 1):
            records = np.fromfile(self.path / f"chunk-{index:05d}.bin", dtype=POINT_DTYPE)
            yield PointBatch(
                positions=records["position"].astype(np.float64),
                colors=records["color"].copy(),
                confidence=records["confidence"].copy(),
                timestamps=records["timestamp"].copy(),
            )

    def points(self, minimum_confidence: int = 0) -> PointBatch:  # Zero includes every confidence level.
        batches = list(self.batches())
        if not batches:
            empty = np.empty((0, 3), dtype=np.float64)
            return PointBatch(empty, np.empty((0, 3), dtype=np.uint8), np.empty(0, dtype=np.uint8), np.empty(0))
        positions = np.concatenate([batch.positions for batch in batches])
        colors = np.concatenate([batch.colors for batch in batches])
        confidence = np.concatenate([batch.confidence for batch in batches])
        timestamps = np.concatenate([batch.timestamps for batch in batches])
        keep = confidence >= minimum_confidence
        return PointBatch(positions[keep], colors[keep], confidence[keep], timestamps[keep])


def transform_points(points: NDArray[np.float64], transform: NDArray[np.float64]) -> NDArray[np.float64]:
    """Apply a homogeneous 4x4 transform to an N-by-3 point array."""
    return points @ transform[:3, :3].T + transform[:3, 3]
