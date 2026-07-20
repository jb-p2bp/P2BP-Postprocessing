"""Reconstruct a colored surface mesh from a merged LAS/LAZ point cloud.

The registration pipeline writes projected coordinates whose magnitudes are too
large for the float-based geometry used by glTF.  Stitching therefore recenters
the cloud before surface reconstruction and writes the removed origin and CRS to
a JSON sidecar.  Consumers can display the GLB in its stable local coordinates
or restore its geographic placement from that metadata.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path
import struct
from typing import Any

import laspy
import numpy as np
from scipy.spatial import cKDTree


STITCHED_MESH_FILENAME = "stitched-mesh.glb"
STITCHED_MESH_METADATA_FILENAME = "stitched-mesh.metadata.json"


@dataclass(frozen=True)
class StitchingParams:
    """Tuning for Open3D Screened Poisson surface reconstruction."""

    voxel_size: float = 0.05
    poisson_depth: int = 9
    density_quantile: float = 0.02
    target_triangles: int = 1_000_000
    normal_radius_multiplier: float = 4.0
    normal_max_neighbors: int = 60
    normal_consistency_neighbors: int = 30
    outlier_neighbors: int = 20
    outlier_std_ratio: float = 2.5
    poisson_scale: float = 1.1
    minimum_component_triangles: int = 100
    read_chunk_points: int = 500_000
    maximum_stitching_points: int = 1_000_000

    def __post_init__(self) -> None:
        finite_fields = {
            "voxel_size": self.voxel_size,
            "density_quantile": self.density_quantile,
            "normal_radius_multiplier": self.normal_radius_multiplier,
            "outlier_std_ratio": self.outlier_std_ratio,
            "poisson_scale": self.poisson_scale,
        }
        for name, value in finite_fields.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"{name} must be a finite number")

        integer_fields = {
            "poisson_depth": self.poisson_depth,
            "target_triangles": self.target_triangles,
            "normal_max_neighbors": self.normal_max_neighbors,
            "normal_consistency_neighbors": self.normal_consistency_neighbors,
            "outlier_neighbors": self.outlier_neighbors,
            "minimum_component_triangles": self.minimum_component_triangles,
            "read_chunk_points": self.read_chunk_points,
            "maximum_stitching_points": self.maximum_stitching_points,
        }
        for name, value in integer_fields.items():
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f"{name} must be an integer")

        if self.voxel_size <= 0:
            raise ValueError("voxel_size must be positive")
        if not 3 <= self.poisson_depth <= 14:
            raise ValueError("poisson_depth must be between 3 and 14")
        if not 0 <= self.density_quantile < 1:
            raise ValueError("density_quantile must be in [0, 1)")
        if self.target_triangles < 0:
            raise ValueError("target_triangles cannot be negative")
        if self.normal_radius_multiplier <= 0:
            raise ValueError("normal_radius_multiplier must be positive")
        if self.normal_max_neighbors < 3:
            raise ValueError("normal_max_neighbors must be at least 3")
        if self.normal_consistency_neighbors < 3:
            raise ValueError("normal_consistency_neighbors must be at least 3")
        if self.outlier_neighbors < 0:
            raise ValueError("outlier_neighbors cannot be negative")
        if self.outlier_neighbors in (1, 2):
            raise ValueError("outlier_neighbors must be zero or at least 3")
        if self.outlier_std_ratio <= 0:
            raise ValueError("outlier_std_ratio must be positive")
        if self.poisson_scale <= 1:
            raise ValueError("poisson_scale must be greater than 1")
        if self.minimum_component_triangles < 0:
            raise ValueError("minimum_component_triangles cannot be negative")
        if self.read_chunk_points <= 0:
            raise ValueError("read_chunk_points must be positive")
        if self.maximum_stitching_points < 10:
            raise ValueError("maximum_stitching_points must be at least 10")


DEFAULT_STITCHING_PARAMS = StitchingParams()


@dataclass(frozen=True)
class StitchingResult:
    mesh: Path
    metadata: Path
    input_points: int
    stitched_points: int
    vertices: int
    triangles: int
    origin: tuple[float, float, float]
    epsg: int | None


def _load_open3d() -> Any:
    # Imported lazily so queue/config code remains importable in lightweight
    # environments that do not execute stitching.
    import open3d as o3d

    return o3d


def _temporary_sibling(output: Path) -> Path:
    """Return a unique sibling that retains the real format suffix."""

    return output.with_suffix(f".{os.getpid()}.tmp{output.suffix}")


def _point_format_has_colors(point_format: laspy.PointFormat) -> bool:
    dimensions = set(point_format.dimension_names)
    return {"red", "green", "blue"}.issubset(dimensions)


def _chunk_colors(points: Any, has_colors: bool) -> np.ndarray:
    if not has_colors:
        return np.full((len(points), 3), 0.7, dtype=np.float64)

    colors = np.empty((len(points), 3), dtype=np.float64)
    colors[:, 0] = points.red
    colors[:, 1] = points.green
    colors[:, 2] = points.blue
    return colors


def _load_downsampled_cloud(
    source: Path,
    params: StitchingParams,
    o3d: Any,
) -> tuple[Any, Any, np.ndarray, int]:
    """Read and voxel-reduce a LAS/LAZ without retaining every source point."""

    input_points = 0
    maximum_color = 0.0
    point_chunks: list[np.ndarray] = []
    color_chunks: list[np.ndarray] = []
    retained_keys: set[tuple[int, int, int]] = set()

    with laspy.open(source) as reader:
        crs = reader.header.parse_crs()
        bounds = np.vstack((reader.header.mins, reader.header.maxs)).astype(
            np.float64
        )
        if not np.isfinite(bounds).all():
            raise ValueError("point cloud header contains non-finite bounds")
        origin = (bounds[0] + bounds[1]) / 2.0
        has_colors = _point_format_has_colors(reader.header.point_format)

        for points in reader.chunk_iterator(params.read_chunk_points):
            xyz = np.empty((len(points), 3), dtype=np.float64)
            xyz[:, 0] = points.x
            xyz[:, 1] = points.y
            xyz[:, 2] = points.z
            colors = _chunk_colors(points, has_colors)

            finite = np.isfinite(xyz).all(axis=1)
            if not finite.all():
                xyz = xyz[finite]
                colors = colors[finite]
            input_points += len(xyz)
            if not len(xyz):
                continue

            if has_colors:
                maximum_color = max(maximum_color, float(colors.max()))

            chunk_keys = np.floor(xyz / params.voxel_size).astype(np.int64)
            _, first = np.unique(chunk_keys, axis=0, return_index=True)
            first.sort()
            chunk_keys = chunk_keys[first]
            retained = []
            for index, values in enumerate(chunk_keys):
                key = (int(values[0]), int(values[1]), int(values[2]))
                if key in retained_keys:
                    continue
                retained_keys.add(key)
                if len(retained_keys) > params.maximum_stitching_points:
                    raise ValueError(
                        "voxel downsampling retained more than "
                        f"{params.maximum_stitching_points} points; increase "
                        "voxel_size or maximum_stitching_points"
                    )
                retained.append(first[index])

            if retained:
                retained_indices = np.asarray(retained, dtype=np.intp)
                point_chunks.append(xyz[retained_indices] - origin)
                color_chunks.append(colors[retained_indices])

    if input_points < 10:
        raise ValueError("at least 10 finite points are required for stitching")

    if has_colors:
        # LAS RGB dimensions are uint16, but some producers store unexpanded
        # 8-bit values. Scaling after chunk aggregation makes this decision
        # consistently across the entire file.
        divisor = 255.0 if maximum_color <= 255 else 65535.0
    local_points = np.concatenate(point_chunks)
    retained_colors = np.concatenate(color_chunks)
    if has_colors:
        retained_colors = np.clip(retained_colors / divisor, 0.0, 1.0)

    del retained_keys, point_chunks, color_chunks
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(local_points)
    point_cloud.colors = o3d.utility.Vector3dVector(retained_colors)
    del local_points, retained_colors

    return point_cloud, crs, origin, input_points


def _remove_small_components(mesh: Any, minimum_triangles: int) -> None:
    if minimum_triangles <= 0 or len(mesh.triangles) == 0:
        return

    triangle_clusters, cluster_sizes, _ = mesh.cluster_connected_triangles()
    clusters = np.asarray(triangle_clusters)
    sizes = np.asarray(cluster_sizes)
    if len(clusters):
        mesh.remove_triangles_by_mask(sizes[clusters] < minimum_triangles)


def _write_mesh_atomic(mesh: Any, output: Path) -> None:
    """Write a self-contained glTF 2.0 binary without ASSIMP side effects."""

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_sibling(output)
    try:
        positions = np.ascontiguousarray(
            np.asarray(mesh.vertices),
            dtype=np.float32,
        )
        normals = np.ascontiguousarray(
            np.asarray(mesh.vertex_normals),
            dtype=np.float32,
        )
        colors = np.ascontiguousarray(
            np.rint(np.clip(np.asarray(mesh.vertex_colors), 0, 1) * 255),
            dtype=np.uint8,
        )
        indices = np.ascontiguousarray(
            np.asarray(mesh.triangles).reshape(-1),
            dtype=np.uint32,
        )

        binary = bytearray()
        buffer_views: list[dict[str, Any]] = []
        accessors: list[dict[str, Any]] = []

        def append_accessor(
            data: np.ndarray,
            *,
            target: int,
            component_type: int,
            accessor_type: str,
            include_bounds: bool = False,
            normalized: bool = False,
        ) -> int:
            while len(binary) % 4:
                binary.append(0)
            offset = len(binary)
            payload = data.tobytes()
            binary.extend(payload)
            view = {
                "buffer": 0,
                "byteOffset": offset,
                "byteLength": len(payload),
                "target": target,
            }
            buffer_views.append(view)
            accessor: dict[str, Any] = {
                "bufferView": len(buffer_views) - 1,
                "componentType": component_type,
                "count": len(data),
                "type": accessor_type,
            }
            if include_bounds:
                minimum = np.asarray(data.min(axis=0)).reshape(-1)
                maximum = np.asarray(data.max(axis=0)).reshape(-1)
                accessor["min"] = minimum.astype(float).tolist()
                accessor["max"] = maximum.astype(float).tolist()
                if np.issubdtype(data.dtype, np.integer):
                    accessor["min"] = minimum.astype(int).tolist()
                    accessor["max"] = maximum.astype(int).tolist()
            if normalized:
                accessor["normalized"] = True
            accessors.append(accessor)
            return len(accessors) - 1

        index_accessor = append_accessor(
            indices,
            target=34963,
            component_type=5125,
            accessor_type="SCALAR",
            include_bounds=True,
        )
        position_accessor = append_accessor(
            positions,
            target=34962,
            component_type=5126,
            accessor_type="VEC3",
            include_bounds=True,
        )
        normal_accessor = append_accessor(
            normals,
            target=34962,
            component_type=5126,
            accessor_type="VEC3",
        )
        color_accessor = append_accessor(
            colors,
            target=34962,
            component_type=5121,
            accessor_type="VEC3",
            normalized=True,
        )
        while len(binary) % 4:
            binary.append(0)

        gltf = {
            "asset": {
                "version": "2.0",
                "generator": "P2BP stitching",
            },
            "scene": 0,
            "scenes": [{"nodes": [0]}],
            "nodes": [
                {
                    "mesh": 0,
                    # Convert the projected Z-up frame to glTF's Y-up frame
                    # without modifying the local-coordinate vertex payload.
                    "rotation": [-2**-0.5, 0, 0, 2**-0.5],
                }
            ],
            "meshes": [
                {
                    "primitives": [
                        {
                            "attributes": {
                                "POSITION": position_accessor,
                                "NORMAL": normal_accessor,
                                "COLOR_0": color_accessor,
                            },
                            "indices": index_accessor,
                            "material": 0,
                            "mode": 4,
                        }
                    ]
                }
            ],
            "materials": [
                {
                    "doubleSided": True,
                    "pbrMetallicRoughness": {
                        "baseColorFactor": [1, 1, 1, 1],
                        "metallicFactor": 0,
                        "roughnessFactor": 1,
                    },
                }
            ],
            "buffers": [{"byteLength": len(binary)}],
            "bufferViews": buffer_views,
            "accessors": accessors,
        }
        json_chunk = json.dumps(
            gltf,
            separators=(",", ":"),
        ).encode("utf-8")
        json_chunk += b" " * (-len(json_chunk) % 4)

        total_length = 12 + 8 + len(json_chunk) + 8 + len(binary)
        with temporary.open("wb") as handle:
            handle.write(struct.pack("<4sII", b"glTF", 2, total_length))
            handle.write(struct.pack("<I4s", len(json_chunk), b"JSON"))
            handle.write(json_chunk)
            handle.write(struct.pack("<I4s", len(binary), b"BIN\x00"))
            handle.write(binary)
        if temporary.stat().st_size != total_length:
            raise RuntimeError(f"GLB length mismatch while writing: {output}")
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _write_metadata_atomic(
    output: Path,
    *,
    mesh_filename: str,
    origin: np.ndarray,
    epsg: int | None,
    crs_wkt: str | None,
    result_counts: dict[str, int],
    params: StitchingParams,
) -> None:
    metadata = {
        "formatVersion": 1,
        "mesh": mesh_filename,
        "coordinateSystem": {
            "epsg": epsg,
            "wkt": crs_wkt,
            "localOrigin": origin.tolist(),
            "units": "meters",
        },
        **result_counts,
        "stitching": {
            "method": "screened-poisson",
            **asdict(params),
        },
    }
    temporary = _temporary_sibling(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        temporary.write_text(
            json.dumps(metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def stitch_point_cloud(
    input_cloud: str | Path,
    output_mesh: str | Path,
    *,
    metadata_output: str | Path | None = None,
    params: StitchingParams = DEFAULT_STITCHING_PARAMS,
) -> StitchingResult:
    """Stitch a merged LAS/LAZ cloud into a colored, local-coordinate GLB.

    Screened Poisson produces a continuous surface from consistently oriented
    normals. Low-support vertices and tiny components are removed, the result
    is cropped to the observed bounds, and an optional triangle budget keeps
    the GLB practical to transfer and render.
    """

    source = Path(input_cloud)
    mesh_path = Path(output_mesh)
    if mesh_path.suffix.lower() != ".glb":
        raise ValueError("stitched mesh output must use the .glb extension")
    metadata_path = (
        Path(metadata_output)
        if metadata_output is not None
        else mesh_path.with_name(STITCHED_MESH_METADATA_FILENAME)
    )

    o3d = _load_open3d()
    point_cloud, crs, origin, input_points = _load_downsampled_cloud(
        source,
        params,
        o3d,
    )
    if params.outlier_neighbors:
        point_cloud, _ = point_cloud.remove_statistical_outlier(
            nb_neighbors=params.outlier_neighbors,
            std_ratio=params.outlier_std_ratio,
        )
    stitched_points = len(point_cloud.points)
    if stitched_points < 10:
        raise ValueError("stitching filters left fewer than 10 points")

    normal_radius = params.voxel_size * params.normal_radius_multiplier
    point_cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius,
            max_nn=params.normal_max_neighbors,
        )
    )
    point_cloud.orient_normals_consistent_tangent_plane(
        params.normal_consistency_neighbors
    )

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        point_cloud,
        depth=params.poisson_depth,
        scale=params.poisson_scale,
        linear_fit=False,
    )
    density_values = np.asarray(densities)
    if params.density_quantile and len(density_values):
        threshold = float(np.quantile(density_values, params.density_quantile))
        mesh.remove_vertices_by_mask(density_values < threshold)

    mesh = mesh.crop(point_cloud.get_axis_aligned_bounding_box())
    _remove_small_components(mesh, params.minimum_component_triangles)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()

    if (
        params.target_triangles
        and len(mesh.triangles) > params.target_triangles
    ):
        mesh = mesh.simplify_quadric_decimation(params.target_triangles)
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_unreferenced_vertices()

    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        raise ValueError("stitching produced an empty mesh")

    # Reapply colors after reconstruction/decimation so the surface retains the
    # nearest observed RGB sample regardless of Open3D exporter behavior.
    observed_points = np.asarray(point_cloud.points)
    observed_colors = np.asarray(point_cloud.colors)
    _, nearest = cKDTree(observed_points).query(
        np.asarray(mesh.vertices),
        workers=-1,
    )
    mesh.vertex_colors = o3d.utility.Vector3dVector(observed_colors[nearest])
    mesh.compute_vertex_normals()
    _write_mesh_atomic(mesh, mesh_path)

    epsg = crs.to_epsg() if crs is not None else None
    counts = {
        "inputPointCount": input_points,
        "stitchedPointCount": stitched_points,
        "vertexCount": len(mesh.vertices),
        "triangleCount": len(mesh.triangles),
    }
    _write_metadata_atomic(
        metadata_path,
        mesh_filename=mesh_path.name,
        origin=origin,
        epsg=epsg,
        crs_wkt=crs.to_wkt() if crs is not None else None,
        result_counts=counts,
        params=params,
    )

    return StitchingResult(
        mesh=mesh_path,
        metadata=metadata_path,
        input_points=input_points,
        stitched_points=stitched_points,
        vertices=len(mesh.vertices),
        triangles=len(mesh.triangles),
        origin=tuple(float(value) for value in origin),
        epsg=epsg,
    )
