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

    def __post_init__(self) -> None:
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


def _las_colors(cloud: laspy.LasData) -> np.ndarray:
    dimension_names = set(cloud.point_format.dimension_names)
    if not {"red", "green", "blue"}.issubset(dimension_names):
        return np.full((len(cloud.points), 3), 0.7, dtype=np.float64)

    rgb = np.column_stack((cloud.red, cloud.green, cloud.blue)).astype(np.float64)
    # LAS RGB dimensions are uint16, but some producers store unexpanded 8-bit
    # values. Detect that convention so those clouds do not render nearly black.
    divisor = 255.0 if len(rgb) and float(rgb.max()) <= 255 else 65535.0
    return np.clip(rgb / divisor, 0.0, 1.0)


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

        def append_view(data: bytes, *, target: int) -> int:
            while len(binary) % 4:
                binary.append(0)
            offset = len(binary)
            binary.extend(data)
            view = {
                "buffer": 0,
                "byteOffset": offset,
                "byteLength": len(data),
                "target": target,
            }
            buffer_views.append(view)
            return len(buffer_views) - 1

        index_view = append_view(indices.tobytes(), target=34963)
        position_view = append_view(positions.tobytes(), target=34962)
        normal_view = append_view(normals.tobytes(), target=34962)
        color_view = append_view(colors.tobytes(), target=34962)
        while len(binary) % 4:
            binary.append(0)

        vertex_count = len(positions)
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
                                "POSITION": 1,
                                "NORMAL": 2,
                                "COLOR_0": 3,
                            },
                            "indices": 0,
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
            "accessors": [
                {
                    "bufferView": index_view,
                    "componentType": 5125,
                    "count": len(indices),
                    "type": "SCALAR",
                    "min": [int(indices.min())],
                    "max": [int(indices.max())],
                },
                {
                    "bufferView": position_view,
                    "componentType": 5126,
                    "count": vertex_count,
                    "type": "VEC3",
                    "min": positions.min(axis=0).astype(float).tolist(),
                    "max": positions.max(axis=0).astype(float).tolist(),
                },
                {
                    "bufferView": normal_view,
                    "componentType": 5126,
                    "count": vertex_count,
                    "type": "VEC3",
                },
                {
                    "bufferView": color_view,
                    "componentType": 5121,
                    "count": vertex_count,
                    "type": "VEC3",
                    "normalized": True,
                },
            ],
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
    params: StitchingParams = StitchingParams(),
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

    cloud = laspy.read(source)
    crs = cloud.header.parse_crs()
    xyz = np.column_stack((cloud.x, cloud.y, cloud.z)).astype(np.float64)
    colors = _las_colors(cloud)
    finite = np.isfinite(xyz).all(axis=1)
    xyz, colors = xyz[finite], colors[finite]
    input_points = len(xyz)
    if input_points < 10:
        raise ValueError("at least 10 finite points are required for stitching")

    minimum = xyz.min(axis=0)
    maximum = xyz.max(axis=0)
    origin = (minimum + maximum) / 2.0
    local_xyz = xyz - origin
    del cloud, xyz

    o3d = _load_open3d()
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(local_xyz)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    del local_xyz, colors

    point_cloud = point_cloud.voxel_down_sample(params.voxel_size)
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
