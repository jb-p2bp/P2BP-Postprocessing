import json
from pathlib import Path

import laspy
import numpy as np
from pyproj import CRS
import pytest

from stitching import (
    StitchingParams,
    _load_downsampled_cloud,
    _load_open3d,
    stitch_point_cloud,
)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("voxel_size", 0, "voxel_size"),
        ("poisson_depth", 2, "poisson_depth"),
        ("density_quantile", 1, "density_quantile"),
        ("target_triangles", -1, "target_triangles"),
        ("outlier_neighbors", 2, "outlier_neighbors"),
        ("poisson_scale", 1, "poisson_scale"),
        ("read_chunk_points", 0, "read_chunk_points"),
        ("maximum_stitching_points", 9, "maximum_stitching_points"),
        ("voxel_size", float("nan"), "voxel_size"),
        ("density_quantile", float("nan"), "density_quantile"),
        (
            "normal_radius_multiplier",
            float("inf"),
            "normal_radius_multiplier",
        ),
        ("outlier_std_ratio", float("-inf"), "outlier_std_ratio"),
        ("poisson_scale", float("nan"), "poisson_scale"),
        ("poisson_depth", 9.5, "poisson_depth"),
        ("target_triangles", True, "target_triangles"),
        ("read_chunk_points", 100.5, "read_chunk_points"),
    ],
)
def test_stitching_params_reject_invalid_values(field, value, message):
    values = {field: value}
    with pytest.raises(ValueError, match=message):
        StitchingParams(**values)


def _write_colored_sphere(path: Path, point_count: int = 1200) -> None:
    indices = np.arange(point_count, dtype=np.float64) + 0.5
    z = 1.0 - 2.0 * indices / point_count
    radius = np.sqrt(1.0 - z * z)
    theta = np.pi * (1.0 + np.sqrt(5.0)) * indices
    local = np.column_stack(
        (
            radius * np.cos(theta),
            radius * np.sin(theta),
            z,
        )
    )
    origin = np.array([465_173.0, 3_139_409.0, 25.5])
    xyz = local + origin

    header = laspy.LasHeader(point_format=7, version="1.4")
    header.scales = np.array([0.001, 0.001, 0.001])
    header.offsets = origin
    header.add_crs(CRS.from_epsg(32617))
    cloud = laspy.LasData(header)
    cloud.x, cloud.y, cloud.z = xyz.T
    rgb = np.clip((local + 1.0) * 0.5 * 65535, 0, 65535).astype(np.uint16)
    cloud.red, cloud.green, cloud.blue = rgb.T
    cloud.write(path)


def test_stitches_laz_to_local_colored_glb_and_metadata(tmp_path):
    source = tmp_path / "merged.laz"
    mesh = tmp_path / "stitched-mesh.glb"
    metadata = tmp_path / "stitched-mesh.metadata.json"
    _write_colored_sphere(source)

    result = stitch_point_cloud(
        source,
        mesh,
        metadata_output=metadata,
        params=StitchingParams(
            voxel_size=0.04,
            poisson_depth=5,
            density_quantile=0,
            target_triangles=0,
            normal_radius_multiplier=6,
            normal_max_neighbors=40,
            normal_consistency_neighbors=20,
            outlier_neighbors=0,
            minimum_component_triangles=0,
            read_chunk_points=113,
        ),
    )

    assert result.mesh == mesh
    assert result.metadata == metadata
    assert result.input_points == 1200
    assert result.stitched_points > 100
    assert result.vertices > 0
    assert result.triangles > 0
    assert result.epsg == 32617
    assert mesh.read_bytes()[:4] == b"glTF"
    import open3d as o3d

    reloaded = o3d.io.read_triangle_mesh(str(mesh), enable_post_processing=True)
    assert len(reloaded.vertices) == result.vertices
    assert len(reloaded.triangles) == result.triangles
    assert reloaded.has_vertex_colors()

    payload = json.loads(metadata.read_text(encoding="utf-8"))
    assert payload["mesh"] == "stitched-mesh.glb"
    assert payload["coordinateSystem"]["epsg"] == 32617
    assert payload["coordinateSystem"]["units"] == "meters"
    assert payload["stitching"]["method"] == "screened-poisson"
    assert payload["vertexCount"] == result.vertices
    assert payload["triangleCount"] == result.triangles


def test_chunked_loading_uses_one_stable_voxel_grid(tmp_path):
    source = tmp_path / "merged.laz"
    _write_colored_sphere(source)
    o3d = _load_open3d()

    small_chunks, _, small_origin, small_count = _load_downsampled_cloud(
        source,
        StitchingParams(voxel_size=0.04, read_chunk_points=113),
        o3d,
    )
    one_chunk, _, one_origin, one_count = _load_downsampled_cloud(
        source,
        StitchingParams(voxel_size=0.04, read_chunk_points=10_000),
        o3d,
    )

    assert small_count == one_count == 1200
    np.testing.assert_allclose(small_origin, one_origin)
    np.testing.assert_allclose(
        np.asarray(small_chunks.points),
        np.asarray(one_chunk.points),
    )
    np.testing.assert_allclose(
        np.asarray(small_chunks.colors),
        np.asarray(one_chunk.colors),
    )


def test_chunked_loading_enforces_stitching_point_limit(tmp_path):
    source = tmp_path / "merged.laz"
    _write_colored_sphere(source, point_count=100)
    o3d = _load_open3d()

    with pytest.raises(ValueError, match="retained more than 10 points"):
        _load_downsampled_cloud(
            source,
            StitchingParams(
                voxel_size=0.001,
                read_chunk_points=7,
                maximum_stitching_points=10,
            ),
            o3d,
        )


def test_stitching_requires_glb_output(tmp_path):
    source = tmp_path / "input.laz"
    _write_colored_sphere(source, point_count=20)

    with pytest.raises(ValueError, match=r"\.glb"):
        stitch_point_cloud(source, tmp_path / "mesh.ply")
