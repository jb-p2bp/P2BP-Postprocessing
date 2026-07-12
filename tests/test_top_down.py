from pathlib import Path

import cv2
import laspy
import numpy as np

from scanproject_merger import export_top_down_view


def write_cloud(path: Path, xyz: np.ndarray, rgb: np.ndarray | None = None, point_format: int = 3) -> None:
    header = laspy.LasHeader(point_format=point_format, version="1.2")
    header.scales = np.array([0.001, 0.001, 0.001])
    cloud = laspy.LasData(header)
    cloud.x, cloud.y, cloud.z = xyz.T
    if rgb is not None:
        cloud.red, cloud.green, cloud.blue = (rgb.astype(np.uint16) * 257).T
    cloud.write(path)


def test_top_down_view_renders_all_outdoor_surfaces_north_up(tmp_path: Path):
    # An L-shaped feature plus surfaces at several elevations. The vertical leg
    # extends north and should therefore rise toward the top of the image.
    wall_heights = np.tile(np.linspace(0.3, 1.7, 40), 30)
    horizontal = np.column_stack((np.linspace(0, 4, 1200), np.zeros(1200), wall_heights))
    vertical = np.column_stack((np.zeros(1200), np.linspace(0, 3, 1200), wall_heights))
    clutter_xy = np.random.default_rng(0).uniform([0, 0], [4, 3], size=(400, 2))
    clutter = np.vstack(
        (np.column_stack((clutter_xy, np.zeros(400))), np.column_stack((clutter_xy, np.full(400, 3))))
    )
    source, output = tmp_path / "merged.laz", tmp_path / "merged.top-down-view.png"
    write_cloud(source, np.vstack((horizontal, vertical, clutter)))

    assert export_top_down_view(source, output, pixels_per_meter=30, padding=8) == 3200

    image = cv2.imread(str(output), cv2.IMREAD_GRAYSCALE)
    assert image is not None
    assert image.shape[1] > image.shape[0]
    assert image.min() < 100
    # The scan occupies only a small part of its bounding rectangle.
    assert np.count_nonzero(image < 245) < image.size // 5


def test_top_down_view_keeps_surfaces_at_different_outdoor_elevations(tmp_path: Path):
    rng = np.random.default_rng(1)
    main_floor = np.column_stack((rng.uniform(0, 4, 3000), rng.uniform(0, 3, 3000), np.full(3000, 3.0)))
    main_walls = np.column_stack(
        (np.zeros(2000), rng.uniform(0, 3, 2000), rng.uniform(3.25, 4.75, 2000))
    )
    lower_cluster = np.column_stack(
        (rng.uniform(8, 9, 300), rng.uniform(0, 1, 300), np.full(300, 0.0))
    )
    source, output = tmp_path / "split-level.laz", tmp_path / "split-level.png"
    write_cloud(source, np.vstack((main_floor, main_walls, lower_cluster)))

    used = export_top_down_view(source, output, pixels_per_meter=30, padding=8)

    # Outdoor rendering retains the main surface, structures above it, and the
    # separate lower cluster rather than applying one global height slice.
    assert used == 5300


def test_top_down_view_falls_back_for_flat_cloud_and_caps_dimensions(tmp_path: Path):
    xyz = np.column_stack((np.linspace(0, 1000, 20), np.linspace(0, 2, 20), np.zeros(20)))
    source, output = tmp_path / "flat.laz", tmp_path / "flat.top-down-view.png"
    write_cloud(source, xyz)

    assert export_top_down_view(source, output, maximum_dimension=256, padding=8) == 20
    image = cv2.imread(str(output), cv2.IMREAD_GRAYSCALE)
    assert image is not None
    assert max(image.shape) <= 256


def test_top_down_view_handles_empty_cloud(tmp_path: Path):
    source, output = tmp_path / "empty.las", tmp_path / "empty.top-down-view.png"
    write_cloud(source, np.empty((0, 3)))

    assert export_top_down_view(source, output, padding=4) == 0
    assert cv2.imread(str(output), cv2.IMREAD_GRAYSCALE).shape == (9, 9)


def test_top_down_view_uses_rgb_from_highest_visible_surface(tmp_path: Path):
    xyz = np.array([[0, 0, 0], [0, 0, 2], [1, 1, 0]], dtype=float)
    rgb = np.array([[255, 0, 0], [0, 255, 255], [0, 0, 255]], dtype=np.uint8)
    source, output = tmp_path / "colored.laz", tmp_path / "colored.png"
    write_cloud(source, xyz, rgb)

    assert export_top_down_view(source, output, pixels_per_meter=10, padding=2) == 3
    image = cv2.imread(str(output), cv2.IMREAD_COLOR)
    # Both points share a pixel; the yellow point at z=2 must hide the red one.
    np.testing.assert_array_equal(image[12, 2], [255, 255, 0])


def test_top_down_view_colors_non_rgb_cloud_by_elevation(tmp_path: Path):
    xyz = np.array([[0, 0, 0], [1, 0, 1], [2, 0, 2]], dtype=float)
    source, output = tmp_path / "elevation.laz", tmp_path / "elevation.png"
    write_cloud(source, xyz, point_format=1)

    assert export_top_down_view(source, output, pixels_per_meter=10, padding=2) == 3
    image = cv2.imread(str(output), cv2.IMREAD_COLOR)
    colors = image[2, [2, 12, 22]]
    assert len(np.unique(colors, axis=0)) == 3

