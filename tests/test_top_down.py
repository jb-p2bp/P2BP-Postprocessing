from pathlib import Path

import cv2
import laspy
import numpy as np

from scanproject_merger import export_top_down_view
from scanproject_merger.top_down import _skeletonize_mask


def write_cloud(path: Path, xyz: np.ndarray, rgb: np.ndarray | None = None, point_format: int = 3) -> None:
    header = laspy.LasHeader(point_format=point_format, version="1.2")
    header.scales = np.array([0.001, 0.001, 0.001])
    cloud = laspy.LasData(header)
    cloud.x, cloud.y, cloud.z = xyz.T
    if rgb is not None:
        cloud.red, cloud.green, cloud.blue = (rgb.astype(np.uint16) * 257).T
    cloud.write(path)


def test_skeletonize_mask_reduces_wall_band_to_one_centerline():
    wall_band = np.zeros((40, 40), dtype=np.uint8)
    wall_band[5:35, 16:25] = 255

    centerline = _skeletonize_mask(wall_band)

    assert np.count_nonzero(centerline[10:30]) == 20
    assert np.all(np.count_nonzero(centerline[10:30], axis=1) == 1)


def test_top_down_view_renders_all_outdoor_surfaces_floor_aligned(tmp_path: Path):
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
    assert np.count_nonzero(image < 245) < image.size // 4


def test_top_down_view_keeps_surfaces_at_different_outdoor_elevations(tmp_path: Path):
    rng = np.random.default_rng(1)
    lower_floor = np.column_stack(
        (rng.uniform(0, 4, 6000), rng.uniform(0, 3, 6000), np.zeros(6000))
    )
    upper_floor = np.column_stack(
        (rng.uniform(8, 9, 1000), rng.uniform(0, 1, 1000), np.full(1000, 3.0))
    )
    source, output = tmp_path / "split-level.laz", tmp_path / "split-level.png"
    write_cloud(source, np.vstack((lower_floor, upper_floor)))

    used = export_top_down_view(source, output, pixels_per_meter=30, padding=8)
    image = cv2.imread(str(output), cv2.IMREAD_GRAYSCALE)

    # Both floor centers remain visible even though their elevations differ by
    # more than the two-meter object/canopy cutoff.
    assert used == 7000
    assert np.count_nonzero(image[48:59, 63:74] < 245) > 10
    assert np.count_nonzero(image[78:89, 258:269] < 245) > 10


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


def test_top_down_view_desaturates_visible_surface_for_plan_base(tmp_path: Path):
    xyz = np.array([[0, 0, 0], [0, 0, 2], [1, 1, 0]], dtype=float)
    rgb = np.array([[255, 0, 0], [0, 255, 255], [0, 0, 255]], dtype=np.uint8)
    source, output = tmp_path / "colored.laz", tmp_path / "colored.png"
    write_cloud(source, xyz, rgb)

    assert export_top_down_view(source, output, pixels_per_meter=10, padding=2) == 3
    image = cv2.imread(str(output), cv2.IMREAD_COLOR)
    # The selected surface is intentionally converted to a light grayscale so
    # plan linework remains dominant.
    assert image[12, 2, 0] == image[12, 2, 1] == image[12, 2, 2]
    assert 150 < image[12, 2, 0] < 245


def test_top_down_view_colors_non_rgb_cloud_by_elevation(tmp_path: Path):
    xyz = np.array([[0, 0, 0], [1, 0, 1], [2, 0, 2]], dtype=float)
    source, output = tmp_path / "elevation.laz", tmp_path / "elevation.png"
    write_cloud(source, xyz, point_format=1)

    assert export_top_down_view(source, output, pixels_per_meter=10, padding=2) == 3
    image = cv2.imread(str(output), cv2.IMREAD_COLOR)
    colors = image[2, [2, 12, 22]]
    assert len(np.unique(colors, axis=0)) == 3


def test_site_plan_draws_solid_walls_and_dotted_overhead_boundary(tmp_path: Path):
    ground_x, ground_y = np.meshgrid(np.linspace(0, 4, 81), np.linspace(0, 4, 81))
    ground = np.column_stack((ground_x.ravel(), ground_y.ravel(), np.zeros(ground_x.size)))
    wall_y, wall_z = np.meshgrid(np.linspace(0.5, 3.5, 121), np.linspace(0.2, 1.8, 41))
    wall = np.column_stack((np.full(wall_y.size, 0.5), wall_y.ravel(), wall_z.ravel()))
    theta, radius = np.meshgrid(np.linspace(0, 2 * np.pi, 180), np.linspace(0, 0.8, 20))
    overhead = np.column_stack(
        (
            2.8 + (radius * np.cos(theta)).ravel(),
            2.2 + (radius * np.sin(theta)).ravel(),
            np.full(theta.size, 3.0),
        )
    )
    xyz = np.vstack((ground, wall, overhead))
    source, output = tmp_path / "site.laz", tmp_path / "site-plan.png"
    write_cloud(source, xyz)

    export_top_down_view(source, output, pixels_per_meter=30, padding=8)
    image = cv2.imread(str(output), cv2.IMREAD_GRAYSCALE)

    # Wall line is continuously dark; the overhead outline contains separated
    # dark dashes with light gaps rather than a filled canopy.
    wall_column = image[23:114, 23]
    assert np.count_nonzero(wall_column < 100) > len(wall_column) * 0.7
    canopy_region = image[35:90, 70:120]
    assert np.count_nonzero(canopy_region < 120) > 5
    assert np.count_nonzero(canopy_region > 120) > 5


def test_top_down_view_detects_y_up_floor_before_projecting(tmp_path: Path):
    # Some scanner exports use Y as vertical. This floor is 3 m wide by 8 m
    # long in X/Z, with a short upright feature extending along Y.
    floor_x, floor_z = np.meshgrid(np.linspace(0, 3, 61), np.linspace(0, 8, 161))
    floor = np.column_stack(
        (floor_x.ravel(), np.zeros(floor_x.size), floor_z.ravel())
    )
    feature_y, feature_z = np.meshgrid(np.linspace(0.1, 2.5, 50), np.linspace(2, 3, 40))
    feature = np.column_stack(
        (np.full(feature_y.size, 1.5), feature_y.ravel(), feature_z.ravel())
    )
    source, output = tmp_path / "y-up.laz", tmp_path / "y-up.png"
    write_cloud(source, np.vstack((floor, feature)))

    export_top_down_view(source, output, pixels_per_meter=20, padding=8)
    image = cv2.imread(str(output), cv2.IMREAD_GRAYSCALE)

    # A floor-first X/Z projection is portrait-shaped. Projecting X/Y instead
    # would produce the side elevation and a landscape/square image.
    assert image.shape[0] > image.shape[1] * 2


def test_top_down_view_repairs_enclosed_floor_holes_but_not_background(tmp_path: Path):
    floor_x, floor_y = np.meshgrid(np.linspace(0, 4, 81), np.linspace(0, 4, 81))
    keep = (floor_x - 2) ** 2 + (floor_y - 2) ** 2 > 0.35**2
    floor = np.column_stack(
        (floor_x[keep], floor_y[keep], np.zeros(np.count_nonzero(keep)))
    )
    source, output = tmp_path / "floor-hole.laz", tmp_path / "floor-hole.png"
    write_cloud(source, floor)

    export_top_down_view(source, output, pixels_per_meter=20, padding=8)
    image = cv2.imread(str(output), cv2.IMREAD_GRAYSCALE)

    # The enclosed missing patch at the floor center is repaired, while the
    # padding outside the scanned floor remains white.
    assert image[48, 48] < 250
    assert image[2, 2] == 255


def test_top_down_view_normalizes_broad_rgb_shadows(tmp_path: Path):
    floor_x, floor_y = np.meshgrid(np.linspace(0, 4, 161), np.linspace(0, 2, 81))
    xyz = np.column_stack(
        (floor_x.ravel(), floor_y.ravel(), np.zeros(floor_x.size))
    )
    # Both halves are the same flat surface; only their captured illumination
    # differs. The plan should not preserve that broad cast-shadow boundary.
    brightness = np.where(floor_x.ravel() < 2, 80, 200).astype(np.uint8)
    rgb = np.repeat(brightness[:, None], 3, axis=1)
    source, output = tmp_path / "shadowed-floor.laz", tmp_path / "shadowed-floor.png"
    write_cloud(source, xyz, rgb)

    export_top_down_view(source, output, pixels_per_meter=30, padding=8)
    image = cv2.imread(str(output), cv2.IMREAD_GRAYSCALE)

    shadow = int(np.median(image[20:55, 20:45]))
    sunlit = int(np.median(image[20:55, 90:115]))
    assert abs(shadow - sunlit) < 20
