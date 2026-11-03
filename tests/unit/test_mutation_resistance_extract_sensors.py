"""Mutation-resistance tests for extract/{lidar,calibration,camera,radar}.py (T5, T6, T7c, T8, T14, T19, T23).

Each test targets a wrong implementation the existing tests let through: channel keys that differ
from their list positions, ring priority, default percentile, non-finite intensities, an asymmetric
ego box, extrinsics variant passthrough, z-only translation errors, camera numeric accuracy with a
strongly distorted lens, a non-0.5 resize factor on a non-uniform image, and radar column order.
"""

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tests.unit.test_calibration import FIXTURE_PATH
from viking_splatad.extract.calibration import compare_transforms, load_extrinsics, parse_extrinsics_text
from viking_splatad.extract.camera import CameraIntrinsics, decode_and_resize_image, undistort_image
from viking_splatad.extract.lidar import assemble_scan_array, assign_channels, normalise_intensity
from viking_splatad.extract.radar import radar_cloud_to_array

# Arbitrary fixed seed: only reproducibility of the random fixtures.
RANDOM_SEED = 777


# --------------------------------------------------------------------------- T19


@pytest.mark.covers("T19")
def test_assign_channels_keys_differ_from_positions_and_ring_has_priority():
    """Channel keys are returned (not list positions), for unsorted keys, and a ring beats elevation."""
    # Keys are neither 0..n-1 nor sorted by elevation; position 0 holds key 10, not key 0.
    mapping = {10: -5.0, 3: 4.0, 7: 0.0, 42: 9.0}
    elevations = np.array([8.6, -4.8, 0.3, 3.1, 9.0])
    # Nearest beams by hand: 9.0 -> key 42 (|8.6-9|=0.4), -5 -> key 10, 0 -> key 7, 4 -> key 3, 9 -> key 42.
    channels = assign_channels(mapping, elevation_deg=elevations)
    np.testing.assert_array_equal(channels, [42, 10, 7, 3, 42])
    assert channels.dtype == np.int64
    # A ring is a channel KEY: key 42 is valid even though only 4 entries exist; 0 is not a key.
    np.testing.assert_array_equal(assign_channels(mapping, ring=np.array([42, 10, 3])), [42, 10, 3])
    with pytest.raises(ValueError):
        assign_channels(mapping, ring=np.array([0]))
    # Ring priority: contradictory elevations (all nearest to key 42) must be ignored when rings are given.
    rings = np.array([10, 3, 7])
    np.testing.assert_array_equal(
        assign_channels(mapping, ring=rings, elevation_deg=np.array([9.0, 9.0, 9.0])), rings
    )
    # Neither input -> error.
    with pytest.raises(ValueError):
        assign_channels(mapping)


# --------------------------------------------------------------------------- T14


@pytest.mark.covers("T14")
def test_normalise_intensity_default_percentile_is_99_and_ignores_nonfinite():
    """Default scale is the 99th percentile of the FINITE values; NaN/inf are not counted or clipped."""
    finite_values = np.arange(1.0, 101.0)  # 1..100, 99th percentile (linear interpolation) = 99.01
    scan_a = np.concatenate([finite_values[:50], [np.nan, np.inf, -np.inf]])
    scan_b = finite_values[50:]
    result = normalise_intensity([scan_a, scan_b])  # no percentile argument: the default
    assert result.scale == pytest.approx(99.01)
    assert result.total_count == 100  # non-finite excluded from the denominator
    assert result.clipped_count == 1  # only the finite 100 exceeds 99.01 (inf / -inf are not counted)
    assert result.clipped_fraction == pytest.approx(0.01)
    # Finite entries are scaled by that scale; the output keeps scan structure and float32 dtype.
    np.testing.assert_allclose(result.normalised_scans[0][:50], finite_values[:50] / 99.01, atol=1e-6)
    assert result.normalised_scans[1].dtype == np.float32 and len(result.normalised_scans) == 2
    # An explicit percentile changes the scale accordingly (50th percentile of 1..100 is 50.5).
    assert normalise_intensity([finite_values], percentile=50.0).scale == pytest.approx(50.5)
    # An explicit dtype_max wins over any percentile.
    assert normalise_intensity([finite_values], percentile=50.0, dtype_max=200.0).scale == 200.0
    # All-zero intensities give scale 0: refused.
    with pytest.raises(ValueError):
        normalise_intensity([np.zeros(5)])


# --------------------------------------------------------------------------- assemble_scan_array


def test_assemble_scan_array_asymmetric_box_zero_range_and_row_alignment():
    """Ego box (dx,dy,dz) is applied per axis in x,y,z order on |p|; every output column stays row-aligned."""
    box = (0.5, 2.0, 3.0)
    xyz = np.array([
        [1.0, 0.1, 0.1],    # outside on x only -> keep
        [0.1, 1.0, 0.1],    # inside on all (y 1.0 < 2.0) -> drop (a box with swapped axes would keep it)
        [0.1, 2.5, 0.1],    # outside on y -> keep
        [0.1, 0.1, 2.9],    # z 2.9 < 3.0 -> drop
        [0.1, 0.1, 3.0],    # exactly on dz -> keep (>=)
        [-1.0, 0.0, 0.1],   # negative x, abs -> keep
        [0.1, -2.0, 0.1],   # negative y exactly on dy -> keep
        [0.0, 0.0, 0.0],    # zero range -> drop
        [4.0, 4.0, 4.0],    # NaN intensity -> drop
        [5.0, 5.0, 5.0],    # NaN time -> drop
    ])
    intensity = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, np.nan, 0.9])
    relative_time = np.array([-0.04, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, np.nan])
    channel = np.arange(10) + 100
    scan = assemble_scan_array(xyz, intensity, relative_time, channel, box)
    kept_rows = [0, 2, 4, 5, 6]
    expected = np.column_stack([xyz[kept_rows], intensity[kept_rows], relative_time[kept_rows], channel[kept_rows]])
    assert scan.dtype == np.float32
    np.testing.assert_allclose(scan, expected.astype(np.float32), atol=1e-7)
    # A zero box disables the ego test but zero-range points are still removed.
    open_box = assemble_scan_array(xyz[:8], intensity[:8], relative_time[:8], channel[:8], (0.0, 0.0, 0.0))
    assert open_box.shape[0] == 7 and not np.any(np.all(open_box[:, :3] == 0.0, axis=1))


# --------------------------------------------------------------------------- T5 / T6


@pytest.mark.covers("T5")
def test_load_extrinsics_passes_the_variant_through(tmp_path):
    """load_extrinsics(high_grass) returns the high-grass radar rotation, not the short-grass one."""
    (tmp_path / "extrinsics").mkdir()
    (tmp_path / "extrinsics" / "extrinsics.txt").write_text(FIXTURE_PATH.read_text())
    short = load_extrinsics(tmp_path, "short_grass")["base_link->hugin_radar"]
    high = load_extrinsics(str(tmp_path), "high_grass")["base_link->hugin_radar"]  # str path accepted
    # Quaternions copied from the fixture file text (x, y, z, w), converted with scipy (independent code).
    np.testing.assert_allclose(high[:3, :3], Rotation.from_quat([0.703, 0.711, 0.004, 0.004]).as_matrix(), atol=1e-9)
    np.testing.assert_allclose(short[:3, :3], Rotation.from_quat([0.701, 0.712, 0.028, 0.027]).as_matrix(), atol=1e-9)
    assert not np.allclose(short, high, atol=1e-4)
    # A variant absent from the file propagates the error.
    with pytest.raises(ValueError):
        load_extrinsics(tmp_path, "mystery_grass")


@pytest.mark.covers("T6")
def test_compare_transforms_z_only_translation_and_random_rotation_angle():
    """Translation error uses all three axes; rotation error is the relative angle of random rotations."""
    rng = np.random.default_rng(RANDOM_SEED)
    rotation_a = Rotation.from_rotvec(rng.normal(size=3))
    rotation_b = Rotation.from_rotvec(rng.normal(size=3))
    transform_a, transform_b = np.eye(4), np.eye(4)
    transform_a[:3, :3], transform_b[:3, :3] = rotation_a.as_matrix(), rotation_b.as_matrix()
    transform_a[:3, 3] = [1.0, 2.0, 3.0]
    transform_b[:3, 3] = [1.0, 2.0, 3.0 + 1.75]  # differs in z only
    translation_error, rotation_error = compare_transforms(transform_a, transform_b)
    assert translation_error == pytest.approx(1.75)
    # Independent angle: magnitude of the relative rotation as computed by scipy's rotvec norm.
    assert rotation_error == pytest.approx((rotation_a.inv() * rotation_b).magnitude(), abs=1e-9)
    # Symmetric in its arguments; not "0 for same translation".
    swapped = compare_transforms(transform_b, transform_a)
    assert swapped[0] == pytest.approx(translation_error) and swapped[1] == pytest.approx(rotation_error)
    assert rotation_error > 0.1  # fixture rotations genuinely differ


# --------------------------------------------------------------------------- T7c


def gaussian_blob_image(centre_x, centre_y, width, height, sigma):
    """float32 image with an analytic Gaussian blob (no rasterisation rounding) as uint8 0..255."""
    column_grid, row_grid = np.meshgrid(np.arange(width), np.arange(height))
    blob = np.exp(-((column_grid - centre_x) ** 2 + (row_grid - centre_y) ** 2) / (2 * sigma**2))
    return np.round(255.0 * blob).astype(np.uint8)


@pytest.mark.covers("T7c")
def test_undistort_strong_distortion_blob_position_crop_and_validity():
    """Strongly distorted blobs land at K_new (X/Z, Y/Z) within 0.4 px; the crop has no invalid border."""
    intrinsics = CameraIntrinsics(
        fx=300.0, fy=310.0, cx=200.5, cy=150.5, width=400, height=300,
        distortion=np.array([-0.35, 0.12, 0.002, -0.003, 0.0]), model="plumb_bob",
    )
    camera_matrix = intrinsics.camera_matrix()
    # Off-axis 3D points (normalised x, y at z = 1), each well inside the valid region after the crop.
    normalised_points = [(-0.30, -0.2), (0.25, 0.15), (0.0, -0.1), (0.2, -0.2)]
    for normalised_x, normalised_y in normalised_points:
        point_3d = np.array([[normalised_x, normalised_y, 1.0]])
        distorted_pixel, _ = cv2.projectPoints(
            point_3d, np.zeros(3), np.zeros(3), camera_matrix, intrinsics.distortion
        )
        distorted_x, distorted_y = distorted_pixel.reshape(2)
        image = gaussian_blob_image(distorted_x, distorted_y, intrinsics.width, intrinsics.height, sigma=3.0)
        undistorted, new_intrinsics = undistort_image(image, intrinsics)
        # Intensity-weighted centroid of the blob in the output (sub-pixel).
        rows, columns = np.mgrid[0 : undistorted.shape[0], 0 : undistorted.shape[1]]
        weights = undistorted.astype(np.float64)
        centroid_x = (columns * weights).sum() / weights.sum()
        centroid_y = (rows * weights).sum() / weights.sum()
        expected_x = new_intrinsics.fx * normalised_x + new_intrinsics.cx
        expected_y = new_intrinsics.fy * normalised_y + new_intrinsics.cy
        assert abs(centroid_x - expected_x) < 0.4
        assert abs(centroid_y - expected_y) < 0.4
    # alpha = 0: the cropped region contains only valid source pixels, so a white image stays white.
    white_undistorted, white_intrinsics = undistort_image(np.full((300, 400), 255, dtype=np.uint8), intrinsics)
    assert white_undistorted.min() >= 254
    assert white_undistorted.shape == (white_intrinsics.height, white_intrinsics.width)
    # alpha = 0 zooms until no invalid pixel is left, so the frame is (nearly) kept whole: at most the
    # last row/column (1 px) is cropped by the ROI. alpha = 1 would keep every source pixel and crop the
    # large black border away instead, shrinking the output far below this bound.
    max_cropped_pixels = 1
    assert white_intrinsics.width >= intrinsics.width - max_cropped_pixels
    assert white_intrinsics.height >= intrinsics.height - max_cropped_pixels
    # The principal point stays inside the output image.
    assert 0 < white_intrinsics.cx < white_intrinsics.width and 0 < white_intrinsics.cy < white_intrinsics.height
    assert np.all(white_intrinsics.distortion == 0.0)


# --------------------------------------------------------------------------- T8


@pytest.mark.covers("T8")
def test_decode_resize_non_half_factor_on_nonuniform_image_per_axis_scale_and_content():
    """Factor 0.37 rounds the two axes differently; intrinsics use the realised per-axis scales."""
    width, height = 320, 240
    intrinsics = CameraIntrinsics(
        fx=250.0, fy=260.0, cx=160.5, cy=118.5, width=width, height=height,
        distortion=np.zeros(5), model="plumb_bob",
        projection_matrix=np.eye(3, 4),
    )
    # Ramps: R grows with the column, G with the row, B constant, so axes and channels are distinguishable.
    rgb = np.zeros((height, width, 3), dtype=np.float64)
    rgb[:, :, 0] = np.linspace(0, 255, width)[None, :]
    rgb[:, :, 1] = np.linspace(0, 255, height)[:, None]
    rgb[:, :, 2] = 128
    _, encoded = cv2.imencode(".jpg", cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 100])
    resize_factor = 0.37
    image, out = decode_and_resize_image(encoded.tobytes(), intrinsics, resize_factor)
    new_width, new_height = int(round(width * resize_factor)), int(round(height * resize_factor))  # 118 x 89
    scale_x, scale_y = new_width / width, new_height / height
    assert scale_x != scale_y  # the point of the factor: rounding makes the axes differ
    assert image.shape == (new_height, new_width, 3)
    assert (out.width, out.height) == (new_width, new_height)
    assert out.fx == pytest.approx(250.0 * scale_x) and out.fy == pytest.approx(260.0 * scale_y)
    assert out.cx == pytest.approx((160.5 + 0.5) * scale_x - 0.5)
    assert out.cy == pytest.approx((118.5 + 0.5) * scale_y - 0.5)
    assert out.projection_matrix is None and out.model == "plumb_bob"
    # Content: area-averaging a linear ramp over output pixel c gives the ramp value at source position
    # (c + 0.5) / scale - 0.5. The ramp spans 255 levels over (extent - 1) source pixels. JPEG q=100 error is
    # a few levels, so 6 levels is the tolerance.
    column_source = (np.arange(new_width) + 0.5) / scale_x - 0.5
    row_source = (np.arange(new_height) + 0.5) / scale_y - 0.5
    expected_red = column_source * 255.0 / (width - 1)
    expected_green = row_source * 255.0 / (height - 1)
    interior_columns, interior_rows = slice(3, new_width - 3), slice(3, new_height - 3)  # skip border blending
    assert np.abs(image[40, interior_columns, 0] - expected_red[interior_columns]).max() < 6
    assert np.abs(image[interior_rows, 50, 1] - expected_green[interior_rows]).max() < 6
    assert np.abs(image[:, :, 2].astype(int) - 128).max() < 6
    # A factor above 1 upsamples and still reports consistent sizes.
    upsampled, up_intrinsics = decode_and_resize_image(encoded.tobytes(), intrinsics, 1.5)
    assert upsampled.shape[:2] == (up_intrinsics.height, up_intrinsics.width) == (360, 480)
    assert up_intrinsics.fx == pytest.approx(250.0 * 1.5)


# --------------------------------------------------------------------------- T23


@pytest.mark.covers("T23")
def test_radar_cloud_to_array_follows_requested_column_order_and_empty_result():
    """Columns come out in the REQUESTED order (not the cloud's or sorted), cast to float32."""
    decoded = {
        "x": np.array([1.0, 2.0, 3.0]),
        "power": np.array([10.0, 20.0, 30.0]),
        "doppler": np.array([100.0, 200.0, 300.0]),
        "azimuth": np.array([0.1, 0.2, 0.3]),
    }
    requested = ["doppler", "x", "power"]  # neither the dict order nor alphabetical
    result = radar_cloud_to_array(decoded, requested)
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result[:, 0], decoded["doppler"])
    np.testing.assert_array_equal(result[:, 1], decoded["x"])
    np.testing.assert_array_equal(result[:, 2], decoded["power"])
    # Single field, and a field requested twice keeps both columns.
    assert radar_cloud_to_array(decoded, ["power"]).shape == (3, 1)
    assert radar_cloud_to_array(decoded, ["x", "x"]).shape == (3, 2)
    # Every row non-finite in a requested field -> empty [0, k], not an error.
    decoded_bad = {"x": np.array([np.nan, np.inf]), "power": np.array([1.0, 2.0])}
    empty = radar_cloud_to_array(decoded_bad, ["power", "x"])
    assert empty.shape == (0, 2) and empty.dtype == np.float32
    # -inf in a requested field also drops the row.
    assert radar_cloud_to_array({"x": np.array([-np.inf, 5.0])}, ["x"]).tolist() == [[5.0]]
