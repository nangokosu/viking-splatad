"""Unit tests for camera intrinsics parsing, distortion, undistortion, decode, shutter."""

import cv2
import numpy as np
import pytest

from viking_splatad.extract.camera import (
    CameraIntrinsics,
    decode_and_resize_image,
    opencv_to_neurad_distortion,
    parse_camera_info,
    rolling_shutter_parameters,
    undistort_image,
)

# Synthetic pinhole used across tests (arbitrary, non-square pixels on purpose so
# fx/fy and cx/cy mix-ups are detected).
SYNTHETIC_FX = 250.0
SYNTHETIC_FY = 260.0
SYNTHETIC_CX = 160.5
SYNTHETIC_CY = 118.5
SYNTHETIC_WIDTH = 320
SYNTHETIC_HEIGHT = 240
# OpenCV order (k1, k2, p1, p2, k3) - also the ROS plumb_bob D order.
SYNTHETIC_DISTORTION_OPENCV = [-0.2, 0.05, 0.001, -0.002, 0.0]


def make_camera_info_dict() -> dict:
    """Build a ROS CameraInfo-like dict from the synthetic pinhole constants.

    Returns:
        Dict with keys k (row-major 9), d, p (row-major 12), distortion_model, width, height.
    """
    return {
        "k": [SYNTHETIC_FX, 0.0, SYNTHETIC_CX, 0.0, SYNTHETIC_FY, SYNTHETIC_CY, 0.0, 0.0, 1.0],
        "d": list(SYNTHETIC_DISTORTION_OPENCV),
        "p": [SYNTHETIC_FX, 0.0, SYNTHETIC_CX, 0.0,
              0.0, SYNTHETIC_FY, SYNTHETIC_CY, 0.0,
              0.0, 0.0, 1.0, 0.0],
        "distortion_model": "plumb_bob",
        "width": SYNTHETIC_WIDTH,
        "height": SYNTHETIC_HEIGHT,
    }


def make_synthetic_intrinsics() -> CameraIntrinsics:
    """Build CameraIntrinsics straight from the synthetic constants."""
    return parse_camera_info(make_camera_info_dict())


@pytest.mark.covers("T7a")
def test_parse_camera_info_known_values():
    """K row-major indices 0,4,2,5 map to fx,fy,cx,cy; D and sizes carried over."""
    intrinsics = parse_camera_info(make_camera_info_dict())
    assert (intrinsics.fx, intrinsics.fy) == (SYNTHETIC_FX, SYNTHETIC_FY)
    assert (intrinsics.cx, intrinsics.cy) == (SYNTHETIC_CX, SYNTHETIC_CY)
    assert (intrinsics.width, intrinsics.height) == (SYNTHETIC_WIDTH, SYNTHETIC_HEIGHT)
    assert isinstance(intrinsics.width, int) and isinstance(intrinsics.height, int)
    assert np.allclose(intrinsics.distortion, SYNTHETIC_DISTORTION_OPENCV)
    assert intrinsics.model == "plumb_bob"
    # Optional P matrix parsed as 3x4.
    assert intrinsics.projection_matrix.shape == (3, 4)
    assert intrinsics.projection_matrix[1, 2] == SYNTHETIC_CY


@pytest.mark.covers("T7a")
def test_parse_camera_info_without_projection_matrix():
    """P is optional."""
    camera_info = make_camera_info_dict()
    del camera_info["p"]
    assert parse_camera_info(camera_info).projection_matrix is None


@pytest.mark.covers("T7a")
def test_parse_camera_info_rejects_fisheye_and_unknown():
    """Anything except plumb_bob/radtan raises a clear error."""
    for unsupported_model in ("equidistant", "fisheye", "rational_polynomial"):
        camera_info = make_camera_info_dict()
        camera_info["distortion_model"] = unsupported_model
        with pytest.raises(ValueError, match="Unsupported distortion model"):
            parse_camera_info(camera_info)
    camera_info = make_camera_info_dict()
    camera_info["distortion_model"] = "radtan"
    assert parse_camera_info(camera_info).model == "radtan"


@pytest.mark.covers("T7b")
def test_opencv_to_neurad_distortion_permutation_and_k4_zero():
    """(k1,k2,p1,p2,k3) -> (k1,k2,k3,k4=0,p1,p2)."""
    k1, k2, p1, p2, k3 = 0.11, 0.22, 0.33, 0.44, 0.55
    neurad_params = opencv_to_neurad_distortion([k1, k2, p1, p2, k3])
    assert neurad_params.shape == (6,)
    assert neurad_params.tolist() == [k1, k2, k3, 0.0, p1, p2]
    # SplatAD asserts distortion_params[3] == 0.
    assert neurad_params[3] == 0.0


def render_polyline_image(distorted_point_lists: list) -> np.ndarray:
    """Draw each point list as a thick white polyline on a black grayscale image.

    Args:
        distorted_point_lists: List of (N,2) float pixel arrays.

    Returns:
        uint8 image (SYNTHETIC_HEIGHT, SYNTHETIC_WIDTH).
    """
    canvas = np.zeros((SYNTHETIC_HEIGHT, SYNTHETIC_WIDTH), dtype=np.uint8)
    for point_list in distorted_point_lists:
        cv2.polylines(canvas, [np.round(point_list).astype(np.int32).reshape(-1, 1, 2)],
                      False, 255, thickness=3)
    return canvas


def project_line(normalized_y: float, distortion: list, intrinsics: CameraIntrinsics) -> np.ndarray:
    """Project a dense straight 3D line (plane z=1, constant y) with distortion.

    Returns:
        (N,2) distorted pixel coordinates.
    """
    sample_count = 4000
    # Wide x range so the line spans the full image width after projection.
    x_values = np.linspace(-1.2, 1.2, sample_count)
    points_3d = np.stack([x_values, np.full(sample_count, normalized_y), np.ones(sample_count)], axis=1)
    camera_matrix = np.array([[intrinsics.fx, 0, intrinsics.cx], [0, intrinsics.fy, intrinsics.cy], [0, 0, 1.0]])
    # OpenCV requires distortion to be float32/float64, not int.
    pixels, _ = cv2.projectPoints(points_3d, np.zeros(3), np.zeros(3), camera_matrix, np.array(distortion, dtype=np.float32))
    return pixels.reshape(-1, 2)


@pytest.mark.covers("T7c")
def test_undistort_makes_distorted_lines_straight():
    """Barrel-distorted straight lines become straight; residual bounded by baseline + 1px."""
    intrinsics = make_synthetic_intrinsics()
    line_heights = [-0.3, -0.1, 0.1, 0.3]

    def max_line_residual(distorted_image: np.ndarray, distortion_free: bool) -> float:
        """Worst deviation (px) of per-column centroids from a best-fit line."""
        result_image, result_intrinsics = (
            undistort_image(distorted_image, intrinsics)
            if not distortion_free else (distorted_image, intrinsics)
        )
        worst = 0.0
        for normalized_y in line_heights:
            expected_row = result_intrinsics.fy * normalized_y + result_intrinsics.cy
            column_indices, centroids = [], []
            for column_index in range(result_image.shape[1]):
                window_top = int(max(expected_row - 10, 0))
                window_bottom = int(min(expected_row + 10, result_image.shape[0]))
                column_values = result_image[window_top:window_bottom, column_index].astype(float)
                if column_values.sum() < 255 * 2:  # need >= ~2 lit pixels for a stable centroid
                    continue
                rows = np.arange(window_top, window_bottom)
                column_indices.append(column_index)
                centroids.append((rows * column_values).sum() / column_values.sum())
            slope, offset = np.polyfit(column_indices, centroids, 1)
            residuals = np.abs(np.array(centroids) - (slope * np.array(column_indices) + offset))
            worst = max(worst, residuals.max())
        return worst

    # Baseline: same lines drawn WITHOUT distortion, run through the same measurement.
    # This is the pure rasterisation/centroid noise floor of the pipeline.
    ideal_image = render_polyline_image(
        [project_line(height, [0, 0, 0, 0, 0], intrinsics) for height in line_heights]
    )
    baseline_residual = max_line_residual(ideal_image, distortion_free=True)
    distorted_image = render_polyline_image(
        [project_line(height, SYNTHETIC_DISTORTION_OPENCV, intrinsics) for height in line_heights]
    )
    distorted_residual = max_line_residual(distorted_image, distortion_free=False)
    # Margin 1 px = 0.5 px integer rasterisation of the polyline + 0.5 px bilinear remap.
    assert distorted_residual <= baseline_residual + 1.0


@pytest.mark.covers("T7c")
def test_undistort_new_intrinsics_reproject_known_point():
    """A dot at the distorted projection of a 3D point lands at Knew*(X/Z,Y/Z)."""
    intrinsics = make_synthetic_intrinsics()
    point_3d = np.array([[0.1, -0.05, 1.0]])
    camera_matrix = np.array([[SYNTHETIC_FX, 0, SYNTHETIC_CX], [0, SYNTHETIC_FY, SYNTHETIC_CY], [0, 0, 1.0]])
    distorted_pixel, _ = cv2.projectPoints(point_3d, np.zeros(3), np.zeros(3), camera_matrix,
                                           np.array(SYNTHETIC_DISTORTION_OPENCV, dtype=np.float32))
    canvas = np.zeros((SYNTHETIC_HEIGHT, SYNTHETIC_WIDTH), dtype=np.uint8)
    cv2.circle(canvas, tuple(np.round(distorted_pixel.reshape(2)).astype(int)), 3, 255, -1)

    undistorted_image, new_intrinsics = undistort_image(canvas, intrinsics)
    # Output metadata contract.
    assert undistorted_image.shape[:2] == (new_intrinsics.height, new_intrinsics.width)
    assert isinstance(new_intrinsics.width, int) and isinstance(new_intrinsics.height, int)
    assert np.allclose(new_intrinsics.distortion, 0.0)
    # Analytic pixel: u = fx' X/Z + cx', v = fy' Y/Z + cy'.
    expected_u = new_intrinsics.fx * 0.1 + new_intrinsics.cx
    expected_v = new_intrinsics.fy * -0.05 + new_intrinsics.cy
    rows, columns = np.nonzero(undistorted_image > 127)
    assert abs(columns.mean() - expected_u) < 1.5
    assert abs(rows.mean() - expected_v) < 1.5


def encode_solid_jpeg(rgb_color: tuple, width: int, height: int) -> bytes:
    """Encode a single-colour RGB image as JPEG bytes."""
    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
    rgb_image[:, :] = rgb_color
    _, encoded = cv2.imencode(".jpg", cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
    return encoded.tobytes()


@pytest.mark.covers("T8")
def test_decode_without_resize_is_rgb_and_shape_consistent():
    """Channel order is RGB, dtype uint8, declared size equals array shape."""
    intrinsics = make_synthetic_intrinsics()
    jpeg_bytes = encode_solid_jpeg((255, 0, 0), SYNTHETIC_WIDTH, SYNTHETIC_HEIGHT)
    image, out_intrinsics = decode_and_resize_image(jpeg_bytes, intrinsics)
    assert image.dtype == np.uint8
    assert image.shape == (SYNTHETIC_HEIGHT, SYNTHETIC_WIDTH, 3)
    assert image[0, 0, 0] > 200 and image[0, 0, 2] < 50  # red stays in channel 0
    assert out_intrinsics == intrinsics


@pytest.mark.covers("T8")
def test_decode_resize_scales_intrinsics_with_half_pixel_convention():
    """fx,fy scale by f; cx' = (cx+0.5)*f - 0.5."""
    intrinsics = make_synthetic_intrinsics()
    resize_factor = 0.5
    jpeg_bytes = encode_solid_jpeg((0, 255, 0), SYNTHETIC_WIDTH, SYNTHETIC_HEIGHT)
    image, out_intrinsics = decode_and_resize_image(jpeg_bytes, intrinsics, resize_factor)
    expected_width = int(SYNTHETIC_WIDTH * resize_factor)
    expected_height = int(SYNTHETIC_HEIGHT * resize_factor)
    assert image.shape == (expected_height, expected_width, 3)
    assert (out_intrinsics.width, out_intrinsics.height) == (expected_width, expected_height)
    assert type(out_intrinsics.width) is int and type(out_intrinsics.height) is int
    assert out_intrinsics.fx == pytest.approx(SYNTHETIC_FX * resize_factor)
    assert out_intrinsics.fy == pytest.approx(SYNTHETIC_FY * resize_factor)
    assert out_intrinsics.cx == pytest.approx((SYNTHETIC_CX + 0.5) * resize_factor - 0.5)
    assert out_intrinsics.cy == pytest.approx((SYNTHETIC_CY + 0.5) * resize_factor - 0.5)


@pytest.mark.covers("T8")
def test_decode_rejects_declared_size_mismatch():
    """Declared intrinsics size must equal decoded array size."""
    intrinsics = make_synthetic_intrinsics()
    wrong_size_jpeg = encode_solid_jpeg((1, 2, 3), SYNTHETIC_WIDTH // 2, SYNTHETIC_HEIGHT)
    with pytest.raises(ValueError, match="does not match"):
        decode_and_resize_image(wrong_size_jpeg, intrinsics)


@pytest.mark.covers("T10")
def test_rolling_shutter_parameters_three_conventions():
    """center -> 0; start -> +readout/2; end -> -readout/2; readout returned as shutter time."""
    readout_seconds = 0.03
    assert rolling_shutter_parameters("center", readout_seconds) == (readout_seconds, 0.0)
    assert rolling_shutter_parameters("start_of_exposure", readout_seconds) == (
        readout_seconds, readout_seconds / 2)
    assert rolling_shutter_parameters("end_of_readout", readout_seconds) == (
        readout_seconds, -readout_seconds / 2)


@pytest.mark.covers("T10")
def test_rolling_shutter_disabled_and_invalid():
    """Readout 0 disables; unknown convention or negative readout raises."""
    for convention in ("center", "start_of_exposure", "end_of_readout"):
        assert rolling_shutter_parameters(convention, 0.0) == (0.0, 0.0)
    with pytest.raises(ValueError):
        rolling_shutter_parameters("middle", 0.01)
    with pytest.raises(ValueError):
        rolling_shutter_parameters("center", -0.01)
