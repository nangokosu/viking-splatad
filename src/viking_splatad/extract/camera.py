"""Camera intrinsics, distortion, undistortion, image decode and shutter timing."""

from dataclasses import dataclass, replace
from typing import Any, Optional, Sequence, Tuple

import cv2
import numpy as np

from viking_splatad.registry import transform

# ROS distortion model names handled as OpenCV radial-tangential (k1,k2,p1,p2,k3).
SUPPORTED_DISTORTION_MODELS = frozenset({"plumb_bob", "radtan"})
# Valid timestamp conventions for rolling-shutter handling (see T10).
SUPPORTED_TIMESTAMP_CONVENTIONS = frozenset({"start_of_exposure", "center", "end_of_readout"})


@dataclass(frozen=True, eq=False)
class CameraIntrinsics:
    """Pinhole intrinsics plus distortion.

    Attributes:
        fx, fy: Focal lengths in pixels.
        cx, cy: Principal point in pixels (pixel centres at integer coordinates).
        width, height: Image size in pixels (Python ints).
        distortion: OpenCV-order vector (k1, k2, p1, p2, k3[, ...]); empty/zeros = none.
        model: Distortion model name.
        projection_matrix: Optional 3x4 ROS ``P`` matrix.
    """

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    distortion: np.ndarray
    model: str
    projection_matrix: Optional[np.ndarray] = None

    def camera_matrix(self) -> np.ndarray:
        """Build the pinhole matrix K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]].

        Returns:
            3x3 float64 array.
        """
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def __eq__(self, other: object) -> bool:
        """Value equality (numpy fields need element-wise comparison).

        Args:
            other: Object to compare against.

        Returns:
            True if scalars and distortion match (projection_matrix is ignored);
            NotImplemented for a different type.
        """
        if not isinstance(other, CameraIntrinsics):
            return NotImplemented
        scalars_equal = (
            (self.fx, self.fy, self.cx, self.cy, self.width, self.height, self.model)
            == (other.fx, other.fy, other.cx, other.cy, other.width, other.height, other.model)
        )
        return scalars_equal and np.array_equal(self.distortion, other.distortion)

    __hash__ = None  # mutable numpy members; not hashable


def _read_field(camera_info: Any, field_name: str) -> Any:
    """Fetch a field from a mapping or attribute object (case-insensitive for ROS ``K``/``k``).

    Args:
        camera_info: dict or object with CameraInfo fields.
        field_name: Lowercase field name.

    Returns:
        Field value, or None if absent.
    """
    if isinstance(camera_info, dict):
        # ROS message dumps use upper-case K/D/P; accept either spelling.
        for key in (field_name, field_name.upper()):
            if key in camera_info:
                return camera_info[key]
        return None
    for attribute_name in (field_name, field_name.upper()):
        if hasattr(camera_info, attribute_name):
            return getattr(camera_info, attribute_name)
    return None


@transform("T7a", "Parse ROS CameraInfo (K, D, P, model, size) into CameraIntrinsics")
def parse_camera_info(camera_info: Any) -> CameraIntrinsics:
    """Parse a ROS sensor_msgs/CameraInfo-like dict or object.

    K is row-major [fx 0 cx; 0 fy cy; 0 0 1], so fx=K[0], cx=K[2], fy=K[4], cy=K[5].

    Args:
        camera_info: Mapping/object with k (9), d, distortion_model, width, height, optional p (12).

    Returns:
        CameraIntrinsics.

    Raises:
        ValueError: Unsupported distortion model (e.g. fisheye) or malformed K.
    """
    # Reject models we cannot represent before reading anything else.
    model_name = _read_field(camera_info, "distortion_model")
    if model_name not in SUPPORTED_DISTORTION_MODELS:
        raise ValueError(
            f"Unsupported distortion model {model_name!r}; only "
            f"{sorted(SUPPORTED_DISTORTION_MODELS)} are supported (fisheye is not)."
        )
    # K must be a flattened 3x3.
    camera_matrix_values = np.asarray(_read_field(camera_info, "k"), dtype=np.float64).reshape(-1)
    expected_k_length = 9  # 3x3 row-major, fixed by the ROS CameraInfo definition
    if camera_matrix_values.size != expected_k_length:
        raise ValueError(f"K must have {expected_k_length} entries, got {camera_matrix_values.size}")
    camera_matrix = camera_matrix_values.reshape(3, 3)
    # Optional 3x4 projection matrix.
    projection_values = _read_field(camera_info, "p")
    projection_matrix = (
        None if projection_values is None
        else np.asarray(projection_values, dtype=np.float64).reshape(3, 4)
    )
    return CameraIntrinsics(
        fx=float(camera_matrix[0, 0]),
        fy=float(camera_matrix[1, 1]),
        cx=float(camera_matrix[0, 2]),
        cy=float(camera_matrix[1, 2]),
        width=int(_read_field(camera_info, "width")),
        height=int(_read_field(camera_info, "height")),
        distortion=np.asarray(_read_field(camera_info, "d"), dtype=np.float64).reshape(-1),
        model=str(model_name),
        projection_matrix=projection_matrix,
    )


@transform("T7b", "OpenCV (k1,k2,p1,p2,k3) -> neurad-studio (k1,k2,k3,k4=0,p1,p2) distortion order")
def opencv_to_neurad_distortion(opencv_distortion: Sequence[float]) -> np.ndarray:
    """Reorder OpenCV distortion into neurad-studio order with k4 = 0.

    SplatAD asserts distortion_params[3] == 0, so the k4 slot is always zero.

    Args:
        opencv_distortion: Exactly (k1, k2, p1, p2, k3).

    Returns:
        float64 array (k1, k2, k3, k4=0, p1, p2).
    """
    opencv_values = np.asarray(opencv_distortion, dtype=np.float64).reshape(-1)
    expected_length = 5  # OpenCV plumb_bob has exactly five coefficients
    if opencv_values.size != expected_length:
        raise ValueError(f"Expected {expected_length} OpenCV coefficients, got {opencv_values.size}")
    radial_k1, radial_k2, tangential_p1, tangential_p2, radial_k3 = opencv_values
    # k4 slot is fixed at 0 (justified above).
    return np.array(
        [radial_k1, radial_k2, radial_k3, 0.0, tangential_p1, tangential_p2], dtype=np.float64
    )


@transform("T7c", "Undistort image (alpha=0 optimal K, crop to valid ROI) and emit new zero-distortion intrinsics")
def undistort_image(image: np.ndarray, intrinsics: CameraIntrinsics) -> Tuple[np.ndarray, CameraIntrinsics]:
    """Undistort an image and crop to the all-valid region.

    alpha=0 makes getOptimalNewCameraMatrix keep only valid pixels; the returned ROI
    (x, y, w, h) is cropped, which shifts the principal point by (-x, -y).

    Args:
        image: HxW or HxWxC array of size intrinsics.height x intrinsics.width.
        intrinsics: Source intrinsics with OpenCV-order distortion.

    Returns:
        (undistorted cropped image, new CameraIntrinsics with zero distortion).
    """
    image_size = (intrinsics.width, intrinsics.height)  # OpenCV expects (w, h)
    if image.shape[:2] != (intrinsics.height, intrinsics.width):
        raise ValueError(f"Image shape {image.shape[:2]} does not match intrinsics {image_size}")
    camera_matrix = intrinsics.camera_matrix()
    # Alpha 0 -> zoom so no invalid (black) border pixels remain.
    new_camera_matrix, valid_roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, intrinsics.distortion, image_size, 0, image_size
    )
    roi_x, roi_y, roi_width, roi_height = (int(value) for value in valid_roi)
    if roi_width <= 0 or roi_height <= 0:
        raise ValueError("Empty valid ROI after undistortion; distortion too strong")
    # Build the remap with the full-size output, then crop.
    map_x, map_y = cv2.initUndistortRectifyMap(
        camera_matrix, intrinsics.distortion, None, new_camera_matrix, image_size, cv2.CV_32FC1
    )
    undistorted = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
    cropped = undistorted[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]
    new_intrinsics = CameraIntrinsics(
        fx=float(new_camera_matrix[0, 0]),
        fy=float(new_camera_matrix[1, 1]),
        # Cropping moves the image origin, so the principal point shifts by the ROI offset.
        cx=float(new_camera_matrix[0, 2]) - roi_x,
        cy=float(new_camera_matrix[1, 2]) - roi_y,
        width=roi_width,
        height=roi_height,
        distortion=np.zeros_like(intrinsics.distortion),
        model=intrinsics.model,
    )
    return np.ascontiguousarray(cropped), new_intrinsics


@transform("T8", "Decode JPEG to uint8 RGB, optional resize with consistent intrinsics scaling")
def decode_and_resize_image(
    jpeg_bytes: bytes, intrinsics: CameraIntrinsics, resize_factor: float = 1.0
) -> Tuple[np.ndarray, CameraIntrinsics]:
    """Decode a CompressedImage JPEG to RGB and optionally resize.

    Pixel-centre convention: pixel i covers [i-0.5, i+0.5], so continuous position
    u_edge = u + 0.5. Scaling by f in edge coordinates and converting back gives
    u' = (u + 0.5) * f - 0.5. Focal lengths are pure scales (fx' = fx * f).

    Args:
        jpeg_bytes: Encoded JPEG.
        intrinsics: Intrinsics whose width/height must match the decoded image.
        resize_factor: Scale factor > 0 (1.0 = no resize).

    Returns:
        (uint8 RGB array HxWx3, intrinsics matching the returned array).

    Raises:
        ValueError: Undecodable data, size mismatch, or non-positive factor.
    """
    if resize_factor <= 0:
        raise ValueError("resize_factor must be positive")
    # Decode JPEG bytes to a BGR array (None on corrupt data).
    bgr_image = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr_image is None:
        raise ValueError("Could not decode JPEG bytes")
    # OpenCV decodes as BGR; SplatAD wants RGB.
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    decoded_height, decoded_width = rgb_image.shape[:2]
    if (decoded_width, decoded_height) != (intrinsics.width, intrinsics.height):
        raise ValueError(
            f"Decoded size {(decoded_width, decoded_height)} does not match declared "
            f"{(intrinsics.width, intrinsics.height)}"
        )
    if resize_factor == 1.0:
        return rgb_image, intrinsics
    # Output size is rounded to whole pixels.
    new_width = int(round(decoded_width * resize_factor))
    new_height = int(round(decoded_height * resize_factor))
    # INTER_AREA is the anti-aliased choice for downscaling.
    resized = cv2.resize(rgb_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    # Rounding makes the true per-axis scale differ slightly from resize_factor; use it.
    scale_x = new_width / decoded_width
    scale_y = new_height / decoded_height
    scaled_intrinsics = replace(
        intrinsics,
        fx=intrinsics.fx * scale_x,
        fy=intrinsics.fy * scale_y,
        cx=(intrinsics.cx + 0.5) * scale_x - 0.5,
        cy=(intrinsics.cy + 0.5) * scale_y - 0.5,
        width=new_width,
        height=new_height,
        projection_matrix=None,  # stale after scaling; drop rather than keep inconsistent
    )
    return resized, scaled_intrinsics


@transform("T10", "Rolling-shutter parameters (readout, time_to_center_pixel) from timestamp convention")
def rolling_shutter_parameters(timestamp_convention: str, readout_seconds: float) -> Tuple[float, float]:
    """Compute SplatAD rolling-shutter values.

    neurad-studio: camera time + time_to_center_pixel = time of the centre row.
    With rows read out uniformly over ``readout_seconds``, the centre row is read
    readout/2 after the first row:
      start_of_exposure: stamp = first row    -> +readout/2
      center:            stamp = centre row   -> 0
      end_of_readout:    stamp = last row     -> -readout/2
    (PandaSet uses readout 0.03 and -0.01.) Readout 0 disables rolling shutter.

    Args:
        timestamp_convention: One of SUPPORTED_TIMESTAMP_CONVENTIONS.
        readout_seconds: Sensor readout duration in seconds (>= 0).

    Returns:
        (rolling_shutter_time, time_to_center_pixel) in seconds.

    Raises:
        ValueError: Unknown convention or negative readout.
    """
    if timestamp_convention not in SUPPORTED_TIMESTAMP_CONVENTIONS:
        raise ValueError(
            f"Unknown timestamp convention {timestamp_convention!r}; "
            f"expected one of {sorted(SUPPORTED_TIMESTAMP_CONVENTIONS)}"
        )
    if readout_seconds < 0:
        raise ValueError("readout_seconds must be >= 0")
    if readout_seconds == 0:
        return 0.0, 0.0  # rolling shutter disabled
    half_readout = readout_seconds / 2.0
    # Sign of the offset per convention (see docstring).
    offset_by_convention = {
        "start_of_exposure": half_readout,
        "center": 0.0,
        "end_of_readout": -half_readout,
    }
    return readout_seconds, offset_by_convention[timestamp_convention]
