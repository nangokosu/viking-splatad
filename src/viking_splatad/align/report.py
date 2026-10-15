"""Lidar/camera alignment report computed from an extracted intermediate dataset.

Memory model: only ``manifest.json``, ``poses.npz`` and the individual frame files that the
report needs are read (a handful of sweeps and images), never the whole dataset.

Conventions: timestamps float64 seconds; poses 4x4 float64 ``T_<target>_<source>``; camera
poses are OpenCV-axes ``T_world_camera``; camera frames are already undistorted (no
distortion model is applied); lidar sweeps are float32 [n,6] = x, y, z, intensity, t, channel
in the lidar sensor frame.
"""

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from viking_splatad.align.spatial_check import edge_alignment_score, project_lidar_to_image
from viking_splatad.align.time_sync import speed_signal_from_poses
from viking_splatad.dataset.writer import FRAME_FILE_PATTERN, POSE_KEY_SEPARATOR

# Number of evenly spaced sweeps examined when the caller does not choose: 5 gives a spread
# over the recording (start, quarter points, end) at a cost of 5 image loads.
DEFAULT_FRAME_COUNT = 5
# Longest overlay side in pixels: 1024 keeps a ~3 MP frame readable while the JPEG stays ~100 kB.
OVERLAY_MAX_SIDE_PIXELS = 1024
# JPEG quality of overlays (visual diagnostic only, size matters more than fidelity).
OVERLAY_JPEG_QUALITY = 90
# Overlay marker radius in (downscaled) pixels: the smallest disc that stays visible.
OVERLAY_POINT_RADIUS_PIXELS = 1
# Depth colour scale tops out at this percentile so a few far returns do not wash out the map.
OVERLAY_DEPTH_PERCENTILE = 98.0
# Yaw sweep covers +-8 camera pixel angles in 1-pixel steps: at the real fx (~1477 px) that is
# +-0.31 deg, i.e. about +-2 lidar beam spacings (~4.5 px each), wide enough to see a peak.
YAW_SWEEP_MAX_PIXELS = 8
# Time sweep: shifts of k * 0.25 camera periods, k = -2..2 (+-0.5 frame). Beyond +-0.5 frame the
# nearest camera frame would change, so larger shifts no longer test the same image.
TIME_SWEEP_STEP_FRAMES = 0.25
TIME_SWEEP_STEPS_EACH_SIDE = 2
# A surface seen at incidence angle up to 80 deg (from its normal) is treated as continuous: two
# neighbouring beams separated by dtheta then differ in range by <= r*dtheta*tan(80 deg), so
# larger relative jumps are real depth discontinuities (occlusion edges), not grazing planes.
GRAZING_INCIDENCE_LIMIT_DEGREES = 80.0
# Speed percentiles reported for the lidar trajectory.
SPEED_PERCENTILES = (50, 90, 99)
# Time sweep is extended to +-2 camera periods when its maximum sits on the initial boundary:
# two periods span the nearest-frame ambiguity (+-0.5) with margin yet keep the pose interpolation
# between neighbouring stamps.
EXTENDED_TIME_SWEEP_PERIODS = 2.0
# Bootstrap replicates for the score noise floor: 20 gives a standard-error estimate with ~16 %
# relative uncertainty (1/sqrt(2(B-1))), enough for a 3-sigma gate at negligible cost.
BOOTSTRAP_REPLICATES = 20
# Fixed seed so a report is reproducible.
BOOTSTRAP_SEED = 0
# A sweep is informative only if its peak prominence exceeds 3 bootstrap standard errors
# (the conventional 3-sigma gate, ~0.3 % false-alarm for Gaussian noise).
NOISE_SIGMAS = 3.0
# Required wording for sweeps that cannot judge the alignment.
UNINFORMATIVE_TEXT = "metric uninformative on this scene; rely on overlays"
# Index of the channel column in a [n,6] sweep.
CHANNEL_COLUMN = 5
# Index of the per-point time column.
TIME_COLUMN = 4
# Number of columns in a sweep that carries channel ids.
SWEEP_COLUMNS_WITH_CHANNEL = 6


def select_evenly_spaced_indices(total_count: int, requested_count: int) -> np.ndarray:
    """Pick ``requested_count`` indices evenly spread over ``range(total_count)``.

    Indices are round(linspace(0, total-1, k)); duplicates (only possible when k > total)
    are removed, so the result never exceeds ``total_count`` entries.

    Args:
        total_count: Number of available frames.
        requested_count: Desired number of picks.

    Returns:
        Strictly increasing int64 array.
    """
    # Evenly spaced real positions, rounded to the nearest frame.
    positions = np.linspace(0, total_count - 1, num=min(requested_count, total_count))
    # unique() sorts and removes duplicates produced by rounding.
    return np.unique(np.rint(positions).astype(np.int64))


def compute_stamp_statistics(times: np.ndarray) -> Dict[str, Any]:
    """Spacing statistics and strict-monotonicity check of a timestamp array.

    Args:
        times: [N] seconds.

    Returns:
        Dict: count, spacing_median/min/max (s) of consecutive differences,
        strictly_increasing (bool; True iff every difference is > 0).
    """
    # Consecutive differences dt_k = t_{k+1} - t_k.
    spacings = np.diff(times)
    return {
        "count": int(times.size),
        "spacing_median": float(np.median(spacings)),
        "spacing_min": float(spacings.min()),
        "spacing_max": float(spacings.max()),
        "strictly_increasing": bool(np.all(spacings > 0.0)),
    }


def compute_nearest_delta_statistics(lidar_times: np.ndarray, camera_times: np.ndarray) -> Dict[str, float]:
    """Distribution of |t_lidar - t_nearest camera| over all lidar frames.

    Uses a binary search on the sorted camera stamps (O(L log C)): the nearest stamp is
    one of the two neighbours of the insertion point.

    Args:
        lidar_times: [L] seconds.
        camera_times: [C] seconds, ascending.

    Returns:
        Dict with median/max/min absolute delta in seconds.
    """
    deltas = np.abs(camera_times[nearest_camera_indices(lidar_times, camera_times)] - lidar_times)
    return {
        "delta_median": float(np.median(deltas)),
        "delta_min": float(deltas.min()),
        "delta_max": float(deltas.max()),
    }


def nearest_camera_indices(query_times: np.ndarray, camera_times: np.ndarray) -> np.ndarray:
    """Index of the camera stamp closest to each query time.

    Args:
        query_times: [M] seconds.
        camera_times: [C] seconds, ascending.

    Returns:
        [M] int64 indices into ``camera_times``.
    """
    # Insertion point; its left and right neighbours are the only nearest candidates.
    right_index = np.clip(np.searchsorted(camera_times, query_times), 1, camera_times.size - 1)
    left_index = right_index - 1
    left_is_closer = np.abs(query_times - camera_times[left_index]) <= np.abs(camera_times[right_index] - query_times)
    return np.where(left_is_closer, left_index, right_index).astype(np.int64)


def compute_motion_statistics(times: np.ndarray, poses_world_sensor: np.ndarray) -> Dict[str, float]:
    """Path length and speed percentiles of a pose trajectory.

    Path length = sum_k ||p_{k+1} - p_k||; speed_k = ||p_{k+1} - p_k|| / dt_k
    (see ``speed_signal_from_poses``).

    Args:
        times: [N] seconds, strictly increasing.
        poses_world_sensor: [N,4,4] poses.

    Returns:
        Dict: path_length [m], speed_p<q> [m/s] for q in SPEED_PERCENTILES, speed_max.
    """
    # Total distance travelled = sum of the step lengths.
    step_lengths = np.linalg.norm(np.diff(poses_world_sensor[:, :3, 3], axis=0), axis=1)
    _, speeds = speed_signal_from_poses(times, poses_world_sensor)
    statistics = {"path_length": float(step_lengths.sum()), "speed_max": float(speeds.max())}
    for percentile in SPEED_PERCENTILES:
        statistics[f"speed_p{percentile}"] = float(np.percentile(speeds, percentile))
    return statistics


def compute_lidar_discontinuity(sweep: np.ndarray) -> Tuple[np.ndarray, float]:
    """Per-point depth-discontinuity indicator from range jumps along each channel.

    Points are grouped by channel (column 5; one group if absent) and ordered by the point
    time (column 4; array order if absent), which is azimuth order within a ring. For
    consecutive points a, b of one ring, with ranges r = ||xyz||:

        jump = |r_a - r_b|,   near = min(r_a, r_b),   beam gap dtheta = |wrapped azimuth diff|.

    The pair is a discontinuity when jump > near * dtheta_median * tan(80 deg). Rationale: on
    a continuous surface with incidence angle alpha, the range changes by ~ r*dtheta*tan(alpha)
    between neighbouring beams, so jumps beyond the alpha = 80 deg limit are occlusion edges.
    dtheta_median is the median azimuth gap of the sweep (measured, not assumed). Both
    endpoints of a discontinuous pair receive weight 1, all other points 0 (a binary weight
    keeps a few huge foliage-to-sky jumps from dominating the score).

    Args:
        sweep: [n,5|6] float sweep in the sensor frame.

    Returns:
        (weights [n] float64 in {0,1} aligned with the input rows, median azimuth gap [rad]).
    """
    point_count = sweep.shape[0]
    weights = np.zeros(point_count, dtype=np.float64)
    ranges = np.linalg.norm(sweep[:, :3].astype(np.float64), axis=1)
    azimuths = np.arctan2(sweep[:, 1], sweep[:, 0]).astype(np.float64)
    # Group key: channel when present; time as the within-ring order; row number breaks ties.
    channels = sweep[:, CHANNEL_COLUMN] if sweep.shape[1] >= SWEEP_COLUMNS_WITH_CHANNEL else np.zeros(point_count)
    point_times = sweep[:, TIME_COLUMN] if sweep.shape[1] > TIME_COLUMN else np.arange(point_count, dtype=np.float64)
    order = np.lexsort((np.arange(point_count), point_times, channels))
    ordered_channels, ordered_ranges, ordered_azimuths = channels[order], ranges[order], azimuths[order]
    # Adjacent pairs inside one channel only.
    same_ring = ordered_channels[1:] == ordered_channels[:-1]
    # Wrapped azimuth gap: angle of exp(i*(a_b - a_a)) lies in (-pi, pi].
    azimuth_gaps = np.abs(np.angle(np.exp(1j * np.diff(ordered_azimuths))))
    valid_pair = same_ring & (ordered_ranges[1:] > 0) & (ordered_ranges[:-1] > 0)
    if not valid_pair.any():
        return weights, 0.0
    median_gap = float(np.median(azimuth_gaps[valid_pair]))
    range_jumps = np.abs(np.diff(ordered_ranges))
    nearer_ranges = np.minimum(ordered_ranges[1:], ordered_ranges[:-1])
    limit_tangent = math.tan(math.radians(GRAZING_INCIDENCE_LIMIT_DEGREES))
    is_discontinuity = valid_pair & (range_jumps > nearer_ranges * median_gap * limit_tangent)
    # Flag both endpoints of every discontinuous pair, then undo the sort permutation.
    ordered_weights = np.zeros(point_count, dtype=np.float64)
    ordered_weights[:-1][is_discontinuity] = 1.0
    ordered_weights[1:][is_discontinuity] = 1.0
    weights[order] = ordered_weights
    return weights, median_gap


def project_sweep_to_pixels(
    points_lidar: np.ndarray,
    pose_world_lidar: np.ndarray,
    pose_world_camera: np.ndarray,
    camera_matrix: np.ndarray,
    image_width: int,
    image_height: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project lidar-frame points into an (undistorted) camera image through the world frame.

    p_world = T_world_lidar p_lidar;  p_cam = inv(T_world_camera) p_world;
    pixel = K p_cam / z. Composing the two transforms gives
    T_camera_lidar = inv(T_world_camera) @ T_world_lidar, which includes ego-motion between
    the lidar and camera stamps because each pose is taken at its own time.

    Args:
        points_lidar: [N,3] points in the lidar sensor frame.
        pose_world_lidar: 4x4 T_world_lidar at the sweep stamp.
        pose_world_camera: 4x4 T_world_camera (OpenCV axes) at the image stamp.
        camera_matrix: 3x3 K.
        image_width, image_height: Image size [px].

    Returns:
        (pixels [N,2] (NaN behind the camera), in_front [N] bool, inside_image [N] bool
        where inside_image implies in_front).
    """
    # Compose into one lidar -> camera transform.
    transform_camera_lidar = np.linalg.inv(pose_world_camera) @ pose_world_lidar
    pixels, inside_image = project_lidar_to_image(
        points_lidar.astype(np.float64), transform_camera_lidar, camera_matrix, image_width, image_height
    )
    # Projection yields NaN pixels exactly for Z <= 0, so finite pixels mean "in front".
    in_front = np.isfinite(pixels[:, 0])
    return pixels, in_front, inside_image


def interpolate_pose(times: np.ndarray, poses: np.ndarray, query_time: float) -> np.ndarray:
    """Pose at an arbitrary time: linear translation, slerp rotation, clamped at the ends.

    Args:
        times: [N] ascending seconds.
        poses: [N,4,4] poses.
        query_time: Time to evaluate [s].

    Returns:
        4x4 interpolated pose.
    """
    # Clamp so out-of-range shifts reuse the end pose instead of extrapolating.
    clamped_time = float(np.clip(query_time, times[0], times[-1]))
    right_index = int(np.clip(np.searchsorted(times, clamped_time, side="right"), 1, times.size - 1))
    left_index = right_index - 1
    # Blend fraction a = (t - t0) / (t1 - t0) in [0, 1].
    blend = (clamped_time - times[left_index]) / (times[right_index] - times[left_index])
    pair_rotations = Rotation.from_matrix(poses[[left_index, right_index], :3, :3])
    rotation = Slerp([0.0, 1.0], pair_rotations)([blend]).as_matrix()[0]
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = (1.0 - blend) * poses[left_index, :3, 3] + blend * poses[right_index, :3, 3]
    return pose


def rotate_pose_about_camera_y(pose_world_camera: np.ndarray, angle_radians: float) -> np.ndarray:
    """Perturb a camera pose by a yaw about its own y axis: T' = T @ Ry(angle).

    OpenCV camera y points down, so this is a heading (yaw) change of the extrinsic.
    Ry = [[c,0,s],[0,1,0],[-s,0,c]].

    Args:
        pose_world_camera: 4x4 T_world_camera.
        angle_radians: Yaw perturbation.

    Returns:
        Perturbed 4x4 pose.
    """
    cosine, sine = math.cos(angle_radians), math.sin(angle_radians)
    perturbation = np.eye(4)
    perturbation[:3, :3] = [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]]
    return pose_world_camera @ perturbation


def render_overlay(
    image_bgr: np.ndarray, pixels: np.ndarray, inside_image: np.ndarray, depths: np.ndarray
) -> np.ndarray:
    """Draw projected lidar points over the image, coloured by depth, downscaled for size.

    Depth d maps to colour via the JET map on d / d_ref clipped to [0, 1], where d_ref is the
    OVERLAY_DEPTH_PERCENTILE of the drawn depths (near = blue, far = red).

    Args:
        image_bgr: [H,W,3] uint8 image.
        pixels: [N,2] projected pixels.
        inside_image: [N] bool mask of points to draw.
        depths: [N] camera-frame depth of each point [m].

    Returns:
        Downscaled uint8 overlay image.
    """
    # Scale so the long side is at most OVERLAY_MAX_SIDE_PIXELS (never upscale).
    scale = min(1.0, OVERLAY_MAX_SIDE_PIXELS / max(image_bgr.shape[:2]))
    overlay = cv2.resize(image_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    drawn_depths = depths[inside_image]
    if drawn_depths.size == 0:
        return overlay
    reference_depth = max(float(np.percentile(drawn_depths, OVERLAY_DEPTH_PERCENTILE)), np.finfo(float).eps)
    # 8-bit colour indices from normalised depth, then the JET lookup.
    colour_indices = np.clip(drawn_depths / reference_depth, 0.0, 1.0) * np.iinfo(np.uint8).max
    colours = cv2.applyColorMap(colour_indices.astype(np.uint8)[:, None], cv2.COLORMAP_JET)[:, 0, :]
    for pixel, colour in zip(pixels[inside_image] * scale, colours):
        cv2.circle(overlay, (int(round(pixel[0])), int(round(pixel[1]))), OVERLAY_POINT_RADIUS_PIXELS,
                   tuple(int(channel) for channel in colour), thickness=-1)
    return overlay


def argmax_nearest_zero(perturbations: np.ndarray, scores: np.ndarray) -> int:
    """Index of the maximum score; exact (within 1e-12 relative) ties resolve toward zero.

    Flat score curves (e.g. a time shift on a static scene) then report 0 instead of an
    arbitrary first element.

    Args:
        perturbations: [K] perturbation values.
        scores: [K] scores.

    Returns:
        Index of the chosen maximum.
    """
    # Floating-point equality slack: a few ulps of the largest score.
    slack = 1e-12 * max(float(np.max(np.abs(scores))), 1.0)
    tied = np.flatnonzero(scores >= scores.max() - slack)
    return int(tied[np.argmin(np.abs(perturbations[tied]))])


def _list_frame_files(dataset_dir: Path, kind: str, sensor_name: str) -> Dict[int, Path]:
    """Map frame index -> file path for one sensor directory.

    Args:
        dataset_dir: Dataset root.
        kind: "camera" or "lidar".
        sensor_name: Sensor subdirectory.

    Returns:
        Dict index -> path (file names ``<idx>_<ts_ns>.<ext>``).
    """
    frame_files = {}
    for path in (dataset_dir / kind / sensor_name).iterdir():
        match = FRAME_FILE_PATTERN.match(path.name)
        if match:
            frame_files[int(match.group(1))] = path
    return frame_files


def _sensor_names_from_manifest(manifest: Dict[str, Any], camera_name: Optional[str], lidar_name: Optional[str]) -> Tuple[str, str]:
    """Resolve camera and lidar sensor names (explicit, else first of each kind in the manifest).

    Args:
        manifest: Parsed manifest.json.
        camera_name, lidar_name: Optional explicit names.

    Returns:
        (camera_name, lidar_name).

    Raises:
        ValueError: If the manifest has no sensor of a required kind.
    """
    names_by_kind: Dict[str, List[str]] = {"camera": [], "lidar": []}
    for sensor in manifest["sensors"]:
        names_by_kind.setdefault(sensor["kind"], []).append(sensor["name"])
    resolved_camera = camera_name or (names_by_kind["camera"][0] if names_by_kind["camera"] else None)
    resolved_lidar = lidar_name or (names_by_kind["lidar"][0] if names_by_kind["lidar"] else None)
    if resolved_camera is None or resolved_lidar is None:
        raise ValueError("manifest must list at least one camera and one lidar sensor")
    return resolved_camera, resolved_lidar


def _camera_matrix(intrinsics: Dict[str, Any]) -> np.ndarray:
    """3x3 pinhole K from a manifest camera entry.

    Args:
        intrinsics: Dict with fx, fy, cx, cy.

    Returns:
        3x3 float64 K.
    """
    return np.array(
        [[intrinsics["fx"], 0.0, intrinsics["cx"]], [0.0, intrinsics["fy"], intrinsics["cy"]], [0.0, 0.0, 1.0]]
    )


def _mean_edge_score(frame_inputs: Sequence[Dict[str, Any]], camera_pose_for_frame, camera_matrix: np.ndarray) -> float:
    """Mean edge-alignment score over frames for a given way of choosing each camera pose.

    Args:
        frame_inputs: Per-frame dicts (sweep points, lidar pose, gray image, weights, size).
        camera_pose_for_frame: Callable(frame_input) -> 4x4 camera pose to use.
        camera_matrix: 3x3 K.

    Returns:
        Mean of ``edge_alignment_score`` over the frames.
    """
    scores = []
    for frame_input in frame_inputs:
        height, width = frame_input["gray"].shape
        pixels, _, inside_image = project_sweep_to_pixels(
            frame_input["points_lidar"], frame_input["pose_world_lidar"],
            camera_pose_for_frame(frame_input), camera_matrix, width, height)
        scores.append(edge_alignment_score(frame_input["gray"], pixels, inside_image, frame_input["weights"]))
    return float(np.mean(scores))


def estimate_score_standard_error(
    frame_inputs: Sequence[Dict[str, Any]], camera_matrix: np.ndarray
) -> float:
    """Bootstrap standard error of the mean edge score at the stored (zero) perturbation.

    Each replicate resamples, with replacement, the projected points of every frame
    (indices drawn uniformly from the frame's points), recomputes the per-frame score and
    averages over frames. The standard deviation over BOOTSTRAP_REPLICATES replicates is the
    sampling noise of the score caused by which lidar points happen to exist; a sweep whose
    variation does not exceed it carries no evidence about alignment.

    Args:
        frame_inputs: Per-frame inputs (see ``_mean_edge_score``).
        camera_matrix: 3x3 K.

    Returns:
        Standard error (same units as the score).
    """
    random_generator = np.random.default_rng(BOOTSTRAP_SEED)
    # Project once; resampling only reindexes the projected arrays.
    projections = []
    for frame_input in frame_inputs:
        height, width = frame_input["gray"].shape
        pixels, _, inside_image = project_sweep_to_pixels(
            frame_input["points_lidar"], frame_input["pose_world_lidar"],
            frame_input["pose_world_camera"], camera_matrix, width, height)
        projections.append((pixels, inside_image))
    replicate_scores = []
    for replicate_index in range(BOOTSTRAP_REPLICATES):
        frame_scores = []
        for frame_input, (pixels, inside_image) in zip(frame_inputs, projections):
            resampled = random_generator.integers(0, pixels.shape[0], size=pixels.shape[0])
            frame_scores.append(edge_alignment_score(
                frame_input["gray"], pixels[resampled], inside_image[resampled], frame_input["weights"][resampled]))
        replicate_scores.append(np.mean(frame_scores))
    return float(np.std(replicate_scores, ddof=1))


def curve_informativeness(scores: np.ndarray, standard_error: float) -> Dict[str, Any]:
    """Contrast and prominence of a sweep curve against the bootstrap noise.

    contrast = (max - min) / mean (scale-free, reported); prominence = max - median (the
    peak height above the typical level; the median is robust to the peak itself). The curve is
    informative iff prominence > NOISE_SIGMAS * standard_error.

    Args:
        scores: [K] sweep scores.
        standard_error: Bootstrap standard error of the score.

    Returns:
        Dict: contrast, prominence, noise_floor (= NOISE_SIGMAS * SE), informative (bool).
    """
    prominence = float(scores.max() - np.median(scores))
    noise_floor = NOISE_SIGMAS * standard_error
    return {
        "contrast": float((scores.max() - scores.min()) / max(scores.mean(), np.finfo(float).eps)),
        "prominence": prominence,
        "noise_floor": float(noise_floor),
        "informative": bool(prominence > noise_floor),
    }


def _time_sweep(
    frame_inputs: Sequence[Dict[str, Any]], camera_matrix: np.ndarray,
    camera_times: np.ndarray, camera_poses: np.ndarray, steps_each_side: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Mean score for camera time shifts k * TIME_SWEEP_STEP_FRAMES, k in [-steps, steps].

    The camera pose is evaluated at t_camera + shift * camera_period; the image stays put.

    Args:
        frame_inputs: Per-frame inputs.
        camera_matrix: 3x3 K.
        camera_times, camera_poses: Camera trajectory samples.
        steps_each_side: Number of grid steps on each side of zero.

    Returns:
        (shifts in camera periods [K], scores [K]).
    """
    camera_period = float(np.median(np.diff(camera_times)))
    shifts = np.arange(-steps_each_side, steps_each_side + 1) * TIME_SWEEP_STEP_FRAMES
    scores = np.array([
        _mean_edge_score(frame_inputs, lambda frame_input, shift=shift: interpolate_pose(
            camera_times, camera_poses, frame_input["camera_time"] + shift * camera_period), camera_matrix)
        for shift in shifts])
    return shifts, scores


def run_sensitivity_sweep(
    frame_inputs: Sequence[Dict[str, Any]],
    camera_matrix: np.ndarray,
    camera_times: np.ndarray,
    camera_poses: np.ndarray,
) -> Dict[str, Any]:
    """Edge score versus camera extrinsic yaw and versus camera time shift, with noise tests.

    Yaw angles: m * atan(1/fx) for integer m in [-M, M], M = YAW_SWEEP_MAX_PIXELS, i.e.
    multiples of the angular size of one pixel at the principal point. Time shifts:
    k * TIME_SWEEP_STEP_FRAMES camera periods for k in [-K, K]. A maximum on the edge of the
    time grid is not a peak (the curve may keep rising), so the grid is then extended to
    +-EXTENDED_TIME_SWEEP_PERIODS periods and the extended argmax is reported; if that is
    still on the boundary the time sweep stays unbracketed (inconclusive). Every sweep is
    also tested for informativeness against the bootstrap noise floor.

    Args:
        frame_inputs: Per-frame inputs (see ``_mean_edge_score``), each with camera_pose/camera_time.
        camera_matrix: 3x3 K.
        camera_times: [C] camera stamps.
        camera_poses: [C,4,4] camera poses.

    Returns:
        Dict with the grids, scores, argmax values, informativeness and status per sweep
        (JSON-ready).
    """
    pixel_angle = math.atan(1.0 / camera_matrix[0, 0])
    yaw_angles = np.arange(-YAW_SWEEP_MAX_PIXELS, YAW_SWEEP_MAX_PIXELS + 1) * pixel_angle
    yaw_scores = np.array([
        _mean_edge_score(frame_inputs, lambda frame_input, angle=angle: rotate_pose_about_camera_y(
            frame_input["pose_world_camera"], angle), camera_matrix)
        for angle in yaw_angles])
    camera_period = float(np.median(np.diff(camera_times)))
    # Initial time grid; extend once when the maximum sits on either end.
    time_shift_frames, time_scores = _time_sweep(
        frame_inputs, camera_matrix, camera_times, camera_poses, TIME_SWEEP_STEPS_EACH_SIDE)
    initial_argmax = float(time_shift_frames[argmax_nearest_zero(time_shift_frames, time_scores)])
    time_extended = abs(initial_argmax) >= time_shift_frames.max() - 1e-12
    if time_extended:
        extended_steps = int(round(EXTENDED_TIME_SWEEP_PERIODS / TIME_SWEEP_STEP_FRAMES))
        time_shift_frames, time_scores = _time_sweep(
            frame_inputs, camera_matrix, camera_times, camera_poses, extended_steps)
    yaw_best = argmax_nearest_zero(yaw_angles, yaw_scores)
    time_best = argmax_nearest_zero(time_shift_frames, time_scores)
    time_on_boundary = abs(float(time_shift_frames[time_best])) >= time_shift_frames.max() - 1e-12
    standard_error = estimate_score_standard_error(frame_inputs, camera_matrix)
    yaw_information = curve_informativeness(yaw_scores, standard_error)
    time_information = curve_informativeness(time_scores, standard_error)
    return {
        "pixel_angle_radians": pixel_angle,
        "score_standard_error": standard_error,
        "yaw_angles_radians": yaw_angles.tolist(),
        "yaw_scores": yaw_scores.tolist(),
        "yaw_argmax_radians": float(yaw_angles[yaw_best]),
        "yaw_argmax_pixels": float(yaw_angles[yaw_best] / pixel_angle),
        "yaw_contrast": yaw_information["contrast"],
        "yaw_prominence": yaw_information["prominence"],
        "yaw_noise_floor": yaw_information["noise_floor"],
        "yaw_informative": yaw_information["informative"],
        "time_shifts_frames": time_shift_frames.tolist(),
        "time_scores": time_scores.tolist(),
        "time_argmax_frames": float(time_shift_frames[time_best]),
        "time_initial_argmax_frames": initial_argmax,
        "time_extended": bool(time_extended),
        "time_boundary_peak": bool(time_on_boundary),
        "time_contrast": time_information["contrast"],
        "time_prominence": time_information["prominence"],
        "time_noise_floor": time_information["noise_floor"],
        "time_informative": time_information["informative"],
        "camera_period": camera_period,
    }


def decide_verdict(
    frames: Sequence[Dict[str, Any]], sensitivity: Dict[str, Any], median_beam_gap_radians: float
) -> Tuple[str, List[str]]:
    """PASS / WARN / INCONCLUSIVE with reasons; also stores per-sweep statuses in ``sensitivity``.

    Per sweep:
      * not informative (prominence <= noise floor) -> INCONCLUSIVE: the metric cannot see
        the alignment on this scene, so nothing is concluded;
      * time sweep whose maximum is still on the boundary after extension -> INCONCLUSIVE
        (peak not bracketed);
      * informative and yaw argmax farther from 0 than max(one lidar beam gap, one pixel
        angle) -> WARN (alignment observable only to the beam spacing and to the 1-pixel grid
        step / nearest-pixel lookup, so smaller offsets equal zero);
      * informative and time argmax beyond one time-sweep step (0.25 frame) -> WARN;
      * otherwise PASS.
    A frame with no lidar point inside the image is a WARN (implausible projection).
    Overall: WARN dominates; else PASS if any sweep passed; else INCONCLUSIVE.

    Args:
        frames: Per-frame result dicts.
        sensitivity: Output of ``run_sensitivity_sweep`` (gets yaw_status / time_status).
        median_beam_gap_radians: Median lidar azimuth gap over the examined sweeps.

    Returns:
        (verdict, list of reason strings).
    """
    reasons, statuses = [], []
    for frame in frames:
        if frame["points_inside_image"] == 0:
            reasons.append(f"WARN lidar frame {frame['lidar_frame_index']}: no lidar point projects inside the image")
            statuses.append("WARN")
    yaw_tolerance = max(median_beam_gap_radians, sensitivity["pixel_angle_radians"])
    # Yaw sweep status.
    if not sensitivity["yaw_informative"]:
        yaw_status = "INCONCLUSIVE"
        reasons.append("INCONCLUSIVE yaw sweep: " + UNINFORMATIVE_TEXT)
    elif abs(sensitivity["yaw_argmax_radians"]) > yaw_tolerance:
        yaw_status = "WARN"
        reasons.append(
            f"WARN yaw score maximum at {sensitivity['yaw_argmax_pixels']:+.1f} px "
            f"({math.degrees(sensitivity['yaw_argmax_radians']):+.3f} deg) exceeds max(beam gap, pixel angle) "
            f"({math.degrees(yaw_tolerance):.3f} deg)")
    else:
        yaw_status = "PASS"
    # Time sweep status.
    if not sensitivity["time_informative"]:
        time_status = "INCONCLUSIVE"
        reasons.append("INCONCLUSIVE time sweep: " + UNINFORMATIVE_TEXT)
    elif sensitivity["time_boundary_peak"]:
        time_status = "INCONCLUSIVE"
        reasons.append("INCONCLUSIVE time sweep: maximum on the boundary of the extended grid (peak not bracketed)")
    elif abs(sensitivity["time_argmax_frames"]) > TIME_SWEEP_STEP_FRAMES:
        time_status = "WARN"
        reasons.append(f"WARN time score maximum at {sensitivity['time_argmax_frames']:+.2f} frames "
                       f"exceeds {TIME_SWEEP_STEP_FRAMES} frame")
    else:
        time_status = "PASS"
    sensitivity["yaw_status"], sensitivity["time_status"] = yaw_status, time_status
    statuses += [yaw_status, time_status]
    if "WARN" in statuses:
        return "WARN", reasons
    return ("PASS" if "PASS" in statuses else "INCONCLUSIVE"), reasons


def render_markdown(report: Dict[str, Any]) -> str:
    """Render the report dict as Markdown.

    Args:
        report: Output of ``generate_align_report``.

    Returns:
        Markdown text.
    """
    time_info, motion, sweep = report["time"], report["motion"], report["sensitivity"]
    lines = [
        f"# Alignment report: {report['session_name']}",
        "",
        f"**Verdict: {report['verdict']}**",
        "",
    ]
    lines += [f"- {reason}" for reason in report["verdict_reasons"]] or ["- no issues"]
    lines += [
        "",
        f"Camera `{report['camera_name']}`, lidar `{report['lidar_name']}`; "
        f"{time_info['camera']['count']} camera / {time_info['lidar']['count']} lidar frames.",
        "",
        "## Per-frame projection",
        "",
        "| lidar idx | camera idx | dt cam-lidar [s] | in front | inside | inside/in-front | edge score | overlay |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for frame in report["frames"]:
        lines.append(
            f"| {frame['lidar_frame_index']} | {frame['camera_frame_index']} | {frame['camera_lidar_time_delta']:+.4f} "
            f"| {frame['points_in_front']} | {frame['points_inside_image']} | {frame['inside_fraction']:.4f} "
            f"| {frame['edge_score']:.4f} | {frame['overlay_file']} |")
    lines += ["", "## Time statistics [s]", "", "| stream | count | spacing median | min | max | strictly increasing |",
              "|---|---|---|---|---|---|"]
    for stream_name in ("camera", "lidar"):
        stream = time_info[stream_name]
        lines.append(f"| {stream_name} | {stream['count']} | {stream['spacing_median']:.5f} | {stream['spacing_min']:.5f} "
                     f"| {stream['spacing_max']:.5f} | {stream['strictly_increasing']} |")
    nearest = time_info["nearest_camera_delta"]
    lines += [
        "",
        f"Nearest camera delta per lidar frame: median {nearest['delta_median']:.5f}, "
        f"min {nearest['delta_min']:.5f}, max {nearest['delta_max']:.5f}.",
        "",
        "## Lidar motion",
        "",
        f"Path length {motion['path_length']:.2f} m; speed p50 {motion['speed_p50']:.3f}, "
        f"p90 {motion['speed_p90']:.3f}, p99 {motion['speed_p99']:.3f}, max {motion['speed_max']:.3f} m/s.",
        "",
        "## Sensitivity sweep",
        "",
        "Score = mean over the sampled frames of `edge_alignment_score` with lidar depth-jump weights "
        f"(a jump above r*dtheta*tan({GRAZING_INCIDENCE_LIMIT_DEGREES:g} deg) between neighbouring beams of one "
        "channel marks both beams). Yaw perturbs T_world_camera by Ry(angle); time evaluates the camera pose at "
        "t + shift*camera_period with the image fixed.",
        "",
        f"Yaw argmax: {sweep['yaw_argmax_pixels']:+.1f} px ({math.degrees(sweep['yaw_argmax_radians']):+.4f} deg). "
        f"Time argmax: {sweep['time_argmax_frames']:+.2f} frames"
        + (f" (initial grid peaked at {sweep['time_initial_argmax_frames']:+.2f}; grid extended to "
           f"+-{max(sweep['time_shifts_frames']):.1f} periods)" if sweep["time_extended"] else "") + ". "
        f"Score standard error (bootstrap) {sweep['score_standard_error']:.5f}; noise floor = "
        f"{NOISE_SIGMAS:g} SE. Yaw: contrast (max-min)/mean {sweep['yaw_contrast']:.3f}, prominence "
        f"{sweep['yaw_prominence']:.5f}, floor {sweep['yaw_noise_floor']:.5f}, status **{sweep['yaw_status']}**. "
        f"Time: contrast {sweep['time_contrast']:.3f}, prominence {sweep['time_prominence']:.5f}, "
        f"floor {sweep['time_noise_floor']:.5f}, status **{sweep['time_status']}**. "
        f"Median lidar beam gap {math.degrees(report['median_beam_gap_radians']):.4f} deg "
        f"= {report['median_beam_gap_radians'] / sweep['pixel_angle_radians']:.1f} px.",
        "",
        "| yaw [px] | score |", "|---|---|",
    ]
    lines += [f"| {angle / sweep['pixel_angle_radians']:+.0f} | {score:.5f} |"
              for angle, score in zip(sweep["yaw_angles_radians"], sweep["yaw_scores"])]
    lines += ["", "| time shift [frames] | score |", "|---|---|"]
    lines += [f"| {shift:+.2f} | {score:.5f} |" for shift, score in zip(sweep["time_shifts_frames"], sweep["time_scores"])]
    return "\n".join(lines) + "\n"


def generate_align_report(
    dataset_dir: Path,
    output_dir: Path,
    requested_frames: int = DEFAULT_FRAME_COUNT,
    camera_name: Optional[str] = None,
    lidar_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute and write the alignment report (overlays, metrics.json, report.md).

    Reads the manifest, poses.npz and only the selected frame files.

    Args:
        dataset_dir: Intermediate dataset root.
        output_dir: Directory to create/fill.
        requested_frames: Number of evenly spaced lidar sweeps to examine.
        camera_name, lidar_name: Sensor names (default: first camera / lidar in the manifest).

    Returns:
        The report dict (also saved as metrics.json).

    Raises:
        FileNotFoundError: If manifest.json / poses.npz is missing.
        ValueError: If the dataset has no usable frames of a sensor.
    """
    dataset_dir, output_dir = Path(dataset_dir), Path(output_dir)
    # Step 1: manifest, poses and file listings (no frame content yet).
    manifest = json.loads((dataset_dir / "manifest.json").read_text())
    camera_name, lidar_name = _sensor_names_from_manifest(manifest, camera_name, lidar_name)
    pose_archive = np.load(dataset_dir / "poses.npz")
    times = {name: pose_archive[f"{name}{POSE_KEY_SEPARATOR}times"] for name in (camera_name, lidar_name)}
    poses = {name: pose_archive[f"{name}{POSE_KEY_SEPARATOR}poses_world_sensor"] for name in (camera_name, lidar_name)}
    camera_files = _list_frame_files(dataset_dir, "camera", camera_name)
    lidar_files = _list_frame_files(dataset_dir, "lidar", lidar_name)
    if not camera_files or not lidar_files or min(times[camera_name].size, times[lidar_name].size) < 2:
        raise ValueError("dataset needs at least two camera and two lidar frames")
    camera_matrix = _camera_matrix(manifest["cameras"][camera_name])
    # Step 2: choose sweeps among the lidar frames that exist on disk, evenly spread.
    available_lidar_indices = np.array(sorted(lidar_files))
    picked = available_lidar_indices[select_evenly_spaced_indices(available_lidar_indices.size, requested_frames)]
    available_camera_indices = np.array(sorted(camera_files))
    available_camera_times = times[camera_name][available_camera_indices]
    output_dir.mkdir(parents=True, exist_ok=True)
    # Step 3: per-frame projection, overlay and statistics.
    frames, frame_inputs, beam_gaps = [], [], []
    for lidar_index in picked:
        lidar_time = float(times[lidar_name][lidar_index])
        camera_index = int(available_camera_indices[nearest_camera_indices(np.array([lidar_time]), available_camera_times)[0]])
        sweep = np.load(lidar_files[int(lidar_index)])
        image_bgr = cv2.imread(str(camera_files[camera_index]))
        image_height, image_width = image_bgr.shape[:2]
        pose_world_lidar, pose_world_camera = poses[lidar_name][lidar_index], poses[camera_name][camera_index]
        pixels, in_front, inside_image = project_sweep_to_pixels(
            sweep[:, :3], pose_world_lidar, pose_world_camera, camera_matrix, image_width, image_height)
        # Camera-frame depth for colouring: z of inv(T_wc) T_wl p.
        transform_camera_lidar = np.linalg.inv(pose_world_camera) @ pose_world_lidar
        depths = sweep[:, :3].astype(np.float64) @ transform_camera_lidar[2, :3] + transform_camera_lidar[2, 3]
        weights, beam_gap = compute_lidar_discontinuity(sweep)
        beam_gaps.append(beam_gap)
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        edge_score = edge_alignment_score(gray, pixels, inside_image, weights)
        overlay_file = f"overlay_frame_{int(lidar_index):06d}.jpg"
        cv2.imwrite(str(output_dir / overlay_file), render_overlay(image_bgr, pixels, inside_image, depths),
                    [cv2.IMWRITE_JPEG_QUALITY, OVERLAY_JPEG_QUALITY])
        in_front_count, inside_count = int(in_front.sum()), int(inside_image.sum())
        frames.append({
            "lidar_frame_index": int(lidar_index),
            "camera_frame_index": camera_index,
            "camera_lidar_time_delta": float(times[camera_name][camera_index] - lidar_time),
            "points_total": int(sweep.shape[0]),
            "points_in_front": in_front_count,
            "points_inside_image": inside_count,
            "inside_fraction": inside_count / in_front_count if in_front_count else 0.0,
            "discontinuity_points": int((weights > 0).sum()),
            "edge_score": edge_score,
            "overlay_file": overlay_file,
        })
        frame_inputs.append({
            "points_lidar": sweep[:, :3], "pose_world_lidar": pose_world_lidar, "gray": gray,
            "weights": weights, "pose_world_camera": pose_world_camera,
            "camera_time": float(times[camera_name][camera_index]),
        })
    # Step 4: global time and motion statistics over every stamp, not only the picked frames.
    time_info = {
        "camera": compute_stamp_statistics(times[camera_name]),
        "lidar": compute_stamp_statistics(times[lidar_name]),
        "nearest_camera_delta": compute_nearest_delta_statistics(times[lidar_name], times[camera_name]),
    }
    motion = compute_motion_statistics(times[lidar_name], poses[lidar_name])
    # Step 5: sensitivity sweep and verdict.
    sensitivity = run_sensitivity_sweep(frame_inputs, camera_matrix, times[camera_name], poses[camera_name])
    verdict, reasons = decide_verdict(frames, sensitivity, float(np.median(beam_gaps)))
    report = {
        "session_name": manifest.get("session", {}).get("name", dataset_dir.name),
        "camera_name": camera_name,
        "lidar_name": lidar_name,
        "frames": frames,
        "time": time_info,
        "motion": motion,
        "sensitivity": sensitivity,
        "median_beam_gap_radians": float(np.median(beam_gaps)),
        "verdict": verdict,
        "verdict_reasons": reasons,
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    (output_dir / "report.md").write_text(render_markdown(report))
    return report
