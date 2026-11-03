"""Contract negative and boundary cases: one fault per dataset, the exact failed check name asserted.

Every case builds a valid in-memory dataset, injects ONE fault, runs every entry of
``CONTRACT_CHECKS`` (so an unexpected second failure or an exception is also caught) and compares
the failed check names with an expected list written by hand from the contract's documented rules.
Cases where the implementation is believed to be wrong live in ``test_contract_suspected_bugs.py``.
"""

import copy
import json

import numpy as np
import pytest

from tests.contract.test_no_return_contract import LIDAR_NAME, dataset_with_valid_sidecar
from tests.unit.test_writer import build_synthetic_dataset
from viking_splatad.contract import splatad_contract
from viking_splatad.contract.splatad_contract import validate_dataset
from viking_splatad.dataset.schema import NoReturnStream
from viking_splatad.dataset.writer import write_dataset

CAMERA_NAME = "cam_front"
# Names of the checks, written out independently of the module (the oracle for the meta-test).
ALL_CHECK_NAMES = {
    "poses_finite_and_count", "intrinsics_integer_size", "image_shape_matches", "distortion_k4_zero",
    "times_strictly_increasing", "sensor_index_contiguity", "lidar_array_format", "lidar_intensity_range",
    "lidar_point_time_range", "lidar_channel_in_mapping", "elevation_mapping_multiple_of_8",
    "azimuth_tile_integral", "lidar_no_return_sidecar", "splits_non_empty", "world_up_is_z",
}
SIDECAR = "lidar_no_return_sidecar"
SIZE_CHECKS = ["intrinsics_integer_size", "image_shape_matches"]  # in CONTRACT_CHECKS order


def failed_names(dataset):
    """Run every contract check in memory and list the failed check names in execution order."""
    results = [check(dataset) for check in splatad_contract.CONTRACT_CHECKS]
    return [result.name for result in results if not result.passed]


def largest_float32_not_above(limit):
    """Largest float32 value that is <= ``limit`` (a float64): the exact upper boundary in float32."""
    candidate = np.float32(limit)
    # float32(limit) may round up past the limit; step down one ulp when it does.
    return candidate if float(candidate) <= limit else np.nextafter(candidate, np.float32(0.0))


def smallest_float32_above(limit):
    """Smallest float32 value strictly greater than ``limit``: one ulp beyond the boundary."""
    return np.nextafter(largest_float32_not_above(limit), np.float32(np.inf))


def add_second_streams(dataset):
    """Add a second camera and a second lidar (deep copies) with their intrinsics, extrinsics, sidecar.

    Args:
        dataset: Dataset with a sidecar for LIDAR_NAME (see dataset_with_valid_sidecar).
    """
    for source_name, new_name in ((CAMERA_NAME, "cam_rear"), (LIDAR_NAME, "lidar_rear")):
        stream = copy.deepcopy(dataset.streams[source_name])
        stream.sensor_name = new_name
        dataset.streams[new_name] = stream
        dataset.extrinsics[new_name] = dataset.extrinsics[source_name].copy()
    dataset.cameras["cam_rear"] = copy.deepcopy(dataset.cameras[CAMERA_NAME])
    dataset.no_return_streams["lidar_rear"] = copy.deepcopy(dataset.no_return_streams[LIDAR_NAME])


def lidar_frame(dataset):
    """First sweep of the first lidar (mutable view into the dataset)."""
    return dataset.streams[LIDAR_NAME].frames[0]


def sidecar_rows(dataset, stream_name=LIDAR_NAME):
    """First no-return file (mutable) of ``stream_name``."""
    return dataset.no_return_streams[stream_name].frames[0]


# ----------------------------- fault injectors (each returns nothing, mutates the dataset) -----------------

def set_intensity(value):
    """Injector: first point's intensity becomes ``value``."""
    return lambda dataset: lidar_frame(dataset).__setitem__((0, splatad_contract.INTENSITY_COLUMN), value)


def set_point_time(value):
    """Injector: first point's per-point time becomes ``value``."""
    return lambda dataset: lidar_frame(dataset).__setitem__((0, splatad_contract.TIME_COLUMN), value)


def set_channel(value):
    """Injector: first point's channel becomes ``value``."""
    return lambda dataset: lidar_frame(dataset).__setitem__((0, splatad_contract.CHANNEL_COLUMN), value)


def set_resolution_for_bins(bin_count):
    """Injector: azimuth resolution that gives exactly ``bin_count`` bins per turn."""
    return lambda dataset: setattr(dataset.lidar, "azimuth_resolution", 2.0 * np.pi / bin_count)


def set_world_up(vector):
    """Injector: world up becomes ``vector``."""
    return lambda dataset: setattr(dataset.session, "world_up", np.array(vector, dtype=np.float64))


def set_camera_size(width, height):
    """Injector: declared camera size becomes (width, height)."""
    def inject(dataset):
        dataset.cameras[CAMERA_NAME].width = width
        dataset.cameras[CAMERA_NAME].height = height
    return inject


def set_lidar_times(new_times):
    """Injector: lidar pose times become ``new_times`` (length kept equal to the frame count)."""
    return lambda dataset: setattr(dataset.streams[LIDAR_NAME], "times", new_times(dataset.streams[LIDAR_NAME].times))


def resize_mapping(length):
    """Injector: elevation mapping with exactly ``length`` channels (angles are irrelevant to the check)."""
    return lambda dataset: setattr(
        dataset.lidar, "elevation_mapping", {channel: 0.0 for channel in range(length)})


def set_sidecar_cell(row_slice, value):
    """Injector: first no-return file, first row, columns ``row_slice`` become ``value``."""
    return lambda dataset: sidecar_rows(dataset).__setitem__((0, row_slice), value)


def scale_sidecar_direction(factor):
    """Injector: first no-return direction is [factor, 0, 0] (norm = |factor|)."""
    def inject(dataset):
        sidecar_rows(dataset)[0, :3] = np.array([factor, 0.0, 0.0], dtype=np.float32)
    return inject


def sidecar_as_float64(dataset):
    """Injector: first no-return file is float64."""
    dataset.no_return_streams[LIDAR_NAME].frames[0] = sidecar_rows(dataset).astype(np.float64)


def sidecar_stamp_only_mismatch(dataset):
    """Injector: same indices, but the first sweep's no-return stamp differs by one nanosecond."""
    dataset.no_return_streams[LIDAR_NAME].frame_timestamps_ns[0] += 1


def poses_shape_3x4(dataset):
    """Injector: camera poses stored as [N,3,4] (no homogeneous row)."""
    stream = dataset.streams[CAMERA_NAME]
    stream.poses_world_sensor = stream.poses_world_sensor[:, :3, :].copy()


def poses_bad_last_row(dataset):
    """Injector: one lidar pose has last row [0,0,0,2]."""
    dataset.streams[LIDAR_NAME].poses_world_sensor[3, 3, 3] = 2.0


def poses_inf(dataset):
    """Injector: one camera pose entry is +inf."""
    dataset.streams[CAMERA_NAME].poses_world_sensor[1, 1, 3] = np.inf


def poses_float32_overflow(dataset):
    """Injector: a translation that is finite in float64 but overflows float32 (finite float64 max / 2)."""
    dataset.streams[CAMERA_NAME].poses_world_sensor[1, 0, 3] = np.finfo(np.float64).max / 2.0


def only_two_poses(dataset):
    """Injector: both camera arrays cut to exactly two entries (the boundary that must still pass)."""
    stream = dataset.streams[CAMERA_NAME]
    stream.poses_world_sensor = stream.poses_world_sensor[:2]
    stream.times = stream.times[:2]


def k4_distortion_of_length(length):
    """Injector: distortion vector of ``length`` entries with a non-zero value at index 3 when it fits."""
    def inject(dataset):
        vector = np.array([0.01, -0.002, 0.0, 0.1, 0.0005, -0.0003])[:length]
        dataset.cameras[CAMERA_NAME].distortion = vector
    return inject


def eval_split_one_frame(dataset):
    """Injector: 0.0 train fraction leaves the train split empty."""
    dataset.session.train_fraction = 0.0


# Injectors that touch the camera image size need image_shape: width/height pairs below.
# (name, injector, expected failed check names in CONTRACT_CHECKS order)
NEGATIVE_CASES = [
    # --- lidar intensity: NaN / inf / out of range -------------------------------------------------------
    ("intensity_nan", set_intensity(np.nan), ["lidar_intensity_range"]),
    ("intensity_inf", set_intensity(np.inf), ["lidar_intensity_range"]),
    ("intensity_minus_inf", set_intensity(-np.inf), ["lidar_intensity_range"]),
    ("intensity_negative", set_intensity(-0.01), ["lidar_intensity_range"]),
    # --- per-point time: abs, inf; negative beyond one period ---------------------------------------------
    ("point_time_negative_beyond_period", lambda dataset: set_point_time(
        np.float32(-3.0 * dataset.lidar.sweep_period))(dataset), ["lidar_point_time_range"]),
    ("point_time_positive_inf", set_point_time(np.inf), ["lidar_point_time_range"]),
    ("point_time_negative_inf", set_point_time(-np.inf), ["lidar_point_time_range"]),
    ("point_time_one_ulp_above_period", lambda dataset: set_point_time(
        smallest_float32_above(dataset.lidar.sweep_period))(dataset), ["lidar_point_time_range"]),
    ("point_time_one_ulp_below_minus_period", lambda dataset: set_point_time(
        -smallest_float32_above(dataset.lidar.sweep_period))(dataset), ["lidar_point_time_range"]),
    # --- channel ---------------------------------------------------------------------------------------
    ("channel_non_integral", set_channel(2.5), ["lidar_channel_in_mapping"]),
    ("channel_barely_non_integral", set_channel(np.float32(3.0) + np.float32(1e-3)), ["lidar_channel_in_mapping"]),
    ("channel_nan", set_channel(np.nan), ["lidar_channel_in_mapping"]),
    ("channel_negative", set_channel(-1.0), ["lidar_channel_in_mapping"]),
    ("channel_one_past_mapping", lambda dataset: set_channel(
        max(dataset.lidar.elevation_mapping) + 1)(dataset), ["lidar_channel_in_mapping"]),
    # --- azimuth tiling -------------------------------------------------------------------------------
    # 500 bins is an integer but 500 % 8 = 4: divisibility by the tile width (8) must fail.
    ("azimuth_bins_not_divisible_by_tile", set_resolution_for_bins(500), ["azimuth_tile_integral"]),
    # 516 = 8 * 64 + 4: also not divisible; neighbours of a valid count (512) on both sides.
    ("azimuth_bins_511", set_resolution_for_bins(511), ["azimuth_tile_integral"]),
    ("azimuth_bins_513", set_resolution_for_bins(513), ["azimuth_tile_integral"]),
    # Non-integer bin count: 2*pi / 0.0175 = 359.04...
    ("azimuth_bins_fractional", lambda dataset: setattr(dataset.lidar, "azimuth_resolution", 0.0175),
     ["azimuth_tile_integral"]),
    # A larger tile width (16) breaks 8-divisible-only counts: 8 bins with tile 16.
    ("azimuth_tile_wider_than_bins", lambda dataset: (set_resolution_for_bins(8)(dataset),
                                                       setattr(dataset.lidar, "azimuth_tile_width", 16)),
     ["azimuth_tile_integral"]),
    # 512.5 bins: not an integer, yet rint() lands on the divisible count 512, so only integrality can fail it.
    ("azimuth_bins_half_off_a_valid_count", lambda dataset: setattr(
        dataset.lidar, "azimuth_resolution", 2.0 * np.pi / 512.5), ["azimuth_tile_integral"]),
    # --- world up ---------------------------------------------------------------------------------------
    ("world_up_minus_z", set_world_up([0.0, 0.0, -1.0]), ["world_up_is_z"]),
    ("world_up_tilted", set_world_up([0.1, 0.0, 1.0]), ["world_up_is_z"]),
    ("world_up_not_unit", set_world_up([0.0, 0.0, 2.0]), ["world_up_is_z"]),
    ("world_up_x", set_world_up([1.0, 0.0, 0.0]), ["world_up_is_z"]),
    # --- pose times ------------------------------------------------------------------------------------
    ("times_float32", set_lidar_times(lambda times: times.astype(np.float32)), ["times_strictly_increasing"]),
    ("times_nan", set_lidar_times(lambda times: np.where(np.arange(times.size) == 4, np.nan, times)),
     ["times_strictly_increasing"]),
    ("times_inf", set_lidar_times(lambda times: np.where(np.arange(times.size) == 4, np.inf, times)),
     ["times_strictly_increasing"]),
    ("times_decreasing_step", set_lidar_times(lambda times: np.where(np.arange(times.size) == 3, times[2] - 1e-3, times)),
     ["times_strictly_increasing"]),
    # +inf only in the LAST time: np.diff stays positive (inf - finite = +inf), so only finiteness catches it.
    ("times_inf_last", set_lidar_times(lambda times: np.where(np.arange(times.size) == times.size - 1, np.inf, times)),
     ["times_strictly_increasing"]),
    ("times_int64", set_lidar_times(lambda times: times.astype(np.int64)), ["times_strictly_increasing"]),
    # --- poses ---------------------------------------------------------------------------------------
    ("poses_single_pose", lambda dataset: (setattr(dataset.streams[CAMERA_NAME], "poses_world_sensor",
                                                  dataset.streams[CAMERA_NAME].poses_world_sensor[:1]),
                                           setattr(dataset.streams[CAMERA_NAME], "times",
                                                   dataset.streams[CAMERA_NAME].times[:1])),
     ["poses_finite_and_count"]),
    ("poses_3x4_shape", poses_shape_3x4, ["poses_finite_and_count"]),
    ("poses_last_row_not_rigid", poses_bad_last_row, ["poses_finite_and_count"]),
    ("poses_inf", poses_inf, ["poses_finite_and_count"]),
    ("poses_overflow_in_float32", poses_float32_overflow, ["poses_finite_and_count"]),
    # --- declared sizes: zero / negative fail both the size check and the image match; bool only the size ---
    ("width_zero", set_camera_size(0, 24), SIZE_CHECKS),
    ("height_zero", set_camera_size(32, 0), SIZE_CHECKS),
    ("width_negative", set_camera_size(-32, 24), SIZE_CHECKS),
    ("width_bool_true", set_camera_size(True, 24), ["intrinsics_integer_size"]),
    ("height_bool_true", set_camera_size(32, True), ["intrinsics_integer_size"]),
    ("height_float", set_camera_size(32, 24.0), ["intrinsics_integer_size"]),
    ("height_mismatch_only", set_camera_size(32, 25), ["image_shape_matches"]),
    # --- distortion: k4 non-zero with exactly 4 entries (the shortest vector that contains k4) -----------------
    ("distortion_length_4_with_k4", k4_distortion_of_length(4), ["distortion_k4_zero"]),
    ("distortion_full_with_tiny_k4", lambda dataset: dataset.cameras[CAMERA_NAME].distortion.__setitem__(3, 1e-12),
     ["distortion_k4_zero"]),
    # --- lidar array format: float64 sweep, 7 columns (the allowed widths are 5 and 6) -----------------------
    ("lidar_sweep_float64", lambda dataset: dataset.streams[LIDAR_NAME].frames.__setitem__(
        0, lidar_frame(dataset).astype(np.float64)), ["lidar_array_format"]),
    ("lidar_seven_columns", lambda dataset: dataset.streams[LIDAR_NAME].frames.__setitem__(
        0, np.column_stack([lidar_frame(dataset), np.zeros(len(lidar_frame(dataset)), np.float32)])),
     ["lidar_array_format"]),
    # --- splits -------------------------------------------------------------------------------------------
    ("train_fraction_zero", eval_split_one_frame, ["splits_non_empty"]),
    # --- elevation mapping ---------------------------------------------------------------------------
    ("mapping_length_zero", resize_mapping(0), ["lidar_channel_in_mapping", "elevation_mapping_multiple_of_8"]),
    ("mapping_length_7", resize_mapping(7), ["lidar_channel_in_mapping", "elevation_mapping_multiple_of_8"]),
    ("mapping_length_12_multiple_of_4_only", resize_mapping(12), ["elevation_mapping_multiple_of_8"]),
    ("mapping_length_9", resize_mapping(9), ["elevation_mapping_multiple_of_8"]),
]

# Cases that need the no-return sidecar fixture (all expect the sidecar check, alone).
SIDECAR_CASES = [
    # Directions: x0.9 is shorter than unit by 0.1 (>> tolerance); x1.1 is the existing longer case.
    ("direction_shrunk_x0_9", scale_sidecar_direction(0.9), [SIDECAR]),
    ("direction_zero", scale_sidecar_direction(0.0), [SIDECAR]),
    # float32 spacing above 1 is eps; 1 + 2 eps deviates by 2 eps > eps (tolerance), so it must fail.
    ("direction_two_eps_long", scale_sidecar_direction(1.0 + 2.0 * float(np.finfo(np.float32).eps)), [SIDECAR]),
    # Time: negative beyond one period, and the audit's 1.5x boundary (limit is 1x, so 1.5x fails).
    ("time_negative_two_periods", lambda dataset: sidecar_rows(dataset).__setitem__(
        (0, 3), np.float32(-2.0 * dataset.lidar.sweep_period)), [SIDECAR]),
    ("time_one_and_half_periods", lambda dataset: sidecar_rows(dataset).__setitem__(
        (0, 3), np.float32(1.5 * dataset.lidar.sweep_period)), [SIDECAR]),
    ("time_one_ulp_above_period", lambda dataset: sidecar_rows(dataset).__setitem__(
        (0, 3), smallest_float32_above(dataset.lidar.sweep_period)), [SIDECAR]),
    ("time_nan", set_sidecar_cell(3, np.nan), [SIDECAR]),
    # Channel: unknown, negative, non-integral, NaN.
    ("channel_negative", set_sidecar_cell(4, -1.0), [SIDECAR]),
    ("channel_non_integral", set_sidecar_cell(4, 2.5), [SIDECAR]),
    ("channel_nan", set_sidecar_cell(4, np.nan), [SIDECAR]),
    # Columns: negative, non-integer, NaN, inf.
    ("column_negative", set_sidecar_cell(5, -1.0), [SIDECAR]),
    ("column_non_integer", set_sidecar_cell(5, 3.5), [SIDECAR]),
    ("column_nan", set_sidecar_cell(5, np.nan), [SIDECAR]),
    ("column_inf", set_sidecar_cell(5, np.inf), [SIDECAR]),
    ("direction_nan", set_sidecar_cell(0, np.nan), [SIDECAR]),
    # Dtype and layout.
    ("rows_float64", sidecar_as_float64, [SIDECAR]),
    ("rows_one_dimensional", lambda dataset: dataset.no_return_streams[LIDAR_NAME].frames.__setitem__(
        0, sidecar_rows(dataset)[0].copy()), [SIDECAR]),
    # Same indices, differing stamp only.
    ("stamp_only_mismatch", sidecar_stamp_only_mismatch, [SIDECAR]),
    # Same stamps, differing index only (a file numbered 0..n-1 shifted by one).
    ("index_only_mismatch", lambda dataset: dataset.no_return_streams[LIDAR_NAME].frame_indices.__setitem__(
        -1, dataset.no_return_streams[LIDAR_NAME].frame_indices[-1] + 1), [SIDECAR]),
    # A column equal to columns-per-ring is one past the last valid column (valid: 0..columns-1).
    ("column_equal_to_columns_per_ring", lambda dataset: sidecar_rows(dataset).__setitem__(
        (0, 5), np.float32(np.rint(2.0 * np.pi / dataset.lidar.azimuth_resolution))), [SIDECAR]),
]

# Boundary cases that must PASS everything (value exactly at a limit).
BOUNDARY_PASS_CASES = [
    ("intensity_exactly_zero", set_intensity(0.0)),
    ("intensity_exactly_one", set_intensity(1.0)),
    ("point_time_exactly_period", lambda dataset: set_point_time(
        largest_float32_not_above(dataset.lidar.sweep_period))(dataset)),
    ("point_time_exactly_minus_period", lambda dataset: set_point_time(
        -largest_float32_not_above(dataset.lidar.sweep_period))(dataset)),
    # 0.125 = 2**-3 is exactly representable in float32, so |t| == period holds with true equality.
    ("point_time_equal_to_representable_period", lambda dataset: (
        setattr(dataset.lidar, "sweep_period", 0.125), set_point_time(np.float32(0.125))(dataset))),
    ("point_time_equal_to_minus_representable_period", lambda dataset: (
        setattr(dataset.lidar, "sweep_period", 0.125), set_point_time(np.float32(-0.125))(dataset))),
    ("sidecar_time_equal_to_representable_period", lambda dataset: (
        setattr(dataset.lidar, "sweep_period", 0.125), sidecar_rows(dataset).__setitem__((0, 3), np.float32(0.125)))),
    ("channel_smallest_key", lambda dataset: set_channel(min(dataset.lidar.elevation_mapping))(dataset)),
    ("channel_largest_key", lambda dataset: set_channel(max(dataset.lidar.elevation_mapping))(dataset)),
    ("azimuth_bins_1024", set_resolution_for_bins(1024)),
    ("mapping_length_16", resize_mapping(16)),
    ("exactly_two_poses", only_two_poses),
    ("distortion_none", lambda dataset: setattr(dataset.cameras[CAMERA_NAME], "distortion", None)),
    ("distortion_length_3_cannot_hold_k4", k4_distortion_of_length(3)),
    ("world_up_within_float_noise", set_world_up([0.0, 1e-12, 1.0])),
    ("sidecar_direction_exactly_at_tolerance",
     scale_sidecar_direction(1.0 + float(np.finfo(np.float32).eps))),
    ("sidecar_direction_shorter_by_half_eps",
     scale_sidecar_direction(1.0 - float(np.finfo(np.float32).eps) / 2.0)),
    ("sidecar_time_exactly_period", lambda dataset: sidecar_rows(dataset).__setitem__(
        (0, 3), largest_float32_not_above(dataset.lidar.sweep_period))),
    ("sidecar_time_exactly_minus_period", lambda dataset: sidecar_rows(dataset).__setitem__(
        (0, 3), -largest_float32_not_above(dataset.lidar.sweep_period))),
    ("sidecar_column_zero", set_sidecar_cell(5, 0.0)),
    ("sidecar_column_last", lambda dataset: sidecar_rows(dataset).__setitem__(
        (0, 5), np.float32(np.rint(2.0 * np.pi / dataset.lidar.azimuth_resolution) - 1))),
    ("sidecar_empty_file_is_valid", lambda dataset: dataset.no_return_streams[LIDAR_NAME].frames.__setitem__(
        0, np.zeros((0, 6), dtype=np.float32))),
]


def sidecar_dataset():
    """Valid dataset with a complete no-return sidecar."""
    return dataset_with_valid_sidecar()


@pytest.mark.parametrize("case_name, injector, expected", NEGATIVE_CASES, ids=[case[0] for case in NEGATIVE_CASES])
def test_single_fault_fails_exactly_the_expected_checks(case_name, injector, expected):
    """One fault in a valid dataset fails exactly the listed checks (and nothing else)."""
    dataset = build_synthetic_dataset()
    injector(dataset)
    assert failed_names(dataset) == expected


@pytest.mark.parametrize("case_name, injector, expected", SIDECAR_CASES, ids=[case[0] for case in SIDECAR_CASES])
def test_single_sidecar_fault_fails_only_the_sidecar_check(case_name, injector, expected):
    """One no-return defect fails the sidecar check alone."""
    dataset = sidecar_dataset()
    injector(dataset)
    assert failed_names(dataset) == expected


@pytest.mark.parametrize("case_name, injector", BOUNDARY_PASS_CASES, ids=[case[0] for case in BOUNDARY_PASS_CASES])
def test_values_exactly_at_a_limit_pass(case_name, injector):
    """A value exactly at an inclusive limit keeps the whole contract green."""
    dataset = sidecar_dataset()
    injector(dataset)
    assert failed_names(dataset) == []


def test_azimuth_bins_equal_to_tile_width_pass():
    """Exactly one tile per turn (bins == tile width, 8) is the smallest valid bin count."""
    dataset = build_synthetic_dataset()
    set_resolution_for_bins(dataset.lidar.azimuth_tile_width)(dataset)
    assert failed_names(dataset) == []


def test_boundary_helpers_straddle_the_limit_exactly():
    """The ulp helpers really sit on either side of the float64 limit (guards the boundary cases above)."""
    period = build_synthetic_dataset().lidar.sweep_period
    at_limit, beyond = largest_float32_not_above(period), smallest_float32_above(period)
    assert float(at_limit) <= period < float(beyond)
    # They are adjacent float32 values: nothing representable lies between them.
    assert np.nextafter(at_limit, np.float32(np.inf)) == beyond


# ----------------------------------- faults on a SECOND stream ----------------------------------------------

def second_stream_cases():
    """(case name, injector, expected check, offending stream label that must appear in the message)."""
    return [
        ("second_lidar_intensity", lambda dataset: dataset.streams["lidar_rear"].frames[0].__setitem__(
            (0, splatad_contract.INTENSITY_COLUMN), 1.5), "lidar_intensity_range", "lidar_rear[0]"),
        ("second_lidar_channel", lambda dataset: dataset.streams["lidar_rear"].frames[1].__setitem__(
            (0, splatad_contract.CHANNEL_COLUMN), 2.5), "lidar_channel_in_mapping", "lidar_rear[1]"),
        ("second_lidar_point_time", lambda dataset: dataset.streams["lidar_rear"].frames[2].__setitem__(
            (0, splatad_contract.TIME_COLUMN), 5.0 * dataset.lidar.sweep_period),
         "lidar_point_time_range", "lidar_rear[2]"),
        ("second_lidar_dtype", lambda dataset: dataset.streams["lidar_rear"].frames.__setitem__(
            3, dataset.streams["lidar_rear"].frames[3].astype(np.float64)), "lidar_array_format", "lidar_rear[3]"),
        ("second_camera_times", lambda dataset: dataset.streams["cam_rear"].times.__setitem__(
            4, dataset.streams["cam_rear"].times[3]), "times_strictly_increasing", "cam_rear"),
        ("second_camera_width", lambda dataset: setattr(dataset.cameras["cam_rear"], "width", 40),
         "image_shape_matches", "cam_rear[0]"),
        ("second_camera_size_float", lambda dataset: setattr(dataset.cameras["cam_rear"], "height", 24.5),
         "intrinsics_integer_size", "cam_rear"),
        ("second_camera_k4", lambda dataset: dataset.cameras["cam_rear"].distortion.__setitem__(3, 0.2),
         "distortion_k4_zero", "cam_rear"),
        ("second_camera_index_gap", lambda dataset: dataset.streams["cam_rear"].frame_indices.__setitem__(
            slice(5, None), dataset.streams["cam_rear"].frame_indices[5:] + 1), "sensor_index_contiguity", "cam_rear"),
        ("second_camera_pose_nan", lambda dataset: dataset.streams["cam_rear"].poses_world_sensor.__setitem__(
            (2, 0, 3), np.nan), "poses_finite_and_count", "cam_rear"),
        ("second_sidecar_direction", lambda dataset: sidecar_rows(dataset, "lidar_rear").__setitem__(
            (0, slice(0, 3)), np.array([0.9, 0.0, 0.0], dtype=np.float32)), SIDECAR, "lidar_rear[0]"),
        ("second_sidecar_missing_file", lambda dataset: dataset.no_return_streams.__setitem__(
            "lidar_rear", NoReturnStream(
                dataset.no_return_streams["lidar_rear"].frame_indices[:-1],
                dataset.no_return_streams["lidar_rear"].frame_timestamps_ns[:-1],
                dataset.no_return_streams["lidar_rear"].frames[:-1])), SIDECAR, "lidar_rear"),
    ]


@pytest.mark.parametrize("case", second_stream_cases(), ids=[case[0] for case in second_stream_cases()])
def test_fault_in_second_stream_is_found_and_attributed(case):
    """A fault only in the second stream fails the check and names that stream (not the healthy first)."""
    _, injector, expected_check, offending_label = case
    dataset = sidecar_dataset()
    add_second_streams(dataset)
    # Control: two healthy copies of every stream pass (the fixture itself is sound).
    assert failed_names(dataset) == []
    injector(dataset)
    results = [check(dataset) for check in splatad_contract.CONTRACT_CHECKS]
    failed = [result for result in results if not result.passed]
    assert [result.name for result in failed] == [expected_check]
    assert offending_label in failed[0].message
    # The healthy first stream of the same kind is never blamed.
    first_label = "lidar_top[" if "lidar" in offending_label else "cam_front"
    assert first_label not in failed[0].message


# ------------------------------------------ loadability arms -----------------------------------------------

def write_valid(tmp_path):
    """Write the valid synthetic dataset (with sidecar) under tmp_path/ds and return that root."""
    root = tmp_path / "ds"
    write_dataset(root, sidecar_dataset())
    return root


def delete_manifest(root):
    """Loadability fault: manifest.json missing."""
    (root / "manifest.json").unlink()


def truncate_poses(root):
    """Loadability fault: poses.npz cut to half its bytes."""
    poses_path = root / "poses.npz"
    poses_path.write_bytes(poses_path.read_bytes()[: poses_path.stat().st_size // 2])


def delete_poses(root):
    """Loadability fault: poses.npz missing."""
    (root / "poses.npz").unlink()


def delete_lidar_directory(root):
    """Loadability fault: a sensor's frame directory is gone."""
    for path in (root / "lidar" / LIDAR_NAME).iterdir():
        path.unlink()
    (root / "lidar" / LIDAR_NAME).rmdir()


def break_manifest_json(root):
    """Loadability fault: manifest.json is not valid JSON."""
    (root / "manifest.json").write_text("{ not json")


def drop_manifest_key(root):
    """Loadability fault: manifest lacks the required 'lidar' section."""
    manifest = json.loads((root / "manifest.json").read_text())
    del manifest["lidar"]
    (root / "manifest.json").write_text(json.dumps(manifest))


def truncate_a_lidar_frame(root):
    """Loadability fault: one sweep .npy cut in half."""
    frame_path = sorted((root / "lidar" / LIDAR_NAME).glob("*.npy"))[0]
    frame_path.write_bytes(frame_path.read_bytes()[: frame_path.stat().st_size // 2])


LOADABILITY_FAULTS = [
    delete_manifest, delete_poses, delete_lidar_directory,
    break_manifest_json, drop_manifest_key, truncate_a_lidar_frame,
]


@pytest.mark.parametrize("fault", LOADABILITY_FAULTS, ids=[fault.__name__ for fault in LOADABILITY_FAULTS])
def test_unloadable_dataset_is_reported_not_raised(tmp_path, fault):
    """A dataset that cannot be read yields exactly one failed ``dataset_loadable`` check."""
    root = write_valid(tmp_path)
    fault(root)
    assert validate_dataset(root).failed_check_names() == ["dataset_loadable"]


def test_nonexistent_root_is_reported_not_raised(tmp_path):
    """A directory that does not exist yields the failed loadable check."""
    assert validate_dataset(tmp_path / "does_not_exist").failed_check_names() == ["dataset_loadable"]


def test_missing_camera_frame_file_fails_a_check_without_raising(tmp_path):
    """Deleting one JPEG leaves a loadable dataset whose index/count checks must fail (never pass)."""
    root = write_valid(tmp_path)
    sorted((root / "camera" / CAMERA_NAME).glob("*.jpg"))[3].unlink()
    report = validate_dataset(root)
    assert not report.passed


def test_missing_lidar_frame_file_fails_a_check_without_raising(tmp_path):
    """Deleting one sweep file leaves a loadable dataset whose index contiguity must fail."""
    root = write_valid(tmp_path)
    sorted((root / "lidar" / LIDAR_NAME).glob("*.npy"))[3].unlink()
    report = validate_dataset(root)
    assert "sensor_index_contiguity" in report.failed_check_names()


def test_valid_dataset_round_trip_still_passes(tmp_path):
    """Control for the loadability arms: the untouched written dataset passes everything."""
    assert validate_dataset(write_valid(tmp_path)).passed


# ----------------------------------------------- meta-test -------------------------------------------------

def test_every_contract_check_has_a_negative_case_in_this_file():
    """Each entry of CONTRACT_CHECKS is the target of at least one case that must fail it.

    The expected names come from ``ALL_CHECK_NAMES`` (hand-written above), not from the module, so a
    new check added to CONTRACT_CHECKS without a negative case here makes this test fail.
    """
    valid_names = {check(sidecar_dataset()).name for check in splatad_contract.CONTRACT_CHECKS}
    # The hand-written list is the complete set of executed checks.
    assert valid_names == ALL_CHECK_NAMES
    targeted = set()
    for _, _, expected in NEGATIVE_CASES + SIDECAR_CASES:
        targeted.update(expected)
    assert targeted >= ALL_CHECK_NAMES - {"sensor_index_contiguity"}
    # Index contiguity has no in-memory injector above; the second-stream cases cover it.
    assert any(case[2] == "sensor_index_contiguity" for case in second_stream_cases())
    # Every check is covered by the table for the sidecar fixture's own check as well.
    assert len(splatad_contract.CONTRACT_CHECKS) == len(ALL_CHECK_NAMES)
