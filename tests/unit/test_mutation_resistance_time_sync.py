"""Mutation-resistance tests for align/time_sync.py (T22 sub-sample refinement, T24 extrinsic residual).

The existing T24 test uses an identity rotation and a pure x translation with identical clouds, so
a transposed rotation, an inverted extrinsic, max/median instead of mean, or the swapped
nearest-neighbour direction all pass. These tests use a seeded random rotation, asymmetric
displacements and clouds of different size.
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from viking_splatad.align.time_sync import estimate_time_offset, radar_extrinsic_residual
from tests.unit.test_time_sync import smooth_signal

# Arbitrary fixed seed: only makes the random fixtures reproducible.
RANDOM_SEED = 424242


def random_extrinsic(rng):
    """Random T_lidar_radar with a rotation of 0.5..2.5 rad about a random axis and a 1..3 m shift.

    Returns:
        (rotation_matrix, translation, 4x4 matrix); the angle floor keeps the rotation far from
        identity so transposing it (which negates the angle) is detectable.
    """
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    rotation = Rotation.from_rotvec(axis * rng.uniform(0.5, 2.5)).as_matrix()
    translation = rng.uniform(1.0, 3.0, size=3) * rng.choice([-1.0, 1.0], size=3)
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return rotation, translation, matrix


def well_separated_lidar_cloud(rng, point_count):
    """Lidar cloud on a jittered 20 m lattice so every nearest-neighbour search is unambiguous."""
    lattice_pitch = 20.0  # >> every displacement used below (at most 2 m), justified in each test
    side = int(np.ceil(point_count ** (1 / 3)))
    grid = np.stack(np.meshgrid(*[np.arange(side)] * 3, indexing="ij"), axis=-1).reshape(-1, 3)
    cloud = grid[:point_count] * lattice_pitch + rng.uniform(-1.0, 1.0, size=(point_count, 3))
    return cloud, lattice_pitch


@pytest.mark.covers("T24")
def test_extrinsic_residual_rotated_extrinsic_zero_only_for_the_right_transform():
    """Radar points built as R^T (lidar - t) give residual 0 under T, and a large one under variants."""
    rng = np.random.default_rng(RANDOM_SEED)
    rotation, translation, extrinsic = random_extrinsic(rng)
    lidar_points, _pitch = well_separated_lidar_cloud(rng, 27)
    # Radar frame -> lidar frame is p_l = R p_r + t, so the radar-frame points are R^T (p_l - t).
    radar_points = (lidar_points - translation) @ rotation
    assert radar_extrinsic_residual(radar_points, lidar_points, extrinsic) == pytest.approx(0.0, abs=1e-9)

    # Transposed rotation (R^T instead of R) and inverted extrinsic must be clearly non-zero.
    transposed = extrinsic.copy()
    transposed[:3, :3] = rotation.T
    assert radar_extrinsic_residual(radar_points, lidar_points, transposed) > 1.0
    assert radar_extrinsic_residual(radar_points, lidar_points, np.linalg.inv(extrinsic)) > 1.0
    # Translation-only (rotation ignored) is also wrong.
    translation_only = np.eye(4)
    translation_only[:3, 3] = translation
    assert radar_extrinsic_residual(radar_points, lidar_points, translation_only) > 1.0


@pytest.mark.covers("T24")
def test_extrinsic_residual_is_mean_over_radar_points_with_outliers():
    """Two displaced radar points among ten: residual is their mean-weighted distance, not max or median."""
    rng = np.random.default_rng(RANDOM_SEED + 1)
    rotation, translation, extrinsic = random_extrinsic(rng)
    lidar_points, lattice_pitch = well_separated_lidar_cloud(rng, 27)
    radar_count = 10
    displaced_distances = {2: 0.5, 7: 1.5}  # asymmetric; both << pitch/2 so the nearest lidar point stays the original
    assert max(displaced_distances.values()) < lattice_pitch / 4
    selected = lidar_points[:radar_count].copy()
    for point_index, distance in displaced_distances.items():
        direction = rng.normal(size=3)
        selected[point_index] += distance * direction / np.linalg.norm(direction)
    radar_points = (selected - translation) @ rotation
    residual = radar_extrinsic_residual(radar_points, lidar_points, extrinsic)
    # Mean = (0.5 + 1.5) / 10 = 0.2; median = 0, max = 1.5, sum = 2.0 are all different.
    assert residual == pytest.approx(sum(displaced_distances.values()) / radar_count, abs=1e-9)


@pytest.mark.covers("T24")
def test_extrinsic_residual_queries_from_radar_into_lidar_not_the_reverse():
    """Extra lidar points are free; extra radar points far from any lidar point raise the mean."""
    rng = np.random.default_rng(RANDOM_SEED + 2)
    rotation, translation, extrinsic = random_extrinsic(rng)
    lidar_points, _pitch = well_separated_lidar_cloud(rng, 27)
    # Radar sees only 5 of the 27 lidar points: every radar point has an exact match -> 0.
    # (Reversed tree direction would average distances of the 22 unmatched lidar points.)
    radar_subset = (lidar_points[:5] - translation) @ rotation
    assert radar_extrinsic_residual(radar_subset, lidar_points, extrinsic) == pytest.approx(0.0, abs=1e-9)
    # One extra radar point 3 m from a lidar point adds 3 / 6 to the mean.
    extra_lidar_space = lidar_points[0] + np.array([0.0, 0.0, 3.0])
    radar_plus_extra = np.vstack([radar_subset, (extra_lidar_space - translation) @ rotation])
    assert radar_extrinsic_residual(radar_plus_extra, lidar_points, extrinsic) == pytest.approx(3.0 / 6.0, abs=1e-9)


@pytest.mark.covers("T22")
def test_parabolic_refinement_beats_the_unrefined_grid_error():
    """Tolerance below the grid error the unrefined peak would have: only the refinement meets it."""
    sample_spacing = 0.05
    times = np.arange(0.0, 12.0, sample_spacing)
    # upsample_factor=1 makes the lag grid equal to the sample spacing (documented: step =
    # spacing / factor), so an offset 0.37 steps above a grid multiple has an unrefined error of 0.37 step.
    fractional_steps = 3.37
    injected_offset = fractional_steps * sample_spacing
    nearest_grid_error = (fractional_steps - round(fractional_steps)) * sample_spacing
    offset, _confidence = estimate_time_offset(
        times, smooth_signal(times), times, smooth_signal(times - injected_offset), upsample_factor=1
    )
    tolerance = 0.4 * abs(nearest_grid_error)
    assert abs(nearest_grid_error) > 2.0 * tolerance - 1e-12  # the tolerance really is tighter than the grid error
    assert abs(offset - injected_offset) < tolerance

    # Same at the default upsampling (grid = spacing / 4), half a grid step off the grid.
    default_step = sample_spacing / 4
    half_step_offset = (6 + 0.5) * default_step
    offset_default, _ = estimate_time_offset(
        times, smooth_signal(times), times, smooth_signal(times - half_step_offset)
    )
    assert abs(offset_default - half_step_offset) < 0.3 * (0.5 * default_step)


@pytest.mark.covers("T22")
def test_confidence_separates_a_true_match_from_an_unrelated_signal():
    """Confidence is high for b = a shifted and clearly lower for an unrelated (time-reversed) signal."""
    times = np.arange(0.0, 12.0, 0.05)
    _offset_match, confidence_match = estimate_time_offset(
        times, smooth_signal(times), times, smooth_signal(times - 0.2)
    )
    _offset_other, confidence_other = estimate_time_offset(
        times, smooth_signal(times), times, smooth_signal(times[-1] - times)
    )
    assert confidence_match > 0.5
    assert confidence_other < confidence_match - 0.3
