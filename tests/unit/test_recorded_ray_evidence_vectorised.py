"""T78 vectorised recorded_ray_evidence must equal the pre-vectorisation per-cell loop bit for bit."""

import numpy as np
import pytest
from scipy.spatial import cKDTree

from viking_splatad.eval.hidden_geometry import (
    BLOCKED_COLUMN, EVIDENCE_FLAG_COUNT, NO_RETURN_COLUMN, PASSED_THROUGH_COLUMN, SEEN_COLUMN,
    ray_cube_intervals, ray_cube_intervals_pairs, recorded_ray_evidence)

RAY_COUNT = 2000
CELL_COUNT = 600
NEAR_CELL_COUNT = 40
FAR_CELL_COUNT = 40
RETURN_PROBABILITY = 0.7
DEPTH_RANGE = (1.0, 60.0)
MID_DISTANCE_RANGE = (1.0, 40.0)
FAR_DISTANCE_RANGE = (500.0, 5000.0)
SCENE_SEEDS = (0, 1, 2, 3)
VOXEL_SIZES = (0.5, 1.0)
RANGE_MARGINS = (0.0, 0.2)


def reference_recorded_ray_evidence(centres_relative, voxel_size, directions, depths, did_return, *, range_margin=0.0):
    """Pre-vectorisation T78 (per-cell Python loop with KD-tree cone query); this is the spec."""
    result = np.zeros((len(centres_relative), EVIDENCE_FLAG_COUNT), dtype=bool)
    tree = cKDTree(directions)
    half_width = voxel_size / 2
    half_diagonal = np.sqrt(3) * half_width
    distances = np.linalg.norm(centres_relative, axis=1)
    for cell_index, centre in enumerate(centres_relative):
        if distances[cell_index] <= half_diagonal:
            candidates = np.arange(len(directions))
        else:
            ratio = half_diagonal / distances[cell_index]
            chord = ratio * np.sqrt(2 / (1 + np.sqrt(1 - ratio ** 2)))
            candidates = np.asarray(tree.query_ball_point(centre / distances[cell_index],
                                                          np.nextafter(chord, np.inf)), dtype=np.int64)
        if not len(candidates):
            continue
        entry, exit_depth, hit = ray_cube_intervals(directions[candidates], centre, half_width)
        candidates, entry, exit_depth = candidates[hit], entry[hit], exit_depth[hit]
        returned = did_return[candidates]
        measured = depths[candidates]
        seen_start, seen_end = entry - range_margin, exit_depth + range_margin
        result[cell_index, SEEN_COLUMN] = np.any(returned & (measured >= seen_start) & (measured <= seen_end))
        result[cell_index, BLOCKED_COLUMN] = np.any(returned & (measured < seen_start))
        result[cell_index, NO_RETURN_COLUMN] = np.any(~returned)
        result[cell_index, PASSED_THROUGH_COLUMN] = np.any(returned & (measured > seen_end))
    return result


def build_scene(seed, voxel_size):
    """Seeded random scene: unit rays, depths, return mask and cells at near/mixed/far distances."""
    generator = np.random.default_rng(seed)
    directions = generator.normal(size=(RAY_COUNT, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    # Include axis-parallel rays to exercise the parallel-slab branch.
    directions[:3] = np.eye(3)
    depths = generator.uniform(*DEPTH_RANGE, size=RAY_COUNT)
    did_return = generator.random(RAY_COUNT) < RETURN_PROBABILITY

    def cells_at(count, low, high):
        """Cells at uniform random directions and distances in [low, high)."""
        unit = generator.normal(size=(count, 3))
        unit /= np.linalg.norm(unit, axis=1, keepdims=True)
        return unit * generator.uniform(low, high, size=(count, 1))

    near_cells = cells_at(NEAR_CELL_COUNT, 0.0, np.sqrt(3) * voxel_size / 2)
    # Cells exactly along rays so many cells really are hit.
    along_rays = directions[generator.integers(0, RAY_COUNT, CELL_COUNT)] * generator.uniform(*MID_DISTANCE_RANGE, (CELL_COUNT, 1))
    far_cells = cells_at(FAR_CELL_COUNT, *FAR_DISTANCE_RANGE)
    mid_cells = cells_at(CELL_COUNT, *MID_DISTANCE_RANGE)
    centres = np.concatenate([near_cells, along_rays, far_cells, mid_cells])
    return centres, directions, depths, did_return


@pytest.mark.covers("T78")
@pytest.mark.parametrize("seed", SCENE_SEEDS)
@pytest.mark.parametrize("voxel_size", VOXEL_SIZES)
@pytest.mark.parametrize("range_margin", RANGE_MARGINS)
def test_vectorised_matches_reference_loop(seed, voxel_size, range_margin):
    """Vectorised T78 equals the reference loop exactly on random scenes."""
    centres, directions, depths, did_return = build_scene(seed, voxel_size)
    expected = reference_recorded_ray_evidence(centres, voxel_size, directions, depths, did_return,
                                               range_margin=range_margin)
    actual = recorded_ray_evidence(centres, voxel_size, directions, depths, did_return, range_margin=range_margin)
    assert expected.any(), "scene must produce some evidence"
    assert np.array_equal(actual, expected)


@pytest.mark.covers("T78")
def test_pairs_helper_matches_single_cube_helper():
    """ray_cube_intervals_pairs equals ray_cube_intervals when every pair shares one centre."""
    centres, directions, _, _ = build_scene(SCENE_SEEDS[0], VOXEL_SIZES[0])
    centre = centres[NEAR_CELL_COUNT]
    expected = ray_cube_intervals(directions, centre, VOXEL_SIZES[0] / 2)
    actual = ray_cube_intervals_pairs(directions, np.tile(centre, (len(directions), 1)), VOXEL_SIZES[0] / 2)
    for expected_part, actual_part in zip(expected, actual):
        assert np.array_equal(expected_part, actual_part)


# Chunk sizes far below the scene sizes (and coprime with them), so every chunk and slice boundary of
# the vectorised T78 is crossed many times: near cells split into several chunks, far cells into many
# cone-query chunks, and candidate pairs into many slices.
SMALL_FAR_CELL_CHUNK = 7
SMALL_PAIR_SLICE = 13
SMALL_NEAR_PAIR_BUDGET = 2 * RAY_COUNT + 1


@pytest.mark.covers("T78")
@pytest.mark.parametrize("range_margin", RANGE_MARGINS)
def test_vectorised_matches_reference_across_chunk_boundaries(monkeypatch, range_margin):
    """Results do not depend on the memory policy: tiny chunks give the reference result exactly."""
    import viking_splatad.eval.hidden_geometry as hidden_geometry_module

    centres, directions, depths, did_return = build_scene(SCENE_SEEDS[0], VOXEL_SIZES[0])
    expected = reference_recorded_ray_evidence(centres, VOXEL_SIZES[0], directions, depths, did_return,
                                               range_margin=range_margin)
    monkeypatch.setattr(hidden_geometry_module, "FAR_CELL_CHUNK", SMALL_FAR_CELL_CHUNK)
    monkeypatch.setattr(hidden_geometry_module, "PAIR_SLICE", SMALL_PAIR_SLICE)
    monkeypatch.setattr(hidden_geometry_module, "NEAR_PAIR_BUDGET", SMALL_NEAR_PAIR_BUDGET)
    actual = recorded_ray_evidence(centres, VOXEL_SIZES[0], directions, depths, did_return, range_margin=range_margin)
    assert np.array_equal(actual, expected)
