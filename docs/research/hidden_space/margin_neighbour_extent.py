"""Diagnostic (run from the repo root: uv run python docs/research/hidden_space/margin_neighbour_extent.py <session>): does the T92 range margin depend on how far the plane-fit neighbours reach?

Recomputes the June margin pairs exactly as scripts/24 does, but records for every kept residual
the distance to its farthest neighbour, then reports the 0.95 quantile of residuals for all points
and for points whose neighbours all lie within a bound.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial import cKDTree

from viking_splatad.eval.hidden_settings import parse_hidden_settings
from viking_splatad.eval.range_margin import cross_sweep_plane_residuals, points_within_range, sweep_pair_indices
from viking_splatad.eval.session_rays import iter_session_sweeps

SESSION = sys.argv[1]
NEIGHBOUR_REACH_BOUNDS_M = (0.25, 0.5, 1.0, 2.0)
spec = importlib.util.spec_from_file_location("split_check", "scripts/16_check_reference_split.py")
split_check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(split_check)
settings = parse_hidden_settings(yaml.safe_load(open("configs/eval.yaml"))["eval"])
sweeps = list(iter_session_sweeps(Path("data/intermediate") / SESSION, split_check.session_split_settings(SESSION)))
pairs = sweep_pair_indices(len(sweeps), settings.margin_sweep_stride, settings.margin_sweep_gap)


def to_map(points, pose):
    return points @ pose[:3, :3].T + pose[:3, 3]


all_residuals, all_reach = [], []
for first_index, second_index in pairs:
    first, second = sweeps[first_index], sweeps[second_index]
    near = points_within_range(first.returns_xyz, settings.margin_max_range_m)
    points_a = to_map(first.returns_xyz[near], first.lidar_to_map)
    points_b = to_map(second.returns_xyz, second.lidar_to_map)
    # Same neighbour search T92 uses; keep the farthest-neighbour distance per point.
    reach, _ = cKDTree(points_b).query(points_a, k=settings.margin_neighbour_count)
    farthest = reach[:, -1]
    # T92 itself is then rerun on the subset of points whose neighbours all lie within each bound.
    all_residuals.append((points_a, points_b, farthest))

print("pairs", len(pairs))
for bound in (np.inf,) + NEIGHBOUR_REACH_BOUNDS_M:
    residuals = []
    kept_points = 0
    for points_a, points_b, farthest in all_residuals:
        subset = farthest <= bound
        kept_points += int(subset.sum())
        residuals.append(cross_sweep_plane_residuals(points_a[subset], points_b, settings.margin_neighbour_count,
                                                     settings.margin_planarity_min))
    residuals = np.concatenate(residuals)
    print(f"neighbour reach <= {bound} m: points {kept_points}, residuals {residuals.size}, "
          f"q{settings.margin_quantile} = {np.quantile(residuals, settings.margin_quantile):.4f} m, "
          f"median = {np.median(residuals):.4f} m")
