"""Why was P2 recall low on seen trunk cells? Compare T57 centre occupancy with where opaque Gaussians actually sit.

Run 2026-11-16 on runs/may_short_grass-may_scene_a-20261116-133928-17bb7a1 (output: seen_cell_diagnostic_output.json):
    uv run python docs/research/p2_estimator_diagnostic/seen_cell_diagnostic.py \
        runs/<run>/eval/gaussian_occupancy_cells.npz <factors.npz from scripts/10 --factorized> 0.5
T57 is recomputed here on the seen cells' centres, so the result does not depend on which estimator
wrote the archive's own occupancy column.
Prints, over seen trunk GT cells: fraction with T57 occupancy >= threshold, fraction containing the mean of
>= 1 Gaussian with opacity >= threshold, the max per-cell opacity of contained Gaussians (combined 1-prod(1-o)),
and quantiles of the smallest axis std of those Gaussians vs the voxel edge.
"""
import json
import sys

import numpy as np

from viking_splatad.eval.hidden_geometry import gaussian_occupancy_at_centres

# T57 working batch, memory only (configs/eval.yaml gaussian_occupancy_chunk_size).
CHUNK_SIZE = 4096
# Opacity clamp below 1 so log1p(-o) stays finite; float64 sigmoid can round to exactly 1.
OPACITY_CEILING = 1 - 1e-12

cells = np.load(sys.argv[1])
factors = np.load(sys.argv[2])
threshold = float(sys.argv[3])
voxel_size, origin = float(cells["voxel_size"]), cells["grid_origin"]
trunk_cells = cells["neighbourhood_tree_trunk"]
evidence = cells["evidence"]
# Exclusive label seen = column 0 set (seen has top priority in exclusive_evidence_labels).
seen_truth = cells["truth_tree_trunk"] & trunk_cells & evidence[:, 0]
# Map every Gaussian mean to its flat voxel index on the GT grid (floor((p - origin) / edge)).
grid_extent = cells["cell_indices"].max(axis=0) + 1
gaussian_index = np.floor((factors["means"] - origin) / voxel_size).astype(np.int64)
inside = np.all((gaussian_index >= 0) & (gaussian_index < grid_extent), axis=1)
gaussian_keys = np.ravel_multi_index(gaussian_index[inside].T, grid_extent)
opacities, min_std = factors["opacities"][inside], factors["axis_std"][inside].min(axis=1)
cell_keys = np.ravel_multi_index(cells["cell_indices"].T, grid_extent)
seen_keys = cell_keys[seen_truth]
in_seen = np.isin(gaussian_keys, seen_keys)
# Per-cell combined opacity of the Gaussians whose mean lies in it: 1 - prod(1 - o).
order = np.argsort(gaussian_keys[in_seen])
sorted_keys, sorted_opacity = gaussian_keys[in_seen][order], opacities[in_seen][order]
unique_keys, starts = np.unique(sorted_keys, return_index=True)
survival = np.add.reduceat(np.log1p(-np.minimum(sorted_opacity, OPACITY_CEILING)), starts)
combined = -np.expm1(survival)
print(json.dumps({
    "seen_trunk_cells": int(seen_truth.sum()),
    "t57_centre_occupied_fraction": float((gaussian_occupancy_at_centres(
        cells["centres"][seen_truth], factors["means"], None, factors["opacities"], CHUNK_SIZE,
        principal_axes=factors["principal_axes"], axis_std=factors["axis_std"]) >= threshold).mean()),
    "cells_with_any_gaussian_mean_fraction": float(len(unique_keys) / seen_truth.sum()),
    "cells_with_mean_binned_combined_opacity_ge_threshold_fraction": float((combined >= threshold).sum() / seen_truth.sum()),
    "gaussians_with_mean_in_seen_cells": int(in_seen.sum()),
    "voxel_edge_m": voxel_size,
    "min_axis_std_quantiles_m(10,50,90)": np.quantile(min_std[in_seen], [0.1, 0.5, 0.9]).tolist(),
    "max_axis_std_quantiles_m(10,50,90)": np.quantile(factors["axis_std"][inside][in_seen].max(axis=1), [0.1, 0.5, 0.9]).tolist(),
}, indent=1))
