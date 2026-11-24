"""(Run from the repo root: uv run python docs/research/hidden_space/range_distribution.py <session>.)
Distribution of real Ouster return ranges over every sweep of a session (diagnostic for the scored range limit)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np

from viking_splatad.eval.session_rays import iter_session_sweeps

SESSION = sys.argv[1]
PERCENTILES = (50.0, 99.0, 99.9, 99.99, 99.999, 100.0)
spec = importlib.util.spec_from_file_location("split_check", "scripts/16_check_reference_split.py")
split_check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(split_check)
settings = split_check.session_split_settings(SESSION)
per_sweep_max, all_ranges = [], []
for sweep in iter_session_sweeps(Path("data/intermediate") / SESSION, settings):
    ranges = np.linalg.norm(sweep.returns_xyz, axis=1)
    per_sweep_max.append(ranges.max())
    all_ranges.append(ranges.astype(np.float32))
all_ranges = np.concatenate(all_ranges)
per_sweep_max = np.asarray(per_sweep_max)
print("returns", all_ranges.size, "sweeps", per_sweep_max.size)
print("all-return range percentiles", dict(zip(PERCENTILES, np.round(np.percentile(all_ranges, PERCENTILES), 2))))
print("per-sweep max percentiles", dict(zip(PERCENTILES, np.round(np.percentile(per_sweep_max, PERCENTILES), 2))))
for limit in (60.0, 80.0, 100.0, 150.0):
    print(f"returns beyond {limit} m:", int((all_ranges > limit).sum()), "sweeps with max beyond:", int((per_sweep_max > limit).sum()))
