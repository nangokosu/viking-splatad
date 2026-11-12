# Session summary

Date: 2026-11-12

## Done

- User clarified that running training is insufficient: the experiment must
  answer whether radar loss strengthens weak trunks missed by B. Implemented
  the missing matched P1/P2 occupancy/evidence comparison and per-trunk figures
  in `8c8aa77` (script 17, T57/T78, new analytic tests). Extended only the frozen
  transform manifest; no existing assertions/tolerances were weakened.
- Native validation 244543: 1,185 passed, one allowed traceability skip, eight
  configured deselections. Alignment job 244544 prepared June-only eval GT and
  overview: 175 scored trunks, voxel edge 0.5219 m from existing GT rule. Coarse
  XY correspondence reviewed; exact label/frame correspondence is not certified.
- Input diagnostic 244551 found zero B/full-window-L-smoke observation mismatches,
  but 263,332 invalid float32 covariance exports in full B (five million Gaussians).
  Held comparison 244546 before execution; it must NOT be released.
- Fixed evaluation to re-export full-precision checkpoint axes/scales separately
  and evaluate the same Gaussian without covariance inversion/repair (`c48e60d`),
  plus both stock/radar trainer directory layouts (`f43e055`). New tests added.
  Neither trained models nor successful run exports were changed. These fixes
  are committed but NOT yet native-validated: SSH expired during submission.

## Current state

- Latest verified full L progress: job 243990, gpunode3, coverage step 19,400/20,001,
  nonempty support ~89.03% of anchors. Do not infer its current state from expired
  SSH. Baseline runs and L smoke succeeded. L smoke fetch was restarted; full
  monitor/fetch is configured in ignored `data/cluster-validation/`.
- Training source remains `da76f8d`; env `splatad-3f478d42f203` unchanged. Baseline
  source `0097a66`. Same canonical June split, LiDAR loss, camera-painted stock
  initialization, no radar Gaussians or rendering. No measured radar benefit yet.
- Corrected evaluation source is frozen at `f43e055`. A continuation script/log in
  ignored `data/cluster-validation/finish_hidden_evaluation.py` /
  `hidden-evaluation-watch.log` waits for authenticated SSH, validates that exact
  source, submits replacement comparison after full training success, then fetches
  reports and per-trunk images. It refuses failed validation or missing reports.
  The user was asked to restore SSH via Terminal; no password is stored.
- GT outputs are evaluation-only under cluster `evaluation-inputs/`; diagnostics
  under `evaluations/hidden-trunks/8c8aa77.../alignment`. Small alignment image/JSON
  fetched to `data/evaluation-results/hidden-trunks/alignment/`. Raw data remain
  remote. Docs: `docs/research/hidden_trunk_evaluation.md`. All commits local.

## Next steps

1. Restore authenticated SSH; reconcile runs and inspect continuation/watch logs.
   Confirm native validation of `f43e055` before corrected evaluation submission.
   Original 244546 is held/superseded; it reserves no compute resources.
2. Verify full-run completion milestones and fetch automatically. Inspect fixed
   comparison logs/results and label correspondence in per-trunk views; do not
   interpret numerical results while alignment remains unreviewed.
3. Report B/L P1/P2 recall on blocked/no-return GT cells alongside precision loss,
   off-trunk cells, unknown/conflicting evidence, and pictures of all scored trunks.
   The new metric is occupancy in a frozen trunk neighbourhood, not a global
   unmatched-tree detector; blocked does not independently mean grass, and camera
   visibility is unclassified. One pilot cannot establish repeatable benefit.
4. Fix matched repeat budget/seeds, complete unmatched-trunk false-positive and
   sensitivity analysis, and compare arm gaps with within-arm variation.
