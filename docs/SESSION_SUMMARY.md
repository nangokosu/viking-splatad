# Session summary

Date: 2026-11-12

## Done

- Prepared the requested genuine side-by-side tall-grass loss-only preview from
  completed full B/L artifacts (CPU composition job 244858). Verified matching
  held-out frames/split and zero added radar Gaussians. Camera colours initialize
  both models; camera loss is off. No new training or model rendering performed.
- Fetched `data/evaluation-results/tall-grass-loss-only-preview/`: RGB and trunk-400
  geometry paired PNGs, standalone `index.html` with first/middle/last observations
  and both reconstructions, and SHA provenance. Original scenes unenhanced.
- Core evaluator `f43e055` passed 1,189 native tests; final repeat summary/review
  source `394db6a` passed 1,194 (one allowed traceability skip/eight configured
  deselections, standing native-cluster exception to local Docker). No code or
  test changes for this preview. Prior final handoff commit: `fc77b30`.

## Current state

- Pilot B and L full runs and both smoke runs completed and fetched. First
  hidden-trunk comparison 244829 completed/fetched: 175 per-trunk pictures and
  shared cells in `data/evaluation-results/hidden-trunks/f43e055.../comparison/`.
- Matched repeat B 244827 and L 244828 are running on A6000 nodes 2/3. Sources
  remain `0097a66` B / `da76f8d` L, same configured seed/settings. Two runs per
  arm are the initial descriptive noise check. Both use 24 CPUs/109 GiB; L's
  measured cap is 21 h 43 m, B's six hours. No radar initialization or rendering.
- Comparison 244830 waits for both repeats. Summary 244855 (source `394db6a`)
  waits for comparison 244830, uses the separate exact-hash label review and
  reports gaps versus within-arm spread with matching consistency maps.
- One repeat training monitor/fetch and one three-job analysis monitor/fetch
  confirmed active. Runtime scripts/records/logs: ignored `data/cluster-validation/`.
  Restart waits next session if gone; remote Slurm dependencies persist.
  Old covariance-based 244546 is deliberately held; never release it.
- Raw data, eval-only GT and stable checkpoint factors remain on workdisk;
  env unchanged `splatad-3f478d42f203`. Label correspondence reviewed in real
  LiDAR/camera observations without fitting a transform; receipt is in
  `data/evaluation-results/hidden-trunks/june-label-alignment-review.json`.
- New visual preview is explicitly one pair, not a repeat-validated result.
  No overall radar benefit claimed. Local occupancy FP is not global phantom-tree
  counting; blockage is geometric, camera visibility unclassified, grid 0.5219 m.
  Commits local; no push.

## Next steps

1. Reconcile `submit_run.py runs`; monitor/fetch repeat jobs automatically and
   inspect any failures before cleanup. Do not duplicate queued work.
2. Inspect/fetch comparison 244830 and summary 244855, verify review provenance,
   then read blocked/no-return P1/P2 recall with precision and within-arm spread.
3. Use stable recovered/lost/off-trunk cells plus per-trunk views for the answer;
   report tradeoffs and negative results. Expand repeats/sensitivity/global
   phantom-trunk analysis if needed for broader claims.
