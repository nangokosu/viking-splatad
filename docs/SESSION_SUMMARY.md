# Session summary

Date: 2026-11-12

## Done

- Restored SSH, verified full L 243990 successful (train 35,147 s; camera/LiDAR
  render, Gaussian export, matching split/end), and fetched it completely.
  All four original baseline/L smoke/full runs are now successful and fetched.
- Corrected factor-based evaluation source `f43e055` passed 1,189 native tests
  (244825). First comparison 244829 completed and its report, cell archive and
  all 175 trunk pictures were fetched. No first-pair performance conclusion read.
- Added repeat summaries/consistency maps (`ee286ce`), clarified a guard error
  message without weakening its rejection (`d6b85d1`), and added exact-source
  external alignment-review provenance (`394db6a`). Final validation 244835:
  1,194 passed, one allowed traceability skip, eight configured deselections.
  Native validation uses the standing local-Docker exception, not host pytest.
- Reviewed GT correspondence in real-LiDAR XY/XZ views and five original held-out
  camera observations with T25-projected June boxes. No transform fitted/applied.
  A separate exact-hash receipt records this visual check and its calibration/
  annotation-accuracy limits; original numerical reports remain unchanged.
- Fixed source staging recovery after local git-archive stalls: remote snapshot
  copied from verified tracked source, every mode/blob/path compared with the
  exact commit before native validation. No recipe/dependency/kernel changes.

## Current state

- Initial budget: two runs per arm with identical per-arm pilot sources/settings
  and configured seed. B repeat 244827 runs on gpunode2; L repeat 244828 on
  gpunode3. Both A6000, live-sized 24 CPUs/109 GiB. L reuses measured 21 h 43 m
  budget; B retains six hours. Latest L repeat coverage step 2,200, nonempty
  support 86.37%; initialization fingerprint matches first L and adds zero Gaussians.
- Repeat run IDs:
  `june_tall_grass-june_scene_a-20261112-222351-0097a66`;
  `june_tall_grass-june_loss_only-20261112-222404-da76f8d`.
- Comparison 244830 (source `f43e055`) depends on both repeats. Final summary
  244855 (source `394db6a`) depends on that comparison and uses the documented
  alignment review. It refuses single/reused runs or differing cells/observations,
  recomputes reported metrics, and reports gaps versus spread plus matched maps.
- One repeat-training wait/fetch and one comparison/summary wait/fetch were started.
  Runtime records/logs are ignored under `data/cluster-validation/`; restart waits
  next session. Remote dependencies run independently of the Mac. Old comparison
  244546 is deliberately held/superseded; never release it (rounded covariances).
- Raw data/GT/factor archives stay on workdisk. First comparison is locally in
  `data/evaluation-results/hidden-trunks/f43e055.../comparison/`; reviewed camera
  overlays in `data/evaluation-results/hidden-trunks/label-camera-alignment/`;
  review receipt is `june-label-alignment-review.json` in that evaluation root.
- No radar benefit established yet. False-positive measure is local off-trunk
  occupancy, not global unmatched-tree counts. Blocked is geometric, not a grass
  semantic label; camera visibility unclassified; GT voxel edge 0.5219 m is coarse.
  Two runs per arm are an initial descriptive check, not a powered significance test.
  Training source/env unchanged (`0097a66` B, `da76f8d` L,
  `splatad-3f478d42f203`). Commits local; no push.

## Next steps

1. Run `submit_run.py runs`, inspect repeat progress, restart one monitor/fetch
   per batch. Automatically fetch successes; diagnose failures before cleanup.
2. Inspect 244830 and 244855 outcomes; fetch reports/pictures and verify summary
   review provenance. Avoid restarting the old submission continuation or
   duplicating already queued training/evaluation jobs.
3. Read repeat-aware P1/P2 blocked/no-return recall and precision together. Compare
   arm gaps with within-arm spread, stable recovered/lost/off-trunk cells and the
   per-trunk pictures; report tradeoffs, no gains and unsupported cases honestly.
4. If needed, expand repeat budget and complete global phantom-trunk counting and
   finer-grid sensitivity before broader research claims. No initialization,
   photometric or radar-depth extension without approval.
