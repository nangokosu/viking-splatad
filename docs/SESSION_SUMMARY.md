# Session summary

Date: 2026-11-12

## Done

- Checked current Slurm jobs and diagnostics. Source `da76f8d` remains frozen for
  the loss-only pilot; no training/config changes this session.
- GPU probe 243986 succeeded: full-scene support passes 1.399/0.725 s,
  index construction 3.952 s. Controller 243991 applied measured caps of 8 h 30 m
  smoke / 21 h 43 m full and released training successfully.
- Smoke 243989 completed successfully: train 8,113 s, plot written, camera/LiDAR
  renders, Gaussian export and matching split; end 2026-11-12T08:17:06Z.
  Confirmed zero added radar Gaussians, initial weight 0.0036067375 and nonempty
  support. Background fetch had started; a resumable fetch was restarted to
  ensure all result files arrive.
- Restarted one background monitor/fetch for the remaining full job. Previous
  implementation/handoff commits: `da76f8d`, `bba9fa1`.

## Current state

- Full loss-only job 243990 runs on gpunode3, RTX A6000, 24 CPUs/109 GiB.
  Latest observed coverage step 18,400/20,001 (~92%), 97% GPU utilization,
  11,220 MiB GPU memory. Nonempty support: 4,457,820 Gaussians / 89.01% anchors.
  Training log ETA ~45 min at this check, then plotting/render/export/split gates.
  No full L result or radar benefit established yet.
- Full run: `june_tall_grass-june_loss_only-20261112-012542-da76f8d`.
  Smoke: `june_tall_grass-june_loss_only_smoke-20261112-012541-da76f8d`.
  Baseline smoke/full successful and fetched. Raw data and immutable 201,471-anchor
  artifact remain on cluster workdisk; no raw bags on Mac. Recipe unchanged:
  `splatad-3f478d42f203`. Final native source validation/preflights passed 1,177
  tests, one allowed skip/eight configured deselections under local-Docker exception.
- Scope remains B versus loss-only L, stock initialization, no radar rendering.
  Static persistent occupancy selector is not a semantic trunk detector.
  Hidden-trunk attribution, Gaussian occupancy/false-positive evaluation and
  repeat-aware scientific comparison remain unfinished. Commits remain local.

## Next steps

1. Reconcile `submit_run.py runs`; ensure one background wait for active full job
   and finish smoke fetch. Runtime watch/spec/log files are ignored under
   `data/cluster-validation/`.
2. On full completion verify plot/render/export/split/end milestones, fetch
   automatically, and inspect loss/coverage curves and held-out renders/metrics.
3. Complete and validate hidden-trunk occupancy/visibility/false-positive metrics
   with matching views and GT used only for evaluation.
4. Fix matched B/L repeat budget/seeds before comparing between-arm improvements
   against within-arm variability. The current pair is a pilot, not repeat evidence.
