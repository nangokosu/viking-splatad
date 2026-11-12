# Session summary

Date: 2026-11-11

## Done

- Implemented the confirmed June B-versus-L loss-only ablation. Stock Gaussian
  creation is inherited unchanged; radar is fixed train metadata, never rendered.
  Main commits: `34f9a5c` / `dff9cb2` calibrated artifacts, parser/model/profile
  integration; `512efd6` / `bd58994` measured noise and train-safe clock domain;
  `d00fce2` exact device-side support index; `405fd1c` encoded-precision rotation
  validation; `da76f8d` measured GPU budgets and automatic held-job release.
- Restored cluster access, verified baseline smoke 241036 and full B 241037
  successful, and fetched both outputs/checkpoints. Full B walltime: 11,072 s.
- On `da76f8d`, native validation 243982 and preflights 243984/243985 each passed
  1,177 tests, one allowed traceability skip and eight configured deselections.
  Contract and canonical split passed. Inspected all five camera overlays and
  radar BEV; profile views were byte-identical. This is native-cluster evidence
  under the user's local-Docker exception, not reference-image testing.
- Radar preparation 243983 published 201,471 anchors from 3,328 training sweeps,
  with measured clock/uncertainty, source hashes and exact training membership.
  Scope/limits recorded in `docs/research/loss_only_implementation.md`.

## Current state

- Published June data/raw sidecars and immutable source/artifacts stay on the
  cluster workdisk. Baseline outputs are fetched locally; no raw bags downloaded
  to the Mac. Recipe env unchanged: `splatad-3f478d42f203`.
- Pilot source is frozen at `da76f8db848d54788c27729891a881f2c0c92595`.
  Probe 243986 waits for A6000 resources. CPU controller 243991 depends on probe
  success; it applies measured time budgets to both pending runs and releases
  held smoke 243989. Full pilot 243990 depends on smoke success. Both training
  allocations use live-measured 24 CPUs/109 GiB. Neither has trained yet.
- Run IDs:
  `june_tall_grass-june_loss_only_smoke-20261112-012541-da76f8d` and
  `june_tall_grass-june_loss_only-20261112-012542-da76f8d`.
  Runtime launch/probe specifications and preflight images are in ignored
  `data/cluster-validation/`. A Mac background wait/fetch was started for both;
  restart it next session. The Slurm release chain runs independently of the Mac.
- Selector is static persistent occupancy, not a semantic trunk detector.
  Ground/canopy/sparsity refinements and complete hidden-trunk evaluation remain
  unfinished. Clock posterior uncertainty/coarse alignment are not independent
  timing GT. There is no demonstrated radar improvement or repeated comparison.
- Existing test assertions/tolerances were not weakened. Only the frozen
  transform-manifest fixture was extended to register the new transforms.
  Commits are local; no push was performed.

## Next steps

1. Run `submit_run.py runs`; restart one wait/fetch for active jobs. Inspect GPU
   probe/controller reports, time amendments and job logs. Failure leaves smoke
   held; diagnose rather than releasing it manually or guessing a loss weight.
2. Verify smoke initialization/weight/coverage diagnostics, progress, checkpoints
   and completion milestones. Full is already success-dependent; fetch success.
3. Complete and validate Gaussian occupancy, seen/blocked/no-return attribution,
   false-positive trunk analysis and metric-matched pictures. Keep GT eval-only.
4. Fix a matched B/L repeat budget/seeds, then run the scientific comparison on
   the unchanged June split. Compare between-arm gaps with within-arm spread;
   report unsupported anchors and negative outcomes. No initialization extension.
