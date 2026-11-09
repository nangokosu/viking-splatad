# Session summary

Date: 2026-11-09

## Done

- Native environment recovered with complete fixtures and strict skip audit.
  GPU verification 240770: 1,136 passed; final source CPU job 241031: 1,143
  passed, one allowed traceability skip, eight configured deselections.
  Local Docker tests omitted at the user's request; no existing tests weakened.
- `ab9c275`–`f31cc47`: workdisk-only validation/training, reviewed provenance
  receipts, core B lidar-only launch, smoke dependency, recoverable job settings.
- `fe31e22`: streamed contract validation and recovery of complete extraction.
  Data job 240771 passed contract/reference split and published June data.
- `c36540d`, `0097a66`: validation resource routing, CPU source checks after GPU
  verification, exclusion of preflight jobs from training-run tracking.
- Preflights 241032/241033 passed on `0097a66`; all ten overlays reviewed with
  no obvious gross registration error. Submitted smoke 241036 and full B 241037
  (afterok smoke). Dated native locks and remote workflow documented.

## Current state

- Cluster workdisk holds raw June chunks 20–22, validated `data/june_tall_grass`,
  compiled `splatad-3f478d42f203`, immutable code, and training run directories.
  Canonical June reference split matches; no raw data downloaded to the Mac.
- Both training runs use `0097a66`: smoke A4500 pending resources; full B A6000
  pending smoke success. Mac watcher tool session 76296 waits for both and
  fetches only after successful plot, split and completion milestones.
  Runtime run IDs and evidence are in gitignored `data/cluster-validation/`.
- Radar arm is not ready: calibrated seed builder, Gaussian append hook,
  provenance/contract gates and measured loss-weight calibration remain missing.
  Radar is never rendered; evaluation cuboids remain evaluation-only.
- Native pip locks match the historical record; conda's unpinned dependencies
  differ. New dated locks preserve the actual environment. No recipe changed.

## Next steps

1. Run `submit_run.py runs`; re-establish one watcher if the previous tool session
   ended. Read logs/milestones after queue exit; fetch success, diagnose failures.
2. Confirm smoke completes and full B starts when its A6000 allocation is granted.
3. Finish calibrated, label-free radar initialization/gates/weight calibration,
   validate and smoke R, then run repeated matched B/R experiments.
