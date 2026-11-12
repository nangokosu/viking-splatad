# June radar loss-only implementation

Scope confirmed 2026-11-11: compare B with L, identical stock Gaussian
initialization. `radar_init/` and `radar_seeds.npz` are legacy names for the
offline supervision artifact; neither adds any model Gaussians.

## Implemented path

`prepare-radar` uses a CPU Slurm job, preserves baseline inputs, and publishes
an immutable artifact under `radar-inputs/<session>/<anchor-profile>/<revision>`.
Only training radar observations contribute to clock fitting, selection or targets.
The fitter tests both Doppler signs, uses measured residual/quantization noise,
and derives its admissible search domain from actual training-time corridors.
Every fit observation must stay training throughout the candidate search box.
Unidentifiable, boundary or uncertain fits fail; zero offsets are never substituted.

The source paper's range/angular resolutions are recorded in
`configs/radar/loss_only.yaml` with its [source](https://arxiv.org/html/2606.19154v1#S3.SS1).
The covariance assumes **uniform full-resolution cells**, not that resolution is
standard deviation. The clock-fit posterior timing uncertainty adds translational
uncertainty. The legacy report field `residual_rms_seconds` is explicitly tagged
as RMS posterior timing standard deviation, not independent clock ground truth.

Static returns are pooled by distinct radar sweeps into persistent voxels. Cell
sizes come from observed uncertainty; the persistence operating point is configured.
This is **static persistent occupancy**, not a semantic trunk detector. No GT
boxes, camera segmentation labels, support-based target filtering or Gaussian
initialization are used. Trunk-specific ground/canopy/sparsity refinements remain
separate unfinished work; do not describe this selector as a trunk oracle.

The loader verifies source/data/artifact hashes, frame/time/session tags, covariance,
support identities, exact profile split and training membership. It transforms
means/covariances/origins, crops the window and scene box, and reports retained counts.
Stored float32 rotations are checked at their input precision, without projecting
or rescaling them. Explicit non-unit scale and reflections remain errors.

The model inherits stock Gaussian creation, trainer schedules and all stock model
fields. Its only new persistent tensor is the non-learned calibrated loss weight.
It records initial parameter hashes and zero added radar Gaussians. The hard
ellipsoid union gate is evaluated at current Gaussian means on every LiDAR step.
An immutable sparse anchor-cell index accelerates lookup on the model device;
it never caps neighbors or changes the final Mahalanobis predicate.

For supported static Gaussians S:

$$L = L_B + \lambda\,|S|^{-1}\sum_{k\in S}\operatorname{softplus}(-o_k).$$

At step zero, lambda equals actual stock weighted opacity regularization divided
by the initial unweighted radar term. Missing support or a zero/invalid term is
an explicit failure, not a guessed weight. Subsequent empty support returns zero
and is logged. This loss changes opacity, not Gaussian locations or count.

`radar_diagnostics.jsonl` records initialization, calibration and periodic coverage.
The weight is in checkpoint state so loading cannot silently recalibrate it.

## Run from the Mac

After committing and validating the current revision:

```bash
uv run python scripts/slurm/submit_run.py verify-env --resources validation_cpu
uv run python scripts/slurm/submit_run.py prepare-radar --session june_tall_grass --profile june_loss_only
uv run python scripts/slurm/submit_run.py remote-preflight --session june_tall_grass --profile june_loss_only_smoke --resources validation_cpu
uv run python scripts/slurm/submit_run.py remote-preflight --session june_tall_grass --profile june_loss_only --resources validation_cpu
```

These are sequential gates. Use `--afterok` with the returned successful job
dependency to queue the next stage; a submission message is not a passing result.
CPU validation requires an already GPU-verified recipe environment and never
creates readiness markers. This session uses native validation under the user's
explicit local-Docker exception; it is not reference-image test evidence.

The 15-second L profile is not the selected smoke: real timing calibration there
failed its uncertainty gate. `june_loss_only_smoke` uses the full canonical June
window with the existing short iteration budget on an A6000. It shares the full
L artifact and retains the full held-out evaluation cadence. Its time budget
reuses the full allocation because the measured B run's full-scene evals are costly.

Review the generated camera/LiDAR overlays and `radar_lidar_bev.png` before using
`submit-remote --reviewed-preflight`. In the BEV, gray is sampled training LiDAR,
orange is radar anchors and white is shared XY pixels; none is a GT accuracy label.
Measure device-side gate time/memory before the full run. The earlier CPU-tree
benchmark took roughly 30 s on five million Gaussians/201,471 anchors, motivating
device-side cell lookup. It is not the timing of the current GPU implementation.

## Scientific result requirements still outstanding

A running loss is not a detection result. Complete and validate P2 Gaussian
occupancy, seen/blocked/no-return attribution, false-positive trunk analysis and
matching views. Compare repeated B/L runs on the same June GT/frame record.
No radar-vs-B hidden-trunk improvement has been established by implementation,
input alignment, support coverage or benchmark timing alone.
