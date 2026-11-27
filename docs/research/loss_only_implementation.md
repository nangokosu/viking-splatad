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

### Measured runtime release

`benchmark_radar_support.py` runs inside a short A6000 Slurm allocation. It uses
the completed B run's exported map-frame Gaussians and current map-frame anchors
to measure index construction and two full-scene gate passes. These are workload
measurements only: B's trained geometry never initializes L or sets its loss weight.
The prediction retains the entire measured full B walltime, adds the slower gate
pass times the profile's iteration count and one-time index construction, then
applies the allocation headroom in `configs/slurm.yaml`. Requests are rounded up
to minutes and refused above the cluster's five-day cap.

Submit reviewed smoke with `submit-remote --hold-for-budget --afterok <probe-job>`
and full with `--afterok <smoke-job>`. A small CPU control job, dependent on probe
success, runs `release_radar_budget.py --specification <runtime-json>`. Its runtime
specification records source revision, stage root, benchmark report and both
job/run/profile identities. The controller verifies the exact pending jobs,
updates full then smoke, writes their `allocation.json`, and releases smoke last.
A missing/failed probe, wrong source or excessive budget leaves smoke held.
This requires no Mac session to perform the release; monitoring/fetching uses the Mac.

### November 11 launch

Source `da76f8d`: final native validation and both preflights each passed 1,177
tests, one allowed no-pending-transform skip and eight configured GPU/real-data
deselections. Contract and canonical split passed. All five camera overlays and
the radar/LiDAR BEV were inspected; the two profiles' views were byte-identical.
These are coarse alignment checks, not independent clock or reconstruction GT.
The published artifact contains 201,471 anchors from 3,328 training radar sweeps.

The first L launch is a **pilot**, not a repeated scientific comparison:
GPU probe 243986; budget controller 243991; smoke 243989; full 243990.
Both training jobs use A6000 allocations sized from live node reports (24 CPUs,
109 GiB). At submission the probe waited for resources, smoke was intentionally
held and full waited on smoke. No L training result exists yet. Baseline jobs
241036/241037 were successful and fetched; full B took 11,072 seconds walltime.

## Scientific evaluation and remaining scope

The full L pilot and factor-based P1/P2 evaluation are now complete and fetched.
The evaluation implements recorded-ray seen/blocked/no-return/passed-through/
unobserved evidence, common-cell occupancy, neighbourhood precision and matched
views of all 175 scored June trunks. Float32 covariance export is unsuitable for
thin full-B Gaussians; evaluation therefore re-exports stable checkpoint factors
without changing the trained model or its legacy archive.

Two runs per arm are the initial repeat budget. Additional B/L jobs 244827/244828
are running; comparison 244830 and repeat-aware summary 244855 are success-dependent.
The tested summary compares between-arm gaps against within-arm spread and refuses
a single pair. (The hidden-trunk evaluation note and scripts 17 and 18 were removed 2026-11-27; see git history.)

No radar improvement has been established from the first pair alone. Global
unmatched-trunk false positives, camera occlusion labels, finer-grid sensitivity
and a larger statistical repeat study remain beyond this initial check. A nearer
LiDAR return is geometric blockage, not independently labeled grass.
