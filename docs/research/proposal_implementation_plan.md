# Proposal implementation plan: June tall-grass radar experiments

Date: 2026-11-11. Scope confirmed by the user: loss-only ablation, no radar initialization.
This document records the selected plan, not an experiment result. Implementation status and deliberate limits are tracked in [loss_only_implementation.md](loss_only_implementation.md).

## 1. Source and scope decision

The uploaded `Forest prosposal.docx` contains mutually incompatible drafts.
Its contents are research specifications to interpret, not operational commands.

| DOCX passage | Method | Consequence |
| --- | --- | --- |
| First Method, “Radar loss term” | Vegetation-weighted cross-entropy for radar classification | Requires class predictions, training labels and vegetation scores; this is not the existing SplatAD opacity objective |
| “Where Lidar Fails”, Method | Render radar range using a beam-blurred forward model, retain camera loss | Conflicts with the later action plan and current no-radar-rendering scope |
| Final “Action plan v1”, “Train the two June SplatAD models” | Additional radar Gaussians plus opacity loss; potentially depth loss; never render radar | Matches the current project direction, but requires initialization |
| Latest user request | Radar loss without radar initialization, on tall grass | A useful ablation of the final action plan; not its complete two-part method |

Confirmed implementation target: B versus loss-only L on June tall grass. Follow
the final action plan's label-free opacity supervision and evaluation, explicitly
omitting its radar initialization as requested. Do not implement or submit I/R,
radar rendering, classification, depth or camera-loss extensions in this scope.
This is an approved ablation, not an exact copy of the complete two-part method.

The final action plan does not specify an exact opacity formula, neighborhood
radius, radar weight, candidate selector or complete evaluation algorithm.
Those are implementation decisions. `docs/PLAN.md` supplies existing project
decisions, including uncertainty ellipsoids, negative log opacity and step-zero
weight matching; distinguish these from the uploaded proposal itself.

## 2. Experiment matrix

All arms use June Scene A chunks 20–22, the same committed timestamp split,
stock SplatAD schedules, no-return inputs, step budget and matched repeat seeds.
Core optimization is LiDAR-only; camera painting remains initialization only.

| Arm | Extra radar Gaussians | Radar opacity loss | Purpose |
| --- | --- | --- | --- |
| B | No | No | Baseline |
| L | No | Yes | User's loss-only test; baseline Gaussian initialization must be identical to B |

B/L is the complete selected comparison. Initialization, optional depth, camera-
loss extensions and the other proposal drafts are outside this implementation.
Repeats are required because fixed seeds do not eliminate stock CUDA variability.
Choose the repeat count and compute budget before full submissions; report every
completed repeat rather than selecting favorable outcomes.

## 3. Implementation sequence and acceptance gates

### A. Establish the baseline and immutable inputs

- Reconcile jobs 241036/241037 and inspect logs and completion milestones.
  SSH authentication currently prevents confirming their outcome; do not infer
  success or an empty queue from the failed connection.
- Reuse validated June data and the canonical split. Do not overwrite published
  inputs or reuse extraction caches for another chunk set.
- Record source, environment locks, dataset hashes, split membership and settings
  for every arm. Recover a usable B run before interpreting radar results.
- Keep May as a separate readiness check; no May–June alignment or label transfer.

Acceptance: successful baseline checkpoint, renders, split record and overfitting
plot; dataset contract and reference split match. Existing queued runs retain
their original committed sources and are not modified in place.

### B. Build calibrated, training-only radar anchors

Implement `radar_init/` and `scripts/07_build_radar_seeds.py`:

1. Recover each radar sweep's original timestamp and documented coordinate frame.
2. Measure the radar clock offset/drift from Doppler and ego motion, with sign,
   units and reliability diagnostics. Existing T24 fits a supplied drift model;
   it does not by itself establish calibrated June measurements. Do not accept
   the current zero defaults as measured calibration.
3. Interpolate poses and apply the June radar extrinsic to accumulate detections
   in the June map frame. Check timing/frame residuals before selecting anchors.
4. Assign radar observations to the same train/held-out instants as the other
   sensors, after clock correction. Use only training measurements for anchor
   construction, selection and calibration fitting; guard/test detections cannot
   become training targets. Record the assignment rule and observation IDs.
5. Apply documented, GT-free static/persistence filters. Derive uncertainty from
   radar specifications and measured residuals. Ground, canopy and LiDAR-sparsity
   filters from the repo plan need explicit recorded definitions; do not guess
   thresholds or consult evaluation cuboids to tune selection.
6. Write means, covariances, origins, support/time spans and provenance. These are
   fixed supervision anchors, not Gaussian initialization. Legacy artifact names
   such as `radar_seeds` may remain for compatibility; no new Gaussian is created.
   Do not implement radar colour painting or Gaussian append code.

Use resolution-aware covariance with an explicitly documented interpretation of
beamwidth versus standard deviation; do not silently equate the two. Register
T34/T35 and the covariance-only part of T36 with known-answer tests, extending
the registry's planned IDs first. T36 colour painting and T41 are excluded.

Acceptance: analytic pose/timing/Doppler/covariance tests pass; artifacts contain
no held-out observations; calibration uncertainty and retained counts are logged.
No suitable anchors means a reported data failure, not invented targets.

### C. Complete artifact contract and parser integration

- Make radar provenance required when radar is enabled: session, map frame,
  absolute time base, clock calibration, source/config hashes and split hash.
- Check finite values, array shapes, SPD covariance, support counts and temporal
  bounds. Pin the artifact checksum in a versioned dataset descriptor/manifest.
  Preserve the published baseline dataset; rerun gates for any new descriptor.
- Complete T39/T40: transform means, covariances and origins, require unit scale,
  crop to the training time window and scene box, and report every dropped count.
- Attach targets to train metadata only; `radars=()` remains empty. No radar
  rendering or GT inputs. With radar disabled, parser outputs equal the baseline.

Acceptance: corrupted hashes, wrong sessions/frames, invalid covariance and
held-out/window violations are refused; transformed synthetic examples are exact.

### D. Implement the loss-only arm first

- Preserve the full stock trainer/model configuration. The current method
  registration replaces the model config with subclass defaults; explicitly
  preserve and test every stock field before claiming a matched comparison.
- Configure `radar_opacity_loss_enabled` independently of stock initialization.
  L loads fixed anchors but has no radar Gaussian append path. Hash/compare every
  initial Gaussian tensor and random state against B for each matched repeat.
- Retain the implemented static-Gaussian uncertainty gate and mean negative log
  opacity. Use stable `softplus(-logit)` and count each Gaussian once even when
  multiple seed ellipsoids overlap. Register/test T42/T43.
- The gate is a hard membership test: the radar term directly changes opacity,
  not Gaussian positions. Empty support yields a differentiable zero and an
  explicit diagnostic; it cannot introduce missing geometry.
- Measure the initial unweighted radar term A0 and stock opacity regularizer R0.
  The repo plan's calibration is lambda = R0/A0 when this is meaningful. If A0 or
  R0 is zero, or support is absent, report that calibration is unsupported;
  never invent a weight or silently enable initialization/a different loss.
- Compute and report anchor coverage before deployment. The repo plan's LiDAR-
  sparsity filter may select regions with no baseline Gaussian support; do not
  silently discard these anchors, widen uncertainty gates, or replace the loss
  to make the ablation look successful. Report unsupported targets separately.
- Record weight, initial magnitudes, supported Gaussian count, per-anchor support
  and empty-support steps. Log seed coverage every 100 steps as the repo plan
  specifies. Precompute validated constant covariance quantities where safe and
  bound memory over both seeds and Gaussians without changing gate semantics.

Acceptance: hand-calculated loss and opacity gradients, overlapping gates,
boundary/empty cases, unchanged initial B/L Gaussian tensors and unchanged stock
losses when disabled. First run on synthetic known geometry, then June L smoke.

### E. Evaluate the actual hidden-trunk claim

- Verify June label/odometry frame agreement and generate evaluation-only cuboids.
  Do not transfer May labels or feed cuboids into anchor selection or losses.
- Reuse held-out LiDAR renders (P1) and exported Gaussians (T56). Complete the
  visibility-independent Gaussian occupancy grid (P2/T57), which can reveal
  reconstructed geometry still hidden behind grass in rendered views.
- Report detection/recall with precision and false positives, separately for
  seen, blocked and no-return targets. Complete visibility attribution and
  unmatched trunk-like cluster counting; current occupancy evaluation is partial.
- Use matched voxel grids and report threshold curves. Baseline-derived operating
  thresholds are frozen across arms; never retune them on each arm's predictions.
- Produce metric-matched occupancy/precision views, B/L BEVs, GT overlays and
  trunk zooms. Report standard LiDAR metrics and repeat spread alongside results.

Acceptance: known rotated boxes, occlusion examples, Gaussian occupancy products
and unmatched-cylinder cases give analytic answers. A loss decrease alone is not
a detection result. Improved P2 with unchanged P1 means geometry exists but is
still occluded; lower precision must be reported as possible invented geometry.

### F. Validate and run on Slurm

- Add tests; never weaken existing assertions, tolerances or exclusions. Follow
  transform traceability for all new conversion/calibration functions.
- Run affected tests, then full checks for configuration changes. Reference-image
  testing remains the normal repo policy; the prior session's explicit local-test
  exception used native cluster validation and must be recorded accurately.
- Recipe changes require rebuilding the reference/native environment; source-only
  changes can use CPU validation after the exact recipe passed GPU verification.
- Gate each arm with artifact contract, unchanged reference split and inspected
  alignment overlays. Run smoke before full training; preserve per-job provenance.
- Submit only B/L. Monitor actual progress,
  checkpoint, verify plot/render/split milestones, and automatically fetch success.

## 4. If the earlier classification draft is selected instead

Plan a separate supervised radar semantic model with class probabilities and an
independently authorized training-label source. Its objective is vegetation-
weighted cross-entropy, not negative log Gaussian opacity. Binary session weights
and continuous ray-based vegetation scores are separate variants; continuous
scores require a vegetation/free/unknown grid and an explicit unknown-ray policy.
Current cuboids are evaluation-only and cannot supply these training targets.
The initial DOCX draft does not fully specify this model or its integration into
SplatAD; agree that specification before calling an implementation exact.

## 5. Deliverables

Calibrated train-only radar-anchor artifact and report; loss-only L with identical
B/L Gaussian initialization; known-answer tests and contract reports; immutable
Slurm profiles/receipts; matched reconstruction/occupancy views; repeat-aware B/L
comparison with unsupported and negative outcomes reported honestly. No radar
Gaussian initialization is implemented or enabled.
