# June hidden-trunk evaluation

The experiment's purpose is to test whether radar opacity supervision strengthens
trunk geometry that stock June reconstruction misses. Training progress, anchor
support and RGB renders do not answer that question.

## Matched pilot

`scripts/17_compare_hidden_trunks.py` compares the completed full B and loss-only L
runs. It refuses mismatched data hashes, timestamp splits, seed, iteration budget,
no-return setting, incomplete outputs or unequal recorded test observations.
The L diagnostic must report zero extra radar Gaussians. Both arms optimize LiDAR
loss with camera-coloured initialization; this is not a comparison against a
joint camera-loss training run.

The June labels remain evaluation-only, outside the model dataset. Ground truth
uses the existing script 05 sample selection and voxel rule: half the median
labeled trunk diameter (0.5219 m for the current labels). It is a coarse occupancy
proxy, not millimetric trunk-surface truth. The range/FOV and frozen neighbourhood
are exactly the existing evaluator's definitions, shared across arms. They retain
175 scored June trunks, including trunks with no observed test returns.

## What is measured

P1 uses each arm's own ungated rendered LiDAR points on the common GT grid. P2
computes alpha from the model's Gaussian factors at those same voxel centres,
inside the proposal's three-sigma ellipsoids, then occupancy as

$$p(v)=1-\prod_k\left(1-o_k\exp(-d_k(v)^2/2)\right).$$

The same committed probability threshold, 0.5, binds P1 and P2. Overlapping
Gaussians accumulate log survival; alpha is not added or counted multiple times.
P2 is independent of first-surface rendering and can expose geometry behind grass.

Hidden space (`scripts/24_build_hidden_space.py`, T91, T92). Every recorded Ouster
sweep of the session (train, guard and test, about 2000) is intersected with the
GT-grid cells, with a range margin m measured from the data (95th percentile of
cross-sweep point-to-plane residuals on near planar returns). A cell is seen
(a return within [entry - m, exit + m]), blocked (a return shorter than
entry - m) or passed (a return beyond exit + m). The hidden space is
H = not seen, blocked, not passed; no-return beams do not affect it. It is built
once per session and stored in `data/gt/<session>/hidden_space.{npz,json}` with
input hashes. The headline band H_near is H within `hidden_space.near_band_m`
(20 m) horizontal distance of the nearest sweep origin; full H is reported beside
it. `scripts/25_hidden_label_transitions.py` shows how GT cells move from the
earlier test-sweep labels (seen, blocked, no-return, passed-through, unobserved)
to H (an optional diagnostic; its reports are not kept). A nearer return is
**not independently identified as grass**, and camera occlusion is not classified.
These limits must accompany any result; the full list is in
`docs/research/evaluation_caveats.md` §6-§7.

Reports include P1/P2 voxel recall, precision and IoU, recall_H, precision_H and
IoU_H on H_near and full H (class `obstacles` first, then each class), off-trunk
occupied cells, per-trunk P2 occupied fractions, and P2 cells
present in the model but absent from its P1 render. Off-trunk occupancy in the
frozen trunk neighbourhood is a false-positive proxy; it is not a count of
unmatched trunk-like objects throughout the forest. No radius/threshold is tuned
to make L look better.

Each scored trunk gets one matching picture: recorded LiDAR, B/L rendered LiDAR,
B/L Gaussian occupancy, occupancy change, and recorded evidence. The pictured
masks and numerical captions use the evaluator's own arrays and functions. Every
trunk is included in annotation order, not just favorable examples. Pictures are
box-local XZ projections across depth, not novel camera renderings.

## Why checkpoint factors are necessary

The legacy export stores covariance matrices in float32. A real-artifact check
found 263,332 non-positive-definite matrices among the full B run's five million
exports, despite their originating from positive Gaussian axis scales. Squaring
very different scales, rotating and rounding the covariance destroys small
eigenvalues. Occupancy must not silently clip or repair these matrices.

The corrected export adds `scripts/10_export_gaussians.py --factorized`. It reads
the exact saved checkpoint on CPU and writes a separate evaluation-only archive
of map-frame means, principal-axis bases A, positive standard deviations s and
opacities in float64. The legacy archive and trained model remain unchanged.
Distance is evaluated as

$$d_k(v)^2=\left\|\operatorname{diag}(s_k^{-1})A_k^{-1}(v-\mu_k)\right\|^2.$$

This inverts the well-conditioned basis, not the ill-conditioned squared
covariance. Candidate spheres bound the ellipsoid conservatively; the final
Mahalanobis gate is unchanged. No eigenvalue clipping, added covariance noise,
new Gaussians or retraining is involved. Analytic tests cover a thin rotated
Gaussian whose float32 covariance becomes singular.

## Interpretation and current gate

A recall_H gain on H_near with preserved precision would be positive pilot
evidence of stronger hidden geometry. A precision loss must be reported beside
any gain. No gain is a valid result. One B/L pair cannot establish a repeatable
research conclusion: matched repeats and their spread remain necessary.

Label correspondence must be reviewed before interpretation. The source label
file carries no coordinate-frame declaration. The evaluator writes an alignment
view and hashes; a matching reviewed receipt can be supplied. A coarse view
alone does not prove exact alignment. No ICP, May-to-June mapping or GT-guided
training adjustment is performed.

The corrected factor-based evaluator (`f43e055`) passed 1,189 native tests.
Comparison 244829 completed successfully and its reports, common cell archive and
all 175 per-trunk pictures were fetched. Both full pilot runs are complete and
fetched. The obsolete covariance-based job 244546 remains held; do not release it.

Coordinate correspondence was reviewed in the real-LiDAR XY overview, observed
trunk XZ panels, and five original held-out camera observations (frame indices
10, 520, 1030, 1634, 2151) with the June cuboids projected by existing T25. Boxes
coincide with the corresponding trunks, including vegetation-obscured bases.
No alignment transform was estimated or applied. This is a visual frame check,
not independent certification of calibration or annotation accuracy. A separate
receipt binds the review to the exact dataset, cuboid and label hashes; original
numerical products remain unchanged.

## Initial repeat check

The predeclared initial budget is two runs per arm, using the same configured
seed and each arm's exact pilot training source. This measures accepted stock
CUDA variation; it is not a power-calculated statistical study.

- B repeat 244827 runs `0097a66`, profile `june_scene_a`.
- L repeat 244828 runs `da76f8d`, profile `june_loss_only`. Its step-zero Gaussian
  fingerprint matches the first L run and records zero added radar Gaussians.
- Both request live-measured A6000 resources: 24 CPUs/109 GiB. L reuses the
  successful measured 21 h 43 m allocation; B retains its six-hour allocation.
- Comparison 244830 depends on both repeat runs and uses the same fixed
  `f43e055` evaluator as the first pair.
- Summary 244855 depends on comparison 244830, uses tested source `394db6a`,
  and consumes both matched products plus the separate correspondence review.

`scripts/18_summarize_hidden_trunk_repeats.py` refuses a single pair, reused run
IDs, differing cells/observations/thresholds, or metric JSON that does not match
its cell archive. It reports arm means, sample spread, min/max ranges and paired
gaps; undefined values cannot be dropped. It also draws metric-matched P1/P2
maps of consistent recovery/loss, intermittently present B cells consistently
preserved in L, and consistently added off-trunk cells. These are descriptive
checks, never a significance claim. Final validation 244835 passed **1,194 tests**,
with the one allowed traceability skip and eight configured deselections, under
the user's native-cluster exception to local Docker tests. No existing test
assertion/tolerance was weakened.

One training monitor and one evaluation/summary monitor were started on the Mac.
They automatically fetch completed products, retry SSH failures, and never treat
connection failure as an empty queue. The Slurm dependency chain runs without
the Mac; fetching needs the Mac connected. Runtime records/scripts/logs live in
ignored `data/cluster-validation/`. Restart those monitors next session; do not
restart the old submission continuation and duplicate already queued work.

The repeat jobs were still active at this handoff. No B/L performance conclusion
was read from the first pair alone. Global unmatched-trunk counting and finer-grid
sensitivity remain additional work; the present false-positive measure is the
committed local occupancy protocol.
