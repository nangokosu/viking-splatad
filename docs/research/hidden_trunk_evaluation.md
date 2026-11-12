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

Recorded held-out beams are intersected with GT-grid cells. Evidence distinguishes
returns in the cell, nearer returns (blocked), explicitly recorded no-returns,
returns beyond the cell (passed-through), and no intersecting recorded ray
(unobserved). Evidence flags are retained even when several types occur. The
exclusive report priority is seen, blocked, no-return, passed-through, unobserved.
A nearer return is **not independently identified as grass**, and camera
occlusion is not classified. These limits must accompany any result.

Reports include P1/P2 voxel recall, precision and IoU, recall within every evidence
stratum, off-trunk occupied cells, per-trunk P2 occupied fractions, and P2 cells
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

A blocked/no-return recall gain with preserved precision would be positive pilot
evidence of stronger hidden geometry. A precision loss must be reported beside
any gain. No gain is a valid result. One B/L pair cannot establish a repeatable
research conclusion: matched repeats and their spread remain necessary.

Label correspondence must be reviewed before interpretation. The source label
file carries no coordinate-frame declaration. The evaluator writes an alignment
view and hashes; a matching reviewed receipt can be supplied. A coarse view
alone does not prove exact alignment. No ICP, May-to-June mapping or GT-guided
training adjustment is performed.

The first evaluation implementation (`8c8aa77`) passed 1,185 native tests, with
one allowed traceability skip and eight configured GPU/real-data deselections.
Its queued comparison 244546 was held after the real covariance check. Stable
factor export (`c48e60d`) and both trainer-directory layouts (`f43e055`) require
fresh native validation before the corrected comparison is submitted. SSH access
expired during that submission attempt; training continues independently.

A local background continuation is prepared in ignored
`data/cluster-validation/finish_hidden_evaluation.py`, pinned to `f43e055`.
After authenticated SSH returns it runs native validation, refuses submission on
failure, submits the corrected comparison after full training success, and fetches
reports/cell arrays/pictures (not the large factor archives). Its log is
`data/cluster-validation/hidden-evaluation-watch.log`. The superseded 244546
remains held; do not release it against the rounded covariance artifacts.
