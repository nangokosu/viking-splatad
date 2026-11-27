# Radar loss comparison and deployment readiness

Date: 2026-11-11

The selected experiment is **B versus loss-only L**, on June tall grass, with
stock Gaussian initialization in both arms. This is the user's approved ablation
of the proposal's final action plan. The complete final action plan also adds
radar Gaussians; that initialization is deliberately omitted here.

| Arm or proposal draft | Radar initialization | Added objective | Current state |
| --- | --- | --- | --- |
| B | No | None; stock LiDAR training | Smoke and full runs successful and fetched |
| L, selected | No | Label-free opacity near fixed radar anchors | Full pilot and first evaluation complete; matched repeat check running |
| I / R | Yes | None / opacity | Outside selected scope; no Gaussian append implemented |
| Depth / opacity plus depth | Optional | One-sided radar depth | Outside selected scope |
| Vegetation classification drafts | Separate semantic model | Weighted cross-entropy | Outside selected scope; requires independently authorized labels/head |
| Rendered radar with beam blur | Separate radar renderer | Radar range reconstruction | Outside selected scope; radar remains unrendered |

For existing static Gaussians whose means lie inside at least one radar anchor's
uncertainty ellipsoid, the selected objective is

$$
\mathcal L_L=\mathcal L_B+
\lambda\frac{1}{|S|}\sum_{k\in S}\operatorname{softplus}(-o_k).
$$

Here $o_k$ is an opacity logit. Membership uses the inclusive Mahalanobis gate
with the configured chi-square quantile; overlapping anchors count a Gaussian
once. At step zero, $\lambda$ matches the initial radar term to the stock weighted
opacity regularizer. Missing initial support or an invalid regularizer stops the
run. Later empty support yields zero and is logged.

This directly encourages opacity at supported existing Gaussians. It does not
create Gaussians, attract their means into unsupported regions, or predict a
trunk class. Camera painting still colors stock initialization; camera loss,
radar rendering and GT training supervision are disabled.

Train-only calibration, uncertainty-aware persistent anchors, strict artifact
provenance/split checks, parser transformation/cropping, stock-config preservation,
exact device-side support lookup and diagnostics are implemented. The artifact is
separate from the baseline dataset, which retains its canonical timestamp split.
Final source `da76f8d` passed 1,177 native tests and both exact-profile preflights;
reference-image testing was skipped under the user's local-Docker exception.

The selector finds static persistent radar occupancy; it is not yet a
trunk-specific ground/canopy/sparsity selector. Recorded-ray attribution, visibility-independent Gaussian occupancy, neighbourhood
precision and matching metric views are implemented. The first full pair is
complete; a second matched B/L pair is running under a predeclared two-per-arm
initial repeat budget. The tested summary measures gaps against within-arm
spread. Global unmatched-trunk false positives and finer-grid sensitivity remain
additional work. No benefit is inferred from the first pair alone. (The hidden-trunk evaluation note and scripts 17 and 18 were removed 2026-11-27; see git history.)

Implementation and live launch records: [loss_only_implementation.md](loss_only_implementation.md).
Proposal interpretation and acceptance gates: [proposal_implementation_plan.md](proposal_implementation_plan.md).
