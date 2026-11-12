# Radar loss comparison and deployment readiness

Date: 2026-11-11

The selected experiment is **B versus loss-only L**, on June tall grass, with
stock Gaussian initialization in both arms. This is the user's approved ablation
of the proposal's final action plan. The complete final action plan also adds
radar Gaussians; that initialization is deliberately omitted here.

| Arm or proposal draft | Radar initialization | Added objective | Current state |
| --- | --- | --- | --- |
| B | No | None; stock LiDAR training | Smoke and full runs successful and fetched |
| L, selected | No | Label-free opacity near fixed radar anchors | Implemented, validated and queued behind GPU runtime probe and smoke gates |
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
trunk-specific ground/canopy/sparsity selector. Hidden-trunk attribution,
visibility-independent Gaussian occupancy, false-positive analysis and matching
metric views still need completion. Matched repeat seeds and a repeat budget
must be fixed before the scientific comparison. The queued smoke/full pair is
one pilot and cannot establish a radar benefit.

Implementation and live launch records: [loss_only_implementation.md](loss_only_implementation.md).
Proposal interpretation and acceptance gates: [proposal_implementation_plan.md](proposal_implementation_plan.md).
