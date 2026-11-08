# Radar loss comparison and deployment readiness

Date: 2026-11-08

The objective is to compare the radar approaches in the supplied proposal. The
document contains several drafts: vegetation-weighted semantic classification,
rendered radar range with beam blur, and a later label-free seed opacity method
with optional depth supervision. They require different predictions and targets;
the current repository implements only part of the last approach.

## Comparison matrix

| Experiment | Added initialization | Added objective | Readiness |
| --- | --- | --- | --- |
| B | None | Stock lidar training | Existing stock method; June launch still needs the no-photometric setting |
| I | Radar Gaussians | None | Seed generation and Gaussian append hook are missing |
| O | Same radar Gaussians | Opacity near radar seeds | Loss and model hook exist; initialization and artifact gates remain |
| D | Same radar Gaussians | One-sided radar depth | Not implemented; implement and verify independently of opacity |
| OD | Same radar Gaussians | Opacity plus one-sided depth | Requires O and D; tests whether their effects combine |
| VB | Semantic prediction pipeline | Vegetation-weighted cross-entropy, binary obstruction | No classifier or training-label source exists in this pipeline |
| VC | Same semantic pipeline | Vegetation-weighted cross-entropy, continuous obstruction | Also requires vegetation/free/unknown grid and a policy for rays with no known cells |
| RB | Radar rendering pipeline | Radar range with beam blur | Conflicts with the standing rule that radar is never rendered; excluded from the current deployment |

VB and VC belong to a semantic segmentation experiment unless a semantic head
and independently authorized training supervision are introduced into SplatAD.
They cannot be substituted for the opacity objective: opacity is an occupancy
probability, not a trunk-class probability. Ground-truth evaluation cuboids must
never supply training labels, seed filters, or loss targets. The rendered-radar
draft is recorded for completeness, not enabled by the attached document.

## Existing opacity objective

For static Gaussian k and radar seed j, define

`distance_squared(k,j) = (mean_k - seed_j)^T covariance_j^-1 (mean_k - seed_j)`.

Include Gaussian k if it lies inside at least one seed's uncertainty ellipsoid:
`distance_squared(k,j) <= chi2.ppf(gate_quantile, df=3)`.

The implemented objective is
`loss = weight * mean(softplus(-opacity_logit_k))` over included Gaussians.
`softplus(-o)` equals `-log(sigmoid(o))` without sigmoid underflow. Overlapping
seed ellipsoids do not count a Gaussian multiple times. Empty support produces a
differentiable zero, which cannot attract new Gaussians to an unsupported seed.
This is why initialization and support diagnostics matter.

Code: `src/viking_splatad/splatad/radar_lidar_loss.py` and
`src/viking_splatad/splatad/radar_lidar_model.py`.

The loader currently requires map-frame `means[N,3]` and positive-definite
`covariances[N,3,3]` in `radar_seeds.npz`. It optionally checks session/frame tags
in `radar_inputs.json`, then the dataparser transforms the metadata for the train
split. This is not yet the full provenance contract described in PLAN section 2.

## Conditions for a valid comparison

- Use June Scene A chunks 20–22 and the committed timestamp-based reference split
  for every arm. Do not change extraction or split settings to enable radar.
- Apply the same lidar-only training setting, camera colour initialization,
  no-return rays, iteration budget, and upstream model settings across B/I/O/D/OD.
  The current launch does not yet set `train-lidar-only True`.
- Use one shared, label-free seed artifact for I/O/D/OD, built using only training
  observations, with calibrated timing, uncertainty, and provenance. The seed
  builder and Gaussian initialization hook are not present in this checkout.
- Preserve the stock trainer/model configuration when registering extensions;
  audit the current replacement model config before claiming matched arms.
- For opacity, PLAN section 4.2 specifies measuring the initial unweighted term
  and stock opacity regularizer, then matching their weighted magnitudes at step
  zero. That calibration is not implemented. Do not invent a deployment weight.
- Measure runtime and memory: the current gate compares every static Gaussian
  against every seed, chunked only over Gaussians, and repeats covariance checks
  and inversion each call. It needs a real seed-count smoke test before scaling.
- Run repeats within every arm. Assess between-arm differences against within-arm
  variation; the same random seed does not eliminate CUDA kernel variability.
- Evaluate all reconstruction arms against the same June occupancy/trunk targets
  and matching views. VB/VC need semantic metrics as well; their classification
  scores alone cannot rank reconstruction quality against B/O/D.

## Deployment order

1. Restore local `uv` and Docker, configure trusted cluster SSH, and recreate or
   locate the June dataset. No dataset or seed artifact was found in this checkout.
2. Run `scripts/test_in_image.sh python scripts/affected_tests.py --run` for the
   wiring fixes. It currently cannot start because Docker is unavailable.
3. Complete seed generation/initialization, input-contract gates, matched
   lidar-only configuration, and measured loss-weight calibration. Add synthetic
   known-answer coverage for each implementation before using real data.
4. Pass the full image tests, dataset contract, reference-split check, and image
   CPU preflight; inspect its lidar-on-camera overlays.
5. Commit the validated changes, build the reference image/native environment if
   recipe files changed, and stage the committed code plus dataset. Submit a
   smoke job for each implemented method before full comparison runs.
6. After successful training/render/split milestones, fetch automatically and
   compare repeated runs. Implement D/OD next; treat semantic VB/VC as a separate
   pipeline with explicit supervision rather than silently using evaluation GT.

No job was submitted during this audit. SSH outside the sandbox reached the
connection stage but failed host-key verification; the queue and remote artifacts
were not inspected. Do not bypass host-key checking to proceed.
