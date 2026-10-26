# Ouster no-returns (`add_missing_points`): impact on the radar losses and the evaluation

Status: no-returns are **on** for every run from 2026-10-26 on (`splatad.add_missing_points: true` in both session configs; commits `0c81773`, `f0777e2`). May, B, R and R+depth all use the same setting. This note records what that changes for the radar work and for the judged output, and what to check. Wider evaluation caveats (occlusion, split, Gaussian occupancy): `evaluation_caveats.md`. Code: `extract/no_return.py` (T51–T54), `splatad/viking_conversions.py` (T55). SplatAD internals: `splatad_extension_points.md` §2.

## 1. What enters training
- **Source:** every Ouster cell with range 0, from the sensor itself. Ring, column and time come from the cloud. The direction is fitted from the same cell's returns in other sweeps, because the bags carry no beam-angle metadata (median residual 0.0005–0.005°, column 0.35°).
- **Form:** each no-return becomes a ray with depth 2000 m. SplatAD counts a ray as "returned" only if its depth is ≤ 1000 m (`valid_lidar_distance_threshold`).
- **Volume (measured):**
  - Smoke data: ~13.1k no-returns per sweep, on top of ~17.5k returns, so **~75% more lidar rays per step**.
  - Scene A medians: 13.6k per sweep in May and 9.6k in June. The tall grass returns more of the low rays.
  - Most no-returns are in the upward rings (sky).
- **Which losses see them:** only the **ray-drop loss** (BCE, λ = 0.1, target "did not return"). Depth, intensity and line of sight use returned rays only. So:
  - no loss carves free space along a no-return ray;
  - no loss pulls Gaussians on that ray towards transparency directly.
- **Path to the Gaussians:** the ray-drop logit comes from `lidar_decoder` on the alpha-blended features plus the ray direction. Its gradient reaches the Gaussians only through the blending weights (opacity and position) and the features of Gaussians the ray crosses. The signal is weak and indirect.
- **Seeds:** neurad seeds Gaussians only from points closer than 1000 m. Our seed and colour code (radar seeds, T36) reads the returns files. So the placeholders never create or colour Gaussians.

## 2. Interaction with the radar opacity loss (T42/T43, core R)
- **Usually no overlap.** A laser fired at a trunk normally returns, from the trunk or from grass in front of it. Both are returns, which train depth and line of sight, not ray drop. No-return rays mostly point at the sky, at far open space, or at dark or wet surfaces.
- **Where they can conflict:** a no-return ray that passes through a radar-gated Gaussian. Ray drop then wants that ray to look empty to the decoder. It can push the gated Gaussian's opacity or features down, against the opacity loss's −log σ(o). Possible causes:
  - a radar seed placed off the true trunk (lateral blur, multipath);
  - a real trunk surface that absorbs the laser.
- **Size of the effect:**
  - The ray-drop weight is 0.1, and its gradient is indirect.
  - λ_opa is matched to `mcmc_opacity_reg` at step 0, and acts on every gated Gaussian directly.
  - So the opacity loss should dominate. This is an expectation, not a measurement (see §5).
- **Benefit:** the decoder learns "sky and far space give no return". This can discourage floaters in those directions, again only indirectly. Radar free space stays unused (PLAN §4.2).

## 3. Interaction with the radar depth loss (T44, optional R+depth)
- **No direct coupling.** T44 computes the expected depth along each seed ray in closed form from the Gaussians and compares it with the radar range. It reads no lidar data, so the placeholders never enter it.
- **No term clears space in front of a radar trunk because of no-returns.** T44 is one-sided, and no-returns get no line of sight. The only term that removes opacity in front of something is line of sight on returned lidar rays.
  - A lidar ray that returns from grass in front of a trunk penalises opacity in front of the grass, not at the trunk.
  - A lidar ray that returns from beyond a radar trunk position does penalise radar Gaussians there. That is correct behaviour (lidar saw through), or it exposes a misplaced seed. It is independent of the no-return setting.
- **Indirect coupling:** T44 makes radar Gaussians more opaque along seed rays. Where those Gaussians sit on no-return rays (§2), the ray-drop conflict grows a little. The same diagnostic covers it.
- **Comparability:** R+depth must use the same no-return setting as B and R. It does, because the flag comes from the session config.

## 4. Impact on the evaluation (PLAN §5)
- **The ray-drop gate is no longer a no-op.**
  - Without no-returns, the decoder never predicted a drop (max probability 0.05 in job 222633), so `08`'s gate `sigmoid(ray_drop) ≤ 0.5` kept every ray.
  - With no-returns, the decoder learns where returns fail. On the interleaved test split, the gate then reproduces the neighbouring training frames' drop-outs.
  - A trunk that R recovers from radar, in a direction where the lidar got no return, can be gated out of R's rendered cloud. **That hides exactly the gain the project measures.**
- **Rule (binding, PLAN §5):** the primary trunk metrics (T45 detection, T46 precision, the occupancy comparison) use **ungated** extraction. A ray becomes a point if its accumulation exceeds a threshold from config, applied identically to May, B, R and R+depth. Gated numbers are reported as secondary (what a real sensor would see).
- **First-surface limit:** a rendered lidar ray reports the first opaque surface. A trunk kept by radar behind opaque grass Gaussians stays hidden in the render, whatever the extraction. So the evaluation adds an **occupancy grid from Gaussian density** (opacity-weighted Gaussian mass per voxel), compared with the same GT occupancy grid. It separates "not in the model" from "in the model but hidden".
- **"Missed by lidar" can now be defined.** Using the no-return files of the evaluation poses, each GT trunk cell splits into three groups:
  - **seen:** returns inside the cuboid;
  - **blocked:** rays towards the cuboid returned from something nearer;
  - **no return:** rays towards the cuboid that got no echo.

  Report detection per group. The radar claim is about "blocked" and "no return". This works for every run, including runs with the flag off.
- **T45 `n_min`** comes from the GT June lidar returns. The returns files contain no placeholders, so it is unaffected.
- **`ray_drop_accuracy`** becomes informative. Report it, but it measures the decoder, not trunk geometry.
- **Lidar depth and Chamfer** use returned rays only, so they don't change meaning. Absolute values can still shift between runs with and without no-returns, because the Gaussians train differently. Never compare numbers across that setting; every run from now on has it on.

## 5. Risks and what to log
| risk | check |
|---|---|
| Ray drop erodes radar Gaussians (§2) | R logs: the fraction of no-return rays per step that cross a gated radar Gaussian, and the mean opacity of gated Gaussians along those rays vs elsewhere. If the gated-seed fraction falls while that crossing fraction is high, run R once with the flag off to isolate the effect. |
| The gate hides R's trunks (§4) | Always report ungated and gated T45 together. A large gated-vs-ungated gap for R but not for B points at the gate. |
| Trunk hidden behind grass in the render | Compare the Gaussian-density occupancy with the render-based occupancy per GT cell. |
| Training cost | ~75% more lidar rays per step. Record lidar step time in the smoke run against job 222633. |
| Direction error of the placeholders | Bounded by the T51 gate (median ≤ half a column). The fit residuals are stored in `derived.json`. |
| End-of-session sweeps | Rotation is held at the last odometry pose (T54); error < one sweep of turning, on the last sweep only. |
| Blind-zone no-returns (body parts closer than the Ouster's minimum range) | They become "no return" rays towards the robot body. Harmless to the scene, since ego space is not reconstructed. Ignore. |
| T16 elevation mapping is parallax-biased (ring 31: −38.1° vs −44.5° far-field) | Unrelated to the no-returns, but it affects SplatAD's lidar model. Decide before the full runs (SESSION_SUMMARY). |

## 6. Summary for the radar claim
- No-returns help the decoder learn where the sensor sees nothing. They do not supervise trunk geometry, and they do not clear space.
- They should not undo the radar opacity or depth losses: weight 0.1 and an indirect path, against direct terms. Log the crossing diagnostic to confirm.
- Their real impact is on the **evaluation**: with them on, the ray-drop gate must not decide whether a radar-recovered trunk counts. Hence the ungated primary metrics, the Gaussian-density occupancy, and the seen / blocked / no-return breakdown.
