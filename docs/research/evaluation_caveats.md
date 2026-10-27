# Evaluation caveats: what `render_lidar` can and cannot show about radar-recovered trunks

The research question asks whether radar puts Gaussians at tree trunks that the lidar missed because tall grass blocked it. This doc lists the ways the planned evaluation (PLAN §5) can under- or over-state that effect, and what to measure instead. Items marked **proposal** are not decided yet. PLAN.md stays the authority until they are adopted there.

Related: `no_return_impact.md` (no-returns and the ray-drop gate), `splatad_extension_points.md` §3 (rendered lidar → point cloud).

## 1. How a rendered lidar point is made
`get_lidar_outputs` rasterises the Gaussians along each lidar ray. It does so front to back, in depth order.
- Gaussian i on the ray gets the weight **w_i = α_i · ∏_{j<i} (1 − α_j)**: its own alpha times the transmittance in front of it.
- **Per ray, it returns:**
  - an accumulation, Σ w_i;
  - an expected depth, a weighted average of the Gaussians' depths;
  - a median depth, where the accumulation crosses 0.5;
  - an intensity;
  - a ray-drop logit from `lidar_decoder`.
- **Point conversion** (`filter_lidar_pred_and_gt`, used by `08`): xyz = depth × ray direction (+ velocity × t). A ray is kept only if `sigmoid(ray_drop) ≤ 0.5`.
- **Mode 1** renders the rays of a recorded scan. **Mode 2** renders a synthetic beam grid (`lidar.azimuths × lidar.elevations`) at any pose, without rolling-shutter compensation.

So a rendered point cloud is **what a lidar at that pose would see first**, not what the model contains.

## 2. Caveat A: the ray-drop gate hides radar trunks
- With Ouster no-returns on (every run from 2026-10-26), the decoder learns where the real lidar got no echo.
- On the interleaved test split, eval poses sit between train poses. So the gate replays the neighbouring frames' drop-outs.
- A radar-recovered trunk in a direction where the lidar had no return can be gated out.
- **Adopted in PLAN §5:** the primary metrics use **ungated** extraction (accumulation > a config threshold, the same for every run). Gated numbers are secondary. `08` saves the per-ray accumulation as `pred_accumulation`.

## 3. Caveat B: first surface only (occlusion), the main caveat
For a ray aimed at a trunk base behind grass:

| grass in front | what the ray reports | trunk in render? |
|---|---|---|
| dense (combined α ≈ 1) | grass depth; the trunk's weight ≈ 0 | **no**, however opaque radar made it |
| thin or patchy | expected depth **between** grass and trunk (a spurious mid-air point); median depth snaps to whichever surface crosses 0.5 first | smeared or partial |
| gap, or a view over the grass | trunk depth | yes, but such a gap was usually open to the real lidar too, so B has the trunk as well |

**Why it is systematic, not bad luck.**
- On training rays the lidar returned from the grass. The depth and line-of-sight losses make the grass opaque along those rays.
- Once the grass is opaque, the trunk behind it gets almost no lidar gradient. So nothing fights the radar opacity loss, and the trunk Gaussian stays.
- So for occluded trunk bases, **"kept by radar" and "visible in the render from the training-like poses" nearly exclude each other.**
- On the interleaved split this is the normal case.

**Consequence:** render-based metrics (T45 detection on rendered points, render-based occupancy) measure **visible** trunks. They can understate R, down to no difference from B, even when R contains the trunks. Ungated extraction fixes Caveat A, not this one.

## 4. Caveat C: the held-out split decides what a render can reveal
- **Interleaved (default):** eval views ≈ train views, so the same grass blocks the same trunks. Renders show almost none of R's occluded-trunk gain.
- **Backtrack hold-out** (candidate, T48): the return pass views the trunks a few metres to the side, possibly through thinner grass or at another angle. This is the one planned split where a render can show R's gain.
- **Raised virtual sensor** (proposal, needs approval): a mode-2 render from a pose lifted above the grass height. The grass path length to a trunk base shrinks when looking down. This is a novel pose, outside the plan's held-out-pose ground rule, so it would be an extension, not a core metric.

## 5. Measuring "in the model" directly: occupancy from the Gaussians (adopted: PLAN §5 P2, T56–T57)
**Data.** Every checkpoint (`step-*.ckpt`) holds the Gaussians as tensors. Read them with `torch.load(path, map_location="cpu")["pipeline"]`, keys `_model.gauss_params.<name>`, N rows each:

| key | shape | stored as | use |
|---|---|---|---|
| `means` | [N,3] | centre, **dataparser frame** | inverse(w2m) → map frame (same as `08`, T50) |
| `scales` | [N,3] | **log** std per axis | `exp` (`splatad.py:980`) |
| `quats` | [N,4] | rotation, unnormalised, (w,x,y,z) | normalise; Σ = R·diag(exp(s))²·Rᵀ |
| `opacities` | [N,1] | **logit** | `sigmoid` (`splatad.py:981`) |
| `features_dc`, `features_rest` | [N,…] | decoder features | not needed |
| `id` | [N,1] | actor index, or static | all static here (no actors) |

`ns-export gaussian-splat` asserts a `SplatfactoModel` (`scripts/exporter.py:535`) and rejects SplatAD, so the checkpoint is the only source. `scripts/10_export_gaussians.py` (T56) reads it and writes `renders/gaussians.npz` (map-frame means, covariances, opacities; see `runs/RUN_FOLDER_GLOSSARY.md`).

**Method.**
- **Blocking probability:** Gaussian k's 3-D alpha is α_k(x) = σ(o_k)·exp(−½(x−μ_k)ᵀΣ_k⁻¹(x−μ_k)). This is the 3-D form of the alpha the rasteriser uses. A voxel's occupancy = 1 − ∏_k (1 − α_k(x)) at its centre or a few sub-samples: the chance that a ray through it is stopped.
- **Cost:** only voxels inside each Gaussian's 3σ ellipsoid are visited (outside, α < 1.1% of its peak). This is sparse and runs on CPU in minutes. It needs torch to read the checkpoint, so it runs in the image or as a CPU step on Slurm. A small loader can write `gaussians.npz` (map-frame means, covariances, opacities) for the numpy evaluation code.
- **Threshold:** the same value as the ungated render rule, e.g. 0.5 ("a ray through here would more likely stop than pass"). The render grid and the Gaussian grid then differ only by visibility.
- **Grid:** the same frame, resolution and origin as `labels/occupancy.py`.
- **Test:** a known-answer test. One Gaussian gives α = σ(o) at its centre, exactly its 3σ voxels, and the exact product-rule value for two overlapping Gaussians.

**Its own caveats:**
- **Ignores visibility** by design. It also counts Gaussians in space no sensor saw, such as SplatAD's random init points, including those on the scene-box faces. Score only cells within range and field of view of the eval poses (T32).
- **Faint, large Gaussians (floaters)** add low occupancy over wide areas, and dense haze can cross the threshold. Report the threshold, and check at a second value.
- **Radar-invented trunks** show up here as false positives that a render might hide behind grass. That makes T46-style precision on this grid essential, not optional.
- It is not a rendered sensor measurement, so it cannot be compared with GT lidar scans. Compare it only with the GT occupancy grid.

## 6. Per-cell breakdown: why the lidar missed (adopted: PLAN §5)
Classify each GT trunk cell, from the recorded June scans at the eval poses:
- **seen:** lidar returns inside the cuboid;
- **blocked:** rays towards the cuboid returned from something nearer;
- **no return:** rays towards the cuboid got no echo. This uses the `lidar_no_return/` files.

Report every metric per class. The radar claim concerns **blocked** and **no return**. On **seen** cells, B and R should agree.

## 7. Reading of the results (adopted 2026-10-26: PLAN §5 P1 + P2)
| measure | answers | role |
|---|---|---|
| Gaussian-occupancy detection + precision vs GT grid, per §6 class | is the trunk **in the model**, and does radar invent trunks? | **primary** test of the radar claim |
| Render-based (ungated) occupancy / T45 on rendered points | what a lidar at the eval poses would **see** | secondary, the realistic sensor view |
| Per-cell gap between the two | "present but hidden" (occluded by grass Gaussians) | explains the difference between the two above |
| Gated render metrics, `ray_drop_accuracy`, Chamfer, depth | sensor-faithfulness of the render | report only |

Adopted in PLAN §5 as two primary tests: **P1** (rendered lidar, ungated) and **P2** (Gaussian occupancy, T56–T57). R must beat B on blocked and no-return cells in at least one, and must not lose precision in either. §5 and §6 above are no longer proposals.
