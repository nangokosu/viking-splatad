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
- **Thresholds:** `eval.p2_thresholds: [0.5, 0.8]`. 0.5 is binding (the same value as the ungated render rule: "a ray through here would more likely stop than pass"), so the render grid and the Gaussian grid differ only by visibility. 0.8 is a strictness check, lowered from 0.9 because radar adds faint Gaussians. It can weaken a result but never replace the 0.5 reading.
- **Grid:** the same frame, resolution and origin as `labels/occupancy.py`.
- **Test:** a known-answer test. One Gaussian gives α = σ(o) at its centre, exactly its 3σ voxels, and the exact product-rule value for two overlapping Gaussians.
- **Centre sampling misses thin Gaussians (found 2026-11-16, fixed by T81):** SplatAD's Gaussians are discs millimetres thick and a few centimetres wide, far smaller than the 0.52 m GT voxel. Their 3σ ellipsoids almost never reach a voxel centre, so the centre-sampled grid sees a small fraction of the trunk surface the model holds. The binding estimator is therefore T81 (`peak_alpha_voxel_occupancy`): each Gaussian adds its peak σ(o) to the voxel holding its mean, plus the T57 tails at other centres. The numbers and the diagnostic script are in PLAN §5 and `docs/research/p2_estimator_diagnostic/`.
- **Two thresholds, not a sweep:** the 11-point threshold sweep is retired. scripts/19 reports P2 at the two values above. P1 occupied = an ungated rendered point in the cell.

**Its own caveats:**
- **Ignores visibility** by design. It also counts Gaussians in space no sensor saw, such as SplatAD's random init points, including those on the scene-box faces. Score only cells within the scored range and field of view of any recorded sweep (T32; train, guard and test sweeps, not only the test sweeps).
- **Faint, large Gaussians (floaters)** add low occupancy over wide areas, and dense haze can cross the threshold. Report the threshold, and check at the second value (0.8).
- **Radar-invented trunks** show up here as false positives that a render might hide behind grass. That makes T46-style precision on this grid essential, not optional.
- It is not a rendered sensor measurement, so it cannot be compared with GT lidar scans. Compare it only with the GT occupancy grid.

## 6. Hidden-cell evaluation (H, H_near) removed 2026-11-27 by the user's decision; see git history before this date.

## 7. Reading of the results (PLAN §5 P1 + P2)
| measure | answers | role |
|---|---|---|
| P2 recall and precision over every scored cell, at 0.5; class `obstacles` first | is the trunk **in the model**, and does radar invent geometry? | **primary** test of the radar claim |
| P2 IoU; per-object means with the object count | summary and context for the primary rows | summary |
| P1 recall and precision (same cells) | what a lidar at the eval poses would **see** | secondary, the realistic sensor view |
| Final test-view PSNR / SSIM / LPIPS (training-eval split = test + guard frames) | does radar hurt the render? | guard |
| Lidar render Chamfer and depth | sensor faithfulness | report only |

**Reading rule.**
- The headline is class `obstacles`, P2 recall and precision at 0.5 over every scored cell, with the pooled value and the mean over objects side by side.
- A result needs the variant gap to exceed the spread between repeats (n = 5) **without losing precision**.
- 0.8 can weaken a result but never replace 0.5.
- May is the low-grass comparison.

Adopted in PLAN §5 as two primary tests: **P1** (rendered lidar, ungated) and **P2** (Gaussian occupancy, T56–T57, T81).

### Caveats to state with every result
- **Unannotated objects** lower precision: an occupied cell on a real but unlabelled object counts as a false positive.
- **Box GT caps recall:** the cuboids are coarse, and the cells they mark are not the true surface.
- **Pooled values are dominated by large objects**, hence the per-object mean beside them.
- **The radar response differs by class** (trunk, bush, rock), so the `obstacles` group can hide a class that moves the other way. Report each class.
- **Bushes are porous:** a beam can pass through cells that hold bush GT, so a bush's recall understates how much of it the lidar saw.

## 8. Audit of the P1 script (open, 2026-10-27)
An independent read-only review of `scripts/13_eval_semantic_occupancy.py` (T45, T46, T67, T68) asked two questions:
- Are the rendered points and the GT cuboids compared in the same coordinates?
- Are the right semantic classes scored?

It was run on the May smoke run and the June run. **Nothing below is fixed yet.** The fixes in §8.4 are proposals until they are adopted in PLAN §5.

Results at the time of the audit (ungated, binding):

| session | trunk detection | bush detection | rock detection | trunk point precision |
|---|---|---|---|---|
| May (smoke) | 0.458 | 0.292 | 0.238 | 0.497 |
| June | 0.386 | 0.256 | 0.053 | 0.467 |

### 8.1 Coordinates: OK
- **Frame chain:** each test sweep's `lidar_to_map` matches `poses.npz` to 1.5e-5 on all 52 sweeps. Voxel indexing (centre = origin + (idx + 0.5)·voxel), the quaternion order and the point-in-box test are correct. The median offset between rendered and real points is about 4 mm.
- **Label map vs. bag map:** they agree in XY. Measured from real returns around 73 May and 47 June trunks:
  - the offset is (+0.02, −0.04) m in May and (−0.03, −0.05) m in June;
  - the yaw difference is ≤ 0.19°;
  - shifting the boxes to the best-fit offset adds only 0.1–0.4% more points inside.

  The ±30 cm seen in phase 1 is per-trunk labelling scatter (about 0.2 m), not a shift. This evidence is enough to close the "UNVERIFIED frame" caveat in `data/gt/*/summary.md`. That folder is gitignored, so this section is the tracked record.
- **Box heights (minor):** May boxes start about 0.25 m above the lowest ground returns, and May bush boxes about 0.9 m above. June boxes start at the ground.

### 8.2 Classes: correct ids, but trunk scores are mostly canopy (high)
- **The ids are right:** trunk = 1, `bush_or_small_tree` = 4, rock = 3, in both labels samples. They match PLAN §5.
- **Trunk boxes overlap canopy boxes:**
  - 188 of 190 May trunk boxes lie inside a canopy box (class 2) for about 75% of their volume. June is the same.
  - 77% of the trunk-neighbourhood voxels are canopy.
  - Points inside canopy boxes make up 83% of the May trunk "false positives" and 46% of the June ones. Leaving them out raises trunk point precision from 0.50 to 0.74 in May (0.74–0.98 depending on the run).
  - T45 detection counts foliage inside the 9–10 m tall trunk boxes as trunk.
- **Grass at the trunk base:** in June, 21–23% of the points inside trunk boxes lie within 1 m of the box bottom, where grass and trunk can't be told apart. That band is what the research question is about, and it isn't scored on its own.
- **Bush labelling differs between May and June:** harmless while the two sessions are never compared (CLAUDE.md: no May ↔ June mapping).

### 8.3 Which boxes are scored (high)
- **The range limit comes from one outlier:** `range_limit_from_real_points` takes the farthest real return. That is 150.8 m in May (p99.9: 38 m) and 65.8 m in June (p99.9: 33.6 m). As a result, every one of the 190 May trunks is scored, with a median distance of 32 m, and about 35% of them got no real return at all.
- **The real-lidar ceiling is low:** the real test returns themselves reach a trunk detection rate of only about 0.49. The rendered rates therefore read much worse than they are, and the ungated rate sometimes beats the real rate.
- **June rocks:** no real return lands in any of the 19 rock boxes. n_min falls back to 1 (`MINIMUM_POSSIBLE_POINT_COUNT`), so the June rock row means nothing.

### 8.4 Proposed fixes
| # | fix | addresses | severity |
|---|---|---|---|
| 1 | Range limit from the p99.9 of real ranges, not the maximum | §8.3 outlier | high |
| 2 | Leave out points and voxels inside canopy (and other non-scored labelled) boxes in T46 and voxel precision; for detection, clip trunk boxes at the canopy bottom, or use a below-canopy height band | §8.2 canopy | high |
| 3 | Report trunk scores separately for the band from the box bottom to about 1 m up (the under-grass band) | §8.2 grass | medium |
| 4 | Report "n/a" for a class when the real sweeps hit none of its boxes, instead of falling back to n_min = 1 | §8.3 June rocks | medium |
| 5 | A real-return baseline row with the same metrics, or score only the boxes the real returns hit ≥ n_min times. This reverses the earlier "no reference row" choice, which needs the user's approval. | §8.3 ceiling | high |
| 6 | Close the frame caveat in `data/gt/*/summary.md`, citing §8.1 | §8.1 | low |

The 1 m band in fix 3 and the p99.9 in fix 1 are proposals. When adopted, they get a source in `configs/eval.yaml` like the other settings.
