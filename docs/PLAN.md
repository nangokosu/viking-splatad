# Plan: June Scene A, SplatAD (B) vs SplatAD + radar (R)

> **Ground rules**
> - **No May ↔ June mapping.** No ICP and no cross-run registration. Poses and labels are never transferred between runs. Each run is processed and evaluated in its own frame.
> - **3DGS renders new views of the same scene, not new scenes.** Training and evaluation use the same scene; evaluation uses held-out poses.
> - **Radar is never rendered.** It enters training only as (1) seed Gaussians and (2) loss terms. The judged output is SplatAD's **rendered lidar point cloud**: `render_lidar` turns the scene's Gaussians into a lidar scan.
> - **Training is self-supervised.** The radar losses use no labels. Cuboids are used only for evaluation (CLAUDE.md rule 6).

## Facts the plan relies on (measured)
- **Bag chunks:** May has 49 chunks over 4068 s, so about 83 s each. `metadata.yaml` lists each chunk's `starting_time` and `duration`.
- **Frames per chunk:**
  - `/icp_odom` runs at 10 Hz, so a chunk holds about 830 poses. They are about 7 cm apart at the May chunk-0 speed (73 m in 99.8 s).
  - Camera and lidar also run at 10 Hz, so 3 chunks give about 2,100–2,500 frames per sensor.
- **Labels** (`short_and_tall_grass_labels.json`):
  - There is one sample per run: May, with 467 cuboids, and June, with 549.
  - The classes are Tree_trunk(1), Tree_canopy(2), Rock(3), Bush_or_small_tree(4), Car, Building, Lamp/sign and Ignore. There is **no grass class**.
  - The June map is "registered" to May: the same car cuboid sits at ≈(0.3, 3.3) in both.
  - The labels come from accumulated lidar maps, so they include objects that a single traversal may not see.
- **Already on disk:** May chunks 0 (campus road) and 1 (forest from about +45 s).

## 1. Choose Scene A: guess and check
**Criteria:** a region with tree trunks, where the grass is **low in May** and **high in June**. We use 3 consecutive chunks per run (chunks must be consecutive).

1. **June:** guess a chunk in the forest part of the run. Start at about chunk 18; 18–25 is the expected range.
   - Fetch it with `00_fetch_chunk.py --chunk N` and extract it.
   - Look at the camera contact sheet: we need high grass in front of visible trunks.
   - If it fails, try the next guess.
2. **May:** guess the chunk that covers the same place. Start with the same index.
   - Fetch it and look at its contact sheet: we need low grass and trunks.
   - Same place = the May and June `/icp_odom` xy tracks overlap, which can be checked because the two label maps share coordinates. The overlap is used only to choose bags, not to align data.
3. **More chunks:** extend to neighbouring chunks, before or after, that stay in the region in each run. Prefer an out-and-back (a backtrack pass), because the return pass can serve as a spatial hold-out (§5).
4. Put the chosen chunks in `configs/sessions/*.yaml` (`window.chunk_files`; start/end offsets if needed). Record the choice and the contact sheets in `docs/experiments.md`.
5. **June check:** June `/icp_odom` must lie on the June label map, i.e. the label map frame equals the `/icp_odom` frame.
   - This check is needed for the June evaluation.
   - If it fails, a one-off June-only alignment is allowed, because it is not May↔June.

## 2. Data parser: lidar + camera + radar seeds
- **The parser.** Our `VikingHillDataParser(ADDataParser)` (exists, T27) gains `radar_inputs_dir: Optional[Path]`.
  - `None` is run B, and the outputs are byte-identical to today's.
  - Upstream neurad is not edited beyond the existing `OUSTER_OS0_32` patch.
- **Radar is not a sensor.** `radars=()` stays empty, because SplatAD asserts there are no radars.
  - Radar enters as `metadata["radar_seeds"]`, which `SplatADPipeline` deep-copies into `model.kwargs["metadata"]`.
  - It is attached for the train split only.
- **Offline seeds** (`scripts/07_build_radar_seeds.py` → `data/intermediate/june_tall_grass/radar_inputs/`):
  - built from `/hugin_raf_1/radar_data`;
  - in the map frame, in absolute float64 seconds;
  - stored as `radar_seeds.npz`: means, covariances, rgb, sensor_origin, support_count, time span;
  - with `radar_inputs.json` holding the frame/time tags, the config hash and every derived threshold;
  - with the sha256 recorded in `manifest.json`.
  The steps:
  - **T34 accumulate:**
    - T_map_radar(t) = T_map_base(t+δ_radar)·T_base_radar;
    - the poses are interpolated with T3, and δ_radar comes from T24;
    - T_base_radar is the June `high_grass` extrinsic (T5).
  - **T35 select trunk candidates (GT-free; every threshold derived from data):**
    - static: Doppler residual within k·σ, with σ the MAD of the per-scan residuals;
    - persistent: voxel hit count ≥ a configured percentile;
    - in the trunk height band: above the lidar ground, below the canopy onset;
    - lidar-sparse: no train-split lidar seed within the radar's lateral resolution, so seeds land where vegetation blocks the lidar.
  - **T36 seed Gaussians:**
    - mean = the voxel centroid;
    - covariance = R_ray·diag(σ_r², (rΔaz)², (rΔel)²)·R_rayᵀ, from the radar resolution;
    - colour painted from the nearest-in-time train camera only if that view is unobstructed (no closer lidar return in the seed's pixel footprint, §4.3); otherwise grey. The colour initialises the radar Gaussians in every run, like SplatAD's `paint_points` for the lidar seeds. Only the photometric extension then trains on colour.
- **In the parser, after `super()`** (all pure numpy in `viking_conversions.py`):
  1. **Load and verify** the files: tags, sha256 and session.
  2. **T39 frame:** with `w2m = dataparser_transform`, set μ' = Rμ + t, origins' = R·origins + t and Σ' = RΣRᵀ. Assert scale == 1.
  3. **T40 crop:** shift the time spans by `time_offset` and keep the sensor window. Crop to the `scene_box`, and report the dropped count.
- **Contract additions:**
  - sha256 and tags match;
  - all values finite;
  - every Σ is SPD;
  - the seed spans lie inside the sensor window;
  - the retained fraction is reported.
  No Slurm run without a passing report.

## 3. May Scene A: does SplatAD work out of the box?
- **Data:** the May Scene A chunks. May chunk 1 (forest, on disk) is the first Slurm/Docker smoke test.
- **Run:** stock `splatad`, lidar + camera, with the middle-block train/test split (§5; `train_fraction` and `test_block_count` from the session config).
- **Held-out metrics:**
  - PSNR, SSIM and LPIPS on eval cameras;
  - lidar depth, Chamfer and ray-drop on eval lidars;
  - the rendered lidar point cloud (`08`).
- **Pass:** the held-out renders show the forest, and the metrics beat a mean-colour / mean-depth baseline computed by the same script.
- May is not compared with June numerically. It shows that SplatAD renders this kind of forest acceptably under short grass.

## 4. June models (same chunks, split, seed and step count)
**Core June runs have no photometric loss.** They use SplatAD's built-in `--pipeline.datamanager.train-lidar-only True` (`full_images_lidar_datamanager.py:83, 446`): no training step draws a camera image.
- **The camera is still used to initialise the Gaussians.** SplatAD's `paint_points` colours the lidar seeds, and T36 colours the radar seeds.
- The judged output is lidar geometry, so the core comparison is lidar + radar losses.
- The photometric loss is an extension (§4.3).
- **Model 1 = B:** stock `splatad`, lidar-only. The lidar is partly blocked by high vegetation, so we expect missing or blurred trunk bases.
- **Model 2 = R:** `splatad-radar` = `SplatADRadarModel(SplatADModel)`, lidar-only plus radar.
  - It is registered through `NERFSTUDIO_METHOD_CONFIGS`, so neurad needs no patch.
  - It uses the same trainer config as `splatad`.

### 4.1 Radar initialization (T41)
In `create_gauss_param_dict`:
1. Call `super()` first. The lidar, random and actor groups are built as before.
2. Append one Gaussian per radar seed:
   - mean = the seed mean;
   - scales and rotation from the eigendecomposition of the seed covariance (det forced to +1);
   - colour = the painted rgb;
   - opacity = `logit(radar_seed_opacity)`;
   - static id.
No new per-Gaussian parameters are added, so the MCMC strategy and the optimizers work unchanged.

### 4.2 Radar losses (label-free, computed directly on the Gaussian parameters)
No second rasteriser pass: it would overwrite `self.info` and corrupt the MCMC densification statistics.

**Opacity loss (main term).** "Gaussians near a radar seed should be opaque."
- **T42 gate:** Gaussian k is near seed i if d_kiᵀ Σ_i⁻¹ d_ki ≤ χ²₃(q).
  - d_ki is the vector from seed i to Gaussian k.
  - The quantile q comes from config, and the threshold is computed with `scipy.stats.chi2.ppf`.
  - So the radius **X is the radar's own resolution ellipsoid**, not a hand-set number.
  - The gate is recomputed every step with a chunked kNN over static Gaussians.
- **T43 loss:** L_opa = λ_opa · mean over gated Gaussians of −log σ(o_k). It pulls them towards opacity 1.
- **Why the radar Gaussians stay:** MCMC relocates only low-opacity Gaussians, so opaque Gaussians at radar seeds are not recycled.
- **Setting λ_opa:** the stock `mcmc_opacity_reg` pushes every opacity down. So λ_opa is set so that the step-0 radar term equals the step-0 `mcmc_opacity_reg` (measured and logged).
- **Logged every 100 steps:** the fraction of seeds with ≥1 gated Gaussian.
  - If it collapses, switch to the seed-density form, which still gives a gradient when no Gaussian is near: −log(1 − exp(−D(q_i))), with D(q) = Σ_k σ(o_k)·exp(−½dᵀ(Σ_k+Σ_i)⁻¹d) at the fixed seed means.
- The radar losses only say "occupied here", never "empty here". Radar free space is unreliable (lateral blur, multipath), so free space stays with the lidar. SplatAD's line-of-sight loss carves the space in front of each returned point. The ray-drop loss learns where no return comes back, but only from no-return rays, i.e. with `add_missing_points` and the no-return sidecar (Phase B, T51–T55); it acts on the Gaussians only indirectly, through the lidar decoder.

**Depth loss (T44, optional, added only after R works with the opacity loss).** "Gaussians must agree with the radar range as well as the lidar depth."
- For each seed, cast a ray from `sensor_origin` to the seed.
- Compute the expected termination depth from the static Gaussians near the ray, in closed form (the 1-D integral of a 3-D Gaussian along a ray).
- Penalise |E[depth] − radar range| / σ_r.
- **Risk:** real June grass stands in front of the trunks. A term that clears everything in front of the seed would delete grass that the lidar saw correctly. So it is applied one-sided only (the trunk is not missing).
- An R + depth run is a third June run and needs approval.

**Ouster no-returns (`add_missing_points`, on for every run) and the radar losses.**
- They enter only the ray-drop loss (λ 0.1, indirect via the lidar decoder).
- They can push down a radar Gaussian only where a no-return ray crosses it.
- They never enter T44.
- R logs the fraction of no-return rays crossing gated radar Gaussians. Details: `docs/research/no_return_impact.md`.

### 4.3 Extension: photometric loss for June (B+cam and R+cam, after the core runs)
Two extra runs: the same as core B and R with camera training on (`train-lidar-only False`). They need your approval before running.
- **Why it can help:** the June contact sheets show that the low, upward-looking camera sees over the grass. Upper trunks, canopy and far trunks are clearly visible; the grass hides only the trunk bases and the near field. So the RGB loss can supervise real trunk geometry and colour above the grass.
- **Caveat 1, below the grass the camera sees grass.** This is consistent with the scene: grass Gaussians in front occlude a radar-kept trunk base in the render, so the RGB loss puts almost no gradient on it and does not fight the radar opacity loss.
  - The risk is seed colour. T36 paints a seed only if the camera's view of it is unobstructed: no lidar return from the same camera frame projects within the seed's pixel footprint at a smaller depth. Otherwise the seed is grey, so trunk-base seeds don't start out grass-coloured.
- **Caveat 2, image metrics cannot show the radar gain where it matters.** Below the grass the GT pixels are grass, so PSNR there measures grass.
  - June image metrics (extension runs only) are a "radar does not hurt" check.
  - They are reported for the full frame and inside the **trunk-above-grass mask**: June trunk cuboids, clipped above the grass height measured from the lidar (§5), projected into eval cameras.
- **Caveat 3, what the extension answers:** does adding the RGB loss improve or degrade the lidar geometry of B and R? Compare the §5 lidar metrics of core vs +cam for each model.

## 5. Evaluation: Model 1 vs Model 2 (June only, held-out June poses)
**Held-out poses: middle-block split (decided 2026-10-26; May, B and R all use it):**
- **Middle blocks (T60, in use):** the test set is two contiguous time stretches, centred at 1/3 and 2/3 of the run's window, each (1 − `train_fraction`)/2 of it long. They are the same for camera and lidar. Config: `split.test_block_count: 2` in the session config. It replaced the interleaved split, whose eval frames sat 0.1 s from a train frame and were chosen per sensor (a test camera image usually had a training lidar sweep from the same instant). Test poses still lie on the training trajectory, inside a stretch of a few metres the model never trained on.
- **Interleaved (retired):** SplatAD's LINSPACE split. Still the dataparser default when `test_block_count` is unset, so pre-T60 runs keep their meaning.
- **Backtrack hold-out (candidate):** June 21 → 22 is an out-and-back through the dense stand. Train on chunks 20+21, evaluate on the return pass (chunk 22), which views the same trunks from a few metres to the side.
  - It is harder (lateral novel views) but closer to the question: does the trunk geometry hold from a viewpoint the model never trained on?
  - It needs a split override in `VikingHillDataParser`: eval = frames inside a configured time range, i.e. the backtrack chunk. That is a new transform, T48, with a known-answer test.
  - Radar seeds come from the train split only (CLAUDE.md rule 5), so in this split they come from chunks 20+21.
  - May has a similar return section (the chunk-8 loop and chunk 9), if May needs the same split.

**Two primary tests.** `render_lidar` reports only the first opaque surface on each ray. A trunk base that radar kept behind grass Gaussians is therefore invisible from the training-like eval poses, even though it is in the model: the occluding grass is what lets it survive the lidar losses (`docs/research/evaluation_caveats.md` §3). So the radar claim is judged by two primary tests, always run together on the same GT, cells and thresholds:
- **P1, rendered lidar** asks what a lidar at the eval poses would see.
- **P2, Gaussian occupancy** asks what the model contains.

R must beat B on at least one of them on blocked and no-return cells (below), and must not lose precision on either.

**P1, rendered lidar point cloud** (`08_render_lidar_pointcloud.py`, in the SplatAD job on Slurm). For each eval lidar pose:
1. call `get_lidar_outputs` on the scan's rays;
2. **ungated extraction (binding):** keep a ray as a point if its accumulation is above `eval.render_accumulation_threshold` (config), the same for May, B, R and R+depth. SplatAD's ray-drop gate (`sigmoid(ray_drop) ≤ 0.5`) is primary for no metric:
   - with Ouster no-returns on, it replays the neighbouring frames' drop-outs on the interleaved split;
   - so it would remove radar-recovered trunks in exactly the directions the lidar missed (`docs/research/no_return_impact.md`).
   - Gated numbers are reported beside the ungated ones.
3. go lidar → world, then apply `inv(w2m)` to reach the map frame (T50);
4. accumulate over the eval poses, then voxelise onto the GT grid.

**P2, occupancy from the Gaussians** (from the run's checkpoint; no rendering, so no visibility):
- **T56, load:** `torch.load(step-*.ckpt)["pipeline"]`, keys `_model.gauss_params.{means, scales, quats, opacities}`. Static Gaussians only (all of them; there are no actors). Convert:
  - Σ = R(q̂)·diag(exp(s))²·R(q̂)ᵀ;
  - o = sigmoid(opacities);
  - means and Σ to the map frame with the same `inv(w2m)` as P1.
  - Write `gaussians.npz` next to the lidar renders (`scripts/10_export_gaussians.py`, a job step after `08`).
- **T57, grid:** Gaussian k's 3-D alpha is α_k(x) = o_k·exp(−½(x−μ_k)ᵀΣ_k⁻¹(x−μ_k)), the 3-D form of the alpha the rasteriser uses.
  - Voxel occupancy = 1 − ∏_k(1 − α_k(x)) at the voxel centre, visiting only voxels inside each Gaussian's 3σ ellipsoid.
  - A voxel is occupied if occupancy ≥ `eval.render_accumulation_threshold`, the same value as P1. So the two grids differ only by visibility.
- **Scored cells:** only cells within lidar range and FOV of at least one eval pose (T32). P2 also counts Gaussians in space no sensor saw (e.g. SplatAD's random init points).
- **Precision is essential here:** P2 exposes radar-invented trunks that a render could hide behind grass.

**Why the lidar missed each GT cell.** Every metric is reported per GT cell class, from the recorded June scans at the eval poses:
- **seen:** returns inside the cuboid;
- **blocked:** rays towards it returned from something nearer;
- **no return:** rays towards it got no echo (the `lidar_no_return/` files).

The radar claim concerns blocked and no-return cells. B and R should agree on seen cells.

**Ground truth: files and frame** (evaluation only; never a SplatAD input, seed, seed filter or loss target, CLAUDE.md rule 6).
- **Source labels:** `short_and_tall_grass_labels.json`. This is one file, with byte-identical copies at `data/raw/{may_short_grass,june_tall_grass}/extracted/ros2_jazzy/cuboid_labels/` (config `session.labels_json`). It holds two samples:
  - **May:** `5cm_map_2024_05_dense.ply`, 467 cuboids, selected by `session.labels_sample_key: "2024_05"`;
  - **June:** `5cm_map_2024_06_high_grass_registered_subsampled.ply`, 549 cuboids, selected by `"2024_06"`.

  **The evaluation uses only the June sample.** The May sample is never compared with June (no May ↔ June mapping).
- **Derived GT:** `scripts/05_extract_cuboids.py --config configs/sessions/june_tall_grass.yaml` writes `data/gt/june_tall_grass/` (config `session.gt_dir`):
  - `cuboids.json`: per box `box_id, class_id, class_name, center_xyz, dimensions_xyz, quaternion_xyzw, corners_xyz`;
  - `occupancy.npz`: the GT voxel grid, a class id per voxel; voxel = 0.5 × median trunk diameter; its origin and shape define the grid for P1 and P2;
  - `summary.md`.

  As of 2026-10-26 only `data/gt/may_short_grass/` exists; the June GT must be generated before evaluation.
- **Frame:** the label file carries no frame information. Before evaluation, the June check (§1 step 5) must confirm that the June label map frame equals the June `/icp_odom` map frame. P1 and P2 both land in that frame via `inv(w2m)`.
- **Classes scored:** tree_trunk (1), bush_or_small_tree (4, young trees) and rock (3). Not scored: canopy (2), car, building, lamp/sign and ignore (8); grass has no class.
- **Kept cuboids:** only those within lidar range and FOV of at least one eval pose (T32). Objects this traversal's lidar never saw are kept on purpose, because that is where radar should help.

| metric | definition | role / expectation |
|---|---|---|
| **Detection rate per class (T45)**, P1 and P2, per seen / blocked / no-return | P1: fraction of kept cuboids with ≥ n_min ungated rendered points inside; n_min comes from the GT June lidar point density on the same cuboids at the same poses. P2: fraction of kept cuboids whose occupied-voxel fraction is ≥ f_min, reported as a curve over f_min (no single threshold tuned on GT) | **primary**: R > B on blocked/no-return in P1 or P2; R ≈ B on seen |
| **Trunk precision / false-positive trunks (T46)**, P1 and P2 | (a) fraction of rendered points (P1) or occupied voxels (P2) in the trunk height band that fall inside a GT cuboid; (b) trunk-like vertical clusters matching no cuboid | **primary**: R ≥ B on both (radar must not invent trunks) |
| Present but hidden | per GT cell: occupied in P2 but no rendered point in P1 | explains P1 vs P2; expected larger for R (trunks behind grass Gaussians) |
| Gated P1 metrics | as T45/T46 with SplatAD's ray-drop gate | report (sensor-faithful view) |
| Lidar depth / Chamfer / ray-drop | SplatAD built-in, on eval lidars | report |
| Camera PSNR / SSIM / LPIPS | extension runs only: full frame and inside the trunk-above-grass mask (§4.3) | R+cam ≈ B+cam (radar must not hurt) |

**Reading the results:**
- **R > B in P2 but not in P1:** the radar trunks exist in the model but are hidden behind grass from these poses. The claim "radar puts trunks into the reconstruction" holds. Showing them in a render needs other views: the backtrack split, or an approved raised-sensor extension (`evaluation_caveats.md` §4).
- **R > B in P1 too:** visible gain.
- **R > B in neither:** no gain.
- **R precision < B in either test:** radar invents geometry.

**Figures (`09_compare_runs.py`):**
- **BEV** (T47): rendered points B | R | GT June lidar, with cuboid outlines;
- occupancy grids against `data/gt/june_tall_grass/occupancy.npz`: P1 and P2 for B and R, coloured seen / blocked / no return;
- zooms on trunks that B misses and R recovers.

## 6. New transforms and tests (one known-answer test each; add tests, never edit existing ones)
| id | what | known-answer test |
|---|---|---|
| T34 | radar → map accumulation | a known pose + extrinsic give the analytic map point |
| T35 | trunk-candidate selection | the Doppler filter keeps a static target and drops a mover; height band and lidar-sparse filters on synthetic points |
| T36 | seed covariance + occlusion-aware colour | eigenvectors lie along r̂, with eigenvalues σ_r², (rΔaz)², (rΔel)²; a seed behind a synthetic closer lidar point gets grey, an unobstructed one gets its pixel colour |
| T39 | seeds → dataparser frame | w2m applied to Σ gives RΣRᵀ |
| T40 | time / scene-box crop | exact kept and dropped counts |
| T41 | radar Gaussian group | count = M; scales from Σ, not kNN; without seeds = stock SplatAD |
| T42 | Mahalanobis gate | a point on the q-ellipsoid is in, slightly outside is out |
| T43 | opacity loss | hand-computed mean of −log σ(o); zero gated Gaussians gives 0 and is flagged |
| T44 | ray expected depth (optional) | a single Gaussian on the ray gives its mean depth |
| T45 | cuboid detection rate | points inside / outside a rotated box give the exact rate |
| T46 | trunk precision / FP clusters | one matched and one unmatched synthetic cylinder give precision 0.5 and 1 FP |
| T47 | BEV raster | a known point set gives exact pixel counts |
| T49 (done) | manifest sweep period covers the per-point time of every extracted sweep | derived period kept unless a longer \|t\| is written; a synthetic chunk with stretched sweeps passes the contract after an incremental, fresh or resumed run |
| T60 (done) | middle-block test split shared by every sensor | camera 0..12 s and lidar 0.5..11.5 s, f = 0.5, K = 2 give blocks [2.5, 5.5) and [6.5, 9.5) with the exact test frames (half-open edges); the real dataparser and scripts/09 pick identical frames |
| T61 (done) | manifest elevation mapping = T51 far-field elevation per ring (replaces T16's parallax-biased medians; T16 now only supplies the ring set) | synthetic beams with elevation_inf + k/r give back elevation_inf exactly while T16's median sits at least k/r_max above it; rings come out ascending; the pipeline writes the cached beam model's far-field angles |
| T48 | time-range split override (backtrack hold-out, only if chosen) | frames inside the eval range go to eval and all others to train; both splits non-empty and disjoint |
| T51 | Ouster beam directions fitted from raw returns (Phase B) | synthetic beams with a known parallax and either column direction give back the exact far-field angles; a cloud the model cannot describe fails the gate |
| T52 | (ring, column) → unit direction | known angles give the exact vectors |
| T53 | no-return cells of an organized cloud | known range-0 cells give their exact ring, column, time; an unorganized cloud raises |
| T54 | no-return directions to the reference pose → sidecar rows | a known yaw rate rotates each direction by rate × time; t = capture − reference |
| T55 | sidecar rows → 2000 m placeholders (dataparser, `add_missing_points`) | xyz = 2000·d, depth beyond SplatAD's return threshold, layout matches the sweep |
| T56 (done) | checkpoint Gaussians → map-frame means, covariances, opacities (P2) | a hand-made state dict: log-scales, an unnormalised quaternion and logit opacity give the exact Σ = R·diag(e^s)²·Rᵀ and sigmoid; `inv(w2m)` moves μ and rotates Σ |
| T57 | Gaussian occupancy grid (P2) | one Gaussian gives occupancy σ(o) at its centre and visits exactly its 3σ voxels; two overlapping Gaussians give 1 − (1−α₁)(1−α₂) |

## 7. File structure (new items marked +)
```
configs/sessions/{may_short_grass,june_tall_grass}.yaml (Scene A chunks)  +configs/radar/june.yaml
+configs/experiments/{may_smoke,may_A,june_B,june_R}.yaml
src/viking_splatad/
  +radar_init/accumulate.py select.py seeds.py      # T34–T36
  +radar_loss/opacity.py                            # T42, T43 (pure torch, CPU-testable); +depth.py (T44, optional)
  splatad/viking_conversions.py (+T39, T40)  splatad/viking_dataparser.py (radar_inputs_dir)
  +splatad/radar_model.py (T41 + losses)  +splatad/radar_method.py
  dataset/schema.py (optional radar_inputs entry)  contract/splatad_contract.py (radar checks)
  +eval/rendered_pointcloud.py detection.py bev.py  # T45–T47
  +eval/gaussian_occupancy.py                       # T56 (checkpoint → gaussians.npz, torch, CPU) + T57 (grid, numpy)
scripts/+07_build_radar_seeds.py +08_render_lidar_pointcloud.py +09_compare_runs.py +slurm/{splatad_run.sbatch,submit_run.py}
```

## 8. Execution order
1. **Choose Scene A** (§1): guess and fetch the chunks, check the contact sheets, set the session configs, then extract them and pass the contract.
   - In parallel: the May chunk 1 Slurm smoke test (Docker image, `ns-train splatad viking-data`).
2. **May Scene A** run with held-out metrics (§3).
3. **Build with parallel subagents**, each told to add tests and never edit existing ones; `transform-test-runner` runs after each:
   - (a) radar seeds T34–T36 + `07`;
   - (b) parser/contract/model T39–T43;
   - (c) eval `08`/`09` T45–T47, plus the P2 Gaussian occupancy T56–T57;
   - (d) the June GT: `05_extract_cuboids.py` for `june_tall_grass`, after the §1 June frame check.
4. **June B and R** (core, lidar-only) on Slurm: a smoke run each, then the full step count.
5. **Evaluate** (§5) and write `docs/experiments.md`.
6. **Extensions, each with approval:**
   - B+cam and R+cam (§4.3);
   - R + depth loss (T44).

## Verification
- `uv run pytest` green, including traceability. `affected_tests.py` shows no modified existing tests.
- `06_validate_contract.py` passes for May and June Scene A, including the radar checks.
- The chosen chunks' contact sheets show low grass + trunks (May) and high grass + trunks (June). The June `/icp_odom` lies on the June label map.
- May: held-out metrics beat the baselines, and the renders show the forest.
- With `radar_inputs_dir=None`, the parser gives outputs identical to the stock path.
- R logs: the radar group count equals M, the opacity loss decreases, and the gated-seed fraction stays high.
- B's rendered point cloud overlays the June GT lidar, and its Chamfer matches the built-in metric.
