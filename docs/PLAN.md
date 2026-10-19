<!-- In-repo execution plan (approved 2026-10-18). Supersedes ~/.claude/plans/start-a-new-claude-cosmic-cascade.md,
     which keeps the Phase-1 findings, the contract table and T1–T33. -->

# June: SplatAD baseline vs SplatAD + radar init + radar loss (proposal §3)

> **Radar's role, stated plainly.** We are **not** rendering radar. We do not add a radar sensor, a radar renderer, radar images, or radar evaluation metrics.
>
> Radar is used in exactly two ways, both in service of the **lidar and camera** reconstruction:
> 1. **Initialization.** Extra seed Gaussians are placed where radar detects trunks that the lidar misses.
> 2. **Geometric regularizer.** A loss term asks the scene to contain opaque geometry at those radar-detected locations. This compensates for lidar returns that tall grass blocks.
>
> The outputs we judge are SplatAD's **rendered lidar (→ point cloud)** and **rendered camera images**.

## Context
The team proposal (`docs/proposal/Final_Forest_Reconstruction_Proposal.pdf`, transcription in `docs/proposal/proposal.md`) asks whether adding 4D radar to LiDAR+camera 3DGS improves trunk reconstruction under foliage occlusion. Until now the repo only targeted May rendering.

**Scope (user decisions):**
- **May** is a readiness check only. Rendering a May forest scene is the evidence that poses, calibration and the processing scripts are correct.
- **June (tall grass)** gets exactly **two runs**:
  - **B** = SplatAD with lidar + camera;
  - **R** = SplatAD + radar-guided init (§3.1) + a radar loss inside SplatAD training (§3.2).
- **Goal:** SplatAD's rendered lidar turned into a point cloud, plus held-out camera renders.

**The radar loss is label-free, like the rest of SplatAD's training.** Every stock SplatAD loss is self-supervised against a raw sensor measurement:
- camera RGB against the captured pixel;
- lidar depth, intensity and ray-drop against the measured scan.

The radar loss follows the same pattern. The radar return is the target, meaning "something reflective is here". It keeps the proposal's form:

- **Per-point loss:** L_i = −log p_i(y_i), with y_i = *occupied* for every radar seed.
- **Occupancy probability:** p_i(occupied) = 1 − exp(−D(p_i)), the standard volumetric opacity of the Gaussian density D at the seed.
- **Weight:** w_i = 1 + α·V_i.
- **Total:** L_radar = (1/N) Σ w_i·L_i.

The radar term **compensates for occluded lidar**, and the vegetation weight concentrates it where grass blocks the lidar. Two consequences:
- **No cuboid labels and no semantic head.** The trunk indicator [y_i = trunk] is replaced by the GT-free trunk-candidate selection applied to the seeds.
- **V_i:** in June-only training the binary seasonal V_i = 1 everywhere, so w_i = 1 + α is a constant that only rescales λ. The **continuous line-of-sight V_i = N_v/N_k** (proposal stretch goal) is what makes the weight spatially selective. It is scheduled after the core R run works.

The proposal's α/λ typo is resolved as α.

## Research findings (neurad-studio @ `8ba9b51`; gsplat = carlinds/splatad, unpinned upstream)
These go into `docs/research/splatad_extension_points.md` with file:line citations.

- **Seeding path.** The AD dataparser never sets `points3D_xyz`. Instead, `SplatADPipeline` (`pipelines/splatad_pipeline.py:106-154`):
  1. takes the train lidar scans and keeps points with norm < `valid_lidar_distance_threshold`;
  2. converts them to world coordinates and adds the `paint_points` RGB;
  3. passes `seed_points=(xyz, rgb, times)` to `model.setup(..., metadata=train_dataset.metadata, …)`.
  `train_dataset.metadata` is a **deepcopy of the dataparser metadata**. Any key we add in the dataparser therefore reaches the model as `self.kwargs["metadata"]`.
- **Model init.** `SplatADModel.populate_modules` (`models/splatad.py:302-335`):
  1. splits the seeds into static and actor points;
  2. appends 2×300k random far and close points to the static set;
  3. calls `create_gauss_param_dict(dynamic, [static], flip)`.
  Each static group gets its own joint kNN-3 scale, random quats, rgb/255, `features_rest` ~ N(0,1), `init_opacities`, and `id = n_actors`. The strategy state is initialised afterwards (:400-430), and the optimizers are built later from `get_param_groups`. **So Gaussians appended inside `create_gauss_param_dict` are picked up by the strategy and optimizers automatically.**
- **Dataparser frame** (`ad_dataparser.py:176-244`):
  - `_adjust_times` makes all times relative to `time_offset = min(camera ∪ lidar times)`.
  - `_adjust_poses` left-multiplies every pose by `w2m = inv(mean-trajectory pose)` (3×4), returned as `dataparser_transform`. Scale is 1.
  - The metadata dict is open.
  - `assert radars is None` (:180) blocks radar *as a sensor*.
- **Densification.** gsplat ops carry every `gauss_params` entry, but MCMC relocation recycles low-opacity Gaussians elsewhere. Radar Gaussians can therefore move or vanish. That is why the loss queries the *fixed seed locations*, not the Gaussians themselves. N changes every 100 steps between steps 500 and 15000.
- **Losses.**
  - The trainer sums every `get_loss_dict` entry (`engine/trainer.py:551`).
  - The stock terms are: camera L1 + SSIM; MCMC scale and opacity regularisers; camera velocity; lidar depth L1, intensity MSE, ray-drop BCE, and line-of-sight opacity.
  - Methods register via `NERFSTUDIO_METHOD_CONFIGS`, so no neurad patch is needed.
  - A second rasteriser call per step would corrupt `self.info` and the strategy, so the radar loss is computed directly on the Gaussian parameters, without rasterising.
- **Lidar → point cloud.**
  - `ns-render --render-point-clouds` fails for SplatAD (`Unknown sensor type`).
  - The working path is `filter_lidar_pred_and_gt(..., output_point_cloud=True)` (`splatad.py:1294-1314`). It keeps rays with ray-drop ≤ 0.5 and returns points in the lidar frame.
  - Alternatively, `get_average_eval_image_metrics(dump_img_to_disk=True)` writes the `pred_points` npz files.

## Dataparser extension for radar (detailed)

### Which dataparser: **yes, we inherit SplatAD's dataparser**
SplatAD's dataparser is `ADDataParser`, a generic base class with no Viking support. Our parser is `class VikingHillDataParser(ADDataParser)`:
- this class already exists in our repo;
- radar support is added to it;
- `ADDataParser` itself is not edited; all of its logic is reused through `super()`.

```
neurad-studio (upstream, pinned 8ba9b51, NOT modified for radar)
  DataParser
   └─ ADDataParser                 nerfstudio/data/dataparsers/ad_dataparser.py   ← SplatAD's generic AD parser
        ├─ PandaSetDataParser, py123d..., WOD...   (upstream dataset parsers, untouched)
        └─ VikingHillDataParser    src/viking_splatad/splatad/viking_dataparser.py ← OURS, already exists (T27), `viking-data`
                                   → EXTENDED with radar config fields + a post-processing step
model side, same pattern:
  SplatADModel (upstream, untouched) └─ SplatADRadarModel  src/viking_splatad/splatad/radar_model.py ← NEW subclass, method `splatad-radar`
```
- **What our subclass already overrides:** `_get_cameras`, `_get_lidars`, `_read_lidars`, `_add_sensor_velocities` and `_get_actor_trajectories`. Its `_generate_dataparser_outputs` registers the lidar model and then calls `super()`.
- **What changes:**
  - radar is added only in this file, in our repo;
  - new config fields (§B);
  - a step after `super()` returns (§C).
- **What stays as it is:**
  - the only neurad source modification remains the existing `OUSTER_OS0_32` lidar-type patch;
  - everything else is a subclass or a plugin.
- **Runs B and R use the same parser.**
  - B: `radar_inputs_dir=None`, which gives byte-identical behaviour to today.
  - R: `radar_inputs_dir` is set.

**What "radar is not a sensor" means.** `ADDataParser` takes two kinds of input.
1. **Sensors.** These are listed in config (`cameras=`, `lidars=`, `radars=`).
   - Each one gets per-frame poses and times, a `sensor_idx`, a train/eval split, a renderer, and a loss against its own measurements.
   - Listing radar in `radars=` hits `assert radars is None, "Radars not supported yet"` (`ad_dataparser.py:180-181`).
   - SplatAD also has no radar renderer.
   - So `radars=()` stays empty.
2. **Metadata.** This is a free-form dict handed to the model at setup.
   - Radar enters through it as **one** precomputed set, `radar_seeds`, already in SplatAD's frame and clock.
   - Each seed is both an extra Gaussian at init and a query point of the label-free radar loss.
   - Radar is never rendered or held out for evaluation.

### A. Offline radar seeds (CPU, Mac): `scripts/07_build_radar_seeds.py` → `data/intermediate/june_tall_grass/radar_inputs/`
Everything is computed in the **map frame**, in **absolute float64 seconds** (the same clock as `poses.npz`). The dataparser is the only place that applies `w2m` and `time_offset`, and it does so exactly once. This mirrors the rule that the OpenCV→nerfstudio flip is applied exactly once.

| file | content |
|---|---|
| `radar_seeds.npz` | `means_map [M,3] f32`, `covariances_map [M,3,3] f32` (SPD), `rgb [M,3] u8`, `sensor_origin_map [M,3] f32` (mean radar position of the supporting detections; ray start for continuous V), `obstruction_v [M] f32` (NaN = undefined / not computed), `support_count [M] i32`, `first_time_s / last_time_s [M] f64` |
| `radar_inputs.json` | `frame: "map"`, `time_base: "absolute_s"`, config hash, git SHA, `extrinsics_variant: high_grass`, radar clock offset (T24), every derived threshold (Doppler σ, persistence percentile → count, height band, lidar-sparsity radius), M |

`manifest.json` gains an optional `radar_inputs` entry with the relative path and sha256. This is covered by a **new** writer test; the existing tests stay untouched.

Steps inside the script:
1. **T34: accumulate.** For each scan, T_map_radar(t) = T_map_base(t + δ_radar) · T_base_radar.
   - Pose interpolation is T3.
   - δ_radar comes from T24.
   - T_base_radar is the June `high_grass` extrinsic from T5.
2. **T35: trunk candidates, GT-free.** Every threshold is derived from the data:
   - **static:** |v_doppler − (−v_ego·r̂)| ≤ k·σ_doppler, where σ is the robust MAD of the per-scan residuals;
   - **persistent:** voxel hit count ≥ a configured percentile;
   - **trunk height band:** above the lidar ground and below the canopy onset (from the accumulated lidar height histogram);
   - **lidar-sparse (§3.1, "where foliage blocks LiDAR"):** no train-split lidar seed within the radar's lateral resolution. These lidar seeds are rebuilt the way the pipeline does it: same LINSPACE train indices (T21), same `valid_lidar_distance_threshold`.
3. **T36: seed Gaussians.**
   - The mean is the voxel centroid.
   - The covariance is R_ray · diag(σ_r², (r·Δaz)², (r·Δel)²) · R_rayᵀ. Here r is the median range of the supporting detections. Δaz/Δel/σ_r come from `hugin_radar_startup_params.txt` or the measured beam spacing; this is the 0.87 m at 40 m lateral blur.
   - The colour is painted from the nearest-in-time train camera (T25 reprojection, undistorted K). Grey if no camera sees the seed.
4. **(stretch) T44–T46: obstruction_v.**
   - Build a vegetation grid G(c) from the accumulated lidar.
   - Run an Amanatides–Woo DDA from `sensor_origin` to the seed.
   - V = N_v/N_k, or NaN if N_k = 0. NaN gives w = 1.

### B. `VikingHillDataParserConfig`: new fields
- `radar_inputs_dir: Optional[Path] = None`. `None` means run B; the outputs are identical to today's parser.
- `radar_time_window: Literal["sensor_window", "all"] = "sensor_window"`. A seed is kept only if its supporting detections overlap the camera/lidar time window that survives `_filter_based_on_time`.
- `radars` stays `()`.

### C. `VikingHillDataParser._generate_dataparser_outputs`: post-process after `super()`
1. If `radar_inputs_dir is None`, return the outputs unchanged (run B). A test asserts they are identical.
2. **Load and verify.** Check that frame == "map", time_base == "absolute_s", the sha256 matches the manifest, and the session matches.
3. **Pose frame.** `w2m = outputs.dataparser_transform`, with R = w2m[:, :3] and t = w2m[:, 3]. Then:
   - means' = R·means + t;
   - origins' = R·origins + t;
   - Σ' = R·Σ·Rᵀ.
   Assert `dataparser_scale == 1`, so a future upstream scale fails loudly.
4. **Time window.** Shift the seed spans by `outputs.time_offset` (in float64) and drop seeds outside the sensor window. Seeds are static, so no per-seed time goes to the model.
5. **Scene box.** Drop seeds outside `outputs.scene_box.aabb`. Count them and report the count in the contract report.
6. **Split.** Attach the key only when `split == "train"`, because the model reads `train_dataset.metadata` and eval outputs stay radar-free.
7. **Attach** `outputs.metadata["radar_seeds"]` in the **dataparser frame**: {`means [M,3] f32`, `covariances [M,3,3] f32`, `rgb [M,3] u8`, `sensor_origins [M,3] f32`, `obstruction_v [M] f32`}.

The numeric parts are pure numpy in `viking_conversions.py`, so they can be tested in the plain venv:
- **T39** `radar_seeds_to_dataparser_frame(means, covs, origins, w2m)`;
- **T40** `crop_seeds(time window, scene box)`.

### D. Model: `SplatADRadarModel(SplatADModel)` in `src/viking_splatad/splatad/radar_model.py`
- **`create_gauss_param_dict` override (T41).**
  1. Call `super()` first, so the lidar, far and actor groups are built exactly as before.
  2. If `metadata["radar_seeds"]` exists, concatenate a radar group onto every `gauss_params` entry:
     - `means`;
     - `scales = log(√eig(Σ))`;
     - `quats` from the eigenvectors (det forced to +1, wxyz);
     - `features_dc = rgb/255`;
     - `features_rest ~ N(0,1)`;
     - `opacities = logit(config.radar_seed_opacity)`;
     - `id = n_actors` (static).
  No new per-Gaussian parameters are added, so the strategy and optimizers are unchanged.
- **`get_loss_dict` override: `radar_loss` (T42, T43).** It follows the existing pattern of the MCMC regularisers, computed directly on the Gaussian parameters with no rasterisation.
  1. Query points q_i are the **fixed seed means from metadata**. A per-step random subset is used.
  2. The density is D(q_i) = Σ_k σ(o_k)·exp(−½·dᵀ(Σ_k + Σ_seed,i)⁻¹·d), with d = q_i − μ_k, over the K nearest static Gaussians (chunked `torch.cdist`). Adding Σ_seed means the radar blur sets the tolerance, so the loss does not demand sharper geometry than the radar resolves.
  3. p_i = 1 − exp(−D(q_i)).
  4. L = (1/N) Σ w_i · (−log(p_i + ε)), with w_i = 1 + α·V_i, where V_i = 1 (binary June) or `obstruction_v` (stretch, NaN → 0).
  5. `radar_loss = λ_radar · L`. Gradients reach the means, scales, quats and opacities of nearby Gaussians.
  6. **Scope of the loss:**
     - It is defined per radar seed, not per Gaussian. Each seed asks "is there enough opacity where radar saw a reflector?"
     - Seeds come from the accumulated, filtered June scans, not from single raw scans.
     - Only Gaussians among the K nearest to a seed, within the radar blur, receive a gradient. All others are trained by the stock camera and lidar losses only.
     - The loss only says "occupied here". It never says "empty here", because radar free space is unreliable (lateral blur, multipath). Free space stays with the lidar ray-drop and line-of-sight losses.
  7. λ_radar is set so the step-0 radar term equals the step-0 lidar depth term (measured and logged). α comes from config; it only matters with the continuous V.
- **`radar_method.py`.** A `MethodSpecification` named `splatad-radar`: a copy of neurad's `splatad` TrainerConfig with this model and the Viking dataparser. No new optimizer entries are needed.
  - B uses the stock `splatad` with `radar_inputs_dir=None`.
  - R uses `splatad-radar`.
- **Tests (neurad CPU venv, like T27/T28):**
  - the radar group count equals M, and its scales come from Σ, not kNN;
  - with no `radar_seeds`, the model equals stock SplatAD;
  - `radar_loss` is in the loss dict;
  - L → 0 when opaque Gaussians sit on every seed;
  - a single Gaussian gives the analytic D.

### E. Contract (`contract/splatad_contract.py`, new checks)
- `radar_seeds.npz` sha256 matches the manifest, and the frame and time-base tags are correct.
- All values are finite.
- Every Σ is SPD.
- `obstruction_v ∈ [0,1]` or NaN.
- Seed spans lie within the sensor window.
- The post-crop retention fraction is reported.

Rule 4 applies: no Slurm run without a passing report.

## Evaluation (B vs R; same split, same seed, MCMC strategy)
- **Held-out June camera poses:** PSNR, SSIM and LPIPS (proposal §4). Reported for the full frame and inside a trunk mask (June trunk cuboids projected into the eval cameras).
- **Lidar:** the built-in depth, Chamfer and ray-drop metrics.
- **`08_render_lidar_pointcloud.py`** (inside the SplatAD container):
  1. for each eval lidar, call `filter_lidar_pred_and_gt(output_point_cloud=True)`;
  2. apply `lidar_to_world`, then `inv(w2m)`, to get the map frame;
  3. accumulate, then build an occupancy grid with `labels/occupancy.py`;
  4. render side-by-side visualisations with the trunk cuboids overlaid;
  5. also produce qualitative trunk renders.
- Cuboids are used **only for evaluation** (rule 6 unchanged). Training never sees them, so cuboid-based trunk numbers are clean.

## File structure (new items marked +)
```
docs/+proposal/{Final_Forest_Reconstruction_Proposal.pdf, proposal.md}   # byte copy (md5 f4409efc…) + transcription w/ equations
    +research/splatad_extension_points.md   +PLAN.md   +experiments.md
configs/sessions/june_tall_grass.yaml (forest window)  +radar/june.yaml  +experiments/{may_check,june_B,june_R}.yaml
src/viking_splatad/
  +radar_init/accumulate.py select.py seeds.py           # T34–T36
  +radar_loss/occupancy_ce.py                            # D(q), p, weighted CE — pure torch, CPU-testable (T42, T43)
  +radar_loss/obstruction.py vegetation_grid.py ray_traversal.py   # stretch: V_i (T44–T46)
  splatad/viking_conversions.py (+T39, T40)  splatad/viking_dataparser.py (§B–C)
  +splatad/radar_model.py (T41 + loss wiring)  +splatad/radar_method.py
  dataset/schema.py (optional radar_inputs entry)  contract/splatad_contract.py (§E)
  +eval/rendered_pointcloud.py trunk_mask.py
scripts/+07_build_radar_seeds.py +08_render_lidar_pointcloud.py +09_compare_runs.py +slurm/{splatad_run.sbatch,submit_run.py}
```
Every new T-id gets one known-answer test, kept minimal:
- a known pose and extrinsic give the analytic radar point;
- the Doppler filter keeps a static target and drops a mover;
- covariance eigenvectors lie along r̂, with eigenvalues σ_r² and (rΔθ)²;
- w2m applied to Σ gives RΣRᵀ;
- the crop counts are correct;
- a single isotropic Gaussian at distance d gives D = σ(o)·exp(−d²/2(s²+σ²));
- a CE with α=0 and V=1 equals the hand-computed −log(1−e^{−D});
- the DDA along an axis visits ⌈L/voxel⌉ cells (stretch);
- depth×dir reproduces a synthetic range image;
- the trunk-mask silhouette of a box is correct.

## Rule changes (CLAUDE.md, README)
- Goal → the research question. Plan pointer → `docs/PLAN.md`.
- **Rule 5:** radar is never a SplatAD *sensor*. It enters only as `radar_seeds` metadata, which is:
  - built offline in the map frame;
  - transformed into the dataparser frame only by `VikingHillDataParser`;
  - train split only;
  - gated by the contract;
  - used for init and the label-free radar loss.
- **Rule 6** is unchanged: cuboids are evaluation-only.
- gsplat (carlinds/splatad) is unpinned in neurad's pyproject; `docker/Dockerfile.splatad` already pins it (`SPLATAD_COMMIT=6e31ad7…`), which is the commit the research read.

## Execution order
1. Copy the PDF and write `proposal.md`, the research note, `docs/PLAN.md` and the CLAUDE.md/README edits.
2. **May readiness:** run a Slurm smoke test, then a full SplatAD run on the May forest window. Render held-out RGB and lidar. Pass = the forest is visible.
3. **June data:** fetch the forest chunks, extract them, and pass the contract. Verify that the label map frame matches `/icp_odom`; this is needed for the trunk-mask evaluation only.
4. Build with parallel subagents, each told to add tests and never edit existing ones:
   - (a) `radar_init` + `07`;
   - (b) `radar_loss`;
   - (c) dataparser §B–C, contract §E, and `radar_model`/`radar_method`;
   - (d) evaluation, `08` and `09`.
   After each, run `transform-test-runner`.
5. On Slurm: run June B and June R, each as a smoke run and then 30001 steps. Then run `08` and `09`, and write the results to `docs/experiments.md`. Stretch: R with the continuous V.

## Verification
- The md5 of the copied PDF matches the original.
- `uv run pytest` is green, including traceability. `affected_tests.py` shows no modified existing tests.
- `06_validate_contract.py` passes for May and June, including the radar checks.
- The May render shows the forest.
- With `radar_inputs_dir=None`, the dataparser produces outputs identical to the stock parser.
- In R's logs, the radar group count equals M after cropping, and `radar_loss` decreases.
- The `08` point cloud for B overlays the June GT lidar, and its Chamfer matches the built-in metric.
