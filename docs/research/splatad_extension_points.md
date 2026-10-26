# SplatAD extension points for radar init and a radar loss

**Sources:**
- neurad-studio `8ba9b5116b8a2822a80c64f63a4ae64c5871aa68` (`third_party/neurad-studio`, same as `NEURAD_STUDIO_COMMIT` in `docker/Dockerfile.splatad`);
- gsplat fork carlinds/splatad `6e31ad766d39e0c33f9034a2ed772d51364b2343` (= `SPLATAD_COMMIT`).

Paths are relative to the neurad-studio root. The research was read-only, and nothing here has been run under CUDA yet.

**Purpose.** Radar is **not rendered**. It enters SplatAD only as (1) extra seed Gaussians and (2) label-free radar losses: an opacity loss near the seeds, and optionally a depth loss. These compensate for lidar returns blocked by tall grass. This note records where SplatAD lets us do that without modifying upstream code.

## 1. How SplatAD initialises Gaussians

### 1.1 Dataparser → datamanager
**`ADDataParser._generate_dataparser_outputs`** (`nerfstudio/data/dataparsers/ad_dataparser.py:176-244`):
- `assert radars is None, "Radars not supported yet"` (:180-181). Radar cannot be a sensor.
- `_filter_based_on_time` (:269), then `_read_lidars`, then `_remove_ego_points` (:263).
- `_adjust_times` (:300-314): all times become relative to `time_offset = min(camera ∪ lidar times)`, as float32.
- `_adjust_poses` (:316-323): every pose is left-multiplied by `w2m = inv(mean-trajectory pose)` (3×4; `_get_world_to_mean_transform` :649). The result is returned as `dataparser_transform`, with `dataparser_scale=1.0`.
- LINSPACE train/eval split (:205-209).
- `metadata` (:234-243) is an **open dict**: `lidars`, `point_clouds` (sensor frame, float32), `point_clouds_times`, `trajectories`, `sensor_idx_to_name`, `duration`.
- **`points3D_xyz` / `points3D_rgb` are never set by AD parsers.** The pipeline branch that reads them (`pipelines/splatad_pipeline.py:99-105`) would build a 2-tuple that `split_seed_points` cannot consume, because it needs times at `models/splatad.py:645`. Don't use it.

**Datamanager** (`data/datamanagers/full_images_lidar_datamanager.py`). `paint_points` (:513-562):
- uses the `topk = n_cam_sensors × paint_points_topk` (default 2) train cameras closest in time;
- projects the points and colours them, with the closest camera winning;
- leaves unpainted points random;
- writes the result to `train_dataparser_outputs.metadata["point_clouds_rgb"]`.

### 1.2 Pipeline builds the seeds (`pipelines/splatad_pipeline.py:106-167`)
- Train scans only. A point is kept if `norm < lidars.valid_lidar_distance_threshold` (default 1e3, `cameras/lidars.py:116`), which drops `add_missing_points` dummies at 2e3.
- Points are moved to world coordinates with `transform_points(pc[:, :3], l2w)`.
- `seed_pts = (xyz_world, rgb, times)` is passed to `model.setup(..., metadata=self.datamanager.train_dataset.metadata, seed_points=seed_pts)`.
- `train_dataset.metadata` is a **deepcopy made at dataset creation** (`data/datasets/base_dataset.py:54`), before `paint_points` runs. **Every key a dataparser adds to `metadata` reaches the model as `self.kwargs["metadata"]`.** This is our radar entry point.

### 1.3 Model init (`models/splatad.py`)
**`populate_modules`** (:302-335):
- `split_seed_points` (:632-689) separates points inside actor boxes from static points.
- It then **always** adds `n_far_points` (300k, :244) points on a far hemisphere and 300k uniform points in the scene box, both to the static set (:314-329).
- It calls `self.create_gauss_param_dict(dynamic_points, [static_points], flip_actors_at_init)` (:331). **The static groups are passed as a list.**

**`create_gauss_param_dict`** (:544-630) builds one group at a time:

| parameter | init |
|---|---|
| means | seed xyz |
| scales | log(mean kNN-3 distance × `init_scale`), computed jointly over the group (sklearn, :708-726) |
| quats | random |
| features_dc | rgb/255 |
| features_rest | N(0,1), `feature_dim=13` |
| opacities | logit(`init_opacities`): MCMC 0.5 with `init_scale` 0.2 (:277-280); default strategy 0.1 / 1.0 |
| id | `min(group, n_actors)`, not optimised (static = `n_actors`) |

After this:
- the strategy is set up (:400-430);
- the optimizers come from `get_param_groups` (:787-803): means, scales, quats, features_dc, features_rest, opacities.

**⇒ Gaussians appended inside a `create_gauss_param_dict` override are seen by the strategy and the optimizers with no further hooks.** No random-init flag exists in SplatAD; `random_init` is Splatfacto-only. `max_num_seed_points=2e6` subsamples the seeds randomly (:691-706).

**Zero actors** (`load_cuboids=False`): `n_actors=0`, the actor loops are empty, and the static id is 0. Reading the code, this looks fine; it has not been run.

### 1.4 Densification (default `strategy="mcmc"`, `model_components/strategy.py:129-132`)
- Every `refine_every=100` steps, between steps 500 and 15000, Gaussians with opacity ≤ `mcmc_min_opacity` (0.005) are **relocated**: overwritten with copies of live Gaussians sampled by opacity.
- The count grows 5% per refine, up to `mcmc_cap_max=5e6`, and noise is added to positions.
- gsplat ops act on **every** `gauss_params` entry: duplicate copies, split repeats, remove masks, relocate overwrites from the source.
- **Consequence:** radar-seeded Gaussians can be moved or recycled. A per-Gaussian "radar origin" flag is diagnostic at best. Anything that must persist, such as the regularizer's query points, should be stored outside `gauss_params`, i.e. the fixed seed means in metadata.

## 2. How SplatAD computes losses, and how to add one
**`get_loss_dict`** (`models/splatad.py:1356-1432`). Each step is either a camera image or a lidar scan (`next_train`, datamanager :438-456).

| term | formula | weight (default) |
|---|---|---|
| `main_loss` (camera) | (1−λ)·L1 + λ·(1−SSIM) | `ssim_lambda=0.2` |
| `mcmc_scale_reg` | mean(exp(scales)) | 0.001 |
| `mcmc_opacity_reg` | mean(sigmoid(opacities)) | 0.005 |
| `camera_velocity_regularizer` | L2 of velocity adjustments | 1e-6 / 1e-5 |
| `depth_loss` (lidar) | L1 on range, top 5% errors dropped | `depth_lambda=0.1`, quantile 0.95 |
| `intensity_loss` | MSE | 1.0 |
| `ray_drop_loss` | BCE-with-logits | 0.1 |
| `alpha_sum_until_points_loss` | opacity accumulated in front of the measured point (line of sight) | 0.1, `line_of_sight_dist=0.8` |

**Which lidar rays each term sees** (`splatad.py:1265-1291, 1410-1430`; `full_images_lidar_datamanager.py:244-262`). A ray "returned" if its depth ≤ `valid_lidar_distance_threshold` (1e3 m).
- Depth, intensity and line of sight use **returned rays only**.
- Ray drop (BCE, target = "did not return") uses every valid ray. It is the only term that sees no-return rays, which exist only with `add_missing_points` (2e3 m placeholders).
- The ray-drop logit comes from `lidar_decoder` on the alpha-blended features plus the ray direction, so it reaches the Gaussians only indirectly.
- The `*_dropped` predictions are computed but unused.
- Our dataset adds the **recorded** Ouster no-returns from the `lidar_no_return/` sidecar (T51–T55, `extract/no_return.py`), not neurad's `_get_missing_points` gap guess.

**Adding a term:**
- The trainer sums **every** dict entry with no filtering: `loss = functools.reduce(torch.add, loss_dict.values())` (`engine/trainer.py:551-553`). A subclass just does `d = super().get_loss_dict(...); d["radar_loss"] = λ·L; return d`. `metrics_dict` is logged only.
- Every loss in the table is **self-supervised against a raw measurement**, with no semantic labels. The radar loss follows the same pattern: the radar return is the target.
- The MCMC regularizers already compute losses **directly on `self.scales` / `self.opacities`**. The radar term does the same on `self.means`, `self.scales`, `self.quats` and `self.opacities`, with no rasterisation.
- **Do not rasterise radar inside the step.** `lidar_rasterization` (:1135-1166) overwrites `self.info` and calls `strategy.step_pre_backward` (:1173-1176). A second call per step would corrupt the densification statistics. (Arbitrary-ray rendering by filling `raster_pts` is possible in principle, but it is not needed and not used.)
- New learnable modules would need `get_param_groups` and optimizer config entries. The planned radar loss adds none.
- **Registration without patching neurad:** `NERFSTUDIO_METHOD_CONFIGS="splatad-radar=viking_splatad.splatad.radar_method:SPEC"` (`plugins/registry.py:34-79`, merged at `configs/method_configs.py:538`). This is the same mechanism as `NERFSTUDIO_DATAPARSER_CONFIGS` for `viking-data`. The base `splatad` TrainerConfig is at `configs/method_configs.py:319-391`.

## 3. Rendered lidar → point cloud
- `ns-render --render-point-clouds` (`scripts/render.py:1140-1171`) goes through `ADModel.get_outputs_for_lidar` (a RayBundle). SplatAD's `get_outputs` raises `ValueError("Unknown sensor type")` on that path (:1232-1238), so it will most likely fail.
- `ns-export gaussian-splat` asserts a `SplatfactoModel` (`scripts/exporter.py:535`), so it fails for SplatAD.
- **Working path:** `filter_lidar_pred_and_gt(..., output_point_cloud=True)` (`models/splatad.py:1294-1314`):
  - xyz = depth × direction(az, el) + v·t, keeping rays with `sigmoid(ray_drop) ≤ 0.5`;
  - the points are in the lidar frame; to get world coordinates, apply `lidar_to_worlds`, and to get the map frame, apply `inv(w2m)` as well.
- `SplatADPipeline.get_average_eval_image_metrics(..., dump_img_to_disk=True)` writes `fid/lidar/points_{idx}.npz` (`pred_points`, `gt_points`; :337-366). `ns-eval` never sets that flag, so our script calls it directly.
- **Built-in eval metrics** (:1464-1507):
  - camera: `psnr`, `ssim`, `lpips`;
  - lidar: `depth_median_l2`, `depth_mean_rel_l2`, `intensity_rmse`, `ray_drop_accuracy`, `chamfer_distance`, `median_chamfer_distance`.

## 4. Resulting design (details in `docs/PLAN.md`)
- `VikingHillDataParser(ADDataParser)` (ours, existing) gains `radar_inputs_dir`. After `super()`, it maps the offline map-frame seeds through `dataparser_transform` (μ' = Rμ + t, Σ' = RΣRᵀ) and the time window. It then attaches `metadata["radar_seeds"]` for the train split only. `radars=()` stays empty.
- `SplatADRadarModel(SplatADModel)` makes two additions:
  - In `create_gauss_param_dict`, it appends a radar group whose scales and rotations come from the radar covariance, not from kNN.
  - In `get_loss_dict`, it adds a radar **opacity loss**:
    - static Gaussians inside a seed's resolution ellipsoid, dᵀΣ_seed⁻¹d ≤ χ²₃(q), get the loss mean(−log σ(o));
    - this keeps them opaque, so MCMC does not relocate them.
  - Fallback if seeds lose their Gaussians: the seed-density form, D(q) = Σ_k σ(o_k)·exp(−½dᵀ(Σ_k + Σ_seed)⁻¹d) at the fixed seed means q, with loss −log(1 − exp(−D)).
  - Optional: a one-sided radar depth term, computed in closed form along the seed rays with no rasterisation.
  - The radar losses only say "occupied here", never "empty", because radar free space is unreliable. Free space stays with the lidar ray-drop and line-of-sight losses.
