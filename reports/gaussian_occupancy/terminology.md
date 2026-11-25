# Terminology of the evaluation reports

## What is evaluated and why

The research question is: does adding 4D radar to lidar + camera SplatAD improve tree-trunk reconstruction when tall grass occludes the trunks? It is answered by training three arms (stock, radar init, radar init + loss) several times on a May and a June session, scoring each trained model's occupancy of the obstacles (trunks, rocks, bushes) against hand-labelled boxes, and comparing arms only by the gap against the spread over the repeats.

## Glossary

### Experiment

- **Scene A.** The region of the Viking Hills data used by every run: it holds tree trunks, with low grass in May and high grass in June. Each session covers the consecutive bag chunks pinned in its YAML's `window.chunk_files`, chosen by guess and check.
  - Where: `CLAUDE.md` (`Current scope`); `configs/sessions/june_tall_grass.yaml` (`Scene A (PLAN.md`)
- **Session.** One recording campaign of Scene A with its own config: `may_short_grass` (low grass, May) or `june_tall_grass` (tall grass, June). The two sessions are never mapped onto each other (no ICP, no cross-run alignment).
  - Where: `CLAUDE.md` (`No May`); `configs/sessions/may_short_grass.yaml` (`chunk_files`)
- **Bag chunk.** One `.mcap` file of the recorded ROS bag. A session's `window.chunk_files` pins the chunks that extraction converts (see the session YAML).
  - Where: `configs/sessions/june_tall_grass.yaml` (`chunk_files`); `CLAUDE.md` (`Extraction`)
- **Arm (variant).** One way of training SplatAD, compared on the same session. Three arms: **stock** (method `splatad`, no radar), **radar init** (method `splatad-radar-init`) and **radar init + loss** (method `splatad-radar-init-lidar`). Slurm profiles: May `may_scene_a`, `may_scene_a_radar_init`, `may_scene_a_radar_init_loss`; June `june_scene_a_cam`, `june_scene_a_radar_init_cam`, `june_scene_a_radar_init_loss_cam`.
  - Where: `CLAUDE.md` (`Current scope`); `configs/slurm.yaml` (`june_scene_a_radar_init_loss_cam`); `configs/slurm.yaml` (`may_scene_a_radar_init_loss`); `scripts/21_compare_sessions.py` (`METHOD_LABELS`)
- **Repeat.** One training run of an arm; each arm of each session is trained several times, each as a separate Slurm job, to measure run-to-run noise. Repeats share one seed and are not paired.
  - Where: `CLAUDE.md` (`each trained 5 times`); `scripts/21_compare_sessions.py` (`repeats share one seed`)
- **Median repeat (drawn run).** The run drawn in the 3D figures: per arm, the repeat whose value of the configured representative metric is the median, so the drawn run is neither the arm's best nor its worst. The setting is `eval.figures.representative_metric` (headline P2 recall on H_near, mean over objects, binding cut-off).
  - Where: `configs/eval.yaml` (`representative_metric`)
- **Campaign.** The full set of runs the comparison rests on: three arms on each of the two sessions, each trained several times, each a separate Slurm job.
  - Where: `CLAUDE.md` (`make the 30-run repeat campaign`)
- **Camera (photometric) loss.** The training loss that compares rendered camera images with the recorded ones. Both sessions train with it in the campaign (profiles with `train_lidar_only: false`); the lidar-only June profiles are not part of the campaign.
  - Where: `CLAUDE.md` (`Both sessions train with the camera`); `configs/slurm.yaml` (`train_lidar_only: false`)
- **Train / test / guard frames.** Every X-th instant of the lidar is held out (shared interleaved split; the camera holds out its frame nearest each held-out sweep). Of the held-out instants, an evenly spaced fraction is **test** (rendered and scored); the rest is **guard** (never trained on, not rendered). Everything else is **train**.
  - Where: `docs/PLAN.md` (`Render subset (T72`); `docs/PLAN.md` (`Shared interleaved (T71`)
- **Reference split.** The committed record of which frames of a session are test and guard, identified by `timestamp_ns`. A run is comparable with the others only when it trains and tests on the same frames, so submission refuses a run whose split settings differ from the record.
  - Where: `CLAUDE.md` (`Reference splits (comparability`); `configs/reference_splits/may_short_grass.json` (`guard_timestamps_ns`)

### Radar

- **4D radar.** The 4D imaging radar carried by the ground robot, next to a 32-beam lidar and a camera. Its detections have a range, azimuth and elevation resolution, which give each anchor its covariance. It is not a SplatAD sensor.
  - Where: `docs/proposal/proposal.md` (`4D imaging radar`); `docs/PLAN.md` (`R_ray`)
- **Radar anchors / seeds.** Points built offline in the map frame from the radar detections (static, persistent, no ground-truth input), each with a covariance from the radar resolution. In the init arms each is appended to the model as one seed Gaussian whose colour is painted from the nearest unobstructed train camera frame (grey when obstructed).
  - Formula: `covariance = R_ray diag(sigma_r^2, (r d_az)^2, (r d_el)^2) R_ray^T`
  - Where: `src/viking_splatad/radar_init/anchors.py` (`T35`); `src/viking_splatad/radar_init/anchors.py` (`T36`); `src/viking_splatad/radar_init/seeds.py` (`T37`); `src/viking_splatad/splatad/radar_init_model.py` (`RadarInitSplatADModel`); `src/viking_splatad/radar_init/build.py` (`build_radar_anchor_artifact`); `scripts/07_build_radar_seeds.py` (`build_radar_anchor_artifact(`)
- **Label-free radar opacity loss.** An extra training term, used only by the radar init + loss arm: static Gaussians inside a seed's radar resolution ellipsoid are pushed towards opaque. It uses no class label, and it never says a place is empty. Its weight is set at step 0 so that it equals the stock opacity regulariser.
  - Formula: `L = lambda * mean over supported Gaussians of -log sigmoid(o) = softplus(-o)`
  - Where: `src/viking_splatad/splatad/radar_lidar_loss.py` (`T43`); `src/viking_splatad/splatad/radar_lidar_loss.py` (`T42`); `src/viking_splatad/splatad/radar_lidar_loss.py` (`calibrate_opacity_weight`); `docs/PLAN.md` (`Opacity loss (main term)`)
- **Radar is never rendered.** Radar is not a SplatAD sensor (`radars=()` stays empty, since SplatAD asserts there are none). It enters only as the `radar_seeds` metadata of the train split, as extra Gaussians at init and as the anchors of the opacity loss.
  - Where: `CLAUDE.md` (`Radar is never a SplatAD`); `docs/research/splatad_extension_points.md` (`Radar cannot be a sensor`); `src/viking_splatad/splatad/viking_dataparser.py` (`metadata["radar_seeds"]`)

### Model

- **SplatAD.** The lidar and camera renderer based on 3D Gaussian splatting that is trained in every run; it models the scene as translucent 3D Gaussians with a learnable opacity.
  - Where: `src/viking_splatad/eval/evaluation_terminology.py` (`SPLATAD_REFERENCE`)
- **3D Gaussian (mean, covariance, opacity).** One blob of the model: a mean mu (position), a covariance Sigma (shape, stored as log-scales and a rotation) and an opacity. A trained model holds many of them.
  - Formula: `Sigma = R(q) diag(exp(s))^2 R(q)^T`
  - Where: `docs/PLAN.md` (`Convert:`); `src/viking_splatad/splatad/radar_init_model.py` (`covariance_to_log_scales_and_quaternions`)
- **Learned opacity o and sigma(o).** The model stores an opacity logit o per Gaussian; the activated opacity sigma(o) lies in (0, 1) (0 see-through, 1 solid).
  - Formula: `sigma(o) = 1 / (1 + exp(-o))`
  - Where: `docs/PLAN.md` (`o = sigmoid(opacities)`)
- **Alpha-blending.** How SplatAD combines the Gaussians along a ray (front to back): each Gaussian's alpha is weighted by the light still left in front of it.
  - Formula: `w_i = alpha_i * prod_{j<i}(1 - alpha_j)`
  - Where: `src/viking_splatad/eval/evaluation_terminology.py` (`front_to_back_weights`); `src/viking_splatad/eval/evaluation_terminology.py` (`SPLATAD_REFERENCE`)
- **Transmittance.** The share of light that passes the first i Gaussians of a ray, i.e. the probability the ray is not stopped by them.
  - Formula: `T_i = prod_{j<=i}(1 - alpha_j)`
  - Where: `src/viking_splatad/eval/evaluation_terminology.py` (`accumulated_opacity`); `src/viking_splatad/eval/evaluation_terminology.py` (`KERBL_REFERENCE`)
- **Accumulation.** The total opacity a ray collects: the sum of its blending weights, which equals one minus the transmittance of all Gaussians (telescoping sum). It lies in [0, 1] and does not depend on the order of the Gaussians.
  - Formula: `sum_i w_i = 1 - prod_i(1 - alpha_i)`
  - Where: `src/viking_splatad/eval/evaluation_terminology.py` (`accumulated_opacity`); `src/viking_splatad/eval/evaluation_terminology.py` (`front_to_back_weights`)
- **Median range.** SplatAD places a rendered lidar point at the rolling-shutter-corrected range of the first Gaussian at which the transmittance drops below 0.5. It is used at inference because, unlike the expected range, it does not yield depths between Gaussians.
  - Formula: `first i with prod_{j<=i}(1 - alpha_j) < 0.5`
  - Where: `src/viking_splatad/eval/evaluation_terminology.py` (`SPLATAD_REFERENCE`); `src/viking_splatad/eval/evaluation_terminology.py` (`SplatAD's own 0.5 rule`)
- **Rendered lidar.** The lidar sweep SplatAD renders at a held-out pose: for each recorded beam direction a depth and an accumulation; a ray becomes a point at its rendered depth.
  - Where: `src/viking_splatad/eval/rendered_pointcloud.py` (`RenderedSweep`); `docs/research/splatad_extension_points.md` (`Rendered lidar`)

### Ground truth and grid

- **Cuboid (GT box).** A hand-labelled box in the map frame (centre, full sizes along its own axes, rotation, class). Boxes are used only for scoring and never reach training (no seed, filter or loss target).
  - Where: `src/viking_splatad/labels/cuboids.py` (`Cuboid`); `CLAUDE.md` (`Ground-truth cuboids stay separate`)
- **Classes and the obstacles group.** The scored classes are tree_trunk, rock and bush_or_small_tree. The group `obstacles` is the union of the three and is the headline class.
  - Where: `configs/eval.yaml` (`obstacles = tree_trunk`); `configs/eval.yaml` (`class_groups`)
- **Cell (voxel) and its edge.** A cube of the grid the scene is cut into. Its edge is `voxel_size_m` (see the value table); all scoring counts cells.
  - Where: `src/viking_splatad/labels/occupancy.py` (`T31`); `scripts/20_summarize_gaussian_occupancy.py` (`run_provenance`)
- **Centre-of-cell rule (T31).** A cell belongs to a box when its centre lies inside the box. Whether a trunk thinner than one cell covers 0, 1 or 2 cells therefore depends on where the grid lines fall.
  - Formula: `centre = origin + (index + 0.5) * edge`
  - Where: `src/viking_splatad/labels/occupancy.py` (`T31`); `src/viking_splatad/eval/detection.py` (`cuboid_voxel_mask`)
- **Margin (median_half_footprint).** The class's typical half footprint width: the median over the session's boxes of the class of (length + width) / 4. It is the distance by which a box is grown to define the cells next to it.
  - Formula: `m_c = median over boxes of (dx + dy) / 4`
  - Where: `src/viking_splatad/eval/detection.py` (`median_half_footprint`); `src/viking_splatad/eval/detection.py` (`class_margin_table`)
- **Grown box (grow_cuboid_horizontally).** A box with its horizontal sizes enlarged by the margin on each of the four sides, height unchanged.
  - Formula: `dimension_xy' = dimension_xy + 2 m_c`
  - Where: `src/viking_splatad/eval/detection.py` (`grow_cuboid_horizontally`)
- **Scored area N.** The cells that are scored for a class group: the cells whose centre lies in a grown box of the group, within the scored range and vertical field of view of at least one recorded sweep, minus the cells of the other-class mask. Everything else is ignored.
  - Formula: `N = (union_c N_c) minus other-class cells`
  - Where: `src/viking_splatad/eval/detection.py` (`visible_neighbourhood_mask`); `src/viking_splatad/eval/hidden_metrics.py` (`class_group_masks`)
- **On an obstacle (G).** A scored cell whose centre lies inside a box of the group (the ground-truth cells; G is a subset of N).
  - Formula: `G = cells with GT label in the group, inside N`
  - Where: `src/viking_splatad/eval/hidden_metrics.py` (`class_group_masks`)
- **Next to an obstacle (N minus G).** The scored cells that are in the grown area but not inside any box of the group.
  - Formula: `N minus G`
  - Where: `src/viking_splatad/eval/p2_cells.py` (`p2_cell_outcomes`); `src/viking_splatad/eval/evaluation_terminology.py` (`Next to an obstacle`)
- **Other-class mask.** Cells whose centre lies in a box of an annotated class outside the group are removed from the scored area, so they count neither for nor against the group.
  - Where: `src/viking_splatad/eval/hidden_metrics.py` (`class_group_masks`)
- **Scored range and vertical field of view.** A box or cell is scored only when it is inside the lidar's reach from at least one recorded sweep: within the scored range (a quantile of all return ranges) and within the vertical half field of view (the largest absolute beam elevation).
  - Formula: `vertical half FOV = max |beam elevation|`
  - Where: `src/viking_splatad/eval/detection.py` (`vertical_half_fov_from_elevations`); `src/viking_splatad/eval/detection.py` (`range_limit_from_real_points`); `configs/eval.yaml` (`scored_range_quantile`)

### Occupancy

- **P2 occupancy (T57, T81).** How solid the model says a cell is, from 0 to 1, read from the Gaussians directly (not from a render): the blocking probability of all Gaussians touching the cell. T57 samples each Gaussian's alpha at the cell centre within 3 sigma; T81, the binding estimator, adds the Gaussian's peak opacity to the cell that holds its mean.
  - Formula: `occupancy = 1 - prod_k(1 - alpha_k); alpha_k(x) = sigma(o_k) exp(-0.5 (x - mu_k)^T Sigma_k^-1 (x - mu_k))`
  - Where: `src/viking_splatad/eval/hidden_geometry.py` (`T57`); `src/viking_splatad/eval/hidden_geometry.py` (`T81`); `docs/PLAN.md` (`T81, peak alpha in the voxel`)
- **Peak-alpha rule.** SplatAD's Gaussians are thin discs far smaller than a cell, so almost none reach a cell centre. The Gaussian therefore adds its peak sigma(o) to the cell holding its mean, and its centre-sampled tail to every other cell within 3 sigma.
  - Formula: `alpha_kv = sigma(o_k) in the home cell; sigma(o_k) exp(-d^2 / 2) elsewhere (d = Mahalanobis distance)`
  - Where: `src/viking_splatad/eval/hidden_geometry.py` (`T81`)
- **Blocking probability.** The chance that a ray through the point is stopped, assuming each Gaussian blocks light independently. It is a value at a point of the cell, not the fraction of the cell that is filled.
  - Formula: `1 - prod_k(1 - alpha_k(x))`
  - Where: `docs/research/evaluation_caveats.md` (`Blocking probability`); `src/viking_splatad/eval/evaluation_terminology.py` (`blocking probability at one point`)
- **Cut-off t and filled cell O_t.** A cell with occupancy at or above the cut-off t counts as filled; O_t is the set of filled cells.
  - Formula: `O_t = {cells with occupancy >= t}`
  - Where: `src/viking_splatad/eval/p2_cells.py` (`p2_cell_outcomes`); `docs/PLAN.md` (`Thresholds:`)
- **P2 binding cut-off and strictness check.** P2 scored at the cut-offs of `eval.p2_thresholds`: the first value (binding) decides; the second (strictness check) only checks that a result survives a stricter cut-off. It can weaken a result but never replace the first. The values are in the value table.
  - Where: `docs/PLAN.md` (`Thresholds:`); `configs/eval.yaml` (`p2_thresholds`); `src/viking_splatad/eval/hidden_settings.py` (`binding_threshold`)

### Lidar view

- **P1 (rendered lidar).** The test that asks what a lidar at the held-out poses would see: a cell is hit when at least one ungated rendered point of any test sweep falls in it.
  - Where: `src/viking_splatad/eval/p2_cells.py` (`p1_hit_cells`); `docs/PLAN.md` (`P1, rendered lidar point cloud`)
- **P1cuboid detection rate (T45).** Per class, the share of scored boxes that hold at least n_min points of the rendered cloud.
  - Formula: `rate(n) = |{b : count_b >= n}| / B`
  - Where: `src/viking_splatad/eval/detection.py` (`T45`); `src/viking_splatad/eval/detection.py` (`cuboid_detection_rate`)
- **P1cuboid point precision (T46).** Of the rendered points in the grown boxes of a class, the share that lies inside a box itself.
  - Formula: `precision = |points in a box| / |points in the grown boxes|`
  - Where: `src/viking_splatad/eval/detection.py` (`T46`); `src/viking_splatad/eval/detection.py` (`neighbourhood_point_precision`)
- **Gated vs ungated point sets.** Gated keeps the rays SplatAD itself would keep (ray-drop probability at most 0.5); ungated (binding) keeps every valid ray whose accumulation reaches the render threshold. The gate can remove radar-recovered trunks, so gated numbers are only reported beside the ungated ones.
  - Where: `src/viking_splatad/eval/rendered_pointcloud.py` (`**ungated** (PLAN`); `docs/PLAN.md` (`ungated extraction (binding)`)
- **Render metrics (chamfer distance, median chamfer, depth median L2, depth mean relative L2).** Lidar render error scores of SplatAD's own evaluation on the test sweeps. They are report only and never decide a comparison. Their exact formulas are those of the SplatAD evaluation code.
  - Where: `src/viking_splatad/eval/run_quality.py` (`REPORTED_LIDAR_METRICS`); `docs/research/splatad_extension_points.md` (`chamfer_distance`)

### Hidden space

- **Seen / blocked / passed through.** Evidence a recorded beam gives about a cell, with cube interval [entry, exit] along the beam and range margin m: **seen** when a returned depth d lies within the cube (widened by m), **blocked** when d falls short of it, **passed through** when d lies beyond it.
  - Formula: `seen: entry - m <= d <= exit + m; blocked: d < entry - m; passed: d > exit + m`
  - Where: `src/viking_splatad/eval/hidden_geometry.py` (`recorded_ray_evidence`)
- **Range margin m (T92) and neighbour reach.** The range noise below which a return counts as seen. It is measured from the data as a quantile of point-to-plane residuals between two sweeps of the same scene. The neighbour reach bounds how far the plane fit's neighbours may lie, so the residual measures range noise and not shape.
  - Formula: `residual = |(p - c) . n|; m = quantile of residuals`
  - Where: `src/viking_splatad/eval/range_margin.py` (`T92`); `src/viking_splatad/eval/range_margin.py` (`range_margin`); `src/viking_splatad/eval/range_margin.py` (`max_neighbour_reach_m`)
- **Hidden space H.** A scored cell that, over all recorded sweeps of the session, was never seen, was blocked at least once and was never passed through. A beam without a return neither proves nor disproves it.
  - Formula: `H = NOT seen AND blocked AND NOT passed_through`
  - Where: `src/viking_splatad/eval/hidden_geometry.py` (`hidden_space_mask`); `src/viking_splatad/eval/session_hidden_space.py` (`Session-wide hidden space H`)
- **Near band and H_near.** The near band is the horizontal distance from the driven path that bounds the headline hidden space. H_near is the hidden cells within it.
  - Formula: `H_near = H AND d_nearest <= near band`
  - Where: `src/viking_splatad/eval/session_hidden_space.py` (`near_band_hidden_mask`); `configs/eval.yaml` (`near_band_m`)

### Outcomes and metrics

- **Hit, false alarm, missed, context.** Cell sets behind the ratios: **context** is every scored cell (of the hidden band); **hit** is a filled cell on an obstacle; **false alarm** a filled cell next to an obstacle; **missed** an obstacle cell left empty.
  - Formula: `hit = O & G (& H); false alarm = O & N & ~G (& H); missed = G & ~O (& H)`
  - Where: `src/viking_splatad/eval/p2_cells.py` (`hidden_cell_outcomes`); `src/viking_splatad/eval/p2_cells.py` (`p2_cell_outcomes`)
- **Correct empty.** A cell next to an obstacle that the model left empty. It is the fourth outcome beside hit, false alarm and missed, and it enters neither recall nor precision.
  - Formula: `correct empty = N & ~G & ~O`
  - Where: `src/viking_splatad/eval/p2_cells.py` (`p2_cell_outcomes`); `src/viking_splatad/eval/p2_cells.py` (`hidden_cell_outcomes`)
- **Recall.** Of the obstacle cells, the share the model filled.
  - Formula: `recall = |O & G| / |G|`
  - Where: `src/viking_splatad/eval/hidden_metrics.py` (`T94`)
- **Precision.** Of the filled cells in the scored area, the share that is on an obstacle.
  - Formula: `precision = |O & G| / |O & N|`
  - Where: `src/viking_splatad/eval/hidden_metrics.py` (`T94`)
- **IoU.** Overlap of the filled cells and the obstacle cells inside the scored area.
  - Formula: `IoU = TP / (TP + FP + FN)`
  - Where: `src/viking_splatad/eval/hidden_metrics.py` (`T94`)
- **Pooled (T94).** The metrics counted over all scored cells together, so big objects weigh most.
  - Formula: `TP = |O & G|, FP = |O & N minus G|, FN = |G minus O|`
  - Where: `src/viking_splatad/eval/hidden_metrics.py` (`T94`)
- **Mean over objects (T95).** Each object is scored alone inside its own grown box, and the metric is averaged over the objects where it is defined, so every object weighs the same. A cell in the grown boxes of two objects is scored in both.
  - Formula: `mean_m = (1 / |D_m|) * sum over objects k in D_m of m_k`
  - Where: `src/viking_splatad/eval/hidden_metrics.py` (`T95`)
- **Threshold curve.** Recall and precision (pooled and mean over objects) recomputed at each cut-off of a grid, with filled = occupancy at or above the cut-off.
  - Where: `src/viking_splatad/viz/figure_metrics.py` (`threshold_curves`); `src/viking_splatad/viz/figure_metrics.py` (`occupancy_samples`)

### Statistics

- **Mean +- std (n - 1).** The mean of a metric over an arm's defined repeats and its sample standard deviation (divisor n - 1; undefined with fewer than two repeats).
  - Formula: `std = sqrt(sum (x - mean)^2 / (n - 1))`
  - Where: `src/viking_splatad/eval/p2_repeats.py` (`describe_values`)
- **Min-max band.** The range from the smallest to the largest repeat of an arm; the figures shade it around the median repeat's curve.
  - Where: `src/viking_splatad/eval/p2_repeats.py` (`describe_values`)
- **Gap.** Difference of two arms' means for a metric, printed beside each arm's spread.
  - Formula: `gap = mean(first arm) - mean(second arm)`
  - Where: `src/viking_splatad/eval/p2_repeats.py` (`compare_arms`)
- **sep / ovl.** A reading aid, not a test: **sep** when the two arms' [min, max] ranges do not overlap (every repeat of one arm is above every repeat of the other), **ovl** when they overlap, **n/a** when an arm has fewer than two defined repeats.
  - Formula: `sep: min(first) > max(second) or min(second) > max(first)`
  - Where: `src/viking_splatad/eval/p2_repeats.py` (`compare_arms`); `src/viking_splatad/eval/report_opening.py` (`separation_label`)
- **Run-to-run noise and why repeats are needed.** Identical-seed SplatAD runs differ, on some metrics by a large factor, and the project accepted this noise. So no comparison is read without repeats.
  - Where: `CLAUDE.md` (`run-to-run noise is accepted`)
- **The reading rule.** Compare arms only by the gap against the spread over the repeats: a gap counts only when it is larger than the spread within the arms.
  - Where: `CLAUDE.md` (`Compare arms only by the gap`); `src/viking_splatad/eval/evaluation_terminology.py` (`A gap is read only when`)

### Image metrics (auxiliary)

- **PSNR, SSIM, LPIPS.** Image-quality scores of SplatAD's built-in evaluation of the rendered camera images, read from the training-time eval on the test and guard frames. They are a guard only (the radar claim is judged on occupancy), because below the grass the ground-truth pixels are grass.
  - Where: `src/viking_splatad/eval/run_quality.py` (`IMAGE_METRIC_NAMES`); `docs/research/splatad_extension_points.md` (`psnr`); `docs/PLAN.md` (`image metrics cannot show the radar gain`)

### Bookkeeping

- **Transform id T<n>.** The number under which a conversion, calibration or metric function is registered with `@transform`; each has a unit test that covers it.
  - Where: `src/viking_splatad/registry.py` (`transform`); `CLAUDE.md` (`Every conversion/calibration function`)
- **Summary (scripts/20).** The per-session summary that groups the runs by arm and describes each metric over the repeats.
  - Where: `scripts/20_summarize_gaussian_occupancy.py` (`summarize`)
- **Session comparison (scripts/21).** The report that puts the sessions side by side, arm by arm, and writes this terminology file.
  - Where: `scripts/21_compare_sessions.py` (`render_markdown`)
- **Figure manifest (scripts/26).** The `figures_manifest.json` written beside each session's figures: input hashes, the numbers quoted in captions and each figure's status.
  - Where: `scripts/26_occupancy_figures.py` (`figures_manifest.json`)

*The definitions below are rendered once, with the values of session june_tall_grass; the per-session values table and sections give the other sessions' values.*

## Definitions

Plain words first, the formula last. Every number is read from the run reports, never typed in.

1. **Cells.** The scene is cut into small cubes ("cells") of edge 0.521933 m. Everything below is about cells.
2. **Ground truth, "on an obstacle" and "next to an obstacle".** People drew a box around every tree trunk, rock and bush ("obstacles"). The boxes are used only for scoring, never for training.
   - **On an obstacle (G):** a cell whose **centre** lies inside an obstacle box.
   - **The scored area around a class (N_c):** for each class (trunk, rock, bush) separately:
     1. **Margin:** m_c = half of the class's typical footprint width, i.e. the median over all of the session's boxes of that class of (length + width) / 4. Values for session june_tall_grass: trunk 0.522 m, rock 0.665 m, bush 0.953 m.
     2. **Grow:** widen every box of the class by m_c on each of its four sides, along the box's own axes, leaving its height unchanged. A typical object's grown box is therefore about twice its own width.
     3. **Collect:** N_c = the cells whose centre lies inside a grown box.
   - **Only space the lidar could have covered is scored:** a box counts only when its centre is within the lidar's range (200.985 m) and vertical field of view (±44.8°) of at least one recorded sweep. The same range and field-of-view test is applied to every cell of N_c.
   - **Obstacles as one group:** N = N_trunk ∪ N_rock ∪ N_bush, minus any cell whose centre lies in a box of a different, non-obstacle annotated class. Those cells count neither for nor against.
   - **Next to an obstacle:** N ∖ G, the cells in the grown area that are not inside any obstacle box.
   - **Scored:** only cells in N; everything else in the scene is ignored.
   - **Caveat:** the boxes are rectangular and a cell counts by its centre only. So for a trunk thinner than one cell (0.521933 m), whether the trunk gets 0, 1 or 2 cells across depends on where the grid lines fall.
3. **What the model contains.** A trained SplatAD model is millions of small fuzzy blobs (3D Gaussians). Each blob has an opacity from 0 (see-through) to 1 (solid).
4. **Occupancy of a cell** is how solid the model says the cell is, from 0 (empty) to 1 (solid).
   - Each blob adds its opacity to the cell its centre lies in, and a fading share to nearby cells.
   - Blobs combine like stacked tinted glass: light passes through a blob of opacity α with probability (1 − α). So occupancy = 1 − ∏_k(1 − α_k), which grows with every blob but never passes 1.
   - **Why opacities are combined this way (the source of the definition).** The rule is not new. It reads, at a cell, the same accumulation SplatAD uses to render a lidar ray. The design choice is recorded in PLAN §5 (P2) and `docs/research/evaluation_caveats.md`, "Blocking probability". Quotes are verbatim from the sources below.
     1. **How SplatAD blends Gaussians along a ray.** SplatAD (Hess et al., *SplatAD: Real-Time Lidar and Camera Rendering with 3D Gaussian Splatting for Autonomous Driving*, arXiv:2411.16816, https://arxiv.org/abs/2411.16816, Eqs. 4-5 (section 3.2), median-range rule (section 3.3); passages checked 2026-11-25) models the scene as "translucent 3D Gaussians with learnable occupancy o∈(0,1)" (§3.1).
        - **Eq. 4 (§3.2):** [F_p^rgb, F_p] = Σ_i [f_i^rgb, f_i] α_i ∏_{j=1}^{i−1}(1 − α_j). Each Gaussian's own opacity α_i is multiplied by ∏_{j<i}(1 − α_j), the light still left in front of it (the transmittance).
        - **Eq. 5 (§3.2):** α_i = sqrt(|Σ_i^I| / |Σ_i^I + s·I|) · o_i · exp(−½ Δ_iᵀ (Σ_i^I + s·I)⁻¹ Δ_i). The Gaussian's α_i is its learned opacity o_i times its falloff at the ray, projected onto the sensor plane.
        - **§3.3:** "For each lidar point, we α-blend the Gaussians' features 𝐟_i using Eqs. 4, 5 and 6." So the lidar uses exactly these weights.
     2. **SplatAD's own 0.5 rule, which is our definition.** SplatAD places a rendered lidar point at the median range, "the rolling shutter corrected range of the first Gaussian that satisfies ∏_{j=1}^{i}(1−α_j) < 0.5 during the α-blending in Eq. 4" (§3.3). It is used at inference: "While we use the expected range for training, the median range is used during inference as it, in contrast to the expected range, does not yield depths between Gaussians." (§3.3). In words: SplatAD says a ray has hit something once **1 − ∏(1 − α) passes 0.5**. Our occupancy is that same quantity. Our binding cut-off is that same 0.5. So "the cell is filled" means "SplatAD's own renderer would consider a ray through these Gaussians stopped".
     3. **The lineage.** 3D Gaussian Splatting (Kerbl et al., *3D Gaussian Splatting for Real-Time Radiance Field Rendering*, arXiv:2308.04079, https://arxiv.org/abs/2308.04079, section 2.3, Eqs. 1-3 (ACM TOG 42(4), 2023); passages checked 2026-11-25) gives the same blending, Eq. 3: C = Σ_{i∈N} c_i α_i ∏_{j=1}^{i−1}(1 − α_j). It states: "Point-based α-blending and NeRF-style volumetric rendering share essentially the same image formation model." (§2.3). So ∏(1 − α_j) is the transmittance T_i of volume rendering (its Eqs. 1–2).
     4. **Why 1 − ∏(1 − α), as one number per cell.** The blending weights along a ray always add up to Σ_i α_i ∏_{j<i}(1 − α_j) = 1 − ∏_i(1 − α_i). Proof by telescoping: with T_k = ∏_{i≤k}(1 − α_i) and T_0 = 1, each weight equals T_{k−1} − T_k, so the sum is T_0 − T_n = 1 − T_n. A unit test checks this identity. The product does not depend on the order of the Gaussians, so no particular ray direction is needed. The result is the total opacity a ray collects from these Gaussians, between 0 and 1.
   - **How ours differs from SplatAD's α (stated here, not hidden):**
     - SplatAD evaluates each Gaussian **projected** onto the image or lidar plane, with a small blur term s·I that matches the pixel size or beam divergence (Eq. 5). We evaluate the **3D** Gaussian at a point: α_k(x) = σ(o_k) · exp(−½ (x − μ_k)ᵀ Σ_k⁻¹ (x − μ_k)), at the cell centre, or at its own peak for the cell holding its mean (T81).
     - We combine every Gaussian touching the cell, rather than those along one beam.
     - So occupancy is a blocking probability at one point of the cell, in the spirit of SplatAD's accumulation. It is not the fraction of the cell that is filled. Like the renderer, it assumes each Gaussian blocks light independently.
   - **General reference, not the source of our rule.** Using 3D Gaussians to predict 3D occupancy has been done before (Huang et al., *GaussianFormer: Scene as Gaussians for Vision-Based 3D Semantic Occupancy Prediction*, arXiv:2405.17429, https://arxiv.org/abs/2405.17429, section 3.3, Gaussian-to-Voxel Splatting (general reference only); passages checked 2026-11-25). It adds up the Gaussians' contributions at voxel positions. Ours combines opacities with the renderer's rule above instead, so that the result stays in [0, 1] and matches SplatAD's stop rule.
   - It is read from the blobs directly, not from a rendered image, so something hidden behind grass still counts.
5. **Filled, recall and precision.** Pick a cut-off t; a cell with occupancy ≥ t counts as *filled*. Let O_t be the set of filled cells.
   - **(a) What is scored and what is not.** Only the cells in N are scored: the cells **on** an obstacle (G) and the cells **next to** an obstacle (N ∖ G, the ring, 11751 cells). "Not ground truth" here means only that ring, not the whole scene. Every other cell is ignored, even if the model fills it, so the ring is what makes false alarms countable: a filled cell far from every obstacle is never counted. Cells on another labelled class are excluded from both metrics.
   - **(b) The 2 × 2 table.** Each scored cell is either filled or empty, and either on an obstacle or next to one:

     | | on an obstacle (G) | next to an obstacle (N ∖ G) |
     |---|---|---|
     | **model filled** | hit | false alarm |
     | **model left empty** | missed | correct empty |

     "Correct empty" enters neither metric.
   - **(c) Recall** = hit / (hit + missed) = |O_t ∩ G| / |G|. *Did it rebuild the obstacles?* The denominator is every obstacle cell (12687); it depends only on the ground truth. So recall cannot be gamed by leaving things out, but it ignores false alarms.
   - **(d) Precision** = hit / (hit + false alarm) = |O_t ∩ G| / |O_t ∩ N|. *When it fills something, is it real?* The denominator is every filled scored cell; it depends on the model. Filling everything near obstacles raises recall and lowers precision. This is why both are read together, and why the reading rule asks for higher recall *without lower precision*.
   - **IoU** (intersection over union) = hit / (hit + missed + false alarm) = |O_t ∩ G| / |(O_t ∪ G) ∩ N|. In words: the overlap between what the model filled and the obstacles, divided by everything either of them covers in the scored area. It puts both kinds of error in one number: a miss and a false alarm each lower it. Correct-empty cells do not enter it. It is always ≤ min(recall, precision). It is a summary only: the reading rule uses recall and precision.
   - **(e) Worked example.** The stock arm's drawn run in june_tall_grass at t = 0.5, with its real counts: 4457 hits, 8230 missed, 1238 false alarms → recall 0.351, precision 0.783, IoU 0.320. These are one run; the tables give mean ± std over the repeats.
   - **(f) Pooled vs mean over objects.** *Pooled* counts all cells together, so big objects weigh most. *Mean over objects* scores each object k alone, with its own cells G_k and its own ring N_k, and then averages, so every object weighs the same. An occupied cell inside several rings counts against each of them.
   - **(g) Why precision stays below 1.** "Next to an obstacle" is not "empty space". The ring also holds things nobody labelled: grass (tall in June), ground and small plants, which really are solid. It also catches the edges of a trunk's own Gaussians, because a trunk is thinner than one cell. So precision stays below 1 even for a correct model. This is a likely cause, not a tested finding.
   - **(h) How the threshold changes both.** Raising t moves cells from filled to empty, so recall can only fall; precision usually rises.
6. **Hidden cells (reference for the later round).** We replay every recorded lidar beam.
   - A cell is *hidden* if no beam reached it, none passed through it, and at least one was stopped before it (by grass, for example).
   - *H_near* keeps the hidden cells within 20 m of the driven path.
   - Hidden-cell metrics are the formulas of point 5, restricted to H_near. The figures will take the same form as the whole-scene ones.
   - Stock SplatAD is expected to leave hidden cells mostly empty, since no lidar or camera signal reached them. That gap is what radar is meant to close.
7. **Statistics.** Each arm is trained 5 times. A table cell shows mean ± std over the repeats. Gap = difference of means. "sep" means the two arms' [min, max] ranges do not overlap. A gap is read only when it is larger than the spread within the arms.


## Reading the figures

### How to read: Whole scene in 3D, per arm

**How to read this figure**
- **GT panel:** the obstacle boxes, coloured by class, for orientation.
- **Arm panels:** green = hit (filled, on an obstacle); red = false alarm (filled, next to an obstacle); grey = missed (empty, on an obstacle); faint grey = the scored area, the ring included. Correct-empty cells are not drawn. Cut-off t = 0.5. The terms are defined in point 5.
- **Difference panel:** hits only radar init + loss has, hits only stock has, and hits both have.
- One run per arm (the median repeat), so it illustrates and does not decide anything.


### How to read: Recall and precision against the occupancy threshold

**How to read this figure**
- **What is plotted:** recall (left) and precision (right) recomputed at each cut-off t from 0.2 to 0.9 (15 evenly spaced values). Solid lines are pooled and dashed lines are the mean over objects. The line is the median repeat. The band spans, at each t, the lowest and highest value of both the pooled and the mean-over-objects curve over the 5 repeats of the arm; it is therefore wider than either line's own spread, so "bands do not overlap" is a conservative reading. Dotted lines mark the table cut-offs 0.5, 0.8.
- **Definitions:** see point 5. Only recall and precision are plotted; IoU is in the tables. Recall = hit / (hit + missed) has all 12687 obstacle cells as its denominator; precision = hit / (hit + false alarm) has every filled scored cell as its denominator.
- **How the curves move:** raising t fills fewer cells, so recall can only fall and precision usually rises.
- **When one arm beats another:** its curve is higher and the bands do not overlap.
- **Why t runs from 0.2 to 0.9:** the extremes are left out because they say nothing about the model. At t = 0 every cell counts as filled (recall 1, precision = the share of obstacle cells). At t ≈ 1 precision rests on a handful of cells and jumps.


## Values used, per session

| value | june_tall_grass | may_short_grass |
|---|---|---|
| Voxel (cell) edge (m) | 0.521933 | 0.521933 |
| Margin, tree_trunk (m) | 0.522 | 0.522 |
| Margin, rock (m) | 0.665 | 0.665 |
| Margin, bush_or_small_tree (m) | 0.953 | 1.069 |
| Scored range (m) | 200.985 | 207.364 |
| Vertical half field of view (degrees) | 44.8 | 44.8 |
| Near band (m) | 20 | 20 |
| Cells on obstacles, whole scene | 12687 | 19600 |
| Cells next to obstacles, whole scene | 11751 | 16407 |
| Repeats per arm | 5 | 5 |
| P2 cut-offs (first = binding) | 0.5, 0.8 | 0.5, 0.8 |
| Curve range, from | 0.2 | 0.2 |
| Curve range, to | 0.9 | 0.9 |
| Curve points | 15 | 15 |

## Assumptions and caveats

- **Centre-of-cell rule and thin trunks.** A cell belongs to a box by its centre only, and boxes are rectangular. A trunk thinner than one cell therefore gets 0, 1 or 2 cells across depending on where the grid lines fall.
- **Unlabelled grass and ground next to obstacles.** The ring of cells next to an obstacle also holds things nobody labelled (grass, ground, small plants) that really are solid, so precision stays below 1 even for a correct model. This is a likely cause of low precision, not a tested finding.
- **Occupancy is a blocking probability at a point, not filled volume.** It is evaluated at the cell centre (or at a Gaussian's peak for the cell holding its mean) and assumes each Gaussian blocks light independently.
- **One drawn run per arm in the 3D figure.** The 3D figure shows the median repeat of each arm. It illustrates and decides nothing; arms are separated only by the spread over all repeats.
- **Run-to-run noise.** SplatAD runs with the same seed differ, so every comparison needs repeats and is read as the gap against the spread within the arms.

## References

- Hess et al., *SplatAD: Real-Time Lidar and Camera Rendering with 3D Gaussian Splatting for Autonomous Driving*, arXiv:2411.16816, https://arxiv.org/abs/2411.16816, Eqs. 4-5 (section 3.2), median-range rule (section 3.3); passages checked 2026-11-25
- Kerbl et al., *3D Gaussian Splatting for Real-Time Radiance Field Rendering*, arXiv:2308.04079, https://arxiv.org/abs/2308.04079, section 2.3, Eqs. 1-3 (ACM TOG 42(4), 2023); passages checked 2026-11-25
- Huang et al., *GaussianFormer: Scene as Gaussians for Vision-Based 3D Semantic Occupancy Prediction*, arXiv:2405.17429, https://arxiv.org/abs/2405.17429, section 3.3, Gaussian-to-Voxel Splatting (general reference only); passages checked 2026-11-25

### Code

- Centre-of-cell rule: voxelising cuboids into a class-id grid: `src/viking_splatad/labels/occupancy.py` (`T31`)
- Cuboid detection rate (P1cuboid): `src/viking_splatad/eval/detection.py` (`T45`)
- Neighbourhood point precision (P1cuboid): `src/viking_splatad/eval/detection.py` (`T46`)
- Three-sigma Gaussian occupancy: `src/viking_splatad/eval/hidden_geometry.py` (`T57`)
- Peak-alpha occupancy, the binding estimator: `src/viking_splatad/eval/hidden_geometry.py` (`T81`)
- Lidar range margin m: `src/viking_splatad/eval/range_margin.py` (`T92`)
- Pooled recall, precision and IoU: `src/viking_splatad/eval/hidden_metrics.py` (`T94`)
- Per-object recall, precision and IoU, averaged: `src/viking_splatad/eval/hidden_metrics.py` (`T95`)
- Margin of a class: `src/viking_splatad/eval/detection.py` (`median_half_footprint`)
- Other-class mask and class-group masks: `src/viking_splatad/eval/hidden_metrics.py` (`class_group_masks`)
- Grown box: `src/viking_splatad/eval/detection.py` (`grow_cuboid_horizontally`)
- Scored area of one class: `src/viking_splatad/eval/detection.py` (`visible_neighbourhood_mask`)
- Occupancy values on and next to obstacles: `src/viking_splatad/viz/figure_metrics.py` (`occupancy_samples`)
- Threshold curves: `src/viking_splatad/viz/figure_metrics.py` (`threshold_curves`)
