# P1/P2 over repeats: may_short_grass

## Definitions

Plain words first, the formula last. Every number is read from the run reports, never typed in.

1. **Cells.** The scene is cut into small cubes ("cells") of edge 0.521933 m. Everything below is about cells.
2. **Ground truth, "on an obstacle" and "next to an obstacle".** People drew a box around every tree trunk, rock and bush ("obstacles"). The boxes are used only for scoring, never for training.
   - **On an obstacle (G):** a cell whose **centre** lies inside an obstacle box.
   - **The scored area around a class (N_c):** for each class (trunk, rock, bush) separately:
     1. **Margin:** m_c = half of the class's typical footprint width, i.e. the median over all of the session's boxes of that class of (length + width) / 4. Values for session may_short_grass: trunk 0.522 m, rock 0.665 m, bush 1.069 m.
     2. **Grow:** widen every box of the class by m_c on each of its four sides, along the box's own axes, leaving its height unchanged. A typical object's grown box is therefore about twice its own width.
     3. **Collect:** N_c = the cells whose centre lies inside a grown box.
   - **Only space the lidar could have covered is scored:** a box counts only when its centre is within the lidar's range (207.364 m) and vertical field of view (±44.8°) of at least one recorded sweep. The same range and field-of-view test is applied to every cell of N_c.
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
   - **(a) What is scored and what is not.** Only the cells in N are scored: the cells **on** an obstacle (G) and the cells **next to** an obstacle (N ∖ G, the ring, 16407 cells). "Not ground truth" here means only that ring, not the whole scene. Every other cell is ignored, even if the model fills it, so the ring is what makes false alarms countable: a filled cell far from every obstacle is never counted. Cells on another labelled class are excluded from both metrics.
   - **(b) The 2 × 2 table.** Each scored cell is either filled or empty, and either on an obstacle or next to one:

     | | on an obstacle (G) | next to an obstacle (N ∖ G) |
     |---|---|---|
     | **model filled** | hit | false alarm |
     | **model left empty** | missed | correct empty |

     "Correct empty" enters neither metric.
   - **(c) Recall** = hit / (hit + missed) = |O_t ∩ G| / |G|. *Did it rebuild the obstacles?* The denominator is every obstacle cell (19600); it depends only on the ground truth. So recall cannot be gamed by leaving things out, but it ignores false alarms.
   - **(d) Precision** = hit / (hit + false alarm) = |O_t ∩ G| / |O_t ∩ N|. *When it fills something, is it real?* The denominator is every filled scored cell; it depends on the model. Filling everything near obstacles raises recall and lowers precision. This is why both are read together, and why the reading rule asks for higher recall *without lower precision*.
   - **IoU** (intersection over union) = hit / (hit + missed + false alarm) = |O_t ∩ G| / |(O_t ∪ G) ∩ N|. In words: the overlap between what the model filled and the obstacles, divided by everything either of them covers in the scored area. It puts both kinds of error in one number: a miss and a false alarm each lower it. Correct-empty cells do not enter it. It is always ≤ min(recall, precision). It is a summary only: the reading rule uses recall and precision.
   - **(e) Worked example.** The stock arm's drawn run in may_short_grass at t = 0.5, with its real counts: 5546 hits, 14054 missed, 1074 false alarms → recall 0.283, precision 0.838, IoU 0.268. These are one run; the tables give mean ± std over the repeats.
   - **(f) Pooled vs mean over objects.** Recall, precision and IoU are each reported in two ways. They differ only in how the objects are combined.
     - **Pooled** (T94): add up the cells of all objects first, then divide once. Pooled recall = Σ_k |O_t ∩ G_k| / Σ_k |G_k| = all hits / all obstacle cells; pooled precision = all hits / all filled scored cells. Every *cell* weighs the same, so objects with many cells (tall trunks, big bushes) dominate.
     - **Mean over objects** (T95): score each object k on its own, inside its own grown box N_k with its own obstacle cells G_k, then take the plain average: mean recall = (1/K) Σ_k recall_k, and likewise precision and IoU. Every *object* weighs the same, however small. K counts only the objects where the ratio is defined (a non-zero denominator): an object with no filled cell in its box has no precision and is left out of the precision mean. A filled cell inside several objects' grown boxes counts as a false alarm of each of them.
     - **Example (made-up numbers, for illustration only):** a large trunk with 90 obstacle cells, 30 of them filled and a small rock with 10 obstacle cells, 8 of them filled. The per-object recalls are 0.333 and 0.800. Pooled recall = 0.380, dominated by the large trunk; mean recall = 0.567, where the small rock counts as much as the trunk.
     - **Reading the two together:** for recall, when objects share no cells, the pooled value is the average of the per-object recalls weighted by each object's cell count. So a mean above the pooled value means objects with few cells are recovered better, relative to their size, than objects with many cells; a mean below means the large objects carry the score.
   - **(g) Why precision stays below 1.** "Next to an obstacle" is not "empty space". The ring also holds things nobody labelled: grass (tall in June), ground and small plants, which really are solid. It also catches the edges of a trunk's own Gaussians, because a trunk is thinner than one cell. So precision stays below 1 even for a correct model. This is a likely cause, not a tested finding.
   - **(h) How the threshold changes both.** Raising t moves cells from filled to empty, so recall can only fall; precision usually rises.
6. **Hidden cells (the second part of the evaluation, read after the whole scene).** We replay every recorded lidar beam.
   - A cell is *hidden* if no beam reached it, none passed through it, and at least one was stopped before it (by grass, for example).
   - *H_near* keeps the hidden cells within 20 m of the driven path.
   - Hidden-cell metrics are the formulas of point 5, restricted to H_near. Their figures take the same form as the whole-scene ones, and are read after them.
   - Stock SplatAD is expected to leave hidden cells mostly empty, since no lidar or camera signal reached them. That gap is what radar is meant to close.
7. **Statistics.** Each arm is trained 5 times. A table cell shows mean ± std over the repeats. Gap = difference of means. "sep" means the two arms' [min, max] ranges do not overlap. A gap is read only when it is larger than the spread within the arms.

## Whole scene

### obstacles, P2 occupied at >= 0.5

| metric | stock (`may_scene_a`) | radar init (`may_scene_a_radar_init`) | radar init + loss (`may_scene_a_radar_init_loss`) |
|---|---|---|---|
| recall | 0.278 ± 0.003 [0.275, 0.283] (n=5) | 0.221 ± 0.003 [0.218, 0.225] (n=5) | 0.370 ± 0.003 [0.366, 0.374] (n=5) |
| precision | 0.838 ± 0.004 [0.834, 0.843] (n=5) | 0.810 ± 0.005 [0.805, 0.817] (n=5) | 0.758 ± 0.006 [0.749, 0.762] (n=5) |
| IoU | 0.264 ± 0.003 [0.261, 0.268] (n=5) | 0.210 ± 0.003 [0.208, 0.215] (n=5) | 0.331 ± 0.002 [0.328, 0.333] (n=5) |

### obstacles, P2 occupied at >= 0.8

| metric | stock (`may_scene_a`) | radar init (`may_scene_a_radar_init`) | radar init + loss (`may_scene_a_radar_init_loss`) |
|---|---|---|---|
| recall | 0.244 ± 0.002 [0.241, 0.247] (n=5) | 0.190 ± 0.002 [0.187, 0.193] (n=5) | 0.326 ± 0.001 [0.324, 0.327] (n=5) |
| precision | 0.863 ± 0.003 [0.859, 0.866] (n=5) | 0.831 ± 0.002 [0.827, 0.834] (n=5) | 0.772 ± 0.003 [0.768, 0.774] (n=5) |
| IoU | 0.234 ± 0.002 [0.232, 0.237] (n=5) | 0.183 ± 0.002 [0.180, 0.186] (n=5) | 0.297 ± 0.001 [0.296, 0.298] (n=5) |

### tree_trunk, P2 occupied at >= 0.5

| metric | stock (`may_scene_a`) | radar init (`may_scene_a_radar_init`) | radar init + loss (`may_scene_a_radar_init_loss`) |
|---|---|---|---|
| recall | 0.375 ± 0.002 [0.372, 0.376] (n=5) | 0.302 ± 0.001 [0.300, 0.303] (n=5) | 0.433 ± 0.004 [0.426, 0.438] (n=5) |
| precision | 0.849 ± 0.005 [0.845, 0.856] (n=5) | 0.828 ± 0.002 [0.826, 0.830] (n=5) | 0.811 ± 0.009 [0.795, 0.817] (n=5) |
| IoU | 0.351 ± 0.001 [0.349, 0.352] (n=5) | 0.284 ± 0.001 [0.283, 0.286] (n=5) | 0.393 ± 0.004 [0.389, 0.398] (n=5) |

### tree_trunk, P2 occupied at >= 0.8

| metric | stock (`may_scene_a`) | radar init (`may_scene_a_radar_init`) | radar init + loss (`may_scene_a_radar_init_loss`) |
|---|---|---|---|
| recall | 0.331 ± 0.001 [0.329, 0.332] (n=5) | 0.263 ± 0.003 [0.258, 0.266] (n=5) | 0.379 ± 0.002 [0.377, 0.383] (n=5) |
| precision | 0.872 ± 0.005 [0.868, 0.878] (n=5) | 0.848 ± 0.003 [0.846, 0.852] (n=5) | 0.825 ± 0.003 [0.821, 0.828] (n=5) |
| IoU | 0.316 ± 0.001 [0.314, 0.317] (n=5) | 0.251 ± 0.003 [0.247, 0.253] (n=5) | 0.351 ± 0.002 [0.349, 0.354] (n=5) |

### rock, P2 occupied at >= 0.5

| metric | stock (`may_scene_a`) | radar init (`may_scene_a_radar_init`) | radar init + loss (`may_scene_a_radar_init_loss`) |
|---|---|---|---|
| recall | 0.253 ± 0.018 [0.230, 0.281] (n=5) | 0.206 ± 0.021 [0.180, 0.225] (n=5) | 0.284 ± 0.010 [0.270, 0.298] (n=5) |
| precision | 0.456 ± 0.011 [0.441, 0.472] (n=5) | 0.406 ± 0.034 [0.367, 0.455] (n=5) | 0.429 ± 0.020 [0.395, 0.445] (n=5) |
| IoU | 0.194 ± 0.012 [0.180, 0.214] (n=5) | 0.158 ± 0.017 [0.140, 0.177] (n=5) | 0.206 ± 0.008 [0.198, 0.217] (n=5) |

### rock, P2 occupied at >= 0.8

| metric | stock (`may_scene_a`) | radar init (`may_scene_a_radar_init`) | radar init + loss (`may_scene_a_radar_init_loss`) |
|---|---|---|---|
| recall | 0.228 ± 0.015 [0.202, 0.236] (n=5) | 0.165 ± 0.011 [0.157, 0.180] (n=5) | 0.267 ± 0.010 [0.258, 0.281] (n=5) |
| precision | 0.478 ± 0.022 [0.444, 0.500] (n=5) | 0.408 ± 0.025 [0.373, 0.438] (n=5) | 0.458 ± 0.011 [0.447, 0.475] (n=5) |
| IoU | 0.183 ± 0.012 [0.161, 0.191] (n=5) | 0.133 ± 0.010 [0.124, 0.146] (n=5) | 0.203 ± 0.007 [0.196, 0.212] (n=5) |

### bush_or_small_tree, P2 occupied at >= 0.5

| metric | stock (`may_scene_a`) | radar init (`may_scene_a_radar_init`) | radar init + loss (`may_scene_a_radar_init_loss`) |
|---|---|---|---|
| recall | 0.127 ± 0.007 [0.122, 0.138] (n=5) | 0.093 ± 0.006 [0.088, 0.104] (n=5) | 0.274 ± 0.005 [0.269, 0.280] (n=5) |
| precision | 0.819 ± 0.012 [0.807, 0.835] (n=5) | 0.763 ± 0.020 [0.739, 0.791] (n=5) | 0.663 ± 0.007 [0.655, 0.672] (n=5) |
| IoU | 0.124 ± 0.007 [0.119, 0.134] (n=5) | 0.091 ± 0.006 [0.086, 0.101] (n=5) | 0.241 ± 0.004 [0.236, 0.246] (n=5) |

### bush_or_small_tree, P2 occupied at >= 0.8

| metric | stock (`may_scene_a`) | radar init (`may_scene_a_radar_init`) | radar init + loss (`may_scene_a_radar_init_loss`) |
|---|---|---|---|
| recall | 0.106 ± 0.005 [0.102, 0.113] (n=5) | 0.076 ± 0.003 [0.073, 0.081] (n=5) | 0.244 ± 0.003 [0.241, 0.247] (n=5) |
| precision | 0.853 ± 0.011 [0.839, 0.864] (n=5) | 0.779 ± 0.020 [0.758, 0.803] (n=5) | 0.677 ± 0.004 [0.673, 0.684] (n=5) |
| IoU | 0.104 ± 0.005 [0.100, 0.111] (n=5) | 0.075 ± 0.003 [0.071, 0.079] (n=5) | 0.218 ± 0.002 [0.216, 0.222] (n=5) |

### Gap radar init (`may_scene_a_radar_init`) - stock (`may_scene_a`) (sep = the arms' [min, max] do not overlap, ovl = they overlap)

| metric | gap | sep / ovl |
|---|---|---|
| obstacles/P2@0.5/recall/pooled | -0.058 | sep |
| obstacles/P2@0.5/precision/pooled | -0.027 | sep |
| obstacles/P2@0.5/iou/pooled | -0.054 | sep |
| obstacles/P2@0.8/recall/pooled | -0.053 | sep |
| obstacles/P2@0.8/precision/pooled | -0.032 | sep |
| obstacles/P2@0.8/iou/pooled | -0.051 | sep |
| tree_trunk/P2@0.5/recall/pooled | -0.073 | sep |
| tree_trunk/P2@0.5/precision/pooled | -0.021 | sep |
| tree_trunk/P2@0.5/iou/pooled | -0.067 | sep |
| tree_trunk/P2@0.8/recall/pooled | -0.068 | sep |
| tree_trunk/P2@0.8/precision/pooled | -0.024 | sep |
| tree_trunk/P2@0.8/iou/pooled | -0.064 | sep |
| rock/P2@0.5/recall/pooled | -0.047 | sep |
| rock/P2@0.5/precision/pooled | -0.050 | ovl |
| rock/P2@0.5/iou/pooled | -0.036 | sep |
| rock/P2@0.8/recall/pooled | -0.063 | sep |
| rock/P2@0.8/precision/pooled | -0.070 | sep |
| rock/P2@0.8/iou/pooled | -0.049 | sep |
| bush_or_small_tree/P2@0.5/recall/pooled | -0.034 | sep |
| bush_or_small_tree/P2@0.5/precision/pooled | -0.056 | sep |
| bush_or_small_tree/P2@0.5/iou/pooled | -0.033 | sep |
| bush_or_small_tree/P2@0.8/recall/pooled | -0.030 | sep |
| bush_or_small_tree/P2@0.8/precision/pooled | -0.074 | sep |
| bush_or_small_tree/P2@0.8/iou/pooled | -0.030 | sep |

### Gap radar init + loss (`may_scene_a_radar_init_loss`) - stock (`may_scene_a`) (sep = the arms' [min, max] do not overlap, ovl = they overlap)

| metric | gap | sep / ovl |
|---|---|---|
| obstacles/P2@0.5/recall/pooled | 0.092 | sep |
| obstacles/P2@0.5/precision/pooled | -0.079 | sep |
| obstacles/P2@0.5/iou/pooled | 0.067 | sep |
| obstacles/P2@0.8/recall/pooled | 0.082 | sep |
| obstacles/P2@0.8/precision/pooled | -0.091 | sep |
| obstacles/P2@0.8/iou/pooled | 0.063 | sep |
| tree_trunk/P2@0.5/recall/pooled | 0.058 | sep |
| tree_trunk/P2@0.5/precision/pooled | -0.038 | sep |
| tree_trunk/P2@0.5/iou/pooled | 0.042 | sep |
| tree_trunk/P2@0.8/recall/pooled | 0.048 | sep |
| tree_trunk/P2@0.8/precision/pooled | -0.047 | sep |
| tree_trunk/P2@0.8/iou/pooled | 0.035 | sep |
| rock/P2@0.5/recall/pooled | 0.031 | ovl |
| rock/P2@0.5/precision/pooled | -0.027 | ovl |
| rock/P2@0.5/iou/pooled | 0.012 | ovl |
| rock/P2@0.8/recall/pooled | 0.039 | sep |
| rock/P2@0.8/precision/pooled | -0.021 | ovl |
| rock/P2@0.8/iou/pooled | 0.020 | sep |
| bush_or_small_tree/P2@0.5/recall/pooled | 0.147 | sep |
| bush_or_small_tree/P2@0.5/precision/pooled | -0.156 | sep |
| bush_or_small_tree/P2@0.5/iou/pooled | 0.117 | sep |
| bush_or_small_tree/P2@0.8/recall/pooled | 0.138 | sep |
| bush_or_small_tree/P2@0.8/precision/pooled | -0.176 | sep |
| bush_or_small_tree/P2@0.8/iou/pooled | 0.114 | sep |

### Gap radar init + loss (`may_scene_a_radar_init_loss`) - radar init (`may_scene_a_radar_init`) (sep = the arms' [min, max] do not overlap, ovl = they overlap)

| metric | gap | sep / ovl |
|---|---|---|
| obstacles/P2@0.5/recall/pooled | 0.150 | sep |
| obstacles/P2@0.5/precision/pooled | -0.052 | sep |
| obstacles/P2@0.5/iou/pooled | 0.121 | sep |
| obstacles/P2@0.8/recall/pooled | 0.136 | sep |
| obstacles/P2@0.8/precision/pooled | -0.058 | sep |
| obstacles/P2@0.8/iou/pooled | 0.114 | sep |
| tree_trunk/P2@0.5/recall/pooled | 0.131 | sep |
| tree_trunk/P2@0.5/precision/pooled | -0.017 | sep |
| tree_trunk/P2@0.5/iou/pooled | 0.109 | sep |
| tree_trunk/P2@0.8/recall/pooled | 0.116 | sep |
| tree_trunk/P2@0.8/precision/pooled | -0.023 | sep |
| tree_trunk/P2@0.8/iou/pooled | 0.100 | sep |
| rock/P2@0.5/recall/pooled | 0.079 | sep |
| rock/P2@0.5/precision/pooled | 0.023 | ovl |
| rock/P2@0.5/iou/pooled | 0.048 | sep |
| rock/P2@0.8/recall/pooled | 0.102 | sep |
| rock/P2@0.8/precision/pooled | 0.049 | sep |
| rock/P2@0.8/iou/pooled | 0.070 | sep |
| bush_or_small_tree/P2@0.5/recall/pooled | 0.181 | sep |
| bush_or_small_tree/P2@0.5/precision/pooled | -0.100 | sep |
| bush_or_small_tree/P2@0.5/iou/pooled | 0.150 | sep |
| bush_or_small_tree/P2@0.8/recall/pooled | 0.167 | sep |
| bush_or_small_tree/P2@0.8/precision/pooled | -0.102 | sep |
| bush_or_small_tree/P2@0.8/iou/pooled | 0.144 | sep |

![Whole scene in 3D, per arm](../figures/may_short_grass/arms_scene_3d.png)

**How to read this figure**
- **GT panel:** the obstacle boxes, coloured by class, for orientation.
- **Arm panels:** green = hit (filled, on an obstacle); red = false alarm (filled, next to an obstacle); grey = missed (empty, on an obstacle); faint grey = the scored area, the ring included. Correct-empty cells are not drawn. Cut-off t = 0.5. The terms are defined in point 5.
- **Difference panel:** hits only radar init + loss has, hits only stock has, and hits both have.
- One run per arm (the median repeat), so it illustrates and does not decide anything.


![Recall and precision against the occupancy threshold](../figures/may_short_grass/threshold_curves_scene.png)

**How to read this figure**
- **What is plotted:** recall (left) and precision (right) recomputed at each cut-off t from 0.2 to 0.9 (15 evenly spaced values). Solid lines are pooled and dashed lines are the mean over objects. The line is the median repeat. Each line has its own shaded band: at each t, the lowest and highest value of that same curve (pooled or mean over objects) over the 5 repeats of the arm. The pooled band is shaded more strongly than the mean band. Dotted lines mark the table cut-offs 0.5, 0.8.
- **Solid vs dashed:** solid = *pooled* (all objects' cells counted together, so large objects dominate); dashed = *mean over objects* (each object scored on its own, then averaged, so every object counts once). Full definitions and an example: point 5(f).
- **Definitions:** see point 5. Only recall and precision are plotted; IoU is in the tables. Recall = hit / (hit + missed) has all 19600 obstacle cells as its denominator; precision = hit / (hit + false alarm) has every filled scored cell as its denominator.
- **How the curves move:** raising t fills fewer cells, so recall can only fall and precision usually rises.
- **When one arm beats another:** its line is higher and its band does not overlap the other arm's band of the same kind (pooled with pooled, mean with mean).
- **Why t runs from 0.2 to 0.9:** the extremes are left out because they say nothing about the model. At t = 0 every cell counts as filled (recall 1, precision = the share of obstacle cells). At t ≈ 1 precision rests on a handful of cells and jumps.




Descriptive only: a variant gap counts only when it exceeds the spread between repeats without losing precision_H. Repeats share one seed, so they are not paired. Cells are mean ± sample std (n = repeats).

- **may_scene_a** (`splatad`): may_short_grass-may_scene_a-20261116-133928-17bb7a1 [17bb7a1; split identical to the reference record], may_short_grass-may_scene_a-20261116-135425-17bb7a1 [17bb7a1; split identical to the reference record], may_short_grass-may_scene_a-20261116-135428-17bb7a1 [17bb7a1; split identical to the reference record], may_short_grass-may_scene_a-20261116-135432-17bb7a1 [17bb7a1; split identical to the reference record], may_short_grass-may_scene_a-20261116-135435-17bb7a1 [17bb7a1; split identical to the reference record]
- **may_scene_a_radar_init** (`splatad-radar-init`): may_short_grass-may_scene_a_radar_init-20261116-134527-17bb7a1 [17bb7a1; split identical to the reference record], may_short_grass-may_scene_a_radar_init-20261116-135426-17bb7a1 [17bb7a1; split identical to the reference record], may_short_grass-may_scene_a_radar_init-20261116-135429-17bb7a1 [17bb7a1; split identical to the reference record], may_short_grass-may_scene_a_radar_init-20261116-135433-17bb7a1 [17bb7a1; split identical to the reference record], may_short_grass-may_scene_a_radar_init-20261116-135436-17bb7a1 [17bb7a1; split identical to the reference record]
- **may_scene_a_radar_init_loss** (`splatad-radar-init-lidar`): may_short_grass-may_scene_a_radar_init_loss-20261116-135125-17bb7a1 [17bb7a1; split identical to the reference record], may_short_grass-may_scene_a_radar_init_loss-20261116-135427-17bb7a1 [17bb7a1; split identical to the reference record], may_short_grass-may_scene_a_radar_init_loss-20261116-135431-17bb7a1 [17bb7a1; split identical to the reference record], may_short_grass-may_scene_a_radar_init_loss-20261116-135434-17bb7a1 [17bb7a1; split identical to the reference record], may_short_grass-may_scene_a_radar_init_loss-20261116-135437-17bb7a1 [17bb7a1; split identical to the reference record]

## obstacles, P2 occupied at >= 0.5

| metric | may_scene_a | may_scene_a_radar_init | may_scene_a_radar_init_loss |
|---|---|---|---|
| P2@0.5/recall/pooled | 0.278 ± 0.003 (n=5) | 0.221 ± 0.003 (n=5) | 0.370 ± 0.003 (n=5) |
| P2@0.5/recall/mean | 0.421 ± 0.003 (n=5) | 0.341 ± 0.002 (n=5) | 0.509 ± 0.006 (n=5) |
| P2@0.5/precision/pooled | 0.838 ± 0.004 (n=5) | 0.810 ± 0.005 (n=5) | 0.758 ± 0.006 (n=5) |
| P2@0.5/precision/mean | 0.703 ± 0.012 (n=5) | 0.676 ± 0.020 (n=5) | 0.690 ± 0.011 (n=5) |
| P2@0.5/iou/pooled | 0.264 ± 0.003 (n=5) | 0.210 ± 0.003 (n=5) | 0.331 ± 0.002 (n=5) |
| P2@0.5/iou/mean | 0.343 ± 0.002 (n=5) | 0.275 ± 0.002 (n=5) | 0.375 ± 0.003 (n=5) |
| P1/recall/pooled | 0.299 ± 0.001 (n=5) | 0.273 ± 0.002 (n=5) | 0.298 ± 0.002 (n=5) |
| P1/recall/mean | 0.433 ± 0.003 (n=5) | 0.404 ± 0.002 (n=5) | 0.430 ± 0.002 (n=5) |
| P1/precision/pooled | 0.680 ± 0.004 (n=5) | 0.660 ± 0.004 (n=5) | 0.673 ± 0.003 (n=5) |
| P1/precision/mean | 0.590 ± 0.007 (n=5) | 0.586 ± 0.005 (n=5) | 0.604 ± 0.010 (n=5) |
| P2@0.5/recall_Hnear/pooled | 0.140 ± 0.105 (n=5) | 0.080 ± 0.012 (n=5) | 0.215 ± 0.077 (n=5) |
| P2@0.5/recall_Hnear/mean | 0.194 ± 0.059 (n=5) | 0.190 ± 0.049 (n=5) | 0.388 ± 0.038 (n=5) |
| P2@0.5/precision_Hnear/pooled | 0.610 ± 0.152 (n=5) | 0.630 ± 0.088 (n=5) | 0.536 ± 0.095 (n=5) |
| P2@0.5/precision_Hnear/mean | 0.466 ± 0.041 (n=5) | 0.570 ± 0.084 (n=5) | 0.457 ± 0.038 (n=5) |
| P2@0.5/iou_Hnear/pooled | 0.131 ± 0.101 (n=5) | 0.076 ± 0.012 (n=5) | 0.183 ± 0.066 (n=5) |
| P2@0.5/iou_Hnear/mean | 0.156 ± 0.042 (n=5) | 0.160 ± 0.033 (n=5) | 0.204 ± 0.030 (n=5) |
| P1/recall_Hnear/pooled | 0.033 ± 0.012 (n=5) | 0.031 ± 0.008 (n=5) | 0.046 ± 0.015 (n=5) |
| P1/recall_Hnear/mean | 0.137 ± 0.042 (n=5) | 0.122 ± 0.035 (n=5) | 0.177 ± 0.065 (n=5) |
| hidden_near/gt_cells | 193.000 ± 0.000 (n=5) | 193.000 ± 0.000 (n=5) | 193.000 ± 0.000 (n=5) |
| hidden_near/objects_with_hidden_gt | 13.000 ± 0.000 (n=5) | 13.000 ± 0.000 (n=5) | 13.000 ± 0.000 (n=5) |
| P2@0.5/recall_H/pooled | 0.019 ± 0.005 (n=5) | 0.006 ± 0.002 (n=5) | 0.097 ± 0.002 (n=5) |
| P2@0.5/recall_H/mean | 0.067 ± 0.005 (n=5) | 0.036 ± 0.009 (n=5) | 0.166 ± 0.009 (n=5) |
| P2@0.5/precision_H/pooled | 0.561 ± 0.062 (n=5) | 0.580 ± 0.093 (n=5) | 0.590 ± 0.017 (n=5) |
| P2@0.5/precision_H/mean | 0.513 ± 0.035 (n=5) | 0.548 ± 0.077 (n=5) | 0.579 ± 0.031 (n=5) |
| P2@0.5/iou_H/pooled | 0.019 ± 0.005 (n=5) | 0.006 ± 0.002 (n=5) | 0.091 ± 0.001 (n=5) |
| P2@0.5/iou_H/mean | 0.060 ± 0.004 (n=5) | 0.033 ± 0.008 (n=5) | 0.116 ± 0.008 (n=5) |
| hidden/gt_cells | 6231.000 ± 0.000 (n=5) | 6231.000 ± 0.000 (n=5) | 6231.000 ± 0.000 (n=5) |
| hidden/objects_with_hidden_gt | 92.000 ± 0.000 (n=5) | 92.000 ± 0.000 (n=5) | 92.000 ± 0.000 (n=5) |
| P1/recall_H/pooled | 0.010 ± 0.001 (n=5) | 0.005 ± 0.002 (n=5) | 0.008 ± 0.002 (n=5) |
| P1/recall_H/mean | 0.034 ± 0.005 (n=5) | 0.032 ± 0.013 (n=5) | 0.037 ± 0.010 (n=5) |


## obstacles, P2 occupied at >= 0.8

| metric | may_scene_a | may_scene_a_radar_init | may_scene_a_radar_init_loss |
|---|---|---|---|
| P2@0.8/recall/pooled | 0.244 ± 0.002 (n=5) | 0.190 ± 0.002 (n=5) | 0.326 ± 0.001 (n=5) |
| P2@0.8/recall/mean | 0.379 ± 0.002 (n=5) | 0.297 ± 0.003 (n=5) | 0.469 ± 0.003 (n=5) |
| P2@0.8/precision/pooled | 0.863 ± 0.003 (n=5) | 0.831 ± 0.002 (n=5) | 0.772 ± 0.003 (n=5) |
| P2@0.8/precision/mean | 0.716 ± 0.012 (n=5) | 0.685 ± 0.012 (n=5) | 0.714 ± 0.008 (n=5) |
| P2@0.8/iou/pooled | 0.234 ± 0.002 (n=5) | 0.183 ± 0.002 (n=5) | 0.297 ± 0.001 (n=5) |
| P2@0.8/iou/mean | 0.316 ± 0.002 (n=5) | 0.246 ± 0.003 (n=5) | 0.351 ± 0.003 (n=5) |
| P1/recall/pooled | 0.299 ± 0.001 (n=5) | 0.273 ± 0.002 (n=5) | 0.298 ± 0.002 (n=5) |
| P1/recall/mean | 0.433 ± 0.003 (n=5) | 0.404 ± 0.002 (n=5) | 0.430 ± 0.002 (n=5) |
| P1/precision/pooled | 0.680 ± 0.004 (n=5) | 0.660 ± 0.004 (n=5) | 0.673 ± 0.003 (n=5) |
| P1/precision/mean | 0.590 ± 0.007 (n=5) | 0.586 ± 0.005 (n=5) | 0.604 ± 0.010 (n=5) |
| P2@0.8/recall_Hnear/pooled | 0.095 ± 0.070 (n=5) | 0.062 ± 0.013 (n=5) | 0.128 ± 0.041 (n=5) |
| P2@0.8/recall_Hnear/mean | 0.168 ± 0.062 (n=5) | 0.144 ± 0.021 (n=5) | 0.337 ± 0.021 (n=5) |
| P2@0.8/precision_Hnear/pooled | 0.627 ± 0.164 (n=5) | 0.605 ± 0.105 (n=5) | 0.514 ± 0.082 (n=5) |
| P2@0.8/precision_Hnear/mean | 0.536 ± 0.127 (n=5) | 0.556 ± 0.101 (n=5) | 0.479 ± 0.043 (n=5) |
| P2@0.8/iou_Hnear/pooled | 0.092 ± 0.070 (n=5) | 0.060 ± 0.012 (n=5) | 0.115 ± 0.038 (n=5) |
| P2@0.8/iou_Hnear/mean | 0.145 ± 0.050 (n=5) | 0.119 ± 0.030 (n=5) | 0.174 ± 0.018 (n=5) |
| P1/recall_Hnear/pooled | 0.033 ± 0.012 (n=5) | 0.031 ± 0.008 (n=5) | 0.046 ± 0.015 (n=5) |
| P1/recall_Hnear/mean | 0.137 ± 0.042 (n=5) | 0.122 ± 0.035 (n=5) | 0.177 ± 0.065 (n=5) |
| hidden_near/gt_cells | 193.000 ± 0.000 (n=5) | 193.000 ± 0.000 (n=5) | 193.000 ± 0.000 (n=5) |
| hidden_near/objects_with_hidden_gt | 13.000 ± 0.000 (n=5) | 13.000 ± 0.000 (n=5) | 13.000 ± 0.000 (n=5) |
| P2@0.8/recall_H/pooled | 0.011 ± 0.004 (n=5) | 0.005 ± 0.002 (n=5) | 0.068 ± 0.001 (n=5) |
| P2@0.8/recall_H/mean | 0.047 ± 0.004 (n=5) | 0.027 ± 0.007 (n=5) | 0.114 ± 0.009 (n=5) |
| P2@0.8/precision_H/pooled | 0.582 ± 0.083 (n=5) | 0.589 ± 0.087 (n=5) | 0.564 ± 0.007 (n=5) |
| P2@0.8/precision_H/mean | 0.537 ± 0.055 (n=5) | 0.520 ± 0.086 (n=5) | 0.562 ± 0.025 (n=5) |
| P2@0.8/iou_H/pooled | 0.010 ± 0.004 (n=5) | 0.005 ± 0.002 (n=5) | 0.065 ± 0.001 (n=5) |
| P2@0.8/iou_H/mean | 0.044 ± 0.004 (n=5) | 0.024 ± 0.008 (n=5) | 0.074 ± 0.008 (n=5) |
| hidden/gt_cells | 6231.000 ± 0.000 (n=5) | 6231.000 ± 0.000 (n=5) | 6231.000 ± 0.000 (n=5) |
| hidden/objects_with_hidden_gt | 92.000 ± 0.000 (n=5) | 92.000 ± 0.000 (n=5) | 92.000 ± 0.000 (n=5) |
| P1/recall_H/pooled | 0.010 ± 0.001 (n=5) | 0.005 ± 0.002 (n=5) | 0.008 ± 0.002 (n=5) |
| P1/recall_H/mean | 0.034 ± 0.005 (n=5) | 0.032 ± 0.013 (n=5) | 0.037 ± 0.010 (n=5) |


## tree_trunk, P2 occupied at >= 0.5

| metric | may_scene_a | may_scene_a_radar_init | may_scene_a_radar_init_loss |
|---|---|---|---|
| P2@0.5/recall/pooled | 0.375 ± 0.002 (n=5) | 0.302 ± 0.001 (n=5) | 0.433 ± 0.004 (n=5) |
| P2@0.5/recall/mean | 0.491 ± 0.002 (n=5) | 0.400 ± 0.002 (n=5) | 0.548 ± 0.006 (n=5) |
| P2@0.5/precision/pooled | 0.849 ± 0.005 (n=5) | 0.828 ± 0.002 (n=5) | 0.811 ± 0.009 (n=5) |
| P2@0.5/precision/mean | 0.744 ± 0.014 (n=5) | 0.727 ± 0.020 (n=5) | 0.752 ± 0.011 (n=5) |
| P2@0.5/iou/pooled | 0.351 ± 0.001 (n=5) | 0.284 ± 0.001 (n=5) | 0.393 ± 0.004 (n=5) |
| P2@0.5/iou/mean | 0.406 ± 0.001 (n=5) | 0.328 ± 0.003 (n=5) | 0.432 ± 0.003 (n=5) |
| P1/recall/pooled | 0.379 ± 0.002 (n=5) | 0.343 ± 0.002 (n=5) | 0.376 ± 0.003 (n=5) |
| P1/recall/mean | 0.493 ± 0.002 (n=5) | 0.453 ± 0.002 (n=5) | 0.486 ± 0.003 (n=5) |
| P1/precision/pooled | 0.711 ± 0.004 (n=5) | 0.687 ± 0.006 (n=5) | 0.712 ± 0.004 (n=5) |
| P1/precision/mean | 0.649 ± 0.008 (n=5) | 0.643 ± 0.007 (n=5) | 0.668 ± 0.013 (n=5) |
| P2@0.5/recall_Hnear/pooled | 0.320 ± 0.065 (n=5) | 0.253 ± 0.030 (n=5) | 0.333 ± 0.100 (n=5) |
| P2@0.5/recall_Hnear/mean | 0.315 ± 0.099 (n=5) | 0.311 ± 0.101 (n=5) | 0.288 ± 0.059 (n=5) |
| P2@0.5/precision_Hnear/pooled | 0.785 ± 0.128 (n=5) | 0.672 ± 0.102 (n=5) | 0.760 ± 0.074 (n=5) |
| P2@0.5/precision_Hnear/mean | 0.638 ± 0.109 (n=5) | 0.652 ± 0.047 (n=5) | 0.607 ± 0.048 (n=5) |
| P2@0.5/iou_Hnear/pooled | 0.297 ± 0.070 (n=5) | 0.226 ± 0.034 (n=5) | 0.303 ± 0.091 (n=5) |
| P2@0.5/iou_Hnear/mean | 0.253 ± 0.084 (n=5) | 0.255 ± 0.071 (n=5) | 0.231 ± 0.045 (n=5) |
| P1/recall_Hnear/pooled | 0.147 ± 0.038 (n=5) | 0.093 ± 0.049 (n=5) | 0.147 ± 0.061 (n=5) |
| P1/recall_Hnear/mean | 0.164 ± 0.074 (n=5) | 0.102 ± 0.090 (n=5) | 0.189 ± 0.119 (n=5) |
| hidden_near/gt_cells | 30.000 ± 0.000 (n=5) | 30.000 ± 0.000 (n=5) | 30.000 ± 0.000 (n=5) |
| hidden_near/objects_with_hidden_gt | 7.000 ± 0.000 (n=5) | 7.000 ± 0.000 (n=5) | 7.000 ± 0.000 (n=5) |
| P2@0.5/recall_H/pooled | 0.028 ± 0.003 (n=5) | 0.009 ± 0.004 (n=5) | 0.094 ± 0.003 (n=5) |
| P2@0.5/recall_H/mean | 0.087 ± 0.012 (n=5) | 0.047 ± 0.015 (n=5) | 0.161 ± 0.017 (n=5) |
| P2@0.5/precision_H/pooled | 0.627 ± 0.033 (n=5) | 0.661 ± 0.085 (n=5) | 0.719 ± 0.051 (n=5) |
| P2@0.5/precision_H/mean | 0.581 ± 0.023 (n=5) | 0.625 ± 0.090 (n=5) | 0.666 ± 0.040 (n=5) |
| P2@0.5/iou_H/pooled | 0.027 ± 0.003 (n=5) | 0.009 ± 0.004 (n=5) | 0.091 ± 0.003 (n=5) |
| P2@0.5/iou_H/mean | 0.078 ± 0.010 (n=5) | 0.043 ± 0.013 (n=5) | 0.132 ± 0.015 (n=5) |
| hidden/gt_cells | 2983.000 ± 0.000 (n=5) | 2983.000 ± 0.000 (n=5) | 2983.000 ± 0.000 (n=5) |
| hidden/objects_with_hidden_gt | 63.000 ± 0.000 (n=5) | 63.000 ± 0.000 (n=5) | 63.000 ± 0.000 (n=5) |
| P1/recall_H/pooled | 0.012 ± 0.002 (n=5) | 0.005 ± 0.002 (n=5) | 0.010 ± 0.001 (n=5) |
| P1/recall_H/mean | 0.037 ± 0.010 (n=5) | 0.030 ± 0.019 (n=5) | 0.037 ± 0.015 (n=5) |


## tree_trunk, P2 occupied at >= 0.8

| metric | may_scene_a | may_scene_a_radar_init | may_scene_a_radar_init_loss |
|---|---|---|---|
| P2@0.8/recall/pooled | 0.331 ± 0.001 (n=5) | 0.263 ± 0.003 (n=5) | 0.379 ± 0.002 (n=5) |
| P2@0.8/recall/mean | 0.443 ± 0.001 (n=5) | 0.352 ± 0.003 (n=5) | 0.504 ± 0.004 (n=5) |
| P2@0.8/precision/pooled | 0.872 ± 0.005 (n=5) | 0.848 ± 0.003 (n=5) | 0.825 ± 0.003 (n=5) |
| P2@0.8/precision/mean | 0.751 ± 0.011 (n=5) | 0.738 ± 0.018 (n=5) | 0.770 ± 0.012 (n=5) |
| P2@0.8/iou/pooled | 0.316 ± 0.001 (n=5) | 0.251 ± 0.003 (n=5) | 0.351 ± 0.002 (n=5) |
| P2@0.8/iou/mean | 0.374 ± 0.001 (n=5) | 0.295 ± 0.003 (n=5) | 0.401 ± 0.003 (n=5) |
| P1/recall/pooled | 0.379 ± 0.002 (n=5) | 0.343 ± 0.002 (n=5) | 0.376 ± 0.003 (n=5) |
| P1/recall/mean | 0.493 ± 0.002 (n=5) | 0.453 ± 0.002 (n=5) | 0.486 ± 0.003 (n=5) |
| P1/precision/pooled | 0.711 ± 0.004 (n=5) | 0.687 ± 0.006 (n=5) | 0.712 ± 0.004 (n=5) |
| P1/precision/mean | 0.649 ± 0.008 (n=5) | 0.643 ± 0.007 (n=5) | 0.668 ± 0.013 (n=5) |
| P2@0.8/recall_Hnear/pooled | 0.240 ± 0.080 (n=5) | 0.193 ± 0.055 (n=5) | 0.220 ± 0.061 (n=5) |
| P2@0.8/recall_Hnear/mean | 0.284 ± 0.104 (n=5) | 0.229 ± 0.063 (n=5) | 0.239 ± 0.039 (n=5) |
| P2@0.8/precision_Hnear/pooled | 0.817 ± 0.160 (n=5) | 0.645 ± 0.116 (n=5) | 0.707 ± 0.095 (n=5) |
| P2@0.8/precision_Hnear/mean | 0.725 ± 0.225 (n=5) | 0.623 ± 0.040 (n=5) | 0.595 ± 0.088 (n=5) |
| P2@0.8/iou_Hnear/pooled | 0.229 ± 0.076 (n=5) | 0.175 ± 0.052 (n=5) | 0.201 ± 0.054 (n=5) |
| P2@0.8/iou_Hnear/mean | 0.243 ± 0.081 (n=5) | 0.184 ± 0.055 (n=5) | 0.192 ± 0.027 (n=5) |
| P1/recall_Hnear/pooled | 0.147 ± 0.038 (n=5) | 0.093 ± 0.049 (n=5) | 0.147 ± 0.061 (n=5) |
| P1/recall_Hnear/mean | 0.164 ± 0.074 (n=5) | 0.102 ± 0.090 (n=5) | 0.189 ± 0.119 (n=5) |
| hidden_near/gt_cells | 30.000 ± 0.000 (n=5) | 30.000 ± 0.000 (n=5) | 30.000 ± 0.000 (n=5) |
| hidden_near/objects_with_hidden_gt | 7.000 ± 0.000 (n=5) | 7.000 ± 0.000 (n=5) | 7.000 ± 0.000 (n=5) |
| P2@0.8/recall_H/pooled | 0.017 ± 0.003 (n=5) | 0.007 ± 0.004 (n=5) | 0.048 ± 0.003 (n=5) |
| P2@0.8/recall_H/mean | 0.063 ± 0.006 (n=5) | 0.035 ± 0.013 (n=5) | 0.096 ± 0.014 (n=5) |
| P2@0.8/precision_H/pooled | 0.633 ± 0.074 (n=5) | 0.647 ± 0.133 (n=5) | 0.642 ± 0.021 (n=5) |
| P2@0.8/precision_H/mean | 0.587 ± 0.071 (n=5) | 0.596 ± 0.114 (n=5) | 0.615 ± 0.026 (n=5) |
| P2@0.8/iou_H/pooled | 0.017 ± 0.003 (n=5) | 0.007 ± 0.004 (n=5) | 0.047 ± 0.002 (n=5) |
| P2@0.8/iou_H/mean | 0.059 ± 0.005 (n=5) | 0.031 ± 0.013 (n=5) | 0.077 ± 0.012 (n=5) |
| hidden/gt_cells | 2983.000 ± 0.000 (n=5) | 2983.000 ± 0.000 (n=5) | 2983.000 ± 0.000 (n=5) |
| hidden/objects_with_hidden_gt | 63.000 ± 0.000 (n=5) | 63.000 ± 0.000 (n=5) | 63.000 ± 0.000 (n=5) |
| P1/recall_H/pooled | 0.012 ± 0.002 (n=5) | 0.005 ± 0.002 (n=5) | 0.010 ± 0.001 (n=5) |
| P1/recall_H/mean | 0.037 ± 0.010 (n=5) | 0.030 ± 0.019 (n=5) | 0.037 ± 0.015 (n=5) |


## rock, P2 occupied at >= 0.5

| metric | may_scene_a | may_scene_a_radar_init | may_scene_a_radar_init_loss |
|---|---|---|---|
| P2@0.5/recall/pooled | 0.253 ± 0.018 (n=5) | 0.206 ± 0.021 (n=5) | 0.284 ± 0.010 (n=5) |
| P2@0.5/recall/mean | 0.237 ± 0.016 (n=5) | 0.195 ± 0.019 (n=5) | 0.282 ± 0.022 (n=5) |
| P2@0.5/precision/pooled | 0.456 ± 0.011 (n=5) | 0.406 ± 0.034 (n=5) | 0.429 ± 0.020 (n=5) |
| P2@0.5/precision/mean | 0.507 ± 0.039 (n=5) | 0.470 ± 0.056 (n=5) | 0.387 ± 0.056 (n=5) |
| P2@0.5/iou/pooled | 0.194 ± 0.012 (n=5) | 0.158 ± 0.017 (n=5) | 0.206 ± 0.008 (n=5) |
| P2@0.5/iou/mean | 0.146 ± 0.013 (n=5) | 0.117 ± 0.011 (n=5) | 0.156 ± 0.017 (n=5) |
| P1/recall/pooled | 0.263 ± 0.023 (n=5) | 0.262 ± 0.005 (n=5) | 0.270 ± 0.016 (n=5) |
| P1/recall/mean | 0.242 ± 0.023 (n=5) | 0.246 ± 0.010 (n=5) | 0.253 ± 0.010 (n=5) |
| P1/precision/pooled | 0.304 ± 0.013 (n=5) | 0.299 ± 0.014 (n=5) | 0.308 ± 0.016 (n=5) |
| P1/precision/mean | 0.281 ± 0.031 (n=5) | 0.300 ± 0.031 (n=5) | 0.286 ± 0.026 (n=5) |
| P2@0.5/recall_Hnear/pooled | n/a (n=0) | n/a (n=0) | n/a (n=0) |
| P2@0.5/recall_Hnear/mean | n/a (n=0) | n/a (n=0) | n/a (n=0) |
| P2@0.5/precision_Hnear/pooled | n/a (n=0) | n/a (n=0) | n/a (n=0) |
| P2@0.5/precision_Hnear/mean | n/a (n=0) | n/a (n=0) | n/a (n=0) |
| P2@0.5/iou_Hnear/pooled | n/a (n=0) | n/a (n=0) | n/a (n=0) |
| P2@0.5/iou_Hnear/mean | n/a (n=0) | n/a (n=0) | n/a (n=0) |
| P1/recall_Hnear/pooled | n/a (n=0) | n/a (n=0) | n/a (n=0) |
| P1/recall_Hnear/mean | n/a (n=0) | n/a (n=0) | n/a (n=0) |
| hidden_near/gt_cells | 0.000 ± 0.000 (n=5) | 0.000 ± 0.000 (n=5) | 0.000 ± 0.000 (n=5) |
| hidden_near/objects_with_hidden_gt | 0.000 ± 0.000 (n=5) | 0.000 ± 0.000 (n=5) | 0.000 ± 0.000 (n=5) |
| P2@0.5/recall_H/pooled | 0.012 ± 0.022 (n=5) | 0.000 ± 0.000 (n=5) | 0.010 ± 0.017 (n=5) |
| P2@0.5/recall_H/mean | 0.011 ± 0.015 (n=5) | 0.000 ± 0.000 (n=5) | 0.019 ± 0.036 (n=5) |
| P2@0.5/precision_H/pooled | 0.857 ± 0.202 (n=2) | n/a (n=0) | 0.167 ± 0.236 (n=5) |
| P2@0.5/precision_H/mean | 0.708 ± 0.412 (n=2) | n/a (n=0) | 0.133 ± 0.183 (n=5) |
| P2@0.5/iou_H/pooled | 0.012 ± 0.021 (n=5) | 0.000 ± 0.000 (n=5) | 0.010 ± 0.017 (n=5) |
| P2@0.5/iou_H/mean | 0.010 ± 0.014 (n=5) | 0.000 ± 0.000 (n=5) | 0.013 ± 0.024 (n=5) |
| hidden/gt_cells | 99.000 ± 0.000 (n=5) | 99.000 ± 0.000 (n=5) | 99.000 ± 0.000 (n=5) |
| hidden/objects_with_hidden_gt | 12.000 ± 0.000 (n=5) | 12.000 ± 0.000 (n=5) | 12.000 ± 0.000 (n=5) |
| P1/recall_H/pooled | 0.000 ± 0.000 (n=5) | 0.004 ± 0.006 (n=5) | 0.000 ± 0.000 (n=5) |
| P1/recall_H/mean | 0.000 ± 0.000 (n=5) | 0.008 ± 0.011 (n=5) | 0.000 ± 0.000 (n=5) |


## rock, P2 occupied at >= 0.8

| metric | may_scene_a | may_scene_a_radar_init | may_scene_a_radar_init_loss |
|---|---|---|---|
| P2@0.8/recall/pooled | 0.228 ± 0.015 (n=5) | 0.165 ± 0.011 (n=5) | 0.267 ± 0.010 (n=5) |
| P2@0.8/recall/mean | 0.211 ± 0.017 (n=5) | 0.154 ± 0.011 (n=5) | 0.263 ± 0.010 (n=5) |
| P2@0.8/precision/pooled | 0.478 ± 0.022 (n=5) | 0.408 ± 0.025 (n=5) | 0.458 ± 0.011 (n=5) |
| P2@0.8/precision/mean | 0.526 ± 0.039 (n=5) | 0.477 ± 0.061 (n=5) | 0.460 ± 0.047 (n=5) |
| P2@0.8/iou/pooled | 0.183 ± 0.012 (n=5) | 0.133 ± 0.010 (n=5) | 0.203 ± 0.007 (n=5) |
| P2@0.8/iou/mean | 0.140 ± 0.014 (n=5) | 0.105 ± 0.009 (n=5) | 0.156 ± 0.009 (n=5) |
| P1/recall/pooled | 0.263 ± 0.023 (n=5) | 0.262 ± 0.005 (n=5) | 0.270 ± 0.016 (n=5) |
| P1/recall/mean | 0.242 ± 0.023 (n=5) | 0.246 ± 0.010 (n=5) | 0.253 ± 0.010 (n=5) |
| P1/precision/pooled | 0.304 ± 0.013 (n=5) | 0.299 ± 0.014 (n=5) | 0.308 ± 0.016 (n=5) |
| P1/precision/mean | 0.281 ± 0.031 (n=5) | 0.300 ± 0.031 (n=5) | 0.286 ± 0.026 (n=5) |
| P2@0.8/recall_Hnear/pooled | n/a (n=0) | n/a (n=0) | n/a (n=0) |
| P2@0.8/recall_Hnear/mean | n/a (n=0) | n/a (n=0) | n/a (n=0) |
| P2@0.8/precision_Hnear/pooled | n/a (n=0) | n/a (n=0) | n/a (n=0) |
| P2@0.8/precision_Hnear/mean | n/a (n=0) | n/a (n=0) | n/a (n=0) |
| P2@0.8/iou_Hnear/pooled | n/a (n=0) | n/a (n=0) | n/a (n=0) |
| P2@0.8/iou_Hnear/mean | n/a (n=0) | n/a (n=0) | n/a (n=0) |
| P1/recall_Hnear/pooled | n/a (n=0) | n/a (n=0) | n/a (n=0) |
| P1/recall_Hnear/mean | n/a (n=0) | n/a (n=0) | n/a (n=0) |
| hidden_near/gt_cells | 0.000 ± 0.000 (n=5) | 0.000 ± 0.000 (n=5) | 0.000 ± 0.000 (n=5) |
| hidden_near/objects_with_hidden_gt | 0.000 ± 0.000 (n=5) | 0.000 ± 0.000 (n=5) | 0.000 ± 0.000 (n=5) |
| P2@0.8/recall_H/pooled | 0.004 ± 0.006 (n=5) | 0.000 ± 0.000 (n=5) | 0.004 ± 0.009 (n=5) |
| P2@0.8/recall_H/mean | 0.007 ± 0.012 (n=5) | 0.000 ± 0.000 (n=5) | 0.008 ± 0.019 (n=5) |
| P2@0.8/precision_H/pooled | 0.750 ± 0.354 (n=2) | n/a (n=0) | 0.333 ± 0.577 (n=3) |
| P2@0.8/precision_H/mean | 0.750 ± 0.354 (n=2) | n/a (n=0) | 0.333 ± 0.577 (n=3) |
| P2@0.8/iou_H/pooled | 0.004 ± 0.006 (n=5) | 0.000 ± 0.000 (n=5) | 0.004 ± 0.009 (n=5) |
| P2@0.8/iou_H/mean | 0.007 ± 0.012 (n=5) | 0.000 ± 0.000 (n=5) | 0.008 ± 0.019 (n=5) |
| hidden/gt_cells | 99.000 ± 0.000 (n=5) | 99.000 ± 0.000 (n=5) | 99.000 ± 0.000 (n=5) |
| hidden/objects_with_hidden_gt | 12.000 ± 0.000 (n=5) | 12.000 ± 0.000 (n=5) | 12.000 ± 0.000 (n=5) |
| P1/recall_H/pooled | 0.000 ± 0.000 (n=5) | 0.004 ± 0.006 (n=5) | 0.000 ± 0.000 (n=5) |
| P1/recall_H/mean | 0.000 ± 0.000 (n=5) | 0.008 ± 0.011 (n=5) | 0.000 ± 0.000 (n=5) |


## bush_or_small_tree, P2 occupied at >= 0.5

| metric | may_scene_a | may_scene_a_radar_init | may_scene_a_radar_init_loss |
|---|---|---|---|
| P2@0.5/recall/pooled | 0.127 ± 0.007 (n=5) | 0.093 ± 0.006 (n=5) | 0.274 ± 0.005 (n=5) |
| P2@0.5/recall/mean | 0.185 ± 0.012 (n=5) | 0.131 ± 0.008 (n=5) | 0.451 ± 0.006 (n=5) |
| P2@0.5/precision/pooled | 0.819 ± 0.012 (n=5) | 0.763 ± 0.020 (n=5) | 0.663 ± 0.007 (n=5) |
| P2@0.5/precision/mean | 0.547 ± 0.033 (n=5) | 0.481 ± 0.043 (n=5) | 0.490 ± 0.015 (n=5) |
| P2@0.5/iou/pooled | 0.124 ± 0.007 (n=5) | 0.091 ± 0.006 (n=5) | 0.241 ± 0.004 (n=5) |
| P2@0.5/iou/mean | 0.147 ± 0.008 (n=5) | 0.104 ± 0.007 (n=5) | 0.230 ± 0.003 (n=5) |
| P1/recall/pooled | 0.174 ± 0.002 (n=5) | 0.165 ± 0.003 (n=5) | 0.175 ± 0.001 (n=5) |
| P1/recall/mean | 0.252 ± 0.003 (n=5) | 0.250 ± 0.002 (n=5) | 0.260 ± 0.006 (n=5) |
| P1/precision/pooled | 0.615 ± 0.007 (n=5) | 0.607 ± 0.011 (n=5) | 0.589 ± 0.007 (n=5) |
| P1/precision/mean | 0.376 ± 0.021 (n=5) | 0.398 ± 0.020 (n=5) | 0.360 ± 0.021 (n=5) |
| P2@0.5/recall_Hnear/pooled | 0.107 ± 0.126 (n=5) | 0.048 ± 0.012 (n=5) | 0.193 ± 0.078 (n=5) |
| P2@0.5/recall_Hnear/mean | 0.052 ± 0.047 (n=5) | 0.050 ± 0.038 (n=5) | 0.505 ± 0.029 (n=5) |
| P2@0.5/precision_Hnear/pooled | 0.510 ± 0.234 (n=5) | 0.592 ± 0.100 (n=5) | 0.488 ± 0.100 (n=5) |
| P2@0.5/precision_Hnear/mean | 0.264 ± 0.132 (n=5) | 0.469 ± 0.139 (n=5) | 0.344 ± 0.043 (n=5) |
| P2@0.5/iou_Hnear/pooled | 0.101 ± 0.121 (n=5) | 0.046 ± 0.012 (n=5) | 0.162 ± 0.067 (n=5) |
| P2@0.5/iou_Hnear/mean | 0.041 ± 0.036 (n=5) | 0.041 ± 0.030 (n=5) | 0.172 ± 0.019 (n=5) |
| P1/recall_Hnear/pooled | 0.012 ± 0.011 (n=5) | 0.020 ± 0.007 (n=5) | 0.027 ± 0.014 (n=5) |
| P1/recall_Hnear/mean | 0.106 ± 0.087 (n=5) | 0.145 ± 0.031 (n=5) | 0.163 ± 0.083 (n=5) |
| hidden_near/gt_cells | 163.000 ± 0.000 (n=5) | 163.000 ± 0.000 (n=5) | 163.000 ± 0.000 (n=5) |
| hidden_near/objects_with_hidden_gt | 6.000 ± 0.000 (n=5) | 6.000 ± 0.000 (n=5) | 6.000 ± 0.000 (n=5) |
| P2@0.5/recall_H/pooled | 0.010 ± 0.008 (n=5) | 0.003 ± 0.001 (n=5) | 0.103 ± 0.002 (n=5) |
| P2@0.5/recall_H/mean | 0.030 ± 0.027 (n=5) | 0.020 ± 0.012 (n=5) | 0.289 ± 0.013 (n=5) |
| P2@0.5/precision_H/pooled | 0.427 ± 0.162 (n=5) | 0.445 ± 0.118 (n=5) | 0.514 ± 0.011 (n=5) |
| P2@0.5/precision_H/mean | 0.273 ± 0.123 (n=5) | 0.369 ± 0.096 (n=5) | 0.402 ± 0.040 (n=5) |
| P2@0.5/iou_H/pooled | 0.010 ± 0.007 (n=5) | 0.003 ± 0.001 (n=5) | 0.094 ± 0.002 (n=5) |
| P2@0.5/iou_H/mean | 0.024 ± 0.020 (n=5) | 0.017 ± 0.010 (n=5) | 0.120 ± 0.007 (n=5) |
| hidden/gt_cells | 3149.000 ± 0.000 (n=5) | 3149.000 ± 0.000 (n=5) | 3149.000 ± 0.000 (n=5) |
| hidden/objects_with_hidden_gt | 17.000 ± 0.000 (n=5) | 17.000 ± 0.000 (n=5) | 17.000 ± 0.000 (n=5) |
| P1/recall_H/pooled | 0.007 ± 0.003 (n=5) | 0.005 ± 0.004 (n=5) | 0.005 ± 0.003 (n=5) |
| P1/recall_H/mean | 0.044 ± 0.032 (n=5) | 0.055 ± 0.010 (n=5) | 0.061 ± 0.030 (n=5) |


## bush_or_small_tree, P2 occupied at >= 0.8

| metric | may_scene_a | may_scene_a_radar_init | may_scene_a_radar_init_loss |
|---|---|---|---|
| P2@0.8/recall/pooled | 0.106 ± 0.005 (n=5) | 0.076 ± 0.003 (n=5) | 0.244 ± 0.003 (n=5) |
| P2@0.8/recall/mean | 0.159 ± 0.003 (n=5) | 0.107 ± 0.007 (n=5) | 0.422 ± 0.004 (n=5) |
| P2@0.8/precision/pooled | 0.853 ± 0.011 (n=5) | 0.779 ± 0.020 (n=5) | 0.677 ± 0.004 (n=5) |
| P2@0.8/precision/mean | 0.598 ± 0.027 (n=5) | 0.487 ± 0.019 (n=5) | 0.531 ± 0.021 (n=5) |
| P2@0.8/iou/pooled | 0.104 ± 0.005 (n=5) | 0.075 ± 0.003 (n=5) | 0.218 ± 0.002 (n=5) |
| P2@0.8/iou/mean | 0.131 ± 0.002 (n=5) | 0.090 ± 0.007 (n=5) | 0.224 ± 0.002 (n=5) |
| P1/recall/pooled | 0.174 ± 0.002 (n=5) | 0.165 ± 0.003 (n=5) | 0.175 ± 0.001 (n=5) |
| P1/recall/mean | 0.252 ± 0.003 (n=5) | 0.250 ± 0.002 (n=5) | 0.260 ± 0.006 (n=5) |
| P1/precision/pooled | 0.615 ± 0.007 (n=5) | 0.607 ± 0.011 (n=5) | 0.589 ± 0.007 (n=5) |
| P1/precision/mean | 0.376 ± 0.021 (n=5) | 0.398 ± 0.020 (n=5) | 0.360 ± 0.021 (n=5) |
| P2@0.8/recall_Hnear/pooled | 0.069 ± 0.085 (n=5) | 0.038 ± 0.009 (n=5) | 0.112 ± 0.041 (n=5) |
| P2@0.8/recall_Hnear/mean | 0.033 ± 0.033 (n=5) | 0.045 ± 0.038 (n=5) | 0.451 ± 0.030 (n=5) |
| P2@0.8/precision_Hnear/pooled | 0.498 ± 0.228 (n=5) | 0.568 ± 0.120 (n=5) | 0.468 ± 0.094 (n=5) |
| P2@0.8/precision_Hnear/mean | 0.314 ± 0.161 (n=5) | 0.482 ± 0.162 (n=5) | 0.399 ± 0.044 (n=5) |
| P2@0.8/iou_Hnear/pooled | 0.066 ± 0.084 (n=5) | 0.037 ± 0.009 (n=5) | 0.100 ± 0.037 (n=5) |
| P2@0.8/iou_Hnear/mean | 0.028 ± 0.028 (n=5) | 0.036 ± 0.027 (n=5) | 0.152 ± 0.017 (n=5) |
| P1/recall_Hnear/pooled | 0.012 ± 0.011 (n=5) | 0.020 ± 0.007 (n=5) | 0.027 ± 0.014 (n=5) |
| P1/recall_Hnear/mean | 0.106 ± 0.087 (n=5) | 0.145 ± 0.031 (n=5) | 0.163 ± 0.083 (n=5) |
| hidden_near/gt_cells | 163.000 ± 0.000 (n=5) | 163.000 ± 0.000 (n=5) | 163.000 ± 0.000 (n=5) |
| hidden_near/objects_with_hidden_gt | 6.000 ± 0.000 (n=5) | 6.000 ± 0.000 (n=5) | 6.000 ± 0.000 (n=5) |
| P2@0.8/recall_H/pooled | 0.005 ± 0.005 (n=5) | 0.003 ± 0.001 (n=5) | 0.089 ± 0.001 (n=5) |
| P2@0.8/recall_H/mean | 0.015 ± 0.011 (n=5) | 0.018 ± 0.012 (n=5) | 0.256 ± 0.011 (n=5) |
| P2@0.8/precision_H/pooled | 0.423 ± 0.155 (n=5) | 0.484 ± 0.100 (n=5) | 0.531 ± 0.006 (n=5) |
| P2@0.8/precision_H/mean | 0.335 ± 0.116 (n=5) | 0.371 ± 0.134 (n=5) | 0.473 ± 0.020 (n=5) |
| P2@0.8/iou_H/pooled | 0.005 ± 0.004 (n=5) | 0.003 ± 0.001 (n=5) | 0.082 ± 0.001 (n=5) |
| P2@0.8/iou_H/mean | 0.013 ± 0.010 (n=5) | 0.015 ± 0.009 (n=5) | 0.109 ± 0.005 (n=5) |
| hidden/gt_cells | 3149.000 ± 0.000 (n=5) | 3149.000 ± 0.000 (n=5) | 3149.000 ± 0.000 (n=5) |
| hidden/objects_with_hidden_gt | 17.000 ± 0.000 (n=5) | 17.000 ± 0.000 (n=5) | 17.000 ± 0.000 (n=5) |
| P1/recall_H/pooled | 0.007 ± 0.003 (n=5) | 0.005 ± 0.004 (n=5) | 0.005 ± 0.003 (n=5) |
| P1/recall_H/mean | 0.044 ± 0.032 (n=5) | 0.055 ± 0.010 (n=5) | 0.061 ± 0.030 (n=5) |


## Arm pairs (gap = first - second; separated = the two arms' [min, max] do not overlap)

### may_scene_a_radar_init - may_scene_a

#### obstacles, P2 >= 0.5

| metric | gap | first std | second std | first range | second range | separated |
|---|---|---|---|---|---|---|
| obstacles/P2@0.5/recall/pooled | -0.058 | 0.003 | 0.003 | 0.007 | 0.008 | True |
| obstacles/P2@0.5/recall/mean | -0.081 | 0.002 | 0.003 | 0.005 | 0.007 | True |
| obstacles/P2@0.5/precision/pooled | -0.027 | 0.005 | 0.004 | 0.012 | 0.009 | True |
| obstacles/P2@0.5/precision/mean | -0.027 | 0.020 | 0.012 | 0.052 | 0.035 | False |
| obstacles/P2@0.5/iou/pooled | -0.054 | 0.003 | 0.003 | 0.007 | 0.007 | True |
| obstacles/P2@0.5/iou/mean | -0.068 | 0.002 | 0.002 | 0.006 | 0.005 | True |
| obstacles/P1/recall/pooled | -0.026 | 0.002 | 0.001 | 0.005 | 0.003 | True |
| obstacles/P1/recall/mean | -0.030 | 0.002 | 0.003 | 0.005 | 0.008 | True |
| obstacles/P1/precision/pooled | -0.020 | 0.004 | 0.004 | 0.009 | 0.008 | True |
| obstacles/P1/precision/mean | -0.003 | 0.005 | 0.007 | 0.010 | 0.016 | False |
| obstacles/P2@0.5/recall_Hnear/pooled | -0.060 | 0.012 | 0.105 | 0.031 | 0.249 | False |
| obstacles/P2@0.5/recall_Hnear/mean | -0.003 | 0.049 | 0.059 | 0.108 | 0.161 | False |
| obstacles/P2@0.5/precision_Hnear/pooled | 0.020 | 0.088 | 0.152 | 0.207 | 0.387 | False |
| obstacles/P2@0.5/precision_Hnear/mean | 0.105 | 0.084 | 0.041 | 0.229 | 0.095 | False |
| obstacles/P2@0.5/iou_Hnear/pooled | -0.055 | 0.012 | 0.101 | 0.031 | 0.238 | False |
| obstacles/P2@0.5/iou_Hnear/mean | 0.004 | 0.033 | 0.042 | 0.081 | 0.117 | False |
| obstacles/P1/recall_Hnear/pooled | -0.002 | 0.008 | 0.012 | 0.021 | 0.026 | False |
| obstacles/P1/recall_Hnear/mean | -0.015 | 0.035 | 0.042 | 0.077 | 0.111 | False |
| obstacles/hidden_near/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| obstacles/hidden_near/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| obstacles/P2@0.5/recall_H/pooled | -0.013 | 0.002 | 0.005 | 0.005 | 0.011 | True |
| obstacles/P2@0.5/recall_H/mean | -0.031 | 0.009 | 0.005 | 0.021 | 0.012 | True |
| obstacles/P2@0.5/precision_H/pooled | 0.019 | 0.093 | 0.062 | 0.257 | 0.166 | False |
| obstacles/P2@0.5/precision_H/mean | 0.036 | 0.077 | 0.035 | 0.205 | 0.093 | False |
| obstacles/P2@0.5/iou_H/pooled | -0.012 | 0.002 | 0.005 | 0.005 | 0.011 | True |
| obstacles/P2@0.5/iou_H/mean | -0.027 | 0.008 | 0.004 | 0.021 | 0.011 | True |
| obstacles/hidden/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| obstacles/hidden/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| obstacles/P1/recall_H/pooled | -0.004 | 0.002 | 0.001 | 0.004 | 0.003 | False |
| obstacles/P1/recall_H/mean | -0.002 | 0.013 | 0.005 | 0.032 | 0.013 | False |

#### obstacles, P2 >= 0.8

| metric | gap | first std | second std | first range | second range | separated |
|---|---|---|---|---|---|---|
| obstacles/P2@0.8/recall/pooled | -0.053 | 0.002 | 0.002 | 0.006 | 0.006 | True |
| obstacles/P2@0.8/recall/mean | -0.082 | 0.003 | 0.002 | 0.007 | 0.005 | True |
| obstacles/P2@0.8/precision/pooled | -0.032 | 0.002 | 0.003 | 0.007 | 0.006 | True |
| obstacles/P2@0.8/precision/mean | -0.031 | 0.012 | 0.012 | 0.027 | 0.027 | True |
| obstacles/P2@0.8/iou/pooled | -0.051 | 0.002 | 0.002 | 0.006 | 0.005 | True |
| obstacles/P2@0.8/iou/mean | -0.069 | 0.003 | 0.002 | 0.007 | 0.004 | True |
| obstacles/P1/recall/pooled | -0.026 | 0.002 | 0.001 | 0.005 | 0.003 | True |
| obstacles/P1/recall/mean | -0.030 | 0.002 | 0.003 | 0.005 | 0.008 | True |
| obstacles/P1/precision/pooled | -0.020 | 0.004 | 0.004 | 0.009 | 0.008 | True |
| obstacles/P1/precision/mean | -0.003 | 0.005 | 0.007 | 0.010 | 0.016 | False |
| obstacles/P2@0.8/recall_Hnear/pooled | -0.033 | 0.013 | 0.070 | 0.031 | 0.181 | False |
| obstacles/P2@0.8/recall_Hnear/mean | -0.024 | 0.021 | 0.062 | 0.054 | 0.167 | False |
| obstacles/P2@0.8/precision_Hnear/pooled | -0.022 | 0.105 | 0.164 | 0.234 | 0.413 | False |
| obstacles/P2@0.8/precision_Hnear/mean | 0.020 | 0.101 | 0.127 | 0.273 | 0.320 | False |
| obstacles/P2@0.8/iou_Hnear/pooled | -0.032 | 0.012 | 0.070 | 0.031 | 0.178 | False |
| obstacles/P2@0.8/iou_Hnear/mean | -0.026 | 0.030 | 0.050 | 0.081 | 0.135 | False |
| obstacles/P1/recall_Hnear/pooled | -0.002 | 0.008 | 0.012 | 0.021 | 0.026 | False |
| obstacles/P1/recall_Hnear/mean | -0.015 | 0.035 | 0.042 | 0.077 | 0.111 | False |
| obstacles/hidden_near/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| obstacles/hidden_near/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| obstacles/P2@0.8/recall_H/pooled | -0.006 | 0.002 | 0.004 | 0.004 | 0.008 | True |
| obstacles/P2@0.8/recall_H/mean | -0.020 | 0.007 | 0.004 | 0.019 | 0.010 | True |
| obstacles/P2@0.8/precision_H/pooled | 0.007 | 0.087 | 0.083 | 0.243 | 0.210 | False |
| obstacles/P2@0.8/precision_H/mean | -0.016 | 0.086 | 0.055 | 0.226 | 0.138 | False |
| obstacles/P2@0.8/iou_H/pooled | -0.006 | 0.002 | 0.004 | 0.004 | 0.008 | True |
| obstacles/P2@0.8/iou_H/mean | -0.019 | 0.008 | 0.004 | 0.021 | 0.010 | True |
| obstacles/hidden/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| obstacles/hidden/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| obstacles/P1/recall_H/pooled | -0.004 | 0.002 | 0.001 | 0.004 | 0.003 | False |
| obstacles/P1/recall_H/mean | -0.002 | 0.013 | 0.005 | 0.032 | 0.013 | False |

#### tree_trunk, P2 >= 0.5

| metric | gap | first std | second std | first range | second range | separated |
|---|---|---|---|---|---|---|
| tree_trunk/P2@0.5/recall/pooled | -0.073 | 0.001 | 0.002 | 0.003 | 0.004 | True |
| tree_trunk/P2@0.5/recall/mean | -0.090 | 0.002 | 0.002 | 0.005 | 0.004 | True |
| tree_trunk/P2@0.5/precision/pooled | -0.021 | 0.002 | 0.005 | 0.004 | 0.011 | True |
| tree_trunk/P2@0.5/precision/mean | -0.017 | 0.020 | 0.014 | 0.047 | 0.038 | False |
| tree_trunk/P2@0.5/iou/pooled | -0.067 | 0.001 | 0.001 | 0.003 | 0.003 | True |
| tree_trunk/P2@0.5/iou/mean | -0.078 | 0.003 | 0.001 | 0.006 | 0.004 | True |
| tree_trunk/P1/recall/pooled | -0.037 | 0.002 | 0.002 | 0.005 | 0.005 | True |
| tree_trunk/P1/recall/mean | -0.039 | 0.002 | 0.002 | 0.007 | 0.006 | True |
| tree_trunk/P1/precision/pooled | -0.024 | 0.006 | 0.004 | 0.015 | 0.009 | True |
| tree_trunk/P1/precision/mean | -0.006 | 0.007 | 0.008 | 0.017 | 0.021 | False |
| tree_trunk/P2@0.5/recall_Hnear/pooled | -0.067 | 0.030 | 0.065 | 0.067 | 0.167 | False |
| tree_trunk/P2@0.5/recall_Hnear/mean | -0.004 | 0.101 | 0.099 | 0.252 | 0.245 | False |
| tree_trunk/P2@0.5/precision_Hnear/pooled | -0.113 | 0.102 | 0.128 | 0.280 | 0.333 | False |
| tree_trunk/P2@0.5/precision_Hnear/mean | 0.013 | 0.047 | 0.109 | 0.114 | 0.300 | False |
| tree_trunk/P2@0.5/iou_Hnear/pooled | -0.071 | 0.034 | 0.070 | 0.087 | 0.175 | False |
| tree_trunk/P2@0.5/iou_Hnear/mean | 0.002 | 0.071 | 0.084 | 0.185 | 0.191 | False |
| tree_trunk/P1/recall_Hnear/pooled | -0.053 | 0.049 | 0.038 | 0.133 | 0.100 | False |
| tree_trunk/P1/recall_Hnear/mean | -0.062 | 0.090 | 0.074 | 0.190 | 0.171 | False |
| tree_trunk/hidden_near/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| tree_trunk/hidden_near/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| tree_trunk/P2@0.5/recall_H/pooled | -0.019 | 0.004 | 0.003 | 0.011 | 0.007 | True |
| tree_trunk/P2@0.5/recall_H/mean | -0.040 | 0.015 | 0.012 | 0.037 | 0.031 | True |
| tree_trunk/P2@0.5/precision_H/pooled | 0.034 | 0.085 | 0.033 | 0.221 | 0.086 | False |
| tree_trunk/P2@0.5/precision_H/mean | 0.044 | 0.090 | 0.023 | 0.230 | 0.062 | False |
| tree_trunk/P2@0.5/iou_H/pooled | -0.018 | 0.004 | 0.003 | 0.011 | 0.006 | True |
| tree_trunk/P2@0.5/iou_H/mean | -0.035 | 0.013 | 0.010 | 0.036 | 0.025 | True |
| tree_trunk/hidden/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| tree_trunk/hidden/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| tree_trunk/P1/recall_H/pooled | -0.007 | 0.002 | 0.002 | 0.005 | 0.004 | True |
| tree_trunk/P1/recall_H/mean | -0.007 | 0.019 | 0.010 | 0.047 | 0.024 | False |

#### tree_trunk, P2 >= 0.8

| metric | gap | first std | second std | first range | second range | separated |
|---|---|---|---|---|---|---|
| tree_trunk/P2@0.8/recall/pooled | -0.068 | 0.003 | 0.001 | 0.007 | 0.004 | True |
| tree_trunk/P2@0.8/recall/mean | -0.091 | 0.003 | 0.001 | 0.008 | 0.003 | True |
| tree_trunk/P2@0.8/precision/pooled | -0.024 | 0.003 | 0.005 | 0.007 | 0.010 | True |
| tree_trunk/P2@0.8/precision/mean | -0.013 | 0.018 | 0.011 | 0.048 | 0.026 | False |
| tree_trunk/P2@0.8/iou/pooled | -0.064 | 0.003 | 0.001 | 0.007 | 0.002 | True |
| tree_trunk/P2@0.8/iou/mean | -0.079 | 0.003 | 0.001 | 0.007 | 0.003 | True |
| tree_trunk/P1/recall/pooled | -0.037 | 0.002 | 0.002 | 0.005 | 0.005 | True |
| tree_trunk/P1/recall/mean | -0.039 | 0.002 | 0.002 | 0.007 | 0.006 | True |
| tree_trunk/P1/precision/pooled | -0.024 | 0.006 | 0.004 | 0.015 | 0.009 | True |
| tree_trunk/P1/precision/mean | -0.006 | 0.007 | 0.008 | 0.017 | 0.021 | False |
| tree_trunk/P2@0.8/recall_Hnear/pooled | -0.047 | 0.055 | 0.080 | 0.133 | 0.200 | False |
| tree_trunk/P2@0.8/recall_Hnear/mean | -0.055 | 0.063 | 0.104 | 0.148 | 0.255 | False |
| tree_trunk/P2@0.8/precision_Hnear/pooled | -0.171 | 0.116 | 0.160 | 0.278 | 0.429 | False |
| tree_trunk/P2@0.8/precision_Hnear/mean | -0.102 | 0.040 | 0.225 | 0.083 | 0.600 | False |
| tree_trunk/P2@0.8/iou_Hnear/pooled | -0.053 | 0.052 | 0.076 | 0.117 | 0.191 | False |
| tree_trunk/P2@0.8/iou_Hnear/mean | -0.059 | 0.055 | 0.081 | 0.129 | 0.198 | False |
| tree_trunk/P1/recall_Hnear/pooled | -0.053 | 0.049 | 0.038 | 0.133 | 0.100 | False |
| tree_trunk/P1/recall_Hnear/mean | -0.062 | 0.090 | 0.074 | 0.190 | 0.171 | False |
| tree_trunk/hidden_near/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| tree_trunk/hidden_near/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| tree_trunk/P2@0.8/recall_H/pooled | -0.010 | 0.004 | 0.003 | 0.008 | 0.007 | True |
| tree_trunk/P2@0.8/recall_H/mean | -0.028 | 0.013 | 0.006 | 0.035 | 0.016 | False |
| tree_trunk/P2@0.8/precision_H/pooled | 0.014 | 0.133 | 0.074 | 0.344 | 0.203 | False |
| tree_trunk/P2@0.8/precision_H/mean | 0.009 | 0.114 | 0.071 | 0.269 | 0.192 | False |
| tree_trunk/P2@0.8/iou_H/pooled | -0.010 | 0.004 | 0.003 | 0.008 | 0.007 | True |
| tree_trunk/P2@0.8/iou_H/mean | -0.027 | 0.013 | 0.005 | 0.034 | 0.014 | False |
| tree_trunk/hidden/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| tree_trunk/hidden/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| tree_trunk/P1/recall_H/pooled | -0.007 | 0.002 | 0.002 | 0.005 | 0.004 | True |
| tree_trunk/P1/recall_H/mean | -0.007 | 0.019 | 0.010 | 0.047 | 0.024 | False |

#### rock, P2 >= 0.5

| metric | gap | first std | second std | first range | second range | separated |
|---|---|---|---|---|---|---|
| rock/P2@0.5/recall/pooled | -0.047 | 0.021 | 0.018 | 0.045 | 0.051 | True |
| rock/P2@0.5/recall/mean | -0.042 | 0.019 | 0.016 | 0.038 | 0.037 | True |
| rock/P2@0.5/precision/pooled | -0.050 | 0.034 | 0.011 | 0.088 | 0.031 | False |
| rock/P2@0.5/precision/mean | -0.037 | 0.056 | 0.039 | 0.138 | 0.093 | False |
| rock/P2@0.5/iou/pooled | -0.036 | 0.017 | 0.012 | 0.037 | 0.034 | True |
| rock/P2@0.5/iou/mean | -0.028 | 0.011 | 0.013 | 0.023 | 0.027 | True |
| rock/P1/recall/pooled | -0.001 | 0.005 | 0.023 | 0.011 | 0.051 | False |
| rock/P1/recall/mean | 0.003 | 0.010 | 0.023 | 0.024 | 0.054 | False |
| rock/P1/precision/pooled | -0.005 | 0.014 | 0.013 | 0.030 | 0.035 | False |
| rock/P1/precision/mean | 0.020 | 0.031 | 0.031 | 0.074 | 0.079 | False |
| rock/P2@0.5/recall_Hnear/pooled | n/a | n/a | n/a | n/a | n/a | None |
| rock/P2@0.5/recall_Hnear/mean | n/a | n/a | n/a | n/a | n/a | None |
| rock/P2@0.5/precision_Hnear/pooled | n/a | n/a | n/a | n/a | n/a | None |
| rock/P2@0.5/precision_Hnear/mean | n/a | n/a | n/a | n/a | n/a | None |
| rock/P2@0.5/iou_Hnear/pooled | n/a | n/a | n/a | n/a | n/a | None |
| rock/P2@0.5/iou_Hnear/mean | n/a | n/a | n/a | n/a | n/a | None |
| rock/P1/recall_Hnear/pooled | n/a | n/a | n/a | n/a | n/a | None |
| rock/P1/recall_Hnear/mean | n/a | n/a | n/a | n/a | n/a | None |
| rock/hidden_near/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| rock/hidden_near/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| rock/P2@0.5/recall_H/pooled | -0.012 | 0.000 | 0.022 | 0.000 | 0.051 | False |
| rock/P2@0.5/recall_H/mean | -0.011 | 0.000 | 0.015 | 0.000 | 0.028 | False |
| rock/P2@0.5/precision_H/pooled | n/a | n/a | 0.202 | n/a | 0.286 | None |
| rock/P2@0.5/precision_H/mean | n/a | n/a | 0.412 | n/a | 0.583 | None |
| rock/P2@0.5/iou_H/pooled | -0.012 | 0.000 | 0.021 | 0.000 | 0.050 | False |
| rock/P2@0.5/iou_H/mean | -0.010 | 0.000 | 0.014 | 0.000 | 0.028 | False |
| rock/hidden/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| rock/hidden/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| rock/P1/recall_H/pooled | 0.004 | 0.006 | 0.000 | 0.010 | 0.000 | False |
| rock/P1/recall_H/mean | 0.008 | 0.011 | 0.000 | 0.021 | 0.000 | False |

#### rock, P2 >= 0.8

| metric | gap | first std | second std | first range | second range | separated |
|---|---|---|---|---|---|---|
| rock/P2@0.8/recall/pooled | -0.063 | 0.011 | 0.015 | 0.022 | 0.034 | True |
| rock/P2@0.8/recall/mean | -0.057 | 0.011 | 0.017 | 0.027 | 0.046 | True |
| rock/P2@0.8/precision/pooled | -0.070 | 0.025 | 0.022 | 0.065 | 0.056 | True |
| rock/P2@0.8/precision/mean | -0.049 | 0.061 | 0.039 | 0.141 | 0.108 | False |
| rock/P2@0.8/iou/pooled | -0.049 | 0.010 | 0.012 | 0.022 | 0.029 | True |
| rock/P2@0.8/iou/mean | -0.035 | 0.009 | 0.014 | 0.025 | 0.038 | True |
| rock/P1/recall/pooled | -0.001 | 0.005 | 0.023 | 0.011 | 0.051 | False |
| rock/P1/recall/mean | 0.003 | 0.010 | 0.023 | 0.024 | 0.054 | False |
| rock/P1/precision/pooled | -0.005 | 0.014 | 0.013 | 0.030 | 0.035 | False |
| rock/P1/precision/mean | 0.020 | 0.031 | 0.031 | 0.074 | 0.079 | False |
| rock/P2@0.8/recall_Hnear/pooled | n/a | n/a | n/a | n/a | n/a | None |
| rock/P2@0.8/recall_Hnear/mean | n/a | n/a | n/a | n/a | n/a | None |
| rock/P2@0.8/precision_Hnear/pooled | n/a | n/a | n/a | n/a | n/a | None |
| rock/P2@0.8/precision_Hnear/mean | n/a | n/a | n/a | n/a | n/a | None |
| rock/P2@0.8/iou_Hnear/pooled | n/a | n/a | n/a | n/a | n/a | None |
| rock/P2@0.8/iou_Hnear/mean | n/a | n/a | n/a | n/a | n/a | None |
| rock/P1/recall_Hnear/pooled | n/a | n/a | n/a | n/a | n/a | None |
| rock/P1/recall_Hnear/mean | n/a | n/a | n/a | n/a | n/a | None |
| rock/hidden_near/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| rock/hidden_near/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| rock/P2@0.8/recall_H/pooled | -0.004 | 0.000 | 0.006 | 0.000 | 0.010 | False |
| rock/P2@0.8/recall_H/mean | -0.007 | 0.000 | 0.012 | 0.000 | 0.028 | False |
| rock/P2@0.8/precision_H/pooled | n/a | n/a | 0.354 | n/a | 0.500 | None |
| rock/P2@0.8/precision_H/mean | n/a | n/a | 0.354 | n/a | 0.500 | None |
| rock/P2@0.8/iou_H/pooled | -0.004 | 0.000 | 0.006 | 0.000 | 0.010 | False |
| rock/P2@0.8/iou_H/mean | -0.007 | 0.000 | 0.012 | 0.000 | 0.028 | False |
| rock/hidden/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| rock/hidden/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| rock/P1/recall_H/pooled | 0.004 | 0.006 | 0.000 | 0.010 | 0.000 | False |
| rock/P1/recall_H/mean | 0.008 | 0.011 | 0.000 | 0.021 | 0.000 | False |

#### bush_or_small_tree, P2 >= 0.5

| metric | gap | first std | second std | first range | second range | separated |
|---|---|---|---|---|---|---|
| bush_or_small_tree/P2@0.5/recall/pooled | -0.034 | 0.006 | 0.007 | 0.015 | 0.016 | True |
| bush_or_small_tree/P2@0.5/recall/mean | -0.054 | 0.008 | 0.012 | 0.020 | 0.034 | True |
| bush_or_small_tree/P2@0.5/precision/pooled | -0.056 | 0.020 | 0.012 | 0.052 | 0.028 | True |
| bush_or_small_tree/P2@0.5/precision/mean | -0.067 | 0.043 | 0.033 | 0.118 | 0.087 | False |
| bush_or_small_tree/P2@0.5/iou/pooled | -0.033 | 0.006 | 0.007 | 0.015 | 0.016 | True |
| bush_or_small_tree/P2@0.5/iou/mean | -0.043 | 0.007 | 0.008 | 0.017 | 0.023 | True |
| bush_or_small_tree/P1/recall/pooled | -0.009 | 0.003 | 0.002 | 0.007 | 0.004 | True |
| bush_or_small_tree/P1/recall/mean | -0.002 | 0.002 | 0.003 | 0.005 | 0.009 | False |
| bush_or_small_tree/P1/precision/pooled | -0.008 | 0.011 | 0.007 | 0.028 | 0.017 | False |
| bush_or_small_tree/P1/precision/mean | 0.022 | 0.020 | 0.021 | 0.053 | 0.051 | False |
| bush_or_small_tree/P2@0.5/recall_Hnear/pooled | -0.059 | 0.012 | 0.126 | 0.031 | 0.288 | False |
| bush_or_small_tree/P2@0.5/recall_Hnear/mean | -0.002 | 0.038 | 0.047 | 0.081 | 0.098 | False |
| bush_or_small_tree/P2@0.5/precision_Hnear/pooled | 0.082 | 0.100 | 0.234 | 0.259 | 0.575 | False |
| bush_or_small_tree/P2@0.5/precision_Hnear/mean | 0.205 | 0.139 | 0.132 | 0.333 | 0.304 | False |
| bush_or_small_tree/P2@0.5/iou_Hnear/pooled | -0.055 | 0.012 | 0.121 | 0.030 | 0.278 | False |
| bush_or_small_tree/P2@0.5/iou_Hnear/mean | 0.001 | 0.030 | 0.036 | 0.067 | 0.079 | False |
| bush_or_small_tree/P1/recall_Hnear/pooled | 0.007 | 0.007 | 0.011 | 0.018 | 0.025 | False |
| bush_or_small_tree/P1/recall_Hnear/mean | 0.039 | 0.031 | 0.087 | 0.058 | 0.196 | False |
| bush_or_small_tree/hidden_near/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| bush_or_small_tree/hidden_near/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| bush_or_small_tree/P2@0.5/recall_H/pooled | -0.007 | 0.001 | 0.008 | 0.002 | 0.016 | False |
| bush_or_small_tree/P2@0.5/recall_H/mean | -0.011 | 0.012 | 0.027 | 0.026 | 0.064 | False |
| bush_or_small_tree/P2@0.5/precision_H/pooled | 0.018 | 0.118 | 0.162 | 0.291 | 0.435 | False |
| bush_or_small_tree/P2@0.5/precision_H/mean | 0.096 | 0.096 | 0.123 | 0.242 | 0.343 | False |
| bush_or_small_tree/P2@0.5/iou_H/pooled | -0.007 | 0.001 | 0.007 | 0.002 | 0.016 | False |
| bush_or_small_tree/P2@0.5/iou_H/mean | -0.007 | 0.010 | 0.020 | 0.023 | 0.047 | False |
| bush_or_small_tree/hidden/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| bush_or_small_tree/hidden/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| bush_or_small_tree/P1/recall_H/pooled | -0.002 | 0.004 | 0.003 | 0.009 | 0.008 | False |
| bush_or_small_tree/P1/recall_H/mean | 0.012 | 0.010 | 0.032 | 0.026 | 0.069 | False |

#### bush_or_small_tree, P2 >= 0.8

| metric | gap | first std | second std | first range | second range | separated |
|---|---|---|---|---|---|---|
| bush_or_small_tree/P2@0.8/recall/pooled | -0.030 | 0.003 | 0.005 | 0.008 | 0.011 | True |
| bush_or_small_tree/P2@0.8/recall/mean | -0.051 | 0.007 | 0.003 | 0.018 | 0.007 | True |
| bush_or_small_tree/P2@0.8/precision/pooled | -0.074 | 0.020 | 0.011 | 0.045 | 0.025 | True |
| bush_or_small_tree/P2@0.8/precision/mean | -0.111 | 0.019 | 0.027 | 0.054 | 0.067 | True |
| bush_or_small_tree/P2@0.8/iou/pooled | -0.030 | 0.003 | 0.005 | 0.008 | 0.011 | True |
| bush_or_small_tree/P2@0.8/iou/mean | -0.041 | 0.007 | 0.002 | 0.017 | 0.005 | True |
| bush_or_small_tree/P1/recall/pooled | -0.009 | 0.003 | 0.002 | 0.007 | 0.004 | True |
| bush_or_small_tree/P1/recall/mean | -0.002 | 0.002 | 0.003 | 0.005 | 0.009 | False |
| bush_or_small_tree/P1/precision/pooled | -0.008 | 0.011 | 0.007 | 0.028 | 0.017 | False |
| bush_or_small_tree/P1/precision/mean | 0.022 | 0.020 | 0.021 | 0.053 | 0.051 | False |
| bush_or_small_tree/P2@0.8/recall_Hnear/pooled | -0.031 | 0.009 | 0.085 | 0.025 | 0.202 | False |
| bush_or_small_tree/P2@0.8/recall_Hnear/mean | 0.012 | 0.038 | 0.033 | 0.085 | 0.069 | False |
| bush_or_small_tree/P2@0.8/precision_Hnear/pooled | 0.070 | 0.120 | 0.228 | 0.303 | 0.567 | False |
| bush_or_small_tree/P2@0.8/precision_Hnear/mean | 0.169 | 0.162 | 0.161 | 0.400 | 0.340 | False |
| bush_or_small_tree/P2@0.8/iou_Hnear/pooled | -0.030 | 0.009 | 0.084 | 0.024 | 0.198 | False |
| bush_or_small_tree/P2@0.8/iou_Hnear/mean | 0.009 | 0.027 | 0.028 | 0.060 | 0.058 | False |
| bush_or_small_tree/P1/recall_Hnear/pooled | 0.007 | 0.007 | 0.011 | 0.018 | 0.025 | False |
| bush_or_small_tree/P1/recall_Hnear/mean | 0.039 | 0.031 | 0.087 | 0.058 | 0.196 | False |
| bush_or_small_tree/hidden_near/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| bush_or_small_tree/hidden_near/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| bush_or_small_tree/P2@0.8/recall_H/pooled | -0.002 | 0.001 | 0.005 | 0.002 | 0.010 | False |
| bush_or_small_tree/P2@0.8/recall_H/mean | 0.002 | 0.012 | 0.011 | 0.029 | 0.025 | False |
| bush_or_small_tree/P2@0.8/precision_H/pooled | 0.061 | 0.100 | 0.155 | 0.225 | 0.392 | False |
| bush_or_small_tree/P2@0.8/precision_H/mean | 0.036 | 0.134 | 0.116 | 0.313 | 0.257 | False |
| bush_or_small_tree/P2@0.8/iou_H/pooled | -0.002 | 0.001 | 0.004 | 0.002 | 0.010 | False |
| bush_or_small_tree/P2@0.8/iou_H/mean | 0.002 | 0.009 | 0.010 | 0.022 | 0.023 | False |
| bush_or_small_tree/hidden/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| bush_or_small_tree/hidden/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| bush_or_small_tree/P1/recall_H/pooled | -0.002 | 0.004 | 0.003 | 0.009 | 0.008 | False |
| bush_or_small_tree/P1/recall_H/mean | 0.012 | 0.010 | 0.032 | 0.026 | 0.069 | False |

GT cells occupied in every repeat of one arm and none of the other: {"bush_or_small_tree": {"first_only": {"blocked": 2, "no_return": 0, "passed_through": 0, "seen": 0, "unobserved": 0}, "second_only": {"blocked": 6, "no_return": 1, "passed_through": 0, "seen": 18, "unobserved": 0}}, "obstacles": {"first_only": {"blocked": 7, "no_return": 0, "passed_through": 0, "seen": 1, "unobserved": 0}, "second_only": {"blocked": 48, "no_return": 4, "passed_through": 0, "seen": 62, "unobserved": 0}}, "rock": {"first_only": {"blocked": 0, "no_return": 0, "passed_through": 0, "seen": 0, "unobserved": 0}, "second_only": {"blocked": 0, "no_return": 0, "passed_through": 0, "seen": 0, "unobserved": 0}}, "tree_trunk": {"first_only": {"blocked": 5, "no_return": 0, "passed_through": 0, "seen": 1, "unobserved": 0}, "second_only": {"blocked": 42, "no_return": 3, "passed_through": 0, "seen": 44, "unobserved": 0}}}

### may_scene_a_radar_init_loss - may_scene_a

#### obstacles, P2 >= 0.5

| metric | gap | first std | second std | first range | second range | separated |
|---|---|---|---|---|---|---|
| obstacles/P2@0.5/recall/pooled | 0.092 | 0.003 | 0.003 | 0.008 | 0.008 | True |
| obstacles/P2@0.5/recall/mean | 0.088 | 0.006 | 0.003 | 0.013 | 0.007 | True |
| obstacles/P2@0.5/precision/pooled | -0.079 | 0.006 | 0.004 | 0.014 | 0.009 | True |
| obstacles/P2@0.5/precision/mean | -0.014 | 0.011 | 0.012 | 0.026 | 0.035 | False |
| obstacles/P2@0.5/iou/pooled | 0.067 | 0.002 | 0.003 | 0.005 | 0.007 | True |
| obstacles/P2@0.5/iou/mean | 0.033 | 0.003 | 0.002 | 0.008 | 0.005 | True |
| obstacles/P1/recall/pooled | -0.001 | 0.002 | 0.001 | 0.005 | 0.003 | False |
| obstacles/P1/recall/mean | -0.003 | 0.002 | 0.003 | 0.005 | 0.008 | False |
| obstacles/P1/precision/pooled | -0.007 | 0.003 | 0.004 | 0.007 | 0.008 | False |
| obstacles/P1/precision/mean | 0.014 | 0.010 | 0.007 | 0.024 | 0.016 | False |
| obstacles/P2@0.5/recall_Hnear/pooled | 0.075 | 0.077 | 0.105 | 0.197 | 0.249 | False |
| obstacles/P2@0.5/recall_Hnear/mean | 0.195 | 0.038 | 0.059 | 0.091 | 0.161 | True |
| obstacles/P2@0.5/precision_Hnear/pooled | -0.074 | 0.095 | 0.152 | 0.242 | 0.387 | False |
| obstacles/P2@0.5/precision_Hnear/mean | -0.009 | 0.038 | 0.041 | 0.103 | 0.095 | False |
| obstacles/P2@0.5/iou_Hnear/pooled | 0.051 | 0.066 | 0.101 | 0.171 | 0.238 | False |
| obstacles/P2@0.5/iou_Hnear/mean | 0.048 | 0.030 | 0.042 | 0.069 | 0.117 | False |
| obstacles/P1/recall_Hnear/pooled | 0.012 | 0.015 | 0.012 | 0.036 | 0.026 | False |
| obstacles/P1/recall_Hnear/mean | 0.039 | 0.065 | 0.042 | 0.171 | 0.111 | False |
| obstacles/hidden_near/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| obstacles/hidden_near/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| obstacles/P2@0.5/recall_H/pooled | 0.078 | 0.002 | 0.005 | 0.004 | 0.011 | True |
| obstacles/P2@0.5/recall_H/mean | 0.099 | 0.009 | 0.005 | 0.022 | 0.012 | True |
| obstacles/P2@0.5/precision_H/pooled | 0.029 | 0.017 | 0.062 | 0.045 | 0.166 | False |
| obstacles/P2@0.5/precision_H/mean | 0.067 | 0.031 | 0.035 | 0.069 | 0.093 | True |
| obstacles/P2@0.5/iou_H/pooled | 0.073 | 0.001 | 0.005 | 0.003 | 0.011 | True |
| obstacles/P2@0.5/iou_H/mean | 0.055 | 0.008 | 0.004 | 0.021 | 0.011 | True |
| obstacles/hidden/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| obstacles/hidden/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| obstacles/P1/recall_H/pooled | -0.002 | 0.002 | 0.001 | 0.005 | 0.003 | False |
| obstacles/P1/recall_H/mean | 0.003 | 0.010 | 0.005 | 0.024 | 0.013 | False |

#### obstacles, P2 >= 0.8

| metric | gap | first std | second std | first range | second range | separated |
|---|---|---|---|---|---|---|
| obstacles/P2@0.8/recall/pooled | 0.082 | 0.001 | 0.002 | 0.003 | 0.006 | True |
| obstacles/P2@0.8/recall/mean | 0.091 | 0.003 | 0.002 | 0.007 | 0.005 | True |
| obstacles/P2@0.8/precision/pooled | -0.091 | 0.003 | 0.003 | 0.006 | 0.006 | True |
| obstacles/P2@0.8/precision/mean | -0.002 | 0.008 | 0.012 | 0.019 | 0.027 | False |
| obstacles/P2@0.8/iou/pooled | 0.063 | 0.001 | 0.002 | 0.002 | 0.005 | True |
| obstacles/P2@0.8/iou/mean | 0.036 | 0.003 | 0.002 | 0.006 | 0.004 | True |
| obstacles/P1/recall/pooled | -0.001 | 0.002 | 0.001 | 0.005 | 0.003 | False |
| obstacles/P1/recall/mean | -0.003 | 0.002 | 0.003 | 0.005 | 0.008 | False |
| obstacles/P1/precision/pooled | -0.007 | 0.003 | 0.004 | 0.007 | 0.008 | False |
| obstacles/P1/precision/mean | 0.014 | 0.010 | 0.007 | 0.024 | 0.016 | False |
| obstacles/P2@0.8/recall_Hnear/pooled | 0.033 | 0.041 | 0.070 | 0.098 | 0.181 | False |
| obstacles/P2@0.8/recall_Hnear/mean | 0.169 | 0.021 | 0.062 | 0.051 | 0.167 | True |
| obstacles/P2@0.8/precision_Hnear/pooled | -0.112 | 0.082 | 0.164 | 0.219 | 0.413 | False |
| obstacles/P2@0.8/precision_Hnear/mean | -0.057 | 0.043 | 0.127 | 0.120 | 0.320 | False |
| obstacles/P2@0.8/iou_Hnear/pooled | 0.023 | 0.038 | 0.070 | 0.088 | 0.178 | False |
| obstacles/P2@0.8/iou_Hnear/mean | 0.029 | 0.018 | 0.050 | 0.047 | 0.135 | False |
| obstacles/P1/recall_Hnear/pooled | 0.012 | 0.015 | 0.012 | 0.036 | 0.026 | False |
| obstacles/P1/recall_Hnear/mean | 0.039 | 0.065 | 0.042 | 0.171 | 0.111 | False |
| obstacles/hidden_near/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| obstacles/hidden_near/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| obstacles/P2@0.8/recall_H/pooled | 0.057 | 0.001 | 0.004 | 0.002 | 0.008 | True |
| obstacles/P2@0.8/recall_H/mean | 0.067 | 0.009 | 0.004 | 0.022 | 0.010 | True |
| obstacles/P2@0.8/precision_H/pooled | -0.019 | 0.007 | 0.083 | 0.017 | 0.210 | False |
| obstacles/P2@0.8/precision_H/mean | 0.025 | 0.025 | 0.055 | 0.062 | 0.138 | False |
| obstacles/P2@0.8/iou_H/pooled | 0.054 | 0.001 | 0.004 | 0.002 | 0.008 | True |
| obstacles/P2@0.8/iou_H/mean | 0.030 | 0.008 | 0.004 | 0.019 | 0.010 | True |
| obstacles/hidden/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| obstacles/hidden/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| obstacles/P1/recall_H/pooled | -0.002 | 0.002 | 0.001 | 0.005 | 0.003 | False |
| obstacles/P1/recall_H/mean | 0.003 | 0.010 | 0.005 | 0.024 | 0.013 | False |

#### tree_trunk, P2 >= 0.5

| metric | gap | first std | second std | first range | second range | separated |
|---|---|---|---|---|---|---|
| tree_trunk/P2@0.5/recall/pooled | 0.058 | 0.004 | 0.002 | 0.011 | 0.004 | True |
| tree_trunk/P2@0.5/recall/mean | 0.057 | 0.006 | 0.002 | 0.012 | 0.004 | True |
| tree_trunk/P2@0.5/precision/pooled | -0.038 | 0.009 | 0.005 | 0.022 | 0.011 | True |
| tree_trunk/P2@0.5/precision/mean | 0.008 | 0.011 | 0.014 | 0.027 | 0.038 | False |
| tree_trunk/P2@0.5/iou/pooled | 0.042 | 0.004 | 0.001 | 0.009 | 0.003 | True |
| tree_trunk/P2@0.5/iou/mean | 0.026 | 0.003 | 0.001 | 0.008 | 0.004 | True |
| tree_trunk/P1/recall/pooled | -0.003 | 0.003 | 0.002 | 0.008 | 0.005 | False |
| tree_trunk/P1/recall/mean | -0.007 | 0.003 | 0.002 | 0.009 | 0.006 | False |
| tree_trunk/P1/precision/pooled | 0.001 | 0.004 | 0.004 | 0.008 | 0.009 | False |
| tree_trunk/P1/precision/mean | 0.019 | 0.013 | 0.008 | 0.033 | 0.021 | False |
| tree_trunk/P2@0.5/recall_Hnear/pooled | 0.013 | 0.100 | 0.065 | 0.233 | 0.167 | False |
| tree_trunk/P2@0.5/recall_Hnear/mean | -0.028 | 0.059 | 0.099 | 0.145 | 0.245 | False |
| tree_trunk/P2@0.5/precision_Hnear/pooled | -0.025 | 0.074 | 0.128 | 0.152 | 0.333 | False |
| tree_trunk/P2@0.5/precision_Hnear/mean | -0.032 | 0.048 | 0.109 | 0.117 | 0.300 | False |
| tree_trunk/P2@0.5/iou_Hnear/pooled | 0.006 | 0.091 | 0.070 | 0.212 | 0.175 | False |
| tree_trunk/P2@0.5/iou_Hnear/mean | -0.021 | 0.045 | 0.084 | 0.110 | 0.191 | False |
| tree_trunk/P1/recall_Hnear/pooled | 0.000 | 0.061 | 0.038 | 0.167 | 0.100 | False |
| tree_trunk/P1/recall_Hnear/mean | 0.024 | 0.119 | 0.074 | 0.321 | 0.171 | False |
| tree_trunk/hidden_near/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| tree_trunk/hidden_near/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| tree_trunk/P2@0.5/recall_H/pooled | 0.066 | 0.003 | 0.003 | 0.008 | 0.007 | True |
| tree_trunk/P2@0.5/recall_H/mean | 0.074 | 0.017 | 0.012 | 0.037 | 0.031 | True |
| tree_trunk/P2@0.5/precision_H/pooled | 0.092 | 0.051 | 0.033 | 0.120 | 0.086 | False |
| tree_trunk/P2@0.5/precision_H/mean | 0.086 | 0.040 | 0.023 | 0.076 | 0.062 | True |
| tree_trunk/P2@0.5/iou_H/pooled | 0.064 | 0.003 | 0.003 | 0.008 | 0.006 | True |
| tree_trunk/P2@0.5/iou_H/mean | 0.054 | 0.015 | 0.010 | 0.033 | 0.025 | True |
| tree_trunk/hidden/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| tree_trunk/hidden/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| tree_trunk/P1/recall_H/pooled | -0.002 | 0.001 | 0.002 | 0.003 | 0.004 | False |
| tree_trunk/P1/recall_H/mean | -0.000 | 0.015 | 0.010 | 0.037 | 0.024 | False |

#### tree_trunk, P2 >= 0.8

| metric | gap | first std | second std | first range | second range | separated |
|---|---|---|---|---|---|---|
| tree_trunk/P2@0.8/recall/pooled | 0.048 | 0.002 | 0.001 | 0.006 | 0.004 | True |
| tree_trunk/P2@0.8/recall/mean | 0.061 | 0.004 | 0.001 | 0.008 | 0.003 | True |
| tree_trunk/P2@0.8/precision/pooled | -0.047 | 0.003 | 0.005 | 0.007 | 0.010 | True |
| tree_trunk/P2@0.8/precision/mean | 0.019 | 0.012 | 0.011 | 0.028 | 0.026 | False |
| tree_trunk/P2@0.8/iou/pooled | 0.035 | 0.002 | 0.001 | 0.005 | 0.002 | True |
| tree_trunk/P2@0.8/iou/mean | 0.027 | 0.003 | 0.001 | 0.007 | 0.003 | True |
| tree_trunk/P1/recall/pooled | -0.003 | 0.003 | 0.002 | 0.008 | 0.005 | False |
| tree_trunk/P1/recall/mean | -0.007 | 0.003 | 0.002 | 0.009 | 0.006 | False |
| tree_trunk/P1/precision/pooled | 0.001 | 0.004 | 0.004 | 0.008 | 0.009 | False |
| tree_trunk/P1/precision/mean | 0.019 | 0.013 | 0.008 | 0.033 | 0.021 | False |
| tree_trunk/P2@0.8/recall_Hnear/pooled | -0.020 | 0.061 | 0.080 | 0.133 | 0.200 | False |
| tree_trunk/P2@0.8/recall_Hnear/mean | -0.046 | 0.039 | 0.104 | 0.095 | 0.255 | False |
| tree_trunk/P2@0.8/precision_Hnear/pooled | -0.109 | 0.095 | 0.160 | 0.233 | 0.429 | False |
| tree_trunk/P2@0.8/precision_Hnear/mean | -0.130 | 0.088 | 0.225 | 0.217 | 0.600 | False |
| tree_trunk/P2@0.8/iou_Hnear/pooled | -0.028 | 0.054 | 0.076 | 0.121 | 0.191 | False |
| tree_trunk/P2@0.8/iou_Hnear/mean | -0.051 | 0.027 | 0.081 | 0.064 | 0.198 | False |
| tree_trunk/P1/recall_Hnear/pooled | 0.000 | 0.061 | 0.038 | 0.167 | 0.100 | False |
| tree_trunk/P1/recall_Hnear/mean | 0.024 | 0.119 | 0.074 | 0.321 | 0.171 | False |
| tree_trunk/hidden_near/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| tree_trunk/hidden_near/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| tree_trunk/P2@0.8/recall_H/pooled | 0.031 | 0.003 | 0.003 | 0.007 | 0.007 | True |
| tree_trunk/P2@0.8/recall_H/mean | 0.032 | 0.014 | 0.006 | 0.033 | 0.016 | True |
| tree_trunk/P2@0.8/precision_H/pooled | 0.008 | 0.021 | 0.074 | 0.053 | 0.203 | False |
| tree_trunk/P2@0.8/precision_H/mean | 0.028 | 0.026 | 0.071 | 0.074 | 0.192 | False |
| tree_trunk/P2@0.8/iou_H/pooled | 0.030 | 0.002 | 0.003 | 0.006 | 0.007 | True |
| tree_trunk/P2@0.8/iou_H/mean | 0.018 | 0.012 | 0.005 | 0.027 | 0.014 | True |
| tree_trunk/hidden/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| tree_trunk/hidden/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| tree_trunk/P1/recall_H/pooled | -0.002 | 0.001 | 0.002 | 0.003 | 0.004 | False |
| tree_trunk/P1/recall_H/mean | -0.000 | 0.015 | 0.010 | 0.037 | 0.024 | False |

#### rock, P2 >= 0.5

| metric | gap | first std | second std | first range | second range | separated |
|---|---|---|---|---|---|---|
| rock/P2@0.5/recall/pooled | 0.031 | 0.010 | 0.018 | 0.028 | 0.051 | False |
| rock/P2@0.5/recall/mean | 0.045 | 0.022 | 0.016 | 0.057 | 0.037 | True |
| rock/P2@0.5/precision/pooled | -0.027 | 0.020 | 0.011 | 0.050 | 0.031 | False |
| rock/P2@0.5/precision/mean | -0.120 | 0.056 | 0.039 | 0.131 | 0.093 | True |
| rock/P2@0.5/iou/pooled | 0.012 | 0.008 | 0.012 | 0.019 | 0.034 | False |
| rock/P2@0.5/iou/mean | 0.010 | 0.017 | 0.013 | 0.042 | 0.027 | False |
| rock/P1/recall/pooled | 0.007 | 0.016 | 0.023 | 0.039 | 0.051 | False |
| rock/P1/recall/mean | 0.010 | 0.010 | 0.023 | 0.022 | 0.054 | False |
| rock/P1/precision/pooled | 0.004 | 0.016 | 0.013 | 0.040 | 0.035 | False |
| rock/P1/precision/mean | 0.005 | 0.026 | 0.031 | 0.073 | 0.079 | False |
| rock/P2@0.5/recall_Hnear/pooled | n/a | n/a | n/a | n/a | n/a | None |
| rock/P2@0.5/recall_Hnear/mean | n/a | n/a | n/a | n/a | n/a | None |
| rock/P2@0.5/precision_Hnear/pooled | n/a | n/a | n/a | n/a | n/a | None |
| rock/P2@0.5/precision_Hnear/mean | n/a | n/a | n/a | n/a | n/a | None |
| rock/P2@0.5/iou_Hnear/pooled | n/a | n/a | n/a | n/a | n/a | None |
| rock/P2@0.5/iou_Hnear/mean | n/a | n/a | n/a | n/a | n/a | None |
| rock/P1/recall_Hnear/pooled | n/a | n/a | n/a | n/a | n/a | None |
| rock/P1/recall_Hnear/mean | n/a | n/a | n/a | n/a | n/a | None |
| rock/hidden_near/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| rock/hidden_near/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| rock/P2@0.5/recall_H/pooled | -0.002 | 0.017 | 0.022 | 0.040 | 0.051 | False |
| rock/P2@0.5/recall_H/mean | 0.008 | 0.036 | 0.015 | 0.083 | 0.028 | False |
| rock/P2@0.5/precision_H/pooled | -0.690 | 0.236 | 0.202 | 0.500 | 0.286 | True |
| rock/P2@0.5/precision_H/mean | -0.575 | 0.183 | 0.412 | 0.333 | 0.583 | True |
| rock/P2@0.5/iou_H/pooled | -0.002 | 0.017 | 0.021 | 0.039 | 0.050 | False |
| rock/P2@0.5/iou_H/mean | 0.002 | 0.024 | 0.014 | 0.056 | 0.028 | False |
| rock/hidden/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| rock/hidden/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| rock/P1/recall_H/pooled | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| rock/P1/recall_H/mean | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |

#### rock, P2 >= 0.8

| metric | gap | first std | second std | first range | second range | separated |
|---|---|---|---|---|---|---|
| rock/P2@0.8/recall/pooled | 0.039 | 0.010 | 0.015 | 0.022 | 0.034 | True |
| rock/P2@0.8/recall/mean | 0.051 | 0.010 | 0.017 | 0.026 | 0.046 | True |
| rock/P2@0.8/precision/pooled | -0.021 | 0.011 | 0.022 | 0.028 | 0.056 | False |
| rock/P2@0.8/precision/mean | -0.066 | 0.047 | 0.039 | 0.121 | 0.108 | False |
| rock/P2@0.8/iou/pooled | 0.020 | 0.007 | 0.012 | 0.016 | 0.029 | True |
| rock/P2@0.8/iou/mean | 0.016 | 0.009 | 0.014 | 0.023 | 0.038 | False |
| rock/P1/recall/pooled | 0.007 | 0.016 | 0.023 | 0.039 | 0.051 | False |
| rock/P1/recall/mean | 0.010 | 0.010 | 0.023 | 0.022 | 0.054 | False |
| rock/P1/precision/pooled | 0.004 | 0.016 | 0.013 | 0.040 | 0.035 | False |
| rock/P1/precision/mean | 0.005 | 0.026 | 0.031 | 0.073 | 0.079 | False |
| rock/P2@0.8/recall_Hnear/pooled | n/a | n/a | n/a | n/a | n/a | None |
| rock/P2@0.8/recall_Hnear/mean | n/a | n/a | n/a | n/a | n/a | None |
| rock/P2@0.8/precision_Hnear/pooled | n/a | n/a | n/a | n/a | n/a | None |
| rock/P2@0.8/precision_Hnear/mean | n/a | n/a | n/a | n/a | n/a | None |
| rock/P2@0.8/iou_Hnear/pooled | n/a | n/a | n/a | n/a | n/a | None |
| rock/P2@0.8/iou_Hnear/mean | n/a | n/a | n/a | n/a | n/a | None |
| rock/P1/recall_Hnear/pooled | n/a | n/a | n/a | n/a | n/a | None |
| rock/P1/recall_Hnear/mean | n/a | n/a | n/a | n/a | n/a | None |
| rock/hidden_near/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| rock/hidden_near/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| rock/P2@0.8/recall_H/pooled | 0.000 | 0.009 | 0.006 | 0.020 | 0.010 | False |
| rock/P2@0.8/recall_H/mean | 0.002 | 0.019 | 0.012 | 0.042 | 0.028 | False |
| rock/P2@0.8/precision_H/pooled | -0.417 | 0.577 | 0.354 | 1.000 | 0.500 | False |
| rock/P2@0.8/precision_H/mean | -0.417 | 0.577 | 0.354 | 1.000 | 0.500 | False |
| rock/P2@0.8/iou_H/pooled | 0.000 | 0.009 | 0.006 | 0.020 | 0.010 | False |
| rock/P2@0.8/iou_H/mean | 0.002 | 0.019 | 0.012 | 0.042 | 0.028 | False |
| rock/hidden/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| rock/hidden/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| rock/P1/recall_H/pooled | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| rock/P1/recall_H/mean | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |

#### bush_or_small_tree, P2 >= 0.5

| metric | gap | first std | second std | first range | second range | separated |
|---|---|---|---|---|---|---|
| bush_or_small_tree/P2@0.5/recall/pooled | 0.147 | 0.005 | 0.007 | 0.010 | 0.016 | True |
| bush_or_small_tree/P2@0.5/recall/mean | 0.265 | 0.006 | 0.012 | 0.013 | 0.034 | True |
| bush_or_small_tree/P2@0.5/precision/pooled | -0.156 | 0.007 | 0.012 | 0.018 | 0.028 | True |
| bush_or_small_tree/P2@0.5/precision/mean | -0.057 | 0.015 | 0.033 | 0.037 | 0.087 | False |
| bush_or_small_tree/P2@0.5/iou/pooled | 0.117 | 0.004 | 0.007 | 0.010 | 0.016 | True |
| bush_or_small_tree/P2@0.5/iou/mean | 0.083 | 0.003 | 0.008 | 0.006 | 0.023 | True |
| bush_or_small_tree/P1/recall/pooled | 0.001 | 0.001 | 0.002 | 0.003 | 0.004 | False |
| bush_or_small_tree/P1/recall/mean | 0.008 | 0.006 | 0.003 | 0.016 | 0.009 | False |
| bush_or_small_tree/P1/precision/pooled | -0.026 | 0.007 | 0.007 | 0.016 | 0.017 | True |
| bush_or_small_tree/P1/precision/mean | -0.016 | 0.021 | 0.021 | 0.050 | 0.051 | False |
| bush_or_small_tree/P2@0.5/recall_Hnear/pooled | 0.086 | 0.078 | 0.126 | 0.190 | 0.288 | False |
| bush_or_small_tree/P2@0.5/recall_Hnear/mean | 0.454 | 0.029 | 0.047 | 0.077 | 0.098 | True |
| bush_or_small_tree/P2@0.5/precision_Hnear/pooled | -0.022 | 0.100 | 0.234 | 0.253 | 0.575 | False |
| bush_or_small_tree/P2@0.5/precision_Hnear/mean | 0.080 | 0.043 | 0.132 | 0.101 | 0.304 | False |
| bush_or_small_tree/P2@0.5/iou_Hnear/pooled | 0.061 | 0.067 | 0.121 | 0.164 | 0.278 | False |
| bush_or_small_tree/P2@0.5/iou_Hnear/mean | 0.131 | 0.019 | 0.036 | 0.046 | 0.079 | True |
| bush_or_small_tree/P1/recall_Hnear/pooled | 0.015 | 0.014 | 0.011 | 0.037 | 0.025 | False |
| bush_or_small_tree/P1/recall_Hnear/mean | 0.057 | 0.083 | 0.087 | 0.194 | 0.196 | False |
| bush_or_small_tree/hidden_near/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| bush_or_small_tree/hidden_near/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| bush_or_small_tree/P2@0.5/recall_H/pooled | 0.092 | 0.002 | 0.008 | 0.006 | 0.016 | True |
| bush_or_small_tree/P2@0.5/recall_H/mean | 0.258 | 0.013 | 0.027 | 0.033 | 0.064 | True |
| bush_or_small_tree/P2@0.5/precision_H/pooled | 0.086 | 0.011 | 0.162 | 0.028 | 0.435 | False |
| bush_or_small_tree/P2@0.5/precision_H/mean | 0.129 | 0.040 | 0.123 | 0.098 | 0.343 | False |
| bush_or_small_tree/P2@0.5/iou_H/pooled | 0.083 | 0.002 | 0.007 | 0.005 | 0.016 | True |
| bush_or_small_tree/P2@0.5/iou_H/mean | 0.096 | 0.007 | 0.020 | 0.020 | 0.047 | True |
| bush_or_small_tree/hidden/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| bush_or_small_tree/hidden/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| bush_or_small_tree/P1/recall_H/pooled | -0.002 | 0.003 | 0.003 | 0.008 | 0.008 | False |
| bush_or_small_tree/P1/recall_H/mean | 0.017 | 0.030 | 0.032 | 0.071 | 0.069 | False |

#### bush_or_small_tree, P2 >= 0.8

| metric | gap | first std | second std | first range | second range | separated |
|---|---|---|---|---|---|---|
| bush_or_small_tree/P2@0.8/recall/pooled | 0.138 | 0.003 | 0.005 | 0.006 | 0.011 | True |
| bush_or_small_tree/P2@0.8/recall/mean | 0.264 | 0.004 | 0.003 | 0.007 | 0.007 | True |
| bush_or_small_tree/P2@0.8/precision/pooled | -0.176 | 0.004 | 0.011 | 0.010 | 0.025 | True |
| bush_or_small_tree/P2@0.8/precision/mean | -0.067 | 0.021 | 0.027 | 0.057 | 0.067 | True |
| bush_or_small_tree/P2@0.8/iou/pooled | 0.114 | 0.002 | 0.005 | 0.006 | 0.011 | True |
| bush_or_small_tree/P2@0.8/iou/mean | 0.093 | 0.002 | 0.002 | 0.006 | 0.005 | True |
| bush_or_small_tree/P1/recall/pooled | 0.001 | 0.001 | 0.002 | 0.003 | 0.004 | False |
| bush_or_small_tree/P1/recall/mean | 0.008 | 0.006 | 0.003 | 0.016 | 0.009 | False |
| bush_or_small_tree/P1/precision/pooled | -0.026 | 0.007 | 0.007 | 0.016 | 0.017 | True |
| bush_or_small_tree/P1/precision/mean | -0.016 | 0.021 | 0.021 | 0.050 | 0.051 | False |
| bush_or_small_tree/P2@0.8/recall_Hnear/pooled | 0.043 | 0.041 | 0.085 | 0.098 | 0.202 | False |
| bush_or_small_tree/P2@0.8/recall_Hnear/mean | 0.419 | 0.030 | 0.033 | 0.064 | 0.069 | True |
| bush_or_small_tree/P2@0.8/precision_Hnear/pooled | -0.030 | 0.094 | 0.228 | 0.235 | 0.567 | False |
| bush_or_small_tree/P2@0.8/precision_Hnear/mean | 0.086 | 0.044 | 0.161 | 0.114 | 0.340 | False |
| bush_or_small_tree/P2@0.8/iou_Hnear/pooled | 0.033 | 0.037 | 0.084 | 0.090 | 0.198 | False |
| bush_or_small_tree/P2@0.8/iou_Hnear/mean | 0.124 | 0.017 | 0.028 | 0.041 | 0.058 | True |
| bush_or_small_tree/P1/recall_Hnear/pooled | 0.015 | 0.014 | 0.011 | 0.037 | 0.025 | False |
| bush_or_small_tree/P1/recall_Hnear/mean | 0.057 | 0.083 | 0.087 | 0.194 | 0.196 | False |
| bush_or_small_tree/hidden_near/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| bush_or_small_tree/hidden_near/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| bush_or_small_tree/P2@0.8/recall_H/pooled | 0.084 | 0.001 | 0.005 | 0.002 | 0.010 | True |
| bush_or_small_tree/P2@0.8/recall_H/mean | 0.241 | 0.011 | 0.011 | 0.023 | 0.025 | True |
| bush_or_small_tree/P2@0.8/precision_H/pooled | 0.107 | 0.006 | 0.155 | 0.013 | 0.392 | False |
| bush_or_small_tree/P2@0.8/precision_H/mean | 0.138 | 0.020 | 0.116 | 0.055 | 0.257 | False |
| bush_or_small_tree/P2@0.8/iou_H/pooled | 0.078 | 0.001 | 0.004 | 0.002 | 0.010 | True |
| bush_or_small_tree/P2@0.8/iou_H/mean | 0.096 | 0.005 | 0.010 | 0.012 | 0.023 | True |
| bush_or_small_tree/hidden/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| bush_or_small_tree/hidden/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| bush_or_small_tree/P1/recall_H/pooled | -0.002 | 0.003 | 0.003 | 0.008 | 0.008 | False |
| bush_or_small_tree/P1/recall_H/mean | 0.017 | 0.030 | 0.032 | 0.071 | 0.069 | False |

GT cells occupied in every repeat of one arm and none of the other: {"bush_or_small_tree": {"first_only": {"blocked": 555, "no_return": 75, "passed_through": 0, "seen": 29, "unobserved": 9}, "second_only": {"blocked": 7, "no_return": 0, "passed_through": 0, "seen": 1, "unobserved": 0}}, "obstacles": {"first_only": {"blocked": 817, "no_return": 187, "passed_through": 0, "seen": 129, "unobserved": 9}, "second_only": {"blocked": 88, "no_return": 2, "passed_through": 0, "seen": 61, "unobserved": 0}}, "rock": {"first_only": {"blocked": 4, "no_return": 0, "passed_through": 0, "seen": 0, "unobserved": 0}, "second_only": {"blocked": 0, "no_return": 0, "passed_through": 0, "seen": 0, "unobserved": 0}}, "tree_trunk": {"first_only": {"blocked": 258, "no_return": 112, "passed_through": 0, "seen": 100, "unobserved": 0}, "second_only": {"blocked": 81, "no_return": 2, "passed_through": 0, "seen": 60, "unobserved": 0}}}

### may_scene_a_radar_init_loss - may_scene_a_radar_init

#### obstacles, P2 >= 0.5

| metric | gap | first std | second std | first range | second range | separated |
|---|---|---|---|---|---|---|
| obstacles/P2@0.5/recall/pooled | 0.150 | 0.003 | 0.003 | 0.008 | 0.007 | True |
| obstacles/P2@0.5/recall/mean | 0.168 | 0.006 | 0.002 | 0.013 | 0.005 | True |
| obstacles/P2@0.5/precision/pooled | -0.052 | 0.006 | 0.005 | 0.014 | 0.012 | True |
| obstacles/P2@0.5/precision/mean | 0.014 | 0.011 | 0.020 | 0.026 | 0.052 | False |
| obstacles/P2@0.5/iou/pooled | 0.121 | 0.002 | 0.003 | 0.005 | 0.007 | True |
| obstacles/P2@0.5/iou/mean | 0.101 | 0.003 | 0.002 | 0.008 | 0.006 | True |
| obstacles/P1/recall/pooled | 0.024 | 0.002 | 0.002 | 0.005 | 0.005 | True |
| obstacles/P1/recall/mean | 0.027 | 0.002 | 0.002 | 0.005 | 0.005 | True |
| obstacles/P1/precision/pooled | 0.013 | 0.003 | 0.004 | 0.007 | 0.009 | True |
| obstacles/P1/precision/mean | 0.017 | 0.010 | 0.005 | 0.024 | 0.010 | True |
| obstacles/P2@0.5/recall_Hnear/pooled | 0.135 | 0.077 | 0.012 | 0.197 | 0.031 | True |
| obstacles/P2@0.5/recall_Hnear/mean | 0.198 | 0.038 | 0.049 | 0.091 | 0.108 | True |
| obstacles/P2@0.5/precision_Hnear/pooled | -0.094 | 0.095 | 0.088 | 0.242 | 0.207 | False |
| obstacles/P2@0.5/precision_Hnear/mean | -0.114 | 0.038 | 0.084 | 0.103 | 0.229 | False |
| obstacles/P2@0.5/iou_Hnear/pooled | 0.106 | 0.066 | 0.012 | 0.171 | 0.031 | True |
| obstacles/P2@0.5/iou_Hnear/mean | 0.044 | 0.030 | 0.033 | 0.069 | 0.081 | False |
| obstacles/P1/recall_Hnear/pooled | 0.015 | 0.015 | 0.008 | 0.036 | 0.021 | False |
| obstacles/P1/recall_Hnear/mean | 0.055 | 0.065 | 0.035 | 0.171 | 0.077 | False |
| obstacles/hidden_near/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| obstacles/hidden_near/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| obstacles/P2@0.5/recall_H/pooled | 0.091 | 0.002 | 0.002 | 0.004 | 0.005 | True |
| obstacles/P2@0.5/recall_H/mean | 0.130 | 0.009 | 0.009 | 0.022 | 0.021 | True |
| obstacles/P2@0.5/precision_H/pooled | 0.010 | 0.017 | 0.093 | 0.045 | 0.257 | False |
| obstacles/P2@0.5/precision_H/mean | 0.031 | 0.031 | 0.077 | 0.069 | 0.205 | False |
| obstacles/P2@0.5/iou_H/pooled | 0.085 | 0.001 | 0.002 | 0.003 | 0.005 | True |
| obstacles/P2@0.5/iou_H/mean | 0.083 | 0.008 | 0.008 | 0.021 | 0.021 | True |
| obstacles/hidden/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| obstacles/hidden/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| obstacles/P1/recall_H/pooled | 0.002 | 0.002 | 0.002 | 0.005 | 0.004 | False |
| obstacles/P1/recall_H/mean | 0.005 | 0.010 | 0.013 | 0.024 | 0.032 | False |

#### obstacles, P2 >= 0.8

| metric | gap | first std | second std | first range | second range | separated |
|---|---|---|---|---|---|---|
| obstacles/P2@0.8/recall/pooled | 0.136 | 0.001 | 0.002 | 0.003 | 0.006 | True |
| obstacles/P2@0.8/recall/mean | 0.173 | 0.003 | 0.003 | 0.007 | 0.007 | True |
| obstacles/P2@0.8/precision/pooled | -0.058 | 0.003 | 0.002 | 0.006 | 0.007 | True |
| obstacles/P2@0.8/precision/mean | 0.029 | 0.008 | 0.012 | 0.019 | 0.027 | True |
| obstacles/P2@0.8/iou/pooled | 0.114 | 0.001 | 0.002 | 0.002 | 0.006 | True |
| obstacles/P2@0.8/iou/mean | 0.105 | 0.003 | 0.003 | 0.006 | 0.007 | True |
| obstacles/P1/recall/pooled | 0.024 | 0.002 | 0.002 | 0.005 | 0.005 | True |
| obstacles/P1/recall/mean | 0.027 | 0.002 | 0.002 | 0.005 | 0.005 | True |
| obstacles/P1/precision/pooled | 0.013 | 0.003 | 0.004 | 0.007 | 0.009 | True |
| obstacles/P1/precision/mean | 0.017 | 0.010 | 0.005 | 0.024 | 0.010 | True |
| obstacles/P2@0.8/recall_Hnear/pooled | 0.066 | 0.041 | 0.013 | 0.098 | 0.031 | True |
| obstacles/P2@0.8/recall_Hnear/mean | 0.193 | 0.021 | 0.021 | 0.051 | 0.054 | True |
| obstacles/P2@0.8/precision_Hnear/pooled | -0.090 | 0.082 | 0.105 | 0.219 | 0.234 | False |
| obstacles/P2@0.8/precision_Hnear/mean | -0.077 | 0.043 | 0.101 | 0.120 | 0.273 | False |
| obstacles/P2@0.8/iou_Hnear/pooled | 0.055 | 0.038 | 0.012 | 0.088 | 0.031 | True |
| obstacles/P2@0.8/iou_Hnear/mean | 0.055 | 0.018 | 0.030 | 0.047 | 0.081 | False |
| obstacles/P1/recall_Hnear/pooled | 0.015 | 0.015 | 0.008 | 0.036 | 0.021 | False |
| obstacles/P1/recall_Hnear/mean | 0.055 | 0.065 | 0.035 | 0.171 | 0.077 | False |
| obstacles/hidden_near/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| obstacles/hidden_near/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| obstacles/P2@0.8/recall_H/pooled | 0.063 | 0.001 | 0.002 | 0.002 | 0.004 | True |
| obstacles/P2@0.8/recall_H/mean | 0.087 | 0.009 | 0.007 | 0.022 | 0.019 | True |
| obstacles/P2@0.8/precision_H/pooled | -0.026 | 0.007 | 0.087 | 0.017 | 0.243 | False |
| obstacles/P2@0.8/precision_H/mean | 0.041 | 0.025 | 0.086 | 0.062 | 0.226 | False |
| obstacles/P2@0.8/iou_H/pooled | 0.060 | 0.001 | 0.002 | 0.002 | 0.004 | True |
| obstacles/P2@0.8/iou_H/mean | 0.050 | 0.008 | 0.008 | 0.019 | 0.021 | True |
| obstacles/hidden/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| obstacles/hidden/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| obstacles/P1/recall_H/pooled | 0.002 | 0.002 | 0.002 | 0.005 | 0.004 | False |
| obstacles/P1/recall_H/mean | 0.005 | 0.010 | 0.013 | 0.024 | 0.032 | False |

#### tree_trunk, P2 >= 0.5

| metric | gap | first std | second std | first range | second range | separated |
|---|---|---|---|---|---|---|
| tree_trunk/P2@0.5/recall/pooled | 0.131 | 0.004 | 0.001 | 0.011 | 0.003 | True |
| tree_trunk/P2@0.5/recall/mean | 0.148 | 0.006 | 0.002 | 0.012 | 0.005 | True |
| tree_trunk/P2@0.5/precision/pooled | -0.017 | 0.009 | 0.002 | 0.022 | 0.004 | True |
| tree_trunk/P2@0.5/precision/mean | 0.024 | 0.011 | 0.020 | 0.027 | 0.047 | False |
| tree_trunk/P2@0.5/iou/pooled | 0.109 | 0.004 | 0.001 | 0.009 | 0.003 | True |
| tree_trunk/P2@0.5/iou/mean | 0.104 | 0.003 | 0.003 | 0.008 | 0.006 | True |
| tree_trunk/P1/recall/pooled | 0.034 | 0.003 | 0.002 | 0.008 | 0.005 | True |
| tree_trunk/P1/recall/mean | 0.033 | 0.003 | 0.002 | 0.009 | 0.007 | True |
| tree_trunk/P1/precision/pooled | 0.024 | 0.004 | 0.006 | 0.008 | 0.015 | True |
| tree_trunk/P1/precision/mean | 0.026 | 0.013 | 0.007 | 0.033 | 0.017 | True |
| tree_trunk/P2@0.5/recall_Hnear/pooled | 0.080 | 0.100 | 0.030 | 0.233 | 0.067 | False |
| tree_trunk/P2@0.5/recall_Hnear/mean | -0.023 | 0.059 | 0.101 | 0.145 | 0.252 | False |
| tree_trunk/P2@0.5/precision_Hnear/pooled | 0.088 | 0.074 | 0.102 | 0.152 | 0.280 | False |
| tree_trunk/P2@0.5/precision_Hnear/mean | -0.045 | 0.048 | 0.047 | 0.117 | 0.114 | False |
| tree_trunk/P2@0.5/iou_Hnear/pooled | 0.077 | 0.091 | 0.034 | 0.212 | 0.087 | False |
| tree_trunk/P2@0.5/iou_Hnear/mean | -0.024 | 0.045 | 0.071 | 0.110 | 0.185 | False |
| tree_trunk/P1/recall_Hnear/pooled | 0.053 | 0.061 | 0.049 | 0.167 | 0.133 | False |
| tree_trunk/P1/recall_Hnear/mean | 0.086 | 0.119 | 0.090 | 0.321 | 0.190 | False |
| tree_trunk/hidden_near/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| tree_trunk/hidden_near/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| tree_trunk/P2@0.5/recall_H/pooled | 0.085 | 0.003 | 0.004 | 0.008 | 0.011 | True |
| tree_trunk/P2@0.5/recall_H/mean | 0.114 | 0.017 | 0.015 | 0.037 | 0.037 | True |
| tree_trunk/P2@0.5/precision_H/pooled | 0.058 | 0.051 | 0.085 | 0.120 | 0.221 | False |
| tree_trunk/P2@0.5/precision_H/mean | 0.042 | 0.040 | 0.090 | 0.076 | 0.230 | False |
| tree_trunk/P2@0.5/iou_H/pooled | 0.082 | 0.003 | 0.004 | 0.008 | 0.011 | True |
| tree_trunk/P2@0.5/iou_H/mean | 0.090 | 0.015 | 0.013 | 0.033 | 0.036 | True |
| tree_trunk/hidden/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| tree_trunk/hidden/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| tree_trunk/P1/recall_H/pooled | 0.005 | 0.001 | 0.002 | 0.003 | 0.005 | True |
| tree_trunk/P1/recall_H/mean | 0.007 | 0.015 | 0.019 | 0.037 | 0.047 | False |

#### tree_trunk, P2 >= 0.8

| metric | gap | first std | second std | first range | second range | separated |
|---|---|---|---|---|---|---|
| tree_trunk/P2@0.8/recall/pooled | 0.116 | 0.002 | 0.003 | 0.006 | 0.007 | True |
| tree_trunk/P2@0.8/recall/mean | 0.152 | 0.004 | 0.003 | 0.008 | 0.008 | True |
| tree_trunk/P2@0.8/precision/pooled | -0.023 | 0.003 | 0.003 | 0.007 | 0.007 | True |
| tree_trunk/P2@0.8/precision/mean | 0.032 | 0.012 | 0.018 | 0.028 | 0.048 | False |
| tree_trunk/P2@0.8/iou/pooled | 0.100 | 0.002 | 0.003 | 0.005 | 0.007 | True |
| tree_trunk/P2@0.8/iou/mean | 0.106 | 0.003 | 0.003 | 0.007 | 0.007 | True |
| tree_trunk/P1/recall/pooled | 0.034 | 0.003 | 0.002 | 0.008 | 0.005 | True |
| tree_trunk/P1/recall/mean | 0.033 | 0.003 | 0.002 | 0.009 | 0.007 | True |
| tree_trunk/P1/precision/pooled | 0.024 | 0.004 | 0.006 | 0.008 | 0.015 | True |
| tree_trunk/P1/precision/mean | 0.026 | 0.013 | 0.007 | 0.033 | 0.017 | True |
| tree_trunk/P2@0.8/recall_Hnear/pooled | 0.027 | 0.061 | 0.055 | 0.133 | 0.133 | False |
| tree_trunk/P2@0.8/recall_Hnear/mean | 0.010 | 0.039 | 0.063 | 0.095 | 0.148 | False |
| tree_trunk/P2@0.8/precision_Hnear/pooled | 0.062 | 0.095 | 0.116 | 0.233 | 0.278 | False |
| tree_trunk/P2@0.8/precision_Hnear/mean | -0.028 | 0.088 | 0.040 | 0.217 | 0.083 | False |
| tree_trunk/P2@0.8/iou_Hnear/pooled | 0.026 | 0.054 | 0.052 | 0.121 | 0.117 | False |
| tree_trunk/P2@0.8/iou_Hnear/mean | 0.008 | 0.027 | 0.055 | 0.064 | 0.129 | False |
| tree_trunk/P1/recall_Hnear/pooled | 0.053 | 0.061 | 0.049 | 0.167 | 0.133 | False |
| tree_trunk/P1/recall_Hnear/mean | 0.086 | 0.119 | 0.090 | 0.321 | 0.190 | False |
| tree_trunk/hidden_near/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| tree_trunk/hidden_near/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| tree_trunk/P2@0.8/recall_H/pooled | 0.041 | 0.003 | 0.004 | 0.007 | 0.008 | True |
| tree_trunk/P2@0.8/recall_H/mean | 0.061 | 0.014 | 0.013 | 0.033 | 0.035 | True |
| tree_trunk/P2@0.8/precision_H/pooled | -0.005 | 0.021 | 0.133 | 0.053 | 0.344 | False |
| tree_trunk/P2@0.8/precision_H/mean | 0.019 | 0.026 | 0.114 | 0.074 | 0.269 | False |
| tree_trunk/P2@0.8/iou_H/pooled | 0.040 | 0.002 | 0.004 | 0.006 | 0.008 | True |
| tree_trunk/P2@0.8/iou_H/mean | 0.045 | 0.012 | 0.013 | 0.027 | 0.034 | True |
| tree_trunk/hidden/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| tree_trunk/hidden/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| tree_trunk/P1/recall_H/pooled | 0.005 | 0.001 | 0.002 | 0.003 | 0.005 | True |
| tree_trunk/P1/recall_H/mean | 0.007 | 0.015 | 0.019 | 0.037 | 0.047 | False |

#### rock, P2 >= 0.5

| metric | gap | first std | second std | first range | second range | separated |
|---|---|---|---|---|---|---|
| rock/P2@0.5/recall/pooled | 0.079 | 0.010 | 0.021 | 0.028 | 0.045 | True |
| rock/P2@0.5/recall/mean | 0.088 | 0.022 | 0.019 | 0.057 | 0.038 | True |
| rock/P2@0.5/precision/pooled | 0.023 | 0.020 | 0.034 | 0.050 | 0.088 | False |
| rock/P2@0.5/precision/mean | -0.083 | 0.056 | 0.056 | 0.131 | 0.138 | False |
| rock/P2@0.5/iou/pooled | 0.048 | 0.008 | 0.017 | 0.019 | 0.037 | True |
| rock/P2@0.5/iou/mean | 0.038 | 0.017 | 0.011 | 0.042 | 0.023 | True |
| rock/P1/recall/pooled | 0.008 | 0.016 | 0.005 | 0.039 | 0.011 | False |
| rock/P1/recall/mean | 0.007 | 0.010 | 0.010 | 0.022 | 0.024 | False |
| rock/P1/precision/pooled | 0.009 | 0.016 | 0.014 | 0.040 | 0.030 | False |
| rock/P1/precision/mean | -0.014 | 0.026 | 0.031 | 0.073 | 0.074 | False |
| rock/P2@0.5/recall_Hnear/pooled | n/a | n/a | n/a | n/a | n/a | None |
| rock/P2@0.5/recall_Hnear/mean | n/a | n/a | n/a | n/a | n/a | None |
| rock/P2@0.5/precision_Hnear/pooled | n/a | n/a | n/a | n/a | n/a | None |
| rock/P2@0.5/precision_Hnear/mean | n/a | n/a | n/a | n/a | n/a | None |
| rock/P2@0.5/iou_Hnear/pooled | n/a | n/a | n/a | n/a | n/a | None |
| rock/P2@0.5/iou_Hnear/mean | n/a | n/a | n/a | n/a | n/a | None |
| rock/P1/recall_Hnear/pooled | n/a | n/a | n/a | n/a | n/a | None |
| rock/P1/recall_Hnear/mean | n/a | n/a | n/a | n/a | n/a | None |
| rock/hidden_near/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| rock/hidden_near/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| rock/P2@0.5/recall_H/pooled | 0.010 | 0.017 | 0.000 | 0.040 | 0.000 | False |
| rock/P2@0.5/recall_H/mean | 0.019 | 0.036 | 0.000 | 0.083 | 0.000 | False |
| rock/P2@0.5/precision_H/pooled | n/a | 0.236 | n/a | 0.500 | n/a | None |
| rock/P2@0.5/precision_H/mean | n/a | 0.183 | n/a | 0.333 | n/a | None |
| rock/P2@0.5/iou_H/pooled | 0.010 | 0.017 | 0.000 | 0.039 | 0.000 | False |
| rock/P2@0.5/iou_H/mean | 0.013 | 0.024 | 0.000 | 0.056 | 0.000 | False |
| rock/hidden/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| rock/hidden/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| rock/P1/recall_H/pooled | -0.004 | 0.000 | 0.006 | 0.000 | 0.010 | False |
| rock/P1/recall_H/mean | -0.008 | 0.000 | 0.011 | 0.000 | 0.021 | False |

#### rock, P2 >= 0.8

| metric | gap | first std | second std | first range | second range | separated |
|---|---|---|---|---|---|---|
| rock/P2@0.8/recall/pooled | 0.102 | 0.010 | 0.011 | 0.022 | 0.022 | True |
| rock/P2@0.8/recall/mean | 0.109 | 0.010 | 0.011 | 0.026 | 0.027 | True |
| rock/P2@0.8/precision/pooled | 0.049 | 0.011 | 0.025 | 0.028 | 0.065 | True |
| rock/P2@0.8/precision/mean | -0.017 | 0.047 | 0.061 | 0.121 | 0.141 | False |
| rock/P2@0.8/iou/pooled | 0.070 | 0.007 | 0.010 | 0.016 | 0.022 | True |
| rock/P2@0.8/iou/mean | 0.051 | 0.009 | 0.009 | 0.023 | 0.025 | True |
| rock/P1/recall/pooled | 0.008 | 0.016 | 0.005 | 0.039 | 0.011 | False |
| rock/P1/recall/mean | 0.007 | 0.010 | 0.010 | 0.022 | 0.024 | False |
| rock/P1/precision/pooled | 0.009 | 0.016 | 0.014 | 0.040 | 0.030 | False |
| rock/P1/precision/mean | -0.014 | 0.026 | 0.031 | 0.073 | 0.074 | False |
| rock/P2@0.8/recall_Hnear/pooled | n/a | n/a | n/a | n/a | n/a | None |
| rock/P2@0.8/recall_Hnear/mean | n/a | n/a | n/a | n/a | n/a | None |
| rock/P2@0.8/precision_Hnear/pooled | n/a | n/a | n/a | n/a | n/a | None |
| rock/P2@0.8/precision_Hnear/mean | n/a | n/a | n/a | n/a | n/a | None |
| rock/P2@0.8/iou_Hnear/pooled | n/a | n/a | n/a | n/a | n/a | None |
| rock/P2@0.8/iou_Hnear/mean | n/a | n/a | n/a | n/a | n/a | None |
| rock/P1/recall_Hnear/pooled | n/a | n/a | n/a | n/a | n/a | None |
| rock/P1/recall_Hnear/mean | n/a | n/a | n/a | n/a | n/a | None |
| rock/hidden_near/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| rock/hidden_near/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| rock/P2@0.8/recall_H/pooled | 0.004 | 0.009 | 0.000 | 0.020 | 0.000 | False |
| rock/P2@0.8/recall_H/mean | 0.008 | 0.019 | 0.000 | 0.042 | 0.000 | False |
| rock/P2@0.8/precision_H/pooled | n/a | 0.577 | n/a | 1.000 | n/a | None |
| rock/P2@0.8/precision_H/mean | n/a | 0.577 | n/a | 1.000 | n/a | None |
| rock/P2@0.8/iou_H/pooled | 0.004 | 0.009 | 0.000 | 0.020 | 0.000 | False |
| rock/P2@0.8/iou_H/mean | 0.008 | 0.019 | 0.000 | 0.042 | 0.000 | False |
| rock/hidden/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| rock/hidden/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| rock/P1/recall_H/pooled | -0.004 | 0.000 | 0.006 | 0.000 | 0.010 | False |
| rock/P1/recall_H/mean | -0.008 | 0.000 | 0.011 | 0.000 | 0.021 | False |

#### bush_or_small_tree, P2 >= 0.5

| metric | gap | first std | second std | first range | second range | separated |
|---|---|---|---|---|---|---|
| bush_or_small_tree/P2@0.5/recall/pooled | 0.181 | 0.005 | 0.006 | 0.010 | 0.015 | True |
| bush_or_small_tree/P2@0.5/recall/mean | 0.319 | 0.006 | 0.008 | 0.013 | 0.020 | True |
| bush_or_small_tree/P2@0.5/precision/pooled | -0.100 | 0.007 | 0.020 | 0.018 | 0.052 | True |
| bush_or_small_tree/P2@0.5/precision/mean | 0.009 | 0.015 | 0.043 | 0.037 | 0.118 | False |
| bush_or_small_tree/P2@0.5/iou/pooled | 0.150 | 0.004 | 0.006 | 0.010 | 0.015 | True |
| bush_or_small_tree/P2@0.5/iou/mean | 0.126 | 0.003 | 0.007 | 0.006 | 0.017 | True |
| bush_or_small_tree/P1/recall/pooled | 0.010 | 0.001 | 0.003 | 0.003 | 0.007 | True |
| bush_or_small_tree/P1/recall/mean | 0.010 | 0.006 | 0.002 | 0.016 | 0.005 | True |
| bush_or_small_tree/P1/precision/pooled | -0.018 | 0.007 | 0.011 | 0.016 | 0.028 | False |
| bush_or_small_tree/P1/precision/mean | -0.038 | 0.021 | 0.020 | 0.050 | 0.053 | False |
| bush_or_small_tree/P2@0.5/recall_Hnear/pooled | 0.145 | 0.078 | 0.012 | 0.190 | 0.031 | True |
| bush_or_small_tree/P2@0.5/recall_Hnear/mean | 0.456 | 0.029 | 0.038 | 0.077 | 0.081 | True |
| bush_or_small_tree/P2@0.5/precision_Hnear/pooled | -0.104 | 0.100 | 0.100 | 0.253 | 0.259 | False |
| bush_or_small_tree/P2@0.5/precision_Hnear/mean | -0.125 | 0.043 | 0.139 | 0.101 | 0.333 | False |
| bush_or_small_tree/P2@0.5/iou_Hnear/pooled | 0.116 | 0.067 | 0.012 | 0.164 | 0.030 | True |
| bush_or_small_tree/P2@0.5/iou_Hnear/mean | 0.130 | 0.019 | 0.030 | 0.046 | 0.067 | True |
| bush_or_small_tree/P1/recall_Hnear/pooled | 0.007 | 0.014 | 0.007 | 0.037 | 0.018 | False |
| bush_or_small_tree/P1/recall_Hnear/mean | 0.018 | 0.083 | 0.031 | 0.194 | 0.058 | False |
| bush_or_small_tree/hidden_near/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| bush_or_small_tree/hidden_near/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| bush_or_small_tree/P2@0.5/recall_H/pooled | 0.099 | 0.002 | 0.001 | 0.006 | 0.002 | True |
| bush_or_small_tree/P2@0.5/recall_H/mean | 0.269 | 0.013 | 0.012 | 0.033 | 0.026 | True |
| bush_or_small_tree/P2@0.5/precision_H/pooled | 0.068 | 0.011 | 0.118 | 0.028 | 0.291 | False |
| bush_or_small_tree/P2@0.5/precision_H/mean | 0.032 | 0.040 | 0.096 | 0.098 | 0.242 | False |
| bush_or_small_tree/P2@0.5/iou_H/pooled | 0.090 | 0.002 | 0.001 | 0.005 | 0.002 | True |
| bush_or_small_tree/P2@0.5/iou_H/mean | 0.103 | 0.007 | 0.010 | 0.020 | 0.023 | True |
| bush_or_small_tree/hidden/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| bush_or_small_tree/hidden/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| bush_or_small_tree/P1/recall_H/pooled | 0.000 | 0.003 | 0.004 | 0.008 | 0.009 | False |
| bush_or_small_tree/P1/recall_H/mean | 0.006 | 0.030 | 0.010 | 0.071 | 0.026 | False |

#### bush_or_small_tree, P2 >= 0.8

| metric | gap | first std | second std | first range | second range | separated |
|---|---|---|---|---|---|---|
| bush_or_small_tree/P2@0.8/recall/pooled | 0.167 | 0.003 | 0.003 | 0.006 | 0.008 | True |
| bush_or_small_tree/P2@0.8/recall/mean | 0.315 | 0.004 | 0.007 | 0.007 | 0.018 | True |
| bush_or_small_tree/P2@0.8/precision/pooled | -0.102 | 0.004 | 0.020 | 0.010 | 0.045 | True |
| bush_or_small_tree/P2@0.8/precision/mean | 0.044 | 0.021 | 0.019 | 0.057 | 0.054 | False |
| bush_or_small_tree/P2@0.8/iou/pooled | 0.144 | 0.002 | 0.003 | 0.006 | 0.008 | True |
| bush_or_small_tree/P2@0.8/iou/mean | 0.134 | 0.002 | 0.007 | 0.006 | 0.017 | True |
| bush_or_small_tree/P1/recall/pooled | 0.010 | 0.001 | 0.003 | 0.003 | 0.007 | True |
| bush_or_small_tree/P1/recall/mean | 0.010 | 0.006 | 0.002 | 0.016 | 0.005 | True |
| bush_or_small_tree/P1/precision/pooled | -0.018 | 0.007 | 0.011 | 0.016 | 0.028 | False |
| bush_or_small_tree/P1/precision/mean | -0.038 | 0.021 | 0.020 | 0.050 | 0.053 | False |
| bush_or_small_tree/P2@0.8/recall_Hnear/pooled | 0.074 | 0.041 | 0.009 | 0.098 | 0.025 | True |
| bush_or_small_tree/P2@0.8/recall_Hnear/mean | 0.407 | 0.030 | 0.038 | 0.064 | 0.085 | True |
| bush_or_small_tree/P2@0.8/precision_Hnear/pooled | -0.100 | 0.094 | 0.120 | 0.235 | 0.303 | False |
| bush_or_small_tree/P2@0.8/precision_Hnear/mean | -0.083 | 0.044 | 0.162 | 0.114 | 0.400 | False |
| bush_or_small_tree/P2@0.8/iou_Hnear/pooled | 0.063 | 0.037 | 0.009 | 0.090 | 0.024 | True |
| bush_or_small_tree/P2@0.8/iou_Hnear/mean | 0.115 | 0.017 | 0.027 | 0.041 | 0.060 | True |
| bush_or_small_tree/P1/recall_Hnear/pooled | 0.007 | 0.014 | 0.007 | 0.037 | 0.018 | False |
| bush_or_small_tree/P1/recall_Hnear/mean | 0.018 | 0.083 | 0.031 | 0.194 | 0.058 | False |
| bush_or_small_tree/hidden_near/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| bush_or_small_tree/hidden_near/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| bush_or_small_tree/P2@0.8/recall_H/pooled | 0.086 | 0.001 | 0.001 | 0.002 | 0.002 | True |
| bush_or_small_tree/P2@0.8/recall_H/mean | 0.238 | 0.011 | 0.012 | 0.023 | 0.029 | True |
| bush_or_small_tree/P2@0.8/precision_H/pooled | 0.047 | 0.006 | 0.100 | 0.013 | 0.225 | False |
| bush_or_small_tree/P2@0.8/precision_H/mean | 0.102 | 0.020 | 0.134 | 0.055 | 0.313 | False |
| bush_or_small_tree/P2@0.8/iou_H/pooled | 0.080 | 0.001 | 0.001 | 0.002 | 0.002 | True |
| bush_or_small_tree/P2@0.8/iou_H/mean | 0.094 | 0.005 | 0.009 | 0.012 | 0.022 | True |
| bush_or_small_tree/hidden/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| bush_or_small_tree/hidden/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| bush_or_small_tree/P1/recall_H/pooled | 0.000 | 0.003 | 0.004 | 0.008 | 0.009 | False |
| bush_or_small_tree/P1/recall_H/mean | 0.006 | 0.030 | 0.010 | 0.071 | 0.026 | False |

GT cells occupied in every repeat of one arm and none of the other: {"bush_or_small_tree": {"first_only": {"blocked": 590, "no_return": 70, "passed_through": 0, "seen": 79, "unobserved": 9}, "second_only": {"blocked": 5, "no_return": 0, "passed_through": 0, "seen": 0, "unobserved": 0}}, "obstacles": {"first_only": {"blocked": 995, "no_return": 233, "passed_through": 0, "seen": 321, "unobserved": 9}, "second_only": {"blocked": 50, "no_return": 0, "passed_through": 0, "seen": 21, "unobserved": 0}}, "rock": {"first_only": {"blocked": 5, "no_return": 0, "passed_through": 0, "seen": 1, "unobserved": 0}, "second_only": {"blocked": 0, "no_return": 0, "passed_through": 0, "seen": 0, "unobserved": 0}}, "tree_trunk": {"first_only": {"blocked": 400, "no_return": 163, "passed_through": 0, "seen": 241, "unobserved": 0}, "second_only": {"blocked": 45, "no_return": 0, "passed_through": 0, "seen": 21, "unobserved": 0}}}

## Guard: image quality (test + guard frames)

| metric | may_scene_a | may_scene_a_radar_init | may_scene_a_radar_init_loss |
|---|---|---|---|
| image/psnr | 18.215 ± 0.050 (n=5) | 17.849 ± 0.061 (n=5) | 18.118 ± 0.042 (n=5) |
| image/ssim | 0.535 ± 0.000 (n=5) | 0.531 ± 0.001 (n=5) | 0.523 ± 0.000 (n=5) |
| image/lpips | 0.637 ± 0.001 (n=5) | 0.653 ± 0.002 (n=5) | 0.602 ± 0.001 (n=5) |

## Report only: lidar render

| metric | may_scene_a | may_scene_a_radar_init | may_scene_a_radar_init_loss |
|---|---|---|---|
| lidar_render/chamfer_distance | 1.039 ± 0.039 (n=5) | 1.808 ± 0.113 (n=5) | 0.978 ± 0.010 (n=5) |
| lidar_render/median_chamfer_distance | 1.012 ± 0.051 (n=5) | 1.724 ± 0.132 (n=5) | 1.054 ± 0.312 (n=5) |
| lidar_render/depth_median_l2 | 0.004 ± 0.000 (n=5) | 0.005 ± 0.000 (n=5) | 0.006 ± 0.000 (n=5) |
| lidar_render/depth_mean_rel_l2 | 0.277 ± 0.122 (n=5) | 0.125 ± 0.012 (n=5) | 0.137 ± 0.010 (n=5) |

