# P1/P2 over repeats: june_tall_grass

## Definitions

Plain words first, the formula last. Every number is read from the run reports, never typed in.

1. **Cells.** The scene is cut into small cubes ("cells") of edge 0.521933 m. The edge is half the median trunk width of the session's boxes (width = mean of a box's two horizontal sizes), so a typical trunk box is about two cells across, and the trunk margin below equals the cell edge. Everything below is about cells.
2. **Ground truth, "on an obstacle" and "next to an obstacle".** People drew a box around every tree trunk, rock and bush ("obstacles"). The boxes are used only for scoring, never for training.
   - **On an obstacle (G):** a cell whose **centre** lies inside an obstacle box.
   - **The scored area around a class (N_c):** for each class (trunk, rock, bush) separately:
     1. **Margin:** m_c = half of the class's typical footprint width, i.e. the median over all of the session's boxes of that class of (length + width) / 4. Values for session june_tall_grass: trunk 0.522 m, rock 0.665 m, bush 0.953 m.
     2. **Grow:** widen every box of the class by m_c on each of its four sides, along the box's own axes, leaving its height unchanged. A typical object's grown box is therefore about twice its own width.
     3. **Collect:** N_c = the cells whose centre lies inside a grown box.
   - **Only space the lidar could have covered is scored:** a box counts only when its centre is within 40.18 m (the distance within which 99.99% of the session's lidar returns lie) and the vertical field of view (±44.8°) of at least one recorded sweep. The same range and field-of-view test is applied to every cell of N_c.
   - **Obstacles as one group:** N = N_trunk ∪ N_rock ∪ N_bush, minus any cell whose centre lies in a box of a different, non-obstacle annotated class. Those cells count neither for nor against.
   - **Next to an obstacle:** N ∖ G, the cells in the grown area that are not inside any obstacle box.
   - **Scored:** only cells in N; everything else in the scene is ignored.
   - **Caveat:** the boxes are rectangular and a cell counts by its centre only. So a trunk narrower than the typical one can get one cell across or none, depending on where the grid lines fall.
3. **What the model contains.** A trained SplatAD model is millions of small fuzzy blobs (3D Gaussians). Each blob has an opacity from 0 (see-through) to 1 (solid).
4. **Occupancy of a cell** is how solid the model says the cell is, from 0 (empty) to 1 (solid).
   - Each blob adds its opacity to the cell its centre lies in, and a fading share to nearby cells.
   - Blobs combine like stacked tinted glass: light passes through a blob of opacity α with probability (1 − α). So occupancy = 1 − ∏_k(1 − α_k), which grows with every blob but never passes 1.
   - **Why opacities are combined this way (the source of the definition).** The rule is not new. It reads, at a cell, the same accumulation SplatAD uses to render a lidar ray. The design choice is recorded in PLAN §5 (P2) and `docs/research/evaluation_caveats.md`, "Blocking probability". Quotes are verbatim from the sources below.
     1. **How SplatAD blends Gaussians along a ray.** SplatAD (Hess et al., *SplatAD: Real-Time Lidar and Camera Rendering with 3D Gaussian Splatting for Autonomous Driving*, arXiv:2411.16816, https://arxiv.org/abs/2411.16816, Eqs. 4-5 (section 3.2), median-range rule (section 3.3); passages checked 2026-11-25) models the scene as "translucent 3D Gaussians with learnable occupancy o∈(0,1)" (§3.1).
        - **Eq. 4 (§3.2):** [F_p^rgb, F_p] = Σ_i [f_i^rgb, f_i] α_i ∏_{j=1}^{i−1}(1 − α_j). Each Gaussian's own opacity α_i is multiplied by ∏_{j<i}(1 − α_j), the light still left in front of it (the transmittance).
        - **Eq. 5 (§3.2):** α_i = sqrt(|Σ_i^I| / |Σ_i^I + s·I|) · o_i · exp(−½ Δ_iᵀ (Σ_i^I + s·I)⁻¹ Δ_i). The Gaussian's α_i is its learned opacity o_i times its falloff at the ray, projected onto the sensor plane.
        - **§3.3:** "For each lidar point, we α-blend the Gaussians' features 𝐟_i using Eqs. 4, 5 and 6." So the lidar uses exactly these weights.
     2. **SplatAD's own 0.5 rule, whose value we reuse.** SplatAD places a rendered lidar point at the median range, "the rolling shutter corrected range of the first Gaussian that satisfies ∏_{j=1}^{i}(1−α_j) < 0.5 during the α-blending in Eq. 4" (§3.3). It is used at inference: "While we use the expected range for training, the median range is used during inference as it, in contrast to the expected range, does not yield depths between Gaussians." (§3.3). In words: SplatAD says a ray has hit something once **1 − ∏(1 − α) passes 0.5**. Our occupancy combines opacities with the same formula, but over the Gaussians of a cell instead of along one ray. Our binding cut-off reuses that value, 0.5. The cut-off is the same, but "the cell is filled" does not mean "SplatAD's renderer would consider a ray stopped": the occupancy of a cell is a more generous call than a single ray's accumulation (see "How ours differs" below).
     3. **The lineage.** 3D Gaussian Splatting (Kerbl et al., *3D Gaussian Splatting for Real-Time Radiance Field Rendering*, arXiv:2308.04079, https://arxiv.org/abs/2308.04079, section 2.3, Eqs. 1-3 (ACM TOG 42(4), 2023); passages checked 2026-11-25) gives the same blending, Eq. 3: C = Σ_{i∈N} c_i α_i ∏_{j=1}^{i−1}(1 − α_j). It states: "Point-based α-blending and NeRF-style volumetric rendering share essentially the same image formation model." (§2.3). So ∏(1 − α_j) is the transmittance T_i of volume rendering (its Eqs. 1–2).
     4. **Why 1 − ∏(1 − α), as one number per cell.** The blending weights along a ray always add up to Σ_i α_i ∏_{j<i}(1 − α_j) = 1 − ∏_i(1 − α_i). Proof by telescoping: with T_k = ∏_{i≤k}(1 − α_i) and T_0 = 1, each weight equals T_{k−1} − T_k, so the sum is T_0 − T_n = 1 − T_n. A unit test checks this identity. The product does not depend on the order of the Gaussians, so no particular ray direction is needed. The result is the total opacity a ray collects from these Gaussians, between 0 and 1.
   - **How ours differs from SplatAD's α (stated openly):**
     - SplatAD evaluates each Gaussian **projected** onto the image or lidar plane, with a small blur term s·I that matches the pixel size or beam divergence (Eq. 5). We evaluate the **3D** Gaussian at a point: α_k(x) = σ(o_k) · exp(−½ (x − μ_k)ᵀ Σ_k⁻¹ (x − μ_k)), at the cell centre, or at its own peak for the cell holding its mean (T81).
     - We combine every Gaussian touching the cell, rather than those along one beam.
     - In a Gaussian's own cell, its peak opacity counts (T81), and the peaks of several Gaussians sit at different points of the cell. No single ray through the cell meets all of them at their peaks, so these Gaussians count for more than one ray through the cell would collect from them. "The cell is filled" is therefore a more generous call than "SplatAD's renderer would consider a ray stopped". The same rule is applied to every arm, so arm comparisons are unaffected.
     - Occupancy is not the fraction of the cell that is filled. Like the renderer, it assumes each Gaussian blocks light independently.
   - **General reference, not the source of our rule.** Using 3D Gaussians to predict 3D occupancy has been done before (Huang et al., *GaussianFormer: Scene as Gaussians for Vision-Based 3D Semantic Occupancy Prediction*, arXiv:2405.17429, https://arxiv.org/abs/2405.17429, section 3.3, Gaussian-to-Voxel Splatting (general reference only); passages checked 2026-11-25). It adds up the Gaussians' contributions at voxel positions. Ours combines opacities with the renderer's rule above instead, so that the result stays in [0, 1] and reuses SplatAD's cut-off value.
   - It is read from the blobs directly, not from a rendered image, so something behind grass that no sensor saw still counts.
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
   - **(e) Worked example.** The stock arm's drawn run in june_tall_grass at t = 0.5, with its real counts: 4320 hits, 8367 missed, 1176 false alarms → recall 0.341, precision 0.786, IoU 0.312. These are one run; the tables give mean ± std over the repeats.
   - **(f) Pooled vs mean over objects.** Recall, precision and IoU are each reported in two ways. They differ only in how the objects are combined.
     - **Pooled** (T94): add up the cells of all objects first, then divide once. Pooled recall = Σ_k |O_t ∩ G_k| / Σ_k |G_k| = all hits / all obstacle cells; pooled precision = all hits / all filled scored cells. Every *cell* weighs the same, so objects with many cells (tall trunks, big bushes) dominate.
     - **Mean over objects** (T95): score each object k on its own, inside its own grown box N_k with its own obstacle cells G_k, then take the plain average: mean recall = (1/K) Σ_k recall_k, and likewise precision and IoU. Every *object* weighs the same, however small. K counts only the objects where the ratio is defined (a non-zero denominator): an object with no filled cell in its box has no precision and is left out of the precision mean. A filled cell inside several objects' grown boxes counts as a false alarm of each of them. Likewise, inside object k's grown box, a filled cell on a different obstacle (for example a bush beside the trunk) counts as a false alarm of k, because k's own cells are only those inside k's box.
     - **Example (made-up numbers, for illustration only):** a large trunk with 90 obstacle cells, 30 of them filled and a small rock with 10 obstacle cells, 8 of them filled. The per-object recalls are 0.333 and 0.800. Pooled recall = 0.380, dominated by the large trunk; mean recall = 0.567, where the small rock counts as much as the trunk.
     - **Reading the two together:** for recall, when objects share no cells, the pooled value is the average of the per-object recalls weighted by each object's cell count. So a mean above the pooled value means objects with few cells are recovered better, relative to their size, than objects with many cells; a mean below means the large objects carry the score.
   - **(g) Why precision stays below 1.** "Next to an obstacle" is not "empty space". The ring also holds things nobody labelled: grass (tall in June), ground and small plants, which really are solid. It also catches the Gaussians of an obstacle that spill past its box into the neighbouring cells (cells count by their centre). So precision stays below 1 even for a correct model. This is a likely cause, not a tested finding.
   - **(h) How the threshold changes both.** Raising t moves cells from filled to empty, so recall can only fall; precision usually rises.
6. **Statistics.** Each arm is trained 5 times. A table cell shows mean ± std over the repeats. Gap = difference of means. "sep" means the two arms' [min, max] ranges do not overlap. A variant gap counts only when it exceeds the spread between repeats without losing precision.

## Whole scene

### obstacles, P2 occupied at >= 0.5

| metric | stock (`june_scene_a_cam`) | radar init (`june_scene_a_radar_init_cam`) | radar init + loss (`june_scene_a_radar_init_loss_cam`) |
|---|---|---|---|
| recall | 0.342 ± 0.006 [0.336, 0.351] (n=5) | 0.312 ± 0.002 [0.310, 0.314] (n=5) | 0.380 ± 0.003 [0.376, 0.383] (n=5) |
| precision | 0.789 ± 0.005 [0.783, 0.796] (n=5) | 0.784 ± 0.004 [0.778, 0.788] (n=5) | 0.796 ± 0.004 [0.790, 0.801] (n=5) |
| IoU | 0.313 ± 0.005 [0.309, 0.320] (n=5) | 0.287 ± 0.002 [0.285, 0.290] (n=5) | 0.346 ± 0.003 [0.342, 0.349] (n=5) |

### obstacles, P2 occupied at >= 0.8

| metric | stock (`june_scene_a_cam`) | radar init (`june_scene_a_radar_init_cam`) | radar init + loss (`june_scene_a_radar_init_loss_cam`) |
|---|---|---|---|
| recall | 0.304 ± 0.002 [0.302, 0.307] (n=5) | 0.276 ± 0.002 [0.274, 0.279] (n=5) | 0.338 ± 0.002 [0.335, 0.341] (n=5) |
| precision | 0.813 ± 0.005 [0.806, 0.821] (n=5) | 0.803 ± 0.005 [0.798, 0.807] (n=5) | 0.823 ± 0.003 [0.820, 0.826] (n=5) |
| IoU | 0.284 ± 0.002 [0.282, 0.286] (n=5) | 0.259 ± 0.002 [0.256, 0.261] (n=5) | 0.315 ± 0.002 [0.312, 0.318] (n=5) |

### tree_trunk, P2 occupied at >= 0.5

| metric | stock (`june_scene_a_cam`) | radar init (`june_scene_a_radar_init_cam`) | radar init + loss (`june_scene_a_radar_init_loss_cam`) |
|---|---|---|---|
| recall | 0.435 ± 0.004 [0.431, 0.440] (n=5) | 0.417 ± 0.002 [0.415, 0.419] (n=5) | 0.467 ± 0.004 [0.461, 0.471] (n=5) |
| precision | 0.828 ± 0.003 [0.825, 0.833] (n=5) | 0.823 ± 0.004 [0.817, 0.828] (n=5) | 0.834 ± 0.004 [0.829, 0.839] (n=5) |
| IoU | 0.399 ± 0.003 [0.397, 0.403] (n=5) | 0.382 ± 0.002 [0.380, 0.384] (n=5) | 0.427 ± 0.004 [0.421, 0.430] (n=5) |

### tree_trunk, P2 occupied at >= 0.8

| metric | stock (`june_scene_a_cam`) | radar init (`june_scene_a_radar_init_cam`) | radar init + loss (`june_scene_a_radar_init_loss_cam`) |
|---|---|---|---|
| recall | 0.397 ± 0.003 [0.394, 0.401] (n=5) | 0.376 ± 0.002 [0.373, 0.379] (n=5) | 0.429 ± 0.002 [0.426, 0.431] (n=5) |
| precision | 0.845 ± 0.003 [0.843, 0.850] (n=5) | 0.838 ± 0.004 [0.832, 0.842] (n=5) | 0.847 ± 0.002 [0.845, 0.851] (n=5) |
| IoU | 0.370 ± 0.002 [0.367, 0.374] (n=5) | 0.350 ± 0.003 [0.347, 0.353] (n=5) | 0.398 ± 0.002 [0.395, 0.401] (n=5) |

### rock, P2 occupied at >= 0.5

| metric | stock (`june_scene_a_cam`) | radar init (`june_scene_a_radar_init_cam`) | radar init + loss (`june_scene_a_radar_init_loss_cam`) |
|---|---|---|---|
| recall | 0.198 ± 0.015 [0.176, 0.212] (n=5) | 0.167 ± 0.017 [0.141, 0.188] (n=5) | 0.266 ± 0.006 [0.259, 0.271] (n=5) |
| precision | 0.210 ± 0.024 [0.179, 0.234] (n=5) | 0.209 ± 0.017 [0.185, 0.232] (n=5) | 0.219 ± 0.005 [0.214, 0.228] (n=5) |
| IoU | 0.113 ± 0.012 [0.097, 0.125] (n=5) | 0.102 ± 0.011 [0.087, 0.116] (n=5) | 0.136 ± 0.003 [0.133, 0.141] (n=5) |

### rock, P2 occupied at >= 0.8

| metric | stock (`june_scene_a_cam`) | radar init (`june_scene_a_radar_init_cam`) | radar init + loss (`june_scene_a_radar_init_loss_cam`) |
|---|---|---|---|
| recall | 0.186 ± 0.013 [0.176, 0.200] (n=5) | 0.153 ± 0.019 [0.129, 0.176] (n=5) | 0.240 ± 0.011 [0.224, 0.247] (n=5) |
| precision | 0.224 ± 0.017 [0.203, 0.239] (n=5) | 0.227 ± 0.023 [0.200, 0.263] (n=5) | 0.216 ± 0.006 [0.207, 0.221] (n=5) |
| IoU | 0.113 ± 0.009 [0.104, 0.122] (n=5) | 0.101 ± 0.012 [0.085, 0.118] (n=5) | 0.128 ± 0.005 [0.120, 0.132] (n=5) |

### bush_or_small_tree, P2 occupied at >= 0.5

| metric | stock (`june_scene_a_cam`) | radar init (`june_scene_a_radar_init_cam`) | radar init + loss (`june_scene_a_radar_init_loss_cam`) |
|---|---|---|---|
| recall | 0.144 ± 0.020 [0.132, 0.180] (n=5) | 0.090 ± 0.007 [0.084, 0.099] (n=5) | 0.196 ± 0.008 [0.187, 0.207] (n=5) |
| precision | 0.640 ± 0.015 [0.624, 0.657] (n=5) | 0.575 ± 0.020 [0.550, 0.596] (n=5) | 0.688 ± 0.022 [0.661, 0.719] (n=5) |
| IoU | 0.133 ± 0.017 [0.122, 0.163] (n=5) | 0.084 ± 0.006 [0.078, 0.092] (n=5) | 0.180 ± 0.008 [0.170, 0.192] (n=5) |

### bush_or_small_tree, P2 occupied at >= 0.8

| metric | stock (`june_scene_a_cam`) | radar init (`june_scene_a_radar_init_cam`) | radar init + loss (`june_scene_a_radar_init_loss_cam`) |
|---|---|---|---|
| recall | 0.106 ± 0.008 [0.101, 0.121] (n=5) | 0.065 ± 0.003 [0.062, 0.071] (n=5) | 0.145 ± 0.005 [0.140, 0.152] (n=5) |
| precision | 0.671 ± 0.016 [0.652, 0.694] (n=5) | 0.579 ± 0.011 [0.567, 0.597] (n=5) | 0.760 ± 0.015 [0.740, 0.782] (n=5) |
| IoU | 0.101 ± 0.007 [0.096, 0.113] (n=5) | 0.062 ± 0.003 [0.060, 0.067] (n=5) | 0.139 ± 0.005 [0.134, 0.146] (n=5) |

### Gap radar init (`june_scene_a_radar_init_cam`) - stock (`june_scene_a_cam`) (sep = the arms' [min, max] do not overlap, ovl = they overlap)

| metric | gap | sep / ovl |
|---|---|---|
| obstacles/P2@0.5/recall/pooled | -0.030 | sep |
| obstacles/P2@0.5/precision/pooled | -0.005 | ovl |
| obstacles/P2@0.5/iou/pooled | -0.026 | sep |
| obstacles/P2@0.8/recall/pooled | -0.028 | sep |
| obstacles/P2@0.8/precision/pooled | -0.010 | ovl |
| obstacles/P2@0.8/iou/pooled | -0.025 | sep |
| tree_trunk/P2@0.5/recall/pooled | -0.019 | sep |
| tree_trunk/P2@0.5/precision/pooled | -0.006 | ovl |
| tree_trunk/P2@0.5/iou/pooled | -0.017 | sep |
| tree_trunk/P2@0.8/recall/pooled | -0.022 | sep |
| tree_trunk/P2@0.8/precision/pooled | -0.008 | sep |
| tree_trunk/P2@0.8/iou/pooled | -0.020 | sep |
| rock/P2@0.5/recall/pooled | -0.031 | ovl |
| rock/P2@0.5/precision/pooled | -0.001 | ovl |
| rock/P2@0.5/iou/pooled | -0.011 | ovl |
| rock/P2@0.8/recall/pooled | -0.033 | ovl |
| rock/P2@0.8/precision/pooled | 0.003 | ovl |
| rock/P2@0.8/iou/pooled | -0.012 | ovl |
| bush_or_small_tree/P2@0.5/recall/pooled | -0.054 | sep |
| bush_or_small_tree/P2@0.5/precision/pooled | -0.064 | sep |
| bush_or_small_tree/P2@0.5/iou/pooled | -0.048 | sep |
| bush_or_small_tree/P2@0.8/recall/pooled | -0.041 | sep |
| bush_or_small_tree/P2@0.8/precision/pooled | -0.092 | sep |
| bush_or_small_tree/P2@0.8/iou/pooled | -0.039 | sep |

### Gap radar init + loss (`june_scene_a_radar_init_loss_cam`) - stock (`june_scene_a_cam`) (sep = the arms' [min, max] do not overlap, ovl = they overlap)

| metric | gap | sep / ovl |
|---|---|---|
| obstacles/P2@0.5/recall/pooled | 0.038 | sep |
| obstacles/P2@0.5/precision/pooled | 0.007 | ovl |
| obstacles/P2@0.5/iou/pooled | 0.033 | sep |
| obstacles/P2@0.8/recall/pooled | 0.034 | sep |
| obstacles/P2@0.8/precision/pooled | 0.010 | ovl |
| obstacles/P2@0.8/iou/pooled | 0.031 | sep |
| tree_trunk/P2@0.5/recall/pooled | 0.031 | sep |
| tree_trunk/P2@0.5/precision/pooled | 0.005 | ovl |
| tree_trunk/P2@0.5/iou/pooled | 0.028 | sep |
| tree_trunk/P2@0.8/recall/pooled | 0.032 | sep |
| tree_trunk/P2@0.8/precision/pooled | 0.002 | ovl |
| tree_trunk/P2@0.8/iou/pooled | 0.028 | sep |
| rock/P2@0.5/recall/pooled | 0.068 | sep |
| rock/P2@0.5/precision/pooled | 0.009 | ovl |
| rock/P2@0.5/iou/pooled | 0.023 | sep |
| rock/P2@0.8/recall/pooled | 0.054 | sep |
| rock/P2@0.8/precision/pooled | -0.008 | ovl |
| rock/P2@0.8/iou/pooled | 0.015 | ovl |
| bush_or_small_tree/P2@0.5/recall/pooled | 0.052 | sep |
| bush_or_small_tree/P2@0.5/precision/pooled | 0.048 | sep |
| bush_or_small_tree/P2@0.5/iou/pooled | 0.047 | sep |
| bush_or_small_tree/P2@0.8/recall/pooled | 0.039 | sep |
| bush_or_small_tree/P2@0.8/precision/pooled | 0.088 | sep |
| bush_or_small_tree/P2@0.8/iou/pooled | 0.038 | sep |

### Gap radar init + loss (`june_scene_a_radar_init_loss_cam`) - radar init (`june_scene_a_radar_init_cam`) (sep = the arms' [min, max] do not overlap, ovl = they overlap)

| metric | gap | sep / ovl |
|---|---|---|
| obstacles/P2@0.5/recall/pooled | 0.068 | sep |
| obstacles/P2@0.5/precision/pooled | 0.012 | sep |
| obstacles/P2@0.5/iou/pooled | 0.059 | sep |
| obstacles/P2@0.8/recall/pooled | 0.062 | sep |
| obstacles/P2@0.8/precision/pooled | 0.020 | sep |
| obstacles/P2@0.8/iou/pooled | 0.057 | sep |
| tree_trunk/P2@0.5/recall/pooled | 0.050 | sep |
| tree_trunk/P2@0.5/precision/pooled | 0.011 | sep |
| tree_trunk/P2@0.5/iou/pooled | 0.045 | sep |
| tree_trunk/P2@0.8/recall/pooled | 0.053 | sep |
| tree_trunk/P2@0.8/precision/pooled | 0.010 | sep |
| tree_trunk/P2@0.8/iou/pooled | 0.048 | sep |
| rock/P2@0.5/recall/pooled | 0.099 | sep |
| rock/P2@0.5/precision/pooled | 0.010 | ovl |
| rock/P2@0.5/iou/pooled | 0.034 | sep |
| rock/P2@0.8/recall/pooled | 0.087 | sep |
| rock/P2@0.8/precision/pooled | -0.011 | ovl |
| rock/P2@0.8/iou/pooled | 0.028 | sep |
| bush_or_small_tree/P2@0.5/recall/pooled | 0.106 | sep |
| bush_or_small_tree/P2@0.5/precision/pooled | 0.113 | sep |
| bush_or_small_tree/P2@0.5/iou/pooled | 0.095 | sep |
| bush_or_small_tree/P2@0.8/recall/pooled | 0.080 | sep |
| bush_or_small_tree/P2@0.8/precision/pooled | 0.181 | sep |
| bush_or_small_tree/P2@0.8/iou/pooled | 0.077 | sep |

![Whole scene in 3D, per arm](../figures/june_tall_grass/arms_scene_3d.png)

**How to read this figure**
- **GT panel:** the obstacle boxes, coloured by class, for orientation.
- **Arm panels:** green = hit (filled, on an obstacle); red = false alarm (filled, next to an obstacle); grey = missed (empty, on an obstacle); faint grey = the scored area, the ring included. Correct-empty cells are not drawn. Cut-off t = 0.5. The terms are defined in point 5.
- **Difference panel:** hits only radar init + loss has, hits only stock has, and hits both have.
- One run per arm: the median repeat of whole-scene pooled recall at the binding cut-off t = 0.5 (metric `obstacles/P2@0.5/recall/pooled`), so it illustrates and does not decide anything.


![Recall and precision against the occupancy threshold](../figures/june_tall_grass/threshold_curves_scene.png)

**How to read this figure**
- **What is plotted:** recall (left) and precision (right) recomputed at each cut-off t from 0.2 to 0.9 (15 evenly spaced values). Solid lines are pooled and dashed lines are the mean over objects. The line is the median repeat. Each line has its own shaded band: at each t, the lowest and highest value of that same curve (pooled or mean over objects) over the 5 repeats of the arm. The pooled band is shaded more strongly than the mean band. Dotted lines mark the table cut-offs 0.5, 0.8.
- **Solid vs dashed:** solid = *pooled* (all objects' cells counted together, so large objects dominate); dashed = *mean over objects* (each object scored on its own, then averaged, so every object counts once). Full definitions and an example: point 5(f).
- **Definitions:** see point 5. Only recall and precision are plotted; IoU is in the tables. Recall = hit / (hit + missed) has all 12687 obstacle cells as its denominator; precision = hit / (hit + false alarm) has every filled scored cell as its denominator.
- **How the curves move:** raising t fills fewer cells, so recall can only fall and precision usually rises.
- **When one arm beats another:** its line is higher and its band does not overlap the other arm's band of the same kind (pooled with pooled, mean with mean).
- **Why t runs from 0.2 to 0.9:** the extremes are left out because they say nothing about the model. At t = 0 every cell counts as filled (recall 1, precision = the share of obstacle cells). At t ≈ 1 precision rests on a handful of cells and jumps.




Descriptive only: a variant gap counts only when it exceeds the spread between repeats without losing precision. Repeats share one seed, so they are not paired. Cells are mean ± sample std (n = repeats).

- **june_scene_a_cam** (`splatad`): june_tall_grass-june_scene_a_cam-20261116-134222-17bb7a1 [17bb7a1; split identical to the reference record], june_tall_grass-june_scene_a_cam-20261116-135429-17bb7a1 [17bb7a1; split identical to the reference record], june_tall_grass-june_scene_a_cam-20261116-135432-17bb7a1 [17bb7a1; split identical to the reference record], june_tall_grass-june_scene_a_cam-20261116-135435-17bb7a1 [17bb7a1; split identical to the reference record], june_tall_grass-june_scene_a_cam-20261118-032048-c28a956 [c28a956; split identical to the reference record]
- **june_scene_a_radar_init_cam** (`splatad-radar-init`): june_tall_grass-june_scene_a_radar_init_cam-20261116-134824-17bb7a1 [17bb7a1; split identical to the reference record], june_tall_grass-june_scene_a_radar_init_cam-20261116-135427-17bb7a1 [17bb7a1; split identical to the reference record], june_tall_grass-june_scene_a_radar_init_cam-20261116-135430-17bb7a1 [17bb7a1; split identical to the reference record], june_tall_grass-june_scene_a_radar_init_cam-20261116-135433-17bb7a1 [17bb7a1; split identical to the reference record], june_tall_grass-june_scene_a_radar_init_cam-20261116-135436-17bb7a1 [17bb7a1; split identical to the reference record]
- **june_scene_a_radar_init_loss_cam** (`splatad-radar-init-lidar`): june_tall_grass-june_scene_a_radar_init_loss_cam-20261116-135425-17bb7a1 [17bb7a1; split identical to the reference record], june_tall_grass-june_scene_a_radar_init_loss_cam-20261116-135428-17bb7a1 [17bb7a1; split identical to the reference record], june_tall_grass-june_scene_a_radar_init_loss_cam-20261116-135431-17bb7a1 [17bb7a1; split identical to the reference record], june_tall_grass-june_scene_a_radar_init_loss_cam-20261116-135437-17bb7a1 [17bb7a1; split identical to the reference record], june_tall_grass-june_scene_a_radar_init_loss_cam-20261118-211540-c28a956 [c28a956; split identical to the reference record]

## obstacles, P2 occupied at >= 0.5

| metric | june_scene_a_cam | june_scene_a_radar_init_cam | june_scene_a_radar_init_loss_cam |
|---|---|---|---|
| P2@0.5/recall/pooled | 0.342 ± 0.006 (n=5) | 0.312 ± 0.002 (n=5) | 0.380 ± 0.003 (n=5) |
| P2@0.5/recall/mean | 0.395 ± 0.004 (n=5) | 0.357 ± 0.001 (n=5) | 0.435 ± 0.002 (n=5) |
| P2@0.5/precision/pooled | 0.789 ± 0.005 (n=5) | 0.784 ± 0.004 (n=5) | 0.796 ± 0.004 (n=5) |
| P2@0.5/precision/mean | 0.646 ± 0.004 (n=5) | 0.624 ± 0.015 (n=5) | 0.677 ± 0.007 (n=5) |
| P2@0.5/iou/pooled | 0.313 ± 0.005 (n=5) | 0.287 ± 0.002 (n=5) | 0.346 ± 0.003 (n=5) |
| P2@0.5/iou/mean | 0.305 ± 0.003 (n=5) | 0.279 ± 0.002 (n=5) | 0.337 ± 0.002 (n=5) |
| P1/recall/pooled | 0.322 ± 0.003 (n=5) | 0.311 ± 0.003 (n=5) | 0.329 ± 0.003 (n=5) |
| P1/recall/mean | 0.363 ± 0.004 (n=5) | 0.352 ± 0.002 (n=5) | 0.369 ± 0.002 (n=5) |
| P1/precision/pooled | 0.644 ± 0.003 (n=5) | 0.640 ± 0.003 (n=5) | 0.644 ± 0.002 (n=5) |
| P1/precision/mean | 0.561 ± 0.010 (n=5) | 0.575 ± 0.009 (n=5) | 0.575 ± 0.009 (n=5) |


## obstacles, P2 occupied at >= 0.8

| metric | june_scene_a_cam | june_scene_a_radar_init_cam | june_scene_a_radar_init_loss_cam |
|---|---|---|---|
| P2@0.8/recall/pooled | 0.304 ± 0.002 (n=5) | 0.276 ± 0.002 (n=5) | 0.338 ± 0.002 (n=5) |
| P2@0.8/recall/mean | 0.358 ± 0.001 (n=5) | 0.320 ± 0.002 (n=5) | 0.398 ± 0.002 (n=5) |
| P2@0.8/precision/pooled | 0.813 ± 0.005 (n=5) | 0.803 ± 0.005 (n=5) | 0.823 ± 0.003 (n=5) |
| P2@0.8/precision/mean | 0.663 ± 0.011 (n=5) | 0.632 ± 0.016 (n=5) | 0.702 ± 0.010 (n=5) |
| P2@0.8/iou/pooled | 0.284 ± 0.002 (n=5) | 0.259 ± 0.002 (n=5) | 0.315 ± 0.002 (n=5) |
| P2@0.8/iou/mean | 0.284 ± 0.001 (n=5) | 0.258 ± 0.002 (n=5) | 0.314 ± 0.003 (n=5) |
| P1/recall/pooled | 0.322 ± 0.003 (n=5) | 0.311 ± 0.003 (n=5) | 0.329 ± 0.003 (n=5) |
| P1/recall/mean | 0.363 ± 0.004 (n=5) | 0.352 ± 0.002 (n=5) | 0.369 ± 0.002 (n=5) |
| P1/precision/pooled | 0.644 ± 0.003 (n=5) | 0.640 ± 0.003 (n=5) | 0.644 ± 0.002 (n=5) |
| P1/precision/mean | 0.561 ± 0.010 (n=5) | 0.575 ± 0.009 (n=5) | 0.575 ± 0.009 (n=5) |


## tree_trunk, P2 occupied at >= 0.5

| metric | june_scene_a_cam | june_scene_a_radar_init_cam | june_scene_a_radar_init_loss_cam |
|---|---|---|---|
| P2@0.5/recall/pooled | 0.435 ± 0.004 (n=5) | 0.417 ± 0.002 (n=5) | 0.467 ± 0.004 (n=5) |
| P2@0.5/recall/mean | 0.461 ± 0.007 (n=5) | 0.432 ± 0.003 (n=5) | 0.491 ± 0.004 (n=5) |
| P2@0.5/precision/pooled | 0.828 ± 0.003 (n=5) | 0.823 ± 0.004 (n=5) | 0.834 ± 0.004 (n=5) |
| P2@0.5/precision/mean | 0.714 ± 0.009 (n=5) | 0.703 ± 0.013 (n=5) | 0.749 ± 0.004 (n=5) |
| P2@0.5/iou/pooled | 0.399 ± 0.003 (n=5) | 0.382 ± 0.002 (n=5) | 0.427 ± 0.004 (n=5) |
| P2@0.5/iou/mean | 0.369 ± 0.005 (n=5) | 0.346 ± 0.004 (n=5) | 0.396 ± 0.004 (n=5) |
| P1/recall/pooled | 0.408 ± 0.004 (n=5) | 0.399 ± 0.003 (n=5) | 0.416 ± 0.002 (n=5) |
| P1/recall/mean | 0.424 ± 0.003 (n=5) | 0.414 ± 0.003 (n=5) | 0.431 ± 0.002 (n=5) |
| P1/precision/pooled | 0.680 ± 0.003 (n=5) | 0.678 ± 0.004 (n=5) | 0.680 ± 0.002 (n=5) |
| P1/precision/mean | 0.620 ± 0.012 (n=5) | 0.634 ± 0.011 (n=5) | 0.642 ± 0.007 (n=5) |


## tree_trunk, P2 occupied at >= 0.8

| metric | june_scene_a_cam | june_scene_a_radar_init_cam | june_scene_a_radar_init_loss_cam |
|---|---|---|---|
| P2@0.8/recall/pooled | 0.397 ± 0.003 (n=5) | 0.376 ± 0.002 (n=5) | 0.429 ± 0.002 (n=5) |
| P2@0.8/recall/mean | 0.423 ± 0.004 (n=5) | 0.394 ± 0.003 (n=5) | 0.456 ± 0.003 (n=5) |
| P2@0.8/precision/pooled | 0.845 ± 0.003 (n=5) | 0.838 ± 0.004 (n=5) | 0.847 ± 0.002 (n=5) |
| P2@0.8/precision/mean | 0.722 ± 0.009 (n=5) | 0.706 ± 0.012 (n=5) | 0.766 ± 0.007 (n=5) |
| P2@0.8/iou/pooled | 0.370 ± 0.002 (n=5) | 0.350 ± 0.003 (n=5) | 0.398 ± 0.002 (n=5) |
| P2@0.8/iou/mean | 0.346 ± 0.002 (n=5) | 0.323 ± 0.004 (n=5) | 0.372 ± 0.003 (n=5) |
| P1/recall/pooled | 0.408 ± 0.004 (n=5) | 0.399 ± 0.003 (n=5) | 0.416 ± 0.002 (n=5) |
| P1/recall/mean | 0.424 ± 0.003 (n=5) | 0.414 ± 0.003 (n=5) | 0.431 ± 0.002 (n=5) |
| P1/precision/pooled | 0.680 ± 0.003 (n=5) | 0.678 ± 0.004 (n=5) | 0.680 ± 0.002 (n=5) |
| P1/precision/mean | 0.620 ± 0.012 (n=5) | 0.634 ± 0.011 (n=5) | 0.642 ± 0.007 (n=5) |


## rock, P2 occupied at >= 0.5

| metric | june_scene_a_cam | june_scene_a_radar_init_cam | june_scene_a_radar_init_loss_cam |
|---|---|---|---|
| P2@0.5/recall/pooled | 0.198 ± 0.015 (n=5) | 0.167 ± 0.017 (n=5) | 0.266 ± 0.006 (n=5) |
| P2@0.5/recall/mean | 0.207 ± 0.013 (n=5) | 0.187 ± 0.017 (n=5) | 0.282 ± 0.007 (n=5) |
| P2@0.5/precision/pooled | 0.210 ± 0.024 (n=5) | 0.209 ± 0.017 (n=5) | 0.219 ± 0.005 (n=5) |
| P2@0.5/precision/mean | 0.243 ± 0.082 (n=5) | 0.224 ± 0.076 (n=5) | 0.236 ± 0.018 (n=5) |
| P2@0.5/iou/pooled | 0.113 ± 0.012 (n=5) | 0.102 ± 0.011 (n=5) | 0.136 ± 0.003 (n=5) |
| P2@0.5/iou/mean | 0.073 ± 0.011 (n=5) | 0.072 ± 0.016 (n=5) | 0.100 ± 0.003 (n=5) |
| P1/recall/pooled | 0.242 ± 0.018 (n=5) | 0.221 ± 0.017 (n=5) | 0.256 ± 0.019 (n=5) |
| P1/recall/mean | 0.236 ± 0.019 (n=5) | 0.212 ± 0.014 (n=5) | 0.246 ± 0.021 (n=5) |
| P1/precision/pooled | 0.206 ± 0.013 (n=5) | 0.186 ± 0.013 (n=5) | 0.201 ± 0.015 (n=5) |
| P1/precision/mean | 0.214 ± 0.015 (n=5) | 0.195 ± 0.012 (n=5) | 0.210 ± 0.017 (n=5) |


## rock, P2 occupied at >= 0.8

| metric | june_scene_a_cam | june_scene_a_radar_init_cam | june_scene_a_radar_init_loss_cam |
|---|---|---|---|
| P2@0.8/recall/pooled | 0.186 ± 0.013 (n=5) | 0.153 ± 0.019 (n=5) | 0.240 ± 0.011 (n=5) |
| P2@0.8/recall/mean | 0.197 ± 0.012 (n=5) | 0.168 ± 0.019 (n=5) | 0.257 ± 0.011 (n=5) |
| P2@0.8/precision/pooled | 0.224 ± 0.017 (n=5) | 0.227 ± 0.023 (n=5) | 0.216 ± 0.006 (n=5) |
| P2@0.8/precision/mean | 0.286 ± 0.106 (n=5) | 0.322 ± 0.137 (n=5) | 0.275 ± 0.053 (n=5) |
| P2@0.8/iou/pooled | 0.113 ± 0.009 (n=5) | 0.101 ± 0.012 (n=5) | 0.128 ± 0.005 (n=5) |
| P2@0.8/iou/mean | 0.078 ± 0.014 (n=5) | 0.074 ± 0.014 (n=5) | 0.098 ± 0.006 (n=5) |
| P1/recall/pooled | 0.242 ± 0.018 (n=5) | 0.221 ± 0.017 (n=5) | 0.256 ± 0.019 (n=5) |
| P1/recall/mean | 0.236 ± 0.019 (n=5) | 0.212 ± 0.014 (n=5) | 0.246 ± 0.021 (n=5) |
| P1/precision/pooled | 0.206 ± 0.013 (n=5) | 0.186 ± 0.013 (n=5) | 0.201 ± 0.015 (n=5) |
| P1/precision/mean | 0.214 ± 0.015 (n=5) | 0.195 ± 0.012 (n=5) | 0.210 ± 0.017 (n=5) |


## bush_or_small_tree, P2 occupied at >= 0.5

| metric | june_scene_a_cam | june_scene_a_radar_init_cam | june_scene_a_radar_init_loss_cam |
|---|---|---|---|
| P2@0.5/recall/pooled | 0.144 ± 0.020 (n=5) | 0.090 ± 0.007 (n=5) | 0.196 ± 0.008 (n=5) |
| P2@0.5/recall/mean | 0.217 ± 0.010 (n=5) | 0.141 ± 0.013 (n=5) | 0.282 ± 0.009 (n=5) |
| P2@0.5/precision/pooled | 0.640 ± 0.015 (n=5) | 0.575 ± 0.020 (n=5) | 0.688 ± 0.022 (n=5) |
| P2@0.5/precision/mean | 0.436 ± 0.020 (n=5) | 0.391 ± 0.032 (n=5) | 0.491 ± 0.033 (n=5) |
| P2@0.5/iou/pooled | 0.133 ± 0.017 (n=5) | 0.084 ± 0.006 (n=5) | 0.180 ± 0.008 (n=5) |
| P2@0.5/iou/mean | 0.146 ± 0.005 (n=5) | 0.100 ± 0.006 (n=5) | 0.195 ± 0.006 (n=5) |
| P1/recall/pooled | 0.136 ± 0.006 (n=5) | 0.122 ± 0.004 (n=5) | 0.143 ± 0.004 (n=5) |
| P1/recall/mean | 0.184 ± 0.009 (n=5) | 0.170 ± 0.005 (n=5) | 0.180 ± 0.007 (n=5) |
| P1/precision/pooled | 0.508 ± 0.010 (n=5) | 0.491 ± 0.012 (n=5) | 0.516 ± 0.013 (n=5) |
| P1/precision/mean | 0.331 ± 0.011 (n=5) | 0.338 ± 0.019 (n=5) | 0.331 ± 0.014 (n=5) |


## bush_or_small_tree, P2 occupied at >= 0.8

| metric | june_scene_a_cam | june_scene_a_radar_init_cam | june_scene_a_radar_init_loss_cam |
|---|---|---|---|
| P2@0.8/recall/pooled | 0.106 ± 0.008 (n=5) | 0.065 ± 0.003 (n=5) | 0.145 ± 0.005 (n=5) |
| P2@0.8/recall/mean | 0.170 ± 0.007 (n=5) | 0.101 ± 0.011 (n=5) | 0.233 ± 0.005 (n=5) |
| P2@0.8/precision/pooled | 0.671 ± 0.016 (n=5) | 0.579 ± 0.011 (n=5) | 0.760 ± 0.015 (n=5) |
| P2@0.8/precision/mean | 0.474 ± 0.038 (n=5) | 0.396 ± 0.042 (n=5) | 0.536 ± 0.026 (n=5) |
| P2@0.8/iou/pooled | 0.101 ± 0.007 (n=5) | 0.062 ± 0.003 (n=5) | 0.139 ± 0.005 (n=5) |
| P2@0.8/iou/mean | 0.125 ± 0.005 (n=5) | 0.079 ± 0.007 (n=5) | 0.171 ± 0.005 (n=5) |
| P1/recall/pooled | 0.136 ± 0.006 (n=5) | 0.122 ± 0.004 (n=5) | 0.143 ± 0.004 (n=5) |
| P1/recall/mean | 0.184 ± 0.009 (n=5) | 0.170 ± 0.005 (n=5) | 0.180 ± 0.007 (n=5) |
| P1/precision/pooled | 0.508 ± 0.010 (n=5) | 0.491 ± 0.012 (n=5) | 0.516 ± 0.013 (n=5) |
| P1/precision/mean | 0.331 ± 0.011 (n=5) | 0.338 ± 0.019 (n=5) | 0.331 ± 0.014 (n=5) |


## Arm pairs (gap = first - second; separated = the two arms' [min, max] do not overlap)

### june_scene_a_radar_init_cam - june_scene_a_cam

#### obstacles, P2 >= 0.5

| metric | gap | first std | second std | first range | second range | separated |
|---|---|---|---|---|---|---|
| obstacles/P2@0.5/recall/pooled | -0.030 | 0.002 | 0.006 | 0.005 | 0.015 | True |
| obstacles/P2@0.5/recall/mean | -0.038 | 0.001 | 0.004 | 0.003 | 0.010 | True |
| obstacles/P2@0.5/precision/pooled | -0.005 | 0.004 | 0.005 | 0.010 | 0.013 | False |
| obstacles/P2@0.5/precision/mean | -0.022 | 0.015 | 0.004 | 0.039 | 0.010 | False |
| obstacles/P2@0.5/iou/pooled | -0.026 | 0.002 | 0.005 | 0.005 | 0.011 | True |
| obstacles/P2@0.5/iou/mean | -0.026 | 0.002 | 0.003 | 0.006 | 0.007 | True |
| obstacles/P1/recall/pooled | -0.011 | 0.003 | 0.003 | 0.006 | 0.008 | True |
| obstacles/P1/recall/mean | -0.011 | 0.002 | 0.004 | 0.006 | 0.011 | True |
| obstacles/P1/precision/pooled | -0.003 | 0.003 | 0.003 | 0.008 | 0.008 | False |
| obstacles/P1/precision/mean | 0.014 | 0.009 | 0.010 | 0.019 | 0.028 | False |

#### obstacles, P2 >= 0.8

| metric | gap | first std | second std | first range | second range | separated |
|---|---|---|---|---|---|---|
| obstacles/P2@0.8/recall/pooled | -0.028 | 0.002 | 0.002 | 0.005 | 0.005 | True |
| obstacles/P2@0.8/recall/mean | -0.037 | 0.002 | 0.001 | 0.005 | 0.003 | True |
| obstacles/P2@0.8/precision/pooled | -0.010 | 0.005 | 0.005 | 0.009 | 0.015 | False |
| obstacles/P2@0.8/precision/mean | -0.031 | 0.016 | 0.011 | 0.042 | 0.030 | False |
| obstacles/P2@0.8/iou/pooled | -0.025 | 0.002 | 0.002 | 0.005 | 0.004 | True |
| obstacles/P2@0.8/iou/mean | -0.026 | 0.002 | 0.001 | 0.006 | 0.003 | True |
| obstacles/P1/recall/pooled | -0.011 | 0.003 | 0.003 | 0.006 | 0.008 | True |
| obstacles/P1/recall/mean | -0.011 | 0.002 | 0.004 | 0.006 | 0.011 | True |
| obstacles/P1/precision/pooled | -0.003 | 0.003 | 0.003 | 0.008 | 0.008 | False |
| obstacles/P1/precision/mean | 0.014 | 0.009 | 0.010 | 0.019 | 0.028 | False |

#### tree_trunk, P2 >= 0.5

| metric | gap | first std | second std | first range | second range | separated |
|---|---|---|---|---|---|---|
| tree_trunk/P2@0.5/recall/pooled | -0.019 | 0.002 | 0.004 | 0.005 | 0.009 | True |
| tree_trunk/P2@0.5/recall/mean | -0.029 | 0.003 | 0.007 | 0.007 | 0.019 | True |
| tree_trunk/P2@0.5/precision/pooled | -0.006 | 0.004 | 0.003 | 0.011 | 0.008 | False |
| tree_trunk/P2@0.5/precision/mean | -0.010 | 0.013 | 0.009 | 0.036 | 0.023 | False |
| tree_trunk/P2@0.5/iou/pooled | -0.017 | 0.002 | 0.003 | 0.004 | 0.007 | True |
| tree_trunk/P2@0.5/iou/mean | -0.023 | 0.004 | 0.005 | 0.011 | 0.012 | True |
| tree_trunk/P1/recall/pooled | -0.009 | 0.003 | 0.004 | 0.007 | 0.010 | False |
| tree_trunk/P1/recall/mean | -0.009 | 0.003 | 0.003 | 0.006 | 0.009 | True |
| tree_trunk/P1/precision/pooled | -0.002 | 0.004 | 0.003 | 0.011 | 0.008 | False |
| tree_trunk/P1/precision/mean | 0.014 | 0.011 | 0.012 | 0.028 | 0.032 | False |

#### tree_trunk, P2 >= 0.8

| metric | gap | first std | second std | first range | second range | separated |
|---|---|---|---|---|---|---|
| tree_trunk/P2@0.8/recall/pooled | -0.022 | 0.002 | 0.003 | 0.006 | 0.007 | True |
| tree_trunk/P2@0.8/recall/mean | -0.029 | 0.003 | 0.004 | 0.006 | 0.010 | True |
| tree_trunk/P2@0.8/precision/pooled | -0.008 | 0.004 | 0.003 | 0.010 | 0.007 | True |
| tree_trunk/P2@0.8/precision/mean | -0.016 | 0.012 | 0.009 | 0.029 | 0.023 | False |
| tree_trunk/P2@0.8/iou/pooled | -0.020 | 0.003 | 0.002 | 0.006 | 0.006 | True |
| tree_trunk/P2@0.8/iou/mean | -0.023 | 0.004 | 0.002 | 0.009 | 0.006 | True |
| tree_trunk/P1/recall/pooled | -0.009 | 0.003 | 0.004 | 0.007 | 0.010 | False |
| tree_trunk/P1/recall/mean | -0.009 | 0.003 | 0.003 | 0.006 | 0.009 | True |
| tree_trunk/P1/precision/pooled | -0.002 | 0.004 | 0.003 | 0.011 | 0.008 | False |
| tree_trunk/P1/precision/mean | 0.014 | 0.011 | 0.012 | 0.028 | 0.032 | False |

#### rock, P2 >= 0.5

| metric | gap | first std | second std | first range | second range | separated |
|---|---|---|---|---|---|---|
| rock/P2@0.5/recall/pooled | -0.031 | 0.017 | 0.015 | 0.047 | 0.035 | False |
| rock/P2@0.5/recall/mean | -0.021 | 0.017 | 0.013 | 0.042 | 0.030 | False |
| rock/P2@0.5/precision/pooled | -0.001 | 0.017 | 0.024 | 0.047 | 0.055 | False |
| rock/P2@0.5/precision/mean | -0.018 | 0.076 | 0.082 | 0.198 | 0.173 | False |
| rock/P2@0.5/iou/pooled | -0.011 | 0.011 | 0.012 | 0.029 | 0.028 | False |
| rock/P2@0.5/iou/mean | -0.002 | 0.016 | 0.011 | 0.035 | 0.028 | False |
| rock/P1/recall/pooled | -0.021 | 0.017 | 0.018 | 0.047 | 0.047 | False |
| rock/P1/recall/mean | -0.023 | 0.014 | 0.019 | 0.038 | 0.050 | False |
| rock/P1/precision/pooled | -0.019 | 0.013 | 0.013 | 0.029 | 0.027 | False |
| rock/P1/precision/mean | -0.018 | 0.012 | 0.015 | 0.029 | 0.032 | False |

#### rock, P2 >= 0.8

| metric | gap | first std | second std | first range | second range | separated |
|---|---|---|---|---|---|---|
| rock/P2@0.8/recall/pooled | -0.033 | 0.019 | 0.013 | 0.047 | 0.024 | False |
| rock/P2@0.8/recall/mean | -0.029 | 0.019 | 0.012 | 0.042 | 0.033 | False |
| rock/P2@0.8/precision/pooled | 0.003 | 0.023 | 0.017 | 0.063 | 0.037 | False |
| rock/P2@0.8/precision/mean | 0.036 | 0.137 | 0.106 | 0.317 | 0.261 | False |
| rock/P2@0.8/iou/pooled | -0.012 | 0.012 | 0.009 | 0.033 | 0.018 | False |
| rock/P2@0.8/iou/mean | -0.004 | 0.014 | 0.014 | 0.033 | 0.035 | False |
| rock/P1/recall/pooled | -0.021 | 0.017 | 0.018 | 0.047 | 0.047 | False |
| rock/P1/recall/mean | -0.023 | 0.014 | 0.019 | 0.038 | 0.050 | False |
| rock/P1/precision/pooled | -0.019 | 0.013 | 0.013 | 0.029 | 0.027 | False |
| rock/P1/precision/mean | -0.018 | 0.012 | 0.015 | 0.029 | 0.032 | False |

#### bush_or_small_tree, P2 >= 0.5

| metric | gap | first std | second std | first range | second range | separated |
|---|---|---|---|---|---|---|
| bush_or_small_tree/P2@0.5/recall/pooled | -0.054 | 0.007 | 0.020 | 0.015 | 0.048 | True |
| bush_or_small_tree/P2@0.5/recall/mean | -0.075 | 0.013 | 0.010 | 0.029 | 0.024 | True |
| bush_or_small_tree/P2@0.5/precision/pooled | -0.064 | 0.020 | 0.015 | 0.046 | 0.032 | True |
| bush_or_small_tree/P2@0.5/precision/mean | -0.045 | 0.032 | 0.020 | 0.073 | 0.052 | False |
| bush_or_small_tree/P2@0.5/iou/pooled | -0.048 | 0.006 | 0.017 | 0.014 | 0.041 | True |
| bush_or_small_tree/P2@0.5/iou/mean | -0.046 | 0.006 | 0.005 | 0.014 | 0.014 | True |
| bush_or_small_tree/P1/recall/pooled | -0.015 | 0.004 | 0.006 | 0.011 | 0.015 | True |
| bush_or_small_tree/P1/recall/mean | -0.014 | 0.005 | 0.009 | 0.014 | 0.025 | False |
| bush_or_small_tree/P1/precision/pooled | -0.016 | 0.012 | 0.010 | 0.029 | 0.023 | False |
| bush_or_small_tree/P1/precision/mean | 0.007 | 0.019 | 0.011 | 0.043 | 0.028 | False |

#### bush_or_small_tree, P2 >= 0.8

| metric | gap | first std | second std | first range | second range | separated |
|---|---|---|---|---|---|---|
| bush_or_small_tree/P2@0.8/recall/pooled | -0.041 | 0.003 | 0.008 | 0.008 | 0.020 | True |
| bush_or_small_tree/P2@0.8/recall/mean | -0.069 | 0.011 | 0.007 | 0.027 | 0.018 | True |
| bush_or_small_tree/P2@0.8/precision/pooled | -0.092 | 0.011 | 0.016 | 0.030 | 0.042 | True |
| bush_or_small_tree/P2@0.8/precision/mean | -0.077 | 0.042 | 0.038 | 0.101 | 0.089 | False |
| bush_or_small_tree/P2@0.8/iou/pooled | -0.039 | 0.003 | 0.007 | 0.008 | 0.018 | True |
| bush_or_small_tree/P2@0.8/iou/mean | -0.045 | 0.007 | 0.005 | 0.016 | 0.013 | True |
| bush_or_small_tree/P1/recall/pooled | -0.015 | 0.004 | 0.006 | 0.011 | 0.015 | True |
| bush_or_small_tree/P1/recall/mean | -0.014 | 0.005 | 0.009 | 0.014 | 0.025 | False |
| bush_or_small_tree/P1/precision/pooled | -0.016 | 0.012 | 0.010 | 0.029 | 0.023 | False |
| bush_or_small_tree/P1/precision/mean | 0.007 | 0.019 | 0.011 | 0.043 | 0.028 | False |

### june_scene_a_radar_init_loss_cam - june_scene_a_cam

#### obstacles, P2 >= 0.5

| metric | gap | first std | second std | first range | second range | separated |
|---|---|---|---|---|---|---|
| obstacles/P2@0.5/recall/pooled | 0.038 | 0.003 | 0.006 | 0.007 | 0.015 | True |
| obstacles/P2@0.5/recall/mean | 0.040 | 0.002 | 0.004 | 0.005 | 0.010 | True |
| obstacles/P2@0.5/precision/pooled | 0.007 | 0.004 | 0.005 | 0.011 | 0.013 | False |
| obstacles/P2@0.5/precision/mean | 0.031 | 0.007 | 0.004 | 0.020 | 0.010 | True |
| obstacles/P2@0.5/iou/pooled | 0.033 | 0.003 | 0.005 | 0.008 | 0.011 | True |
| obstacles/P2@0.5/iou/mean | 0.031 | 0.002 | 0.003 | 0.006 | 0.007 | True |
| obstacles/P1/recall/pooled | 0.007 | 0.003 | 0.003 | 0.006 | 0.008 | False |
| obstacles/P1/recall/mean | 0.006 | 0.002 | 0.004 | 0.006 | 0.011 | False |
| obstacles/P1/precision/pooled | 0.000 | 0.002 | 0.003 | 0.005 | 0.008 | False |
| obstacles/P1/precision/mean | 0.014 | 0.009 | 0.010 | 0.021 | 0.028 | False |

#### obstacles, P2 >= 0.8

| metric | gap | first std | second std | first range | second range | separated |
|---|---|---|---|---|---|---|
| obstacles/P2@0.8/recall/pooled | 0.034 | 0.002 | 0.002 | 0.006 | 0.005 | True |
| obstacles/P2@0.8/recall/mean | 0.041 | 0.002 | 0.001 | 0.006 | 0.003 | True |
| obstacles/P2@0.8/precision/pooled | 0.010 | 0.003 | 0.005 | 0.006 | 0.015 | False |
| obstacles/P2@0.8/precision/mean | 0.039 | 0.010 | 0.011 | 0.027 | 0.030 | True |
| obstacles/P2@0.8/iou/pooled | 0.031 | 0.002 | 0.002 | 0.006 | 0.004 | True |
| obstacles/P2@0.8/iou/mean | 0.029 | 0.003 | 0.001 | 0.007 | 0.003 | True |
| obstacles/P1/recall/pooled | 0.007 | 0.003 | 0.003 | 0.006 | 0.008 | False |
| obstacles/P1/recall/mean | 0.006 | 0.002 | 0.004 | 0.006 | 0.011 | False |
| obstacles/P1/precision/pooled | 0.000 | 0.002 | 0.003 | 0.005 | 0.008 | False |
| obstacles/P1/precision/mean | 0.014 | 0.009 | 0.010 | 0.021 | 0.028 | False |

#### tree_trunk, P2 >= 0.5

| metric | gap | first std | second std | first range | second range | separated |
|---|---|---|---|---|---|---|
| tree_trunk/P2@0.5/recall/pooled | 0.031 | 0.004 | 0.004 | 0.010 | 0.009 | True |
| tree_trunk/P2@0.5/recall/mean | 0.030 | 0.004 | 0.007 | 0.009 | 0.019 | True |
| tree_trunk/P2@0.5/precision/pooled | 0.005 | 0.004 | 0.003 | 0.010 | 0.008 | False |
| tree_trunk/P2@0.5/precision/mean | 0.036 | 0.004 | 0.009 | 0.009 | 0.023 | True |
| tree_trunk/P2@0.5/iou/pooled | 0.028 | 0.004 | 0.003 | 0.009 | 0.007 | True |
| tree_trunk/P2@0.5/iou/mean | 0.027 | 0.004 | 0.005 | 0.011 | 0.012 | True |
| tree_trunk/P1/recall/pooled | 0.007 | 0.002 | 0.004 | 0.007 | 0.010 | True |
| tree_trunk/P1/recall/mean | 0.008 | 0.002 | 0.003 | 0.004 | 0.009 | True |
| tree_trunk/P1/precision/pooled | -0.001 | 0.002 | 0.003 | 0.004 | 0.008 | False |
| tree_trunk/P1/precision/mean | 0.022 | 0.007 | 0.012 | 0.018 | 0.032 | False |

#### tree_trunk, P2 >= 0.8

| metric | gap | first std | second std | first range | second range | separated |
|---|---|---|---|---|---|---|
| tree_trunk/P2@0.8/recall/pooled | 0.032 | 0.002 | 0.003 | 0.005 | 0.007 | True |
| tree_trunk/P2@0.8/recall/mean | 0.033 | 0.003 | 0.004 | 0.008 | 0.010 | True |
| tree_trunk/P2@0.8/precision/pooled | 0.002 | 0.002 | 0.003 | 0.006 | 0.007 | False |
| tree_trunk/P2@0.8/precision/mean | 0.044 | 0.007 | 0.009 | 0.014 | 0.023 | True |
| tree_trunk/P2@0.8/iou/pooled | 0.028 | 0.002 | 0.002 | 0.006 | 0.006 | True |
| tree_trunk/P2@0.8/iou/mean | 0.026 | 0.003 | 0.002 | 0.009 | 0.006 | True |
| tree_trunk/P1/recall/pooled | 0.007 | 0.002 | 0.004 | 0.007 | 0.010 | True |
| tree_trunk/P1/recall/mean | 0.008 | 0.002 | 0.003 | 0.004 | 0.009 | True |
| tree_trunk/P1/precision/pooled | -0.001 | 0.002 | 0.003 | 0.004 | 0.008 | False |
| tree_trunk/P1/precision/mean | 0.022 | 0.007 | 0.012 | 0.018 | 0.032 | False |

#### rock, P2 >= 0.5

| metric | gap | first std | second std | first range | second range | separated |
|---|---|---|---|---|---|---|
| rock/P2@0.5/recall/pooled | 0.068 | 0.006 | 0.015 | 0.012 | 0.035 | True |
| rock/P2@0.5/recall/mean | 0.074 | 0.007 | 0.013 | 0.017 | 0.030 | True |
| rock/P2@0.5/precision/pooled | 0.009 | 0.005 | 0.024 | 0.014 | 0.055 | False |
| rock/P2@0.5/precision/mean | -0.006 | 0.018 | 0.082 | 0.045 | 0.173 | False |
| rock/P2@0.5/iou/pooled | 0.023 | 0.003 | 0.012 | 0.009 | 0.028 | True |
| rock/P2@0.5/iou/mean | 0.027 | 0.003 | 0.011 | 0.009 | 0.028 | True |
| rock/P1/recall/pooled | 0.014 | 0.019 | 0.018 | 0.047 | 0.047 | False |
| rock/P1/recall/mean | 0.010 | 0.021 | 0.019 | 0.054 | 0.050 | False |
| rock/P1/precision/pooled | -0.004 | 0.015 | 0.013 | 0.036 | 0.027 | False |
| rock/P1/precision/mean | -0.004 | 0.017 | 0.015 | 0.041 | 0.032 | False |

#### rock, P2 >= 0.8

| metric | gap | first std | second std | first range | second range | separated |
|---|---|---|---|---|---|---|
| rock/P2@0.8/recall/pooled | 0.054 | 0.011 | 0.013 | 0.024 | 0.024 | True |
| rock/P2@0.8/recall/mean | 0.059 | 0.011 | 0.012 | 0.025 | 0.033 | True |
| rock/P2@0.8/precision/pooled | -0.008 | 0.006 | 0.017 | 0.015 | 0.037 | False |
| rock/P2@0.8/precision/mean | -0.011 | 0.053 | 0.106 | 0.108 | 0.261 | False |
| rock/P2@0.8/iou/pooled | 0.015 | 0.005 | 0.009 | 0.012 | 0.018 | False |
| rock/P2@0.8/iou/mean | 0.020 | 0.006 | 0.014 | 0.015 | 0.035 | False |
| rock/P1/recall/pooled | 0.014 | 0.019 | 0.018 | 0.047 | 0.047 | False |
| rock/P1/recall/mean | 0.010 | 0.021 | 0.019 | 0.054 | 0.050 | False |
| rock/P1/precision/pooled | -0.004 | 0.015 | 0.013 | 0.036 | 0.027 | False |
| rock/P1/precision/mean | -0.004 | 0.017 | 0.015 | 0.041 | 0.032 | False |

#### bush_or_small_tree, P2 >= 0.5

| metric | gap | first std | second std | first range | second range | separated |
|---|---|---|---|---|---|---|
| bush_or_small_tree/P2@0.5/recall/pooled | 0.052 | 0.008 | 0.020 | 0.021 | 0.048 | True |
| bush_or_small_tree/P2@0.5/recall/mean | 0.066 | 0.009 | 0.010 | 0.021 | 0.024 | True |
| bush_or_small_tree/P2@0.5/precision/pooled | 0.048 | 0.022 | 0.015 | 0.059 | 0.032 | True |
| bush_or_small_tree/P2@0.5/precision/mean | 0.055 | 0.033 | 0.020 | 0.073 | 0.052 | False |
| bush_or_small_tree/P2@0.5/iou/pooled | 0.047 | 0.008 | 0.017 | 0.021 | 0.041 | True |
| bush_or_small_tree/P2@0.5/iou/mean | 0.048 | 0.006 | 0.005 | 0.013 | 0.014 | True |
| bush_or_small_tree/P1/recall/pooled | 0.007 | 0.004 | 0.006 | 0.011 | 0.015 | False |
| bush_or_small_tree/P1/recall/mean | -0.004 | 0.007 | 0.009 | 0.015 | 0.025 | False |
| bush_or_small_tree/P1/precision/pooled | 0.008 | 0.013 | 0.010 | 0.031 | 0.023 | False |
| bush_or_small_tree/P1/precision/mean | 0.000 | 0.014 | 0.011 | 0.036 | 0.028 | False |

#### bush_or_small_tree, P2 >= 0.8

| metric | gap | first std | second std | first range | second range | separated |
|---|---|---|---|---|---|---|
| bush_or_small_tree/P2@0.8/recall/pooled | 0.039 | 0.005 | 0.008 | 0.012 | 0.020 | True |
| bush_or_small_tree/P2@0.8/recall/mean | 0.063 | 0.005 | 0.007 | 0.012 | 0.018 | True |
| bush_or_small_tree/P2@0.8/precision/pooled | 0.088 | 0.015 | 0.016 | 0.042 | 0.042 | True |
| bush_or_small_tree/P2@0.8/precision/mean | 0.062 | 0.026 | 0.038 | 0.059 | 0.089 | False |
| bush_or_small_tree/P2@0.8/iou/pooled | 0.038 | 0.005 | 0.007 | 0.012 | 0.018 | True |
| bush_or_small_tree/P2@0.8/iou/mean | 0.046 | 0.005 | 0.005 | 0.013 | 0.013 | True |
| bush_or_small_tree/P1/recall/pooled | 0.007 | 0.004 | 0.006 | 0.011 | 0.015 | False |
| bush_or_small_tree/P1/recall/mean | -0.004 | 0.007 | 0.009 | 0.015 | 0.025 | False |
| bush_or_small_tree/P1/precision/pooled | 0.008 | 0.013 | 0.010 | 0.031 | 0.023 | False |
| bush_or_small_tree/P1/precision/mean | 0.000 | 0.014 | 0.011 | 0.036 | 0.028 | False |

### june_scene_a_radar_init_loss_cam - june_scene_a_radar_init_cam

#### obstacles, P2 >= 0.5

| metric | gap | first std | second std | first range | second range | separated |
|---|---|---|---|---|---|---|
| obstacles/P2@0.5/recall/pooled | 0.068 | 0.003 | 0.002 | 0.007 | 0.005 | True |
| obstacles/P2@0.5/recall/mean | 0.078 | 0.002 | 0.001 | 0.005 | 0.003 | True |
| obstacles/P2@0.5/precision/pooled | 0.012 | 0.004 | 0.004 | 0.011 | 0.010 | True |
| obstacles/P2@0.5/precision/mean | 0.054 | 0.007 | 0.015 | 0.020 | 0.039 | True |
| obstacles/P2@0.5/iou/pooled | 0.059 | 0.003 | 0.002 | 0.008 | 0.005 | True |
| obstacles/P2@0.5/iou/mean | 0.058 | 0.002 | 0.002 | 0.006 | 0.006 | True |
| obstacles/P1/recall/pooled | 0.018 | 0.003 | 0.003 | 0.006 | 0.006 | True |
| obstacles/P1/recall/mean | 0.017 | 0.002 | 0.002 | 0.006 | 0.006 | True |
| obstacles/P1/precision/pooled | 0.004 | 0.002 | 0.003 | 0.005 | 0.008 | False |
| obstacles/P1/precision/mean | 0.000 | 0.009 | 0.009 | 0.021 | 0.019 | False |

#### obstacles, P2 >= 0.8

| metric | gap | first std | second std | first range | second range | separated |
|---|---|---|---|---|---|---|
| obstacles/P2@0.8/recall/pooled | 0.062 | 0.002 | 0.002 | 0.006 | 0.005 | True |
| obstacles/P2@0.8/recall/mean | 0.078 | 0.002 | 0.002 | 0.006 | 0.005 | True |
| obstacles/P2@0.8/precision/pooled | 0.020 | 0.003 | 0.005 | 0.006 | 0.009 | True |
| obstacles/P2@0.8/precision/mean | 0.071 | 0.010 | 0.016 | 0.027 | 0.042 | True |
| obstacles/P2@0.8/iou/pooled | 0.057 | 0.002 | 0.002 | 0.006 | 0.005 | True |
| obstacles/P2@0.8/iou/mean | 0.056 | 0.003 | 0.002 | 0.007 | 0.006 | True |
| obstacles/P1/recall/pooled | 0.018 | 0.003 | 0.003 | 0.006 | 0.006 | True |
| obstacles/P1/recall/mean | 0.017 | 0.002 | 0.002 | 0.006 | 0.006 | True |
| obstacles/P1/precision/pooled | 0.004 | 0.002 | 0.003 | 0.005 | 0.008 | False |
| obstacles/P1/precision/mean | 0.000 | 0.009 | 0.009 | 0.021 | 0.019 | False |

#### tree_trunk, P2 >= 0.5

| metric | gap | first std | second std | first range | second range | separated |
|---|---|---|---|---|---|---|
| tree_trunk/P2@0.5/recall/pooled | 0.050 | 0.004 | 0.002 | 0.010 | 0.005 | True |
| tree_trunk/P2@0.5/recall/mean | 0.059 | 0.004 | 0.003 | 0.009 | 0.007 | True |
| tree_trunk/P2@0.5/precision/pooled | 0.011 | 0.004 | 0.004 | 0.010 | 0.011 | True |
| tree_trunk/P2@0.5/precision/mean | 0.046 | 0.004 | 0.013 | 0.009 | 0.036 | True |
| tree_trunk/P2@0.5/iou/pooled | 0.045 | 0.004 | 0.002 | 0.009 | 0.004 | True |
| tree_trunk/P2@0.5/iou/mean | 0.050 | 0.004 | 0.004 | 0.011 | 0.011 | True |
| tree_trunk/P1/recall/pooled | 0.016 | 0.002 | 0.003 | 0.007 | 0.007 | True |
| tree_trunk/P1/recall/mean | 0.017 | 0.002 | 0.003 | 0.004 | 0.006 | True |
| tree_trunk/P1/precision/pooled | 0.002 | 0.002 | 0.004 | 0.004 | 0.011 | False |
| tree_trunk/P1/precision/mean | 0.008 | 0.007 | 0.011 | 0.018 | 0.028 | False |

#### tree_trunk, P2 >= 0.8

| metric | gap | first std | second std | first range | second range | separated |
|---|---|---|---|---|---|---|
| tree_trunk/P2@0.8/recall/pooled | 0.053 | 0.002 | 0.002 | 0.005 | 0.006 | True |
| tree_trunk/P2@0.8/recall/mean | 0.062 | 0.003 | 0.003 | 0.008 | 0.006 | True |
| tree_trunk/P2@0.8/precision/pooled | 0.010 | 0.002 | 0.004 | 0.006 | 0.010 | True |
| tree_trunk/P2@0.8/precision/mean | 0.059 | 0.007 | 0.012 | 0.014 | 0.029 | True |
| tree_trunk/P2@0.8/iou/pooled | 0.048 | 0.002 | 0.003 | 0.006 | 0.006 | True |
| tree_trunk/P2@0.8/iou/mean | 0.049 | 0.003 | 0.004 | 0.009 | 0.009 | True |
| tree_trunk/P1/recall/pooled | 0.016 | 0.002 | 0.003 | 0.007 | 0.007 | True |
| tree_trunk/P1/recall/mean | 0.017 | 0.002 | 0.003 | 0.004 | 0.006 | True |
| tree_trunk/P1/precision/pooled | 0.002 | 0.002 | 0.004 | 0.004 | 0.011 | False |
| tree_trunk/P1/precision/mean | 0.008 | 0.007 | 0.011 | 0.018 | 0.028 | False |

#### rock, P2 >= 0.5

| metric | gap | first std | second std | first range | second range | separated |
|---|---|---|---|---|---|---|
| rock/P2@0.5/recall/pooled | 0.099 | 0.006 | 0.017 | 0.012 | 0.047 | True |
| rock/P2@0.5/recall/mean | 0.095 | 0.007 | 0.017 | 0.017 | 0.042 | True |
| rock/P2@0.5/precision/pooled | 0.010 | 0.005 | 0.017 | 0.014 | 0.047 | False |
| rock/P2@0.5/precision/mean | 0.012 | 0.018 | 0.076 | 0.045 | 0.198 | False |
| rock/P2@0.5/iou/pooled | 0.034 | 0.003 | 0.011 | 0.009 | 0.029 | True |
| rock/P2@0.5/iou/mean | 0.029 | 0.003 | 0.016 | 0.009 | 0.035 | True |
| rock/P1/recall/pooled | 0.035 | 0.019 | 0.017 | 0.047 | 0.047 | False |
| rock/P1/recall/mean | 0.033 | 0.021 | 0.014 | 0.054 | 0.038 | False |
| rock/P1/precision/pooled | 0.015 | 0.015 | 0.013 | 0.036 | 0.029 | False |
| rock/P1/precision/mean | 0.014 | 0.017 | 0.012 | 0.041 | 0.029 | False |

#### rock, P2 >= 0.8

| metric | gap | first std | second std | first range | second range | separated |
|---|---|---|---|---|---|---|
| rock/P2@0.8/recall/pooled | 0.087 | 0.011 | 0.019 | 0.024 | 0.047 | True |
| rock/P2@0.8/recall/mean | 0.088 | 0.011 | 0.019 | 0.025 | 0.042 | True |
| rock/P2@0.8/precision/pooled | -0.011 | 0.006 | 0.023 | 0.015 | 0.063 | False |
| rock/P2@0.8/precision/mean | -0.047 | 0.053 | 0.137 | 0.108 | 0.317 | False |
| rock/P2@0.8/iou/pooled | 0.028 | 0.005 | 0.012 | 0.012 | 0.033 | True |
| rock/P2@0.8/iou/mean | 0.024 | 0.006 | 0.014 | 0.015 | 0.033 | False |
| rock/P1/recall/pooled | 0.035 | 0.019 | 0.017 | 0.047 | 0.047 | False |
| rock/P1/recall/mean | 0.033 | 0.021 | 0.014 | 0.054 | 0.038 | False |
| rock/P1/precision/pooled | 0.015 | 0.015 | 0.013 | 0.036 | 0.029 | False |
| rock/P1/precision/mean | 0.014 | 0.017 | 0.012 | 0.041 | 0.029 | False |

#### bush_or_small_tree, P2 >= 0.5

| metric | gap | first std | second std | first range | second range | separated |
|---|---|---|---|---|---|---|
| bush_or_small_tree/P2@0.5/recall/pooled | 0.106 | 0.008 | 0.007 | 0.021 | 0.015 | True |
| bush_or_small_tree/P2@0.5/recall/mean | 0.141 | 0.009 | 0.013 | 0.021 | 0.029 | True |
| bush_or_small_tree/P2@0.5/precision/pooled | 0.113 | 0.022 | 0.020 | 0.059 | 0.046 | True |
| bush_or_small_tree/P2@0.5/precision/mean | 0.100 | 0.033 | 0.032 | 0.073 | 0.073 | True |
| bush_or_small_tree/P2@0.5/iou/pooled | 0.095 | 0.008 | 0.006 | 0.021 | 0.014 | True |
| bush_or_small_tree/P2@0.5/iou/mean | 0.094 | 0.006 | 0.006 | 0.013 | 0.014 | True |
| bush_or_small_tree/P1/recall/pooled | 0.021 | 0.004 | 0.004 | 0.011 | 0.011 | True |
| bush_or_small_tree/P1/recall/mean | 0.010 | 0.007 | 0.005 | 0.015 | 0.014 | False |
| bush_or_small_tree/P1/precision/pooled | 0.024 | 0.013 | 0.012 | 0.031 | 0.029 | False |
| bush_or_small_tree/P1/precision/mean | -0.007 | 0.014 | 0.019 | 0.036 | 0.043 | False |

#### bush_or_small_tree, P2 >= 0.8

| metric | gap | first std | second std | first range | second range | separated |
|---|---|---|---|---|---|---|
| bush_or_small_tree/P2@0.8/recall/pooled | 0.080 | 0.005 | 0.003 | 0.012 | 0.008 | True |
| bush_or_small_tree/P2@0.8/recall/mean | 0.132 | 0.005 | 0.011 | 0.012 | 0.027 | True |
| bush_or_small_tree/P2@0.8/precision/pooled | 0.181 | 0.015 | 0.011 | 0.042 | 0.030 | True |
| bush_or_small_tree/P2@0.8/precision/mean | 0.139 | 0.026 | 0.042 | 0.059 | 0.101 | True |
| bush_or_small_tree/P2@0.8/iou/pooled | 0.077 | 0.005 | 0.003 | 0.012 | 0.008 | True |
| bush_or_small_tree/P2@0.8/iou/mean | 0.091 | 0.005 | 0.007 | 0.013 | 0.016 | True |
| bush_or_small_tree/P1/recall/pooled | 0.021 | 0.004 | 0.004 | 0.011 | 0.011 | True |
| bush_or_small_tree/P1/recall/mean | 0.010 | 0.007 | 0.005 | 0.015 | 0.014 | False |
| bush_or_small_tree/P1/precision/pooled | 0.024 | 0.013 | 0.012 | 0.031 | 0.029 | False |
| bush_or_small_tree/P1/precision/mean | -0.007 | 0.014 | 0.019 | 0.036 | 0.043 | False |

## Guard: image quality (test + guard frames)

| metric | june_scene_a_cam | june_scene_a_radar_init_cam | june_scene_a_radar_init_loss_cam |
|---|---|---|---|
| image/psnr | 15.917 ± 0.035 (n=5) | 15.861 ± 0.040 (n=5) | 15.887 ± 0.012 (n=5) |
| image/ssim | 0.413 ± 0.000 (n=5) | 0.410 ± 0.000 (n=5) | 0.408 ± 0.000 (n=5) |
| image/lpips | 0.688 ± 0.002 (n=5) | 0.707 ± 0.003 (n=5) | 0.635 ± 0.001 (n=5) |

## Report only: lidar render

| metric | june_scene_a_cam | june_scene_a_radar_init_cam | june_scene_a_radar_init_loss_cam |
|---|---|---|---|
| lidar_render/chamfer_distance | 0.739 ± 0.021 (n=5) | 0.821 ± 0.038 (n=5) | 0.703 ± 0.019 (n=5) |
| lidar_render/median_chamfer_distance | 0.868 ± 0.074 (n=5) | 0.875 ± 0.034 (n=5) | 1.626 ± 0.315 (n=5) |
| lidar_render/depth_median_l2 | 0.037 ± 0.000 (n=5) | 0.041 ± 0.001 (n=5) | 0.040 ± 0.000 (n=5) |
| lidar_render/depth_mean_rel_l2 | 103.530 ± 2.783 (n=5) | 99.787 ± 7.653 (n=5) | 111.345 ± 5.044 (n=5) |

