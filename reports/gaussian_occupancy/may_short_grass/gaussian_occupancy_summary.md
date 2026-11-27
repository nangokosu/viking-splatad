# P1/P2 over repeats: may_short_grass

## Definitions

Plain words first, the formula last. Every number is read from the run reports, never typed in.

1. **Cells.** The scene is cut into small cubes ("cells") of edge 0.521933 m. The edge is half the median trunk width of the session's boxes (width = mean of a box's two horizontal sizes), so a typical trunk box is about two cells across, and the trunk margin below equals the cell edge. Everything below is about cells.
2. **Ground truth, "on an obstacle" and "next to an obstacle".** People drew a box around every tree trunk, rock and bush ("obstacles"). The boxes are used only for scoring, never for training.
   - **On an obstacle (G):** a cell whose **centre** lies inside an obstacle box.
   - **The scored area around a class (N_c):** for each class (trunk, rock, bush) separately:
     1. **Margin:** m_c = half of the class's typical footprint width, i.e. the median over all of the session's boxes of that class of (length + width) / 4. Values for session may_short_grass: trunk 0.522 m, rock 0.665 m, bush 1.069 m.
     2. **Grow:** widen every box of the class by m_c on each of its four sides, along the box's own axes, leaving its height unchanged. A typical object's grown box is therefore about twice its own width.
     3. **Collect:** N_c = the cells whose centre lies inside a grown box.
   - **Only space the lidar could have covered is scored:** a box counts only when its centre is within 56.18 m (the distance within which 99.99% of the session's lidar returns lie) and the vertical field of view (±44.8°) of at least one recorded sweep. The same range and field-of-view test is applied to every cell of N_c.
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
   - **(e) Worked example.** The stock arm's drawn run in may_short_grass at t = 0.5, with its real counts: 5420 hits, 14180 missed, 1011 false alarms → recall 0.277, precision 0.843, IoU 0.263. These are one run; the tables give mean ± std over the repeats.
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
- One run per arm: the median repeat of whole-scene pooled recall at the binding cut-off t = 0.5 (metric `obstacles/P2@0.5/recall/pooled`), so it illustrates and does not decide anything.


![Recall and precision against the occupancy threshold](../figures/may_short_grass/threshold_curves_scene.png)

**How to read this figure**
- **What is plotted:** recall (left) and precision (right) recomputed at each cut-off t from 0.2 to 0.9 (15 evenly spaced values). Solid lines are pooled and dashed lines are the mean over objects. The line is the median repeat. Each line has its own shaded band: at each t, the lowest and highest value of that same curve (pooled or mean over objects) over the 5 repeats of the arm. The pooled band is shaded more strongly than the mean band. Dotted lines mark the table cut-offs 0.5, 0.8.
- **Solid vs dashed:** solid = *pooled* (all objects' cells counted together, so large objects dominate); dashed = *mean over objects* (each object scored on its own, then averaged, so every object counts once). Full definitions and an example: point 5(f).
- **Definitions:** see point 5. Only recall and precision are plotted; IoU is in the tables. Recall = hit / (hit + missed) has all 19600 obstacle cells as its denominator; precision = hit / (hit + false alarm) has every filled scored cell as its denominator.
- **How the curves move:** raising t fills fewer cells, so recall can only fall and precision usually rises.
- **When one arm beats another:** its line is higher and its band does not overlap the other arm's band of the same kind (pooled with pooled, mean with mean).
- **Why t runs from 0.2 to 0.9:** the extremes are left out because they say nothing about the model. At t = 0 every cell counts as filled (recall 1, precision = the share of obstacle cells). At t ≈ 1 precision rests on a handful of cells and jumps.




Descriptive only: a variant gap counts only when it exceeds the spread between repeats without losing precision. Repeats share one seed, so they are not paired. Cells are mean ± sample std (n = repeats).

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

