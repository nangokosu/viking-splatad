"""Glossary of the evaluation terminology, stored as data and rendered by one function.

Each entry was written from its source (the named code docstring or the named doc section), never from
general knowledge. The "where" of an entry lists the sources; ``tests/unit/test_terminology_glossary.py``
checks that every named file exists and that every named Python symbol or text snippet is found in it.
Terms whose source could not be found are not listed here; they are reported as open by whoever edits
this file. No number is typed into a definition: values for a session are in the document's value table.
"""

from typing import Dict, List, NamedTuple, Optional, Tuple

# Topic names in the order the glossary prints them.
TOPIC_EXPERIMENT = "Experiment"
TOPIC_RADAR = "Radar"
TOPIC_MODEL = "Model"
TOPIC_GROUND_TRUTH = "Ground truth and grid"
TOPIC_OCCUPANCY = "Occupancy"
TOPIC_LIDAR_VIEW = "Lidar view"
TOPIC_OUTCOMES = "Outcomes and metrics"
TOPIC_STATISTICS = "Statistics"
TOPIC_IMAGE = "Image metrics (auxiliary)"
TOPIC_BOOKKEEPING = "Bookkeeping"
TOPIC_ORDER = (TOPIC_EXPERIMENT, TOPIC_RADAR, TOPIC_MODEL, TOPIC_GROUND_TRUTH, TOPIC_OCCUPANCY, TOPIC_LIDAR_VIEW,
               TOPIC_OUTCOMES, TOPIC_STATISTICS, TOPIC_IMAGE, TOPIC_BOOKKEEPING)

# File name suffix that marks a Python source (its anchors are defined symbols, transform ids or verbatim phrases;
# any other file's anchors are text snippets).
PYTHON_SUFFIX = ".py"


class WhereRef(NamedTuple):
    """One source of a glossary entry.

    Attributes:
        path: Repo-relative file path.
        anchor: For a ``.py`` file a function, class or constant name, a transform id ``T<n>``, or a phrase (any
            anchor that is not a single identifier) that must occur verbatim; for any other file a text snippet
            (heading or phrase) that must occur in the file.
    """

    path: str
    anchor: str


class GlossaryEntry(NamedTuple):
    """One term of the glossary.

    Attributes:
        topic: One of ``TOPIC_ORDER``.
        term: The term as it appears in the reports.
        definition: Plain-language definition (1-3 sentences).
        formula: Formula as text, or None when the term has none.
        where: Sources (code symbol or doc section) that define the term.
    """

    topic: str
    term: str
    definition: str
    formula: Optional[str]
    where: Tuple[WhereRef, ...]


def _refs(*pairs: Tuple[str, str]) -> Tuple[WhereRef, ...]:
    """Build a ``where`` tuple from (path, anchor) pairs.

    Args:
        *pairs: (path, anchor) pairs.

    Returns:
        The same pairs as ``WhereRef`` tuples.
    """
    return tuple(WhereRef(path, anchor) for path, anchor in pairs)


# Frequently cited sources, named once so the entries stay short.
CLAUDE = "CLAUDE.md"
PLAN = "docs/PLAN.md"
CAVEATS = "docs/research/evaluation_caveats.md"
EXTENSION_POINTS = "docs/research/splatad_extension_points.md"
SLURM_YAML = "configs/slurm.yaml"
EVAL_YAML = "configs/eval.yaml"
TERMINOLOGY_SOURCE = "src/viking_splatad/eval/evaluation_terminology.py"
DETECTION = "src/viking_splatad/eval/detection.py"
SCENE_METRICS = "src/viking_splatad/eval/scene_metrics.py"
VOXEL_OCCUPANCY = "src/viking_splatad/eval/voxel_occupancy.py"
P2_CELLS = "src/viking_splatad/eval/p2_cells.py"
P2_REPEATS = "src/viking_splatad/eval/p2_repeats.py"
RENDERED_POINTCLOUD = "src/viking_splatad/eval/rendered_pointcloud.py"
RUN_QUALITY = "src/viking_splatad/eval/run_quality.py"
FIGURE_METRICS = "src/viking_splatad/viz/figure_metrics.py"
RADAR_LOSS = "src/viking_splatad/splatad/radar_lidar_loss.py"
RADAR_INIT_MODEL = "src/viking_splatad/splatad/radar_init_model.py"

GLOSSARY: Tuple[GlossaryEntry, ...] = (
    # ---- Experiment ----
    GlossaryEntry(TOPIC_EXPERIMENT, "Scene A",
                  "The region of the Viking Hills data used by every run: it holds tree trunks, with low grass in May "
                  "and high grass in June. Each session covers the consecutive bag chunks pinned in its YAML's `window.chunk_files`, chosen by guess and check.",
                  None, _refs((CLAUDE, "Current scope"), ("configs/sessions/june_tall_grass.yaml", "Scene A (PLAN.md"))),
    GlossaryEntry(TOPIC_EXPERIMENT, "Session",
                  "One recording campaign of Scene A with its own config: `may_short_grass` (low grass, May) or "
                  "`june_tall_grass` (tall grass, June). The two sessions are never mapped onto each other (no ICP, no "
                  "cross-run alignment).",
                  None, _refs((CLAUDE, "No May"), ("configs/sessions/may_short_grass.yaml", "chunk_files"))),
    GlossaryEntry(TOPIC_EXPERIMENT, "Bag chunk",
                  "One `.mcap` file of the recorded ROS bag. A session's `window.chunk_files` pins the chunks "
                  "that extraction converts (see the session YAML).",
                  None, _refs(("configs/sessions/june_tall_grass.yaml", "chunk_files"), (CLAUDE, "Extraction"))),
    GlossaryEntry(TOPIC_EXPERIMENT, "Arm (variant)",
                  "One way of training SplatAD, compared on the same session. Three arms: **stock** (method `splatad`, "
                  "no radar), **radar init** (method `splatad-radar-init`) and **radar init + loss** (method "
                  "`splatad-radar-init-lidar`). Slurm profiles: May `may_scene_a`, `may_scene_a_radar_init`, "
                  "`may_scene_a_radar_init_loss`; June `june_scene_a_cam`, `june_scene_a_radar_init_cam`, "
                  "`june_scene_a_radar_init_loss_cam`.",
                  None, _refs((CLAUDE, "Current scope"), (SLURM_YAML, "june_scene_a_radar_init_loss_cam"),
                              (SLURM_YAML, "may_scene_a_radar_init_loss"), ("scripts/21_compare_sessions.py", "METHOD_LABELS"))),
    GlossaryEntry(TOPIC_EXPERIMENT, "Repeat",
                  "One training run of an arm; each arm of each session is trained several times, each as a separate Slurm "
                  "job, to measure run-to-run noise. Repeats share one seed and are not paired.",
                  None, _refs((CLAUDE, "each trained 5 times"), ("scripts/21_compare_sessions.py", "repeats share one seed"))),
    GlossaryEntry(TOPIC_EXPERIMENT, "Median repeat (drawn run)",
                  "The run drawn in the 3D figures: per arm, the repeat whose value of the configured representative metric "
                  "is the median, so the drawn run is neither the arm's best nor its worst. The setting is "
                  "`eval.figures.representative_metric` (whole-scene pooled P2 recall of `obstacles` at the binding cut-off).",
                  None, _refs((EVAL_YAML, "representative_metric"))),
    GlossaryEntry(TOPIC_EXPERIMENT, "Campaign",
                  "The full set of runs the comparison rests on: three arms on each of the two sessions, each trained "
                  "several times, each a separate Slurm job.",
                  None, _refs((CLAUDE, "make the 30-run repeat campaign"))),
    GlossaryEntry(TOPIC_EXPERIMENT, "Camera (photometric) loss",
                  "The training loss that compares rendered camera images with the recorded ones. Both sessions train with it in "
                  "the campaign (profiles with `train_lidar_only: false`); the lidar-only June profiles are not part of the campaign.",
                  None, _refs((CLAUDE, "Both sessions train with the camera"), (SLURM_YAML, "train_lidar_only: false"))),
    GlossaryEntry(TOPIC_EXPERIMENT, "Train / test / guard frames",
                  "Every X-th instant of the lidar is held out (shared interleaved split; the camera holds out its frame "
                  "nearest each held-out sweep). Of the held-out instants, an evenly spaced fraction is **test** (rendered and "
                  "scored); the rest is **guard** (never trained on, not rendered). Everything else is **train**.",
                  None, _refs((PLAN, "Render subset (T72"), (PLAN, "Shared interleaved (T71"))),
    GlossaryEntry(TOPIC_EXPERIMENT, "Reference split",
                  "The committed record of which frames of a session are test and guard, identified by `timestamp_ns`. A run is "
                  "comparable with the others only when it trains and tests on the same frames, so submission refuses a run whose "
                  "split settings differ from the record.",
                  None, _refs((CLAUDE, "Reference splits (comparability"), ("configs/reference_splits/may_short_grass.json", "guard_timestamps_ns"))),
    # ---- Radar ----
    GlossaryEntry(TOPIC_RADAR, "4D radar",
                  "The 4D imaging radar carried by the ground robot, next to a 32-beam lidar and a camera. Its detections have a "
                  "range, azimuth and elevation resolution, which give each anchor its covariance. It is not a SplatAD sensor.",
                  None, _refs(("docs/proposal/proposal.md", "4D imaging radar"), (PLAN, "R_ray"))),
    GlossaryEntry(TOPIC_RADAR, "Radar anchors / seeds",
                  "Points built offline in the map frame from the radar detections (static, persistent, no ground-truth input), "
                  "each with a covariance from the radar resolution. In the init arms each is appended to the model as one seed "
                  "Gaussian whose colour is painted from the nearest unobstructed train camera frame (grey when obstructed).",
                  "covariance = R_ray diag(sigma_r^2, (r d_az)^2, (r d_el)^2) R_ray^T",
                  _refs(("src/viking_splatad/radar_init/anchors.py", "T35"), ("src/viking_splatad/radar_init/anchors.py", "T36"),
                        ("src/viking_splatad/radar_init/seeds.py", "T37"), (RADAR_INIT_MODEL, "RadarInitSplatADModel"),
                        ("src/viking_splatad/radar_init/build.py", "build_radar_anchor_artifact"),
                        ("scripts/07_build_radar_seeds.py", "build_radar_anchor_artifact("))),
    GlossaryEntry(TOPIC_RADAR, "Label-free radar opacity loss",
                  "An extra training term, used only by the radar init + loss arm: static Gaussians inside a seed's radar resolution "
                  "ellipsoid are pushed towards opaque. It uses no class label, and it never says a place is empty. Its weight is set "
                  "at step 0 so that it equals the stock opacity regulariser.",
                  "L = lambda * mean over supported Gaussians of -log sigmoid(o) = softplus(-o)",
                  _refs((RADAR_LOSS, "T43"), (RADAR_LOSS, "T42"), (RADAR_LOSS, "calibrate_opacity_weight"),
                        (PLAN, "Opacity loss (main term)"))),
    GlossaryEntry(TOPIC_RADAR, "Radar is never rendered",
                  "Radar is not a SplatAD sensor (`radars=()` stays empty, since SplatAD asserts there are none). It enters only as the "
                  "`radar_seeds` metadata of the train split, as extra Gaussians at init and as the anchors of the opacity loss.",
                  None, _refs((CLAUDE, "Radar is never a SplatAD"), (EXTENSION_POINTS, "Radar cannot be a sensor"),
                              ("src/viking_splatad/splatad/viking_dataparser.py", 'metadata["radar_seeds"]'))),
    # ---- Model ----
    GlossaryEntry(TOPIC_MODEL, "SplatAD",
                  "The lidar and camera renderer based on 3D Gaussian splatting that is trained in every run; it models the scene "
                  "as translucent 3D Gaussians with a learnable opacity.",
                  None, _refs((TERMINOLOGY_SOURCE, "SPLATAD_REFERENCE"))),
    GlossaryEntry(TOPIC_MODEL, "3D Gaussian (mean, covariance, opacity)",
                  "One blob of the model: a mean mu (position), a covariance Sigma (shape, stored as log-scales and a rotation) and an "
                  "opacity. A trained model holds many of them.",
                  "Sigma = R(q) diag(exp(s))^2 R(q)^T",
                  _refs((PLAN, "Convert:"), ("src/viking_splatad/splatad/radar_init_model.py", "covariance_to_log_scales_and_quaternions"))),
    GlossaryEntry(TOPIC_MODEL, "Learned opacity o and sigma(o)",
                  "The model stores an opacity logit o per Gaussian; the activated opacity sigma(o) lies in (0, 1) (0 see-through, 1 solid).",
                  "sigma(o) = 1 / (1 + exp(-o))",
                  _refs((PLAN, "o = sigmoid(opacities)"))),
    GlossaryEntry(TOPIC_MODEL, "Alpha-blending",
                  "How SplatAD combines the Gaussians along a ray (front to back): each Gaussian's alpha is weighted by the light "
                  "still left in front of it.",
                  "w_i = alpha_i * prod_{j<i}(1 - alpha_j)",
                  _refs((TERMINOLOGY_SOURCE, "front_to_back_weights"), (TERMINOLOGY_SOURCE, "SPLATAD_REFERENCE"))),
    GlossaryEntry(TOPIC_MODEL, "Transmittance",
                  "The share of light that passes the first i Gaussians of a ray, i.e. the probability the ray is not stopped by them.",
                  "T_i = prod_{j<=i}(1 - alpha_j)",
                  _refs((TERMINOLOGY_SOURCE, "accumulated_opacity"), (TERMINOLOGY_SOURCE, "KERBL_REFERENCE"))),
    GlossaryEntry(TOPIC_MODEL, "Accumulation",
                  "The total opacity a ray collects: the sum of its blending weights, which equals one minus the transmittance of all "
                  "Gaussians (telescoping sum). It lies in [0, 1] and does not depend on the order of the Gaussians.",
                  "sum_i w_i = 1 - prod_i(1 - alpha_i)",
                  _refs((TERMINOLOGY_SOURCE, "accumulated_opacity"), (TERMINOLOGY_SOURCE, "front_to_back_weights"))),
    GlossaryEntry(TOPIC_MODEL, "Median range",
                  "SplatAD places a rendered lidar point at the rolling-shutter-corrected range of the first Gaussian at which the "
                  "transmittance drops below 0.5. It is used at inference because, unlike the expected range, it does not yield depths "
                  "between Gaussians.",
                  "first i with prod_{j<=i}(1 - alpha_j) < 0.5",
                  _refs((TERMINOLOGY_SOURCE, "SPLATAD_REFERENCE"), (TERMINOLOGY_SOURCE, "SplatAD's own 0.5 rule"))),
    GlossaryEntry(TOPIC_MODEL, "Rendered lidar",
                  "The lidar sweep SplatAD renders at a held-out pose: for each recorded beam direction a depth and an accumulation; a "
                  "ray becomes a point at its rendered depth.",
                  None, _refs((RENDERED_POINTCLOUD, "RenderedSweep"), (EXTENSION_POINTS, "Rendered lidar"))),
    # ---- Ground truth and grid ----
    GlossaryEntry(TOPIC_GROUND_TRUTH, "Cuboid (GT box)",
                  "A hand-labelled box in the map frame (centre, full sizes along its own axes, rotation, class). Boxes are used only "
                  "for scoring and never reach training (no seed, filter or loss target).",
                  None, _refs(("src/viking_splatad/labels/cuboids.py", "Cuboid"), (CLAUDE, "Ground-truth cuboids stay separate"))),
    GlossaryEntry(TOPIC_GROUND_TRUTH, "Classes and the obstacles group",
                  "The scored classes are tree_trunk, rock and bush_or_small_tree. The group `obstacles` is the union of the three and "
                  "is the class the whole-scene tables lead with.",
                  None, _refs((EVAL_YAML, "obstacles = tree_trunk"), (EVAL_YAML, "class_groups"))),
    GlossaryEntry(TOPIC_GROUND_TRUTH, "Cell (voxel) and its edge",
                  "A cube of the grid the scene is cut into. Its edge is `voxel_size_m` (see the value table); all scoring counts cells.",
                  None, _refs(("src/viking_splatad/labels/occupancy.py", "T31"), ("scripts/20_summarize_gaussian_occupancy.py", "run_provenance"))),
    GlossaryEntry(TOPIC_GROUND_TRUTH, "Centre-of-cell rule (T31)",
                  "A cell belongs to a box when its centre lies inside the box. A trunk narrower than the typical one (about two cells "
                  "across) can therefore get one cell across or none, depending on where the grid lines fall.",
                  "centre = origin + (index + 0.5) * edge",
                  _refs(("src/viking_splatad/labels/occupancy.py", "T31"), (DETECTION, "cuboid_voxel_mask"))),
    GlossaryEntry(TOPIC_GROUND_TRUTH, "Margin (median_half_footprint)",
                  "The class's typical half footprint width: the median over the session's boxes of the class of (length + width) / 4. "
                  "It is the distance by which a box is grown to define the cells next to it.",
                  "m_c = median over boxes of (dx + dy) / 4",
                  _refs((DETECTION, "median_half_footprint"), (DETECTION, "class_margin_table"))),
    GlossaryEntry(TOPIC_GROUND_TRUTH, "Grown box (grow_cuboid_horizontally)",
                  "A box with its horizontal sizes enlarged by the margin on each of the four sides, height unchanged.",
                  "dimension_xy' = dimension_xy + 2 m_c",
                  _refs((DETECTION, "grow_cuboid_horizontally"))),
    GlossaryEntry(TOPIC_GROUND_TRUTH, "Scored area N",
                  "The cells that are scored for a class group: the cells whose centre lies in a grown box of the group, within the scored "
                  "range and vertical field of view of at least one recorded sweep, minus the cells of the other-class mask. Everything "
                  "else is ignored.",
                  "N = (union_c N_c) minus other-class cells",
                  _refs((DETECTION, "visible_neighbourhood_mask"), (SCENE_METRICS, "class_group_masks"))),
    GlossaryEntry(TOPIC_GROUND_TRUTH, "On an obstacle (G)",
                  "A scored cell whose centre lies inside a box of the group (the ground-truth cells; G is a subset of N).",
                  "G = cells with GT label in the group, inside N",
                  _refs((SCENE_METRICS, "class_group_masks"))),
    GlossaryEntry(TOPIC_GROUND_TRUTH, "Next to an obstacle (N minus G)",
                  "The scored cells that are in the grown area but not inside any box of the group.",
                  "N minus G",
                  _refs((P2_CELLS, "p2_cell_outcomes"), (TERMINOLOGY_SOURCE, "Next to an obstacle"))),
    GlossaryEntry(TOPIC_GROUND_TRUTH, "Other-class mask",
                  "Cells whose centre lies in a box of an annotated class outside the group are removed from the scored area, so they count "
                  "neither for nor against the group.",
                  None, _refs((SCENE_METRICS, "class_group_masks"))),
    GlossaryEntry(TOPIC_GROUND_TRUTH, "Scored range and vertical field of view",
                  "A box or cell is scored only when it is inside the lidar's reach from at least one recorded sweep: within the scored range "
                  "(a quantile of all return ranges) and within the vertical half field of view (the largest absolute beam elevation).",
                  "vertical half FOV = max |beam elevation|",
                  _refs((DETECTION, "vertical_half_fov_from_elevations"), (DETECTION, "range_limit_from_real_points"),
                        (EVAL_YAML, "scored_range_quantile"))),
    # ---- Occupancy ----
    GlossaryEntry(TOPIC_OCCUPANCY, "P2 occupancy (T57, T81)",
                  "How solid the model says a cell is, from 0 to 1, read from the Gaussians directly (not from a render): the opacity "
                  "that all Gaussians touching the cell would collect, combined as 1 - prod(1 - alpha). T57 samples each Gaussian's alpha at the cell centre within 3 sigma; "
                  "T81, the binding estimator, adds the Gaussian's peak opacity to the cell that holds its mean.",
                  "occupancy = 1 - prod_k(1 - alpha_k); alpha_k(x) = sigma(o_k) exp(-0.5 (x - mu_k)^T Sigma_k^-1 (x - mu_k))",
                  _refs((VOXEL_OCCUPANCY, "T57"), (VOXEL_OCCUPANCY, "T81"), (PLAN, "T81, peak alpha in the voxel"))),
    GlossaryEntry(TOPIC_OCCUPANCY, "Peak-alpha rule",
                  "SplatAD's Gaussians are thin discs far smaller than a cell, so almost none reach a cell centre. The Gaussian therefore "
                  "adds its peak sigma(o) to the cell holding its mean, and its centre-sampled tail to every other cell within 3 sigma.",
                  "alpha_kv = sigma(o_k) in the home cell; sigma(o_k) exp(-d^2 / 2) elsewhere (d = Mahalanobis distance)",
                  _refs((VOXEL_OCCUPANCY, "T81"))),
    GlossaryEntry(TOPIC_OCCUPANCY, "Blocking probability",
                  "The chance that a ray through a point is stopped, assuming each Gaussian blocks light independently. Our cell occupancy "
                  "applies it with each Gaussian at its peak in its own cell, which counts for more than one ray through the cell would "
                  "collect from it (a more generous call than SplatAD's stop rule, the same for every arm), not the fraction of the cell that is filled.",
                  "1 - prod_k(1 - alpha_k(x))",
                  _refs((CAVEATS, "Blocking probability"), (TERMINOLOGY_SOURCE, "more than one ray through the cell would collect"))),
    GlossaryEntry(TOPIC_OCCUPANCY, "Cut-off t and filled cell O_t",
                  "A cell with occupancy at or above the cut-off t counts as filled; O_t is the set of filled cells.",
                  "O_t = {cells with occupancy >= t}",
                  _refs((P2_CELLS, "p2_cell_outcomes"), (PLAN, "Thresholds:"))),
    GlossaryEntry(TOPIC_OCCUPANCY, "P2 binding cut-off and strictness check",
                  "P2 scored at the cut-offs of `eval.p2_thresholds`: the first value (binding) decides; the second (strictness check) only checks that a "
                  "result survives a stricter cut-off. It can weaken a result but never replace the first. The values are in the value table.",
                  None, _refs((PLAN, "Thresholds:"), (EVAL_YAML, "p2_thresholds"), ("src/viking_splatad/eval/occupancy_settings.py", "binding_threshold"))),
    # ---- Lidar view ----
    GlossaryEntry(TOPIC_LIDAR_VIEW, "P1 (rendered lidar)",
                  "The test that asks what a lidar at the held-out poses would see: a cell is hit when at least one ungated rendered point of "
                  "any test sweep falls in it.",
                  None, _refs((P2_CELLS, "p1_hit_cells"), (PLAN, "P1, rendered lidar point cloud"))),
    GlossaryEntry(TOPIC_LIDAR_VIEW, "P1cuboid detection rate (T45)",
                  "Per class, the share of scored boxes that hold at least n_min points of the rendered cloud.",
                  "rate(n) = |{b : count_b >= n}| / B",
                  _refs((DETECTION, "T45"), (DETECTION, "cuboid_detection_rate"))),
    GlossaryEntry(TOPIC_LIDAR_VIEW, "P1cuboid point precision (T46)",
                  "Of the rendered points in the grown boxes of a class, the share that lies inside a box itself.",
                  "precision = |points in a box| / |points in the grown boxes|",
                  _refs((DETECTION, "T46"), (DETECTION, "neighbourhood_point_precision"))),
    GlossaryEntry(TOPIC_LIDAR_VIEW, "Gated vs ungated point sets",
                  "Gated keeps the rays SplatAD itself would keep (ray-drop probability at most 0.5); ungated (binding) keeps every valid ray "
                  "whose accumulation reaches the render threshold. The gate can remove radar-recovered trunks, so gated numbers are only "
                  "reported beside the ungated ones.",
                  None, _refs((RENDERED_POINTCLOUD, "**ungated** (PLAN"), (PLAN, "ungated extraction (binding)"))),
    GlossaryEntry(TOPIC_LIDAR_VIEW, "Render metrics (chamfer distance, median chamfer, depth median L2, depth mean relative L2)",
                  "Lidar render error scores of SplatAD's own evaluation on the test sweeps. They are report only and never decide a "
                  "comparison. Their exact formulas are those of the SplatAD evaluation code.",
                  None, _refs((RUN_QUALITY, "REPORTED_LIDAR_METRICS"), (EXTENSION_POINTS, "chamfer_distance"))),
    # ---- Outcomes and metrics ----
    GlossaryEntry(TOPIC_OUTCOMES, "Hit, false alarm, missed, context",
                  "Cell sets behind the ratios: **context** is every scored cell; **hit** is a filled cell on an obstacle; "
                  "**false alarm** a filled cell next to an obstacle; **missed** an obstacle cell left empty.",
                  "hit = O & G; false alarm = O & N & ~G; missed = G & ~O",
                  _refs((P2_CELLS, "p2_cell_outcomes"))),
    GlossaryEntry(TOPIC_OUTCOMES, "Correct empty",
                  "A cell next to an obstacle that the model left empty. It is the fourth outcome beside hit, false alarm and missed, "
                  "and it enters neither recall nor precision.",
                  "correct empty = N & ~G & ~O",
                  _refs((P2_CELLS, "p2_cell_outcomes"))),
    GlossaryEntry(TOPIC_OUTCOMES, "Recall",
                  "Of the obstacle cells, the share the model filled.",
                  "recall = |O & G| / |G|",
                  _refs((SCENE_METRICS, "T94"))),
    GlossaryEntry(TOPIC_OUTCOMES, "Precision",
                  "Of the filled cells in the scored area, the share that is on an obstacle.",
                  "precision = |O & G| / |O & N|",
                  _refs((SCENE_METRICS, "T94"))),
    GlossaryEntry(TOPIC_OUTCOMES, "IoU",
                  "Overlap of the filled cells and the obstacle cells inside the scored area.",
                  "IoU = TP / (TP + FP + FN)",
                  _refs((SCENE_METRICS, "T94"))),
    GlossaryEntry(TOPIC_OUTCOMES, "Pooled (T94)",
                  "The metrics counted over all scored cells together, so big objects weigh most.",
                  "TP = |O & G|, FP = |O & N minus G|, FN = |G minus O|",
                  _refs((SCENE_METRICS, "T94"))),
    GlossaryEntry(TOPIC_OUTCOMES, "Mean over objects (T95)",
                  "Each object is scored alone inside its own grown box, and the metric is averaged over the objects where it is defined, "
                  "so every object weighs the same. A cell in the grown boxes of two objects is scored in both, and a filled cell on a "
                  "different obstacle inside an object's grown box is a false alarm of that object.",
                  "mean_m = (1 / |D_m|) * sum over objects k in D_m of m_k",
                  _refs((SCENE_METRICS, "T95"))),
    GlossaryEntry(TOPIC_OUTCOMES, "Threshold curve",
                  "Recall and precision (pooled and mean over objects) recomputed at each cut-off of a grid, with filled = occupancy at or "
                  "above the cut-off.",
                  None, _refs((FIGURE_METRICS, "threshold_curves"), (FIGURE_METRICS, "occupancy_samples"))),
    # ---- Statistics ----
    GlossaryEntry(TOPIC_STATISTICS, "Mean +- std (n - 1)",
                  "The mean of a metric over an arm's defined repeats and its sample standard deviation (divisor n - 1; undefined with fewer "
                  "than two repeats).",
                  "std = sqrt(sum (x - mean)^2 / (n - 1))",
                  _refs((P2_REPEATS, "describe_values"))),
    GlossaryEntry(TOPIC_STATISTICS, "Min-max band",
                  "The range from the smallest to the largest repeat of an arm; the figures shade it around the median repeat's curve.",
                  None, _refs((P2_REPEATS, "describe_values"))),
    GlossaryEntry(TOPIC_STATISTICS, "Gap",
                  "Difference of two arms' means for a metric, printed beside each arm's spread.",
                  "gap = mean(first arm) - mean(second arm)",
                  _refs((P2_REPEATS, "compare_arms"))),
    GlossaryEntry(TOPIC_STATISTICS, "sep / ovl",
                  "A reading aid, not a test: **sep** when the two arms' [min, max] ranges do not overlap (every repeat of one arm is above "
                  "every repeat of the other), **ovl** when they overlap, **n/a** when an arm has fewer than two defined repeats.",
                  "sep: min(first) > max(second) or min(second) > max(first)",
                  _refs((P2_REPEATS, "compare_arms"), ("src/viking_splatad/eval/report_opening.py", "separation_label"))),
    GlossaryEntry(TOPIC_STATISTICS, "Run-to-run noise and why repeats are needed",
                  "Identical-seed SplatAD runs differ, on some metrics by a large factor, and the project accepted this noise. So no "
                  "comparison is read without repeats.",
                  None, _refs((CLAUDE, "run-to-run noise is accepted"))),
    GlossaryEntry(TOPIC_STATISTICS, "The reading rule",
                  "Compare arms only by the gap against the spread over the repeats: a variant gap counts only when it exceeds the spread "
                  "between repeats without losing precision.",
                  None, _refs((CLAUDE, "Compare arms only by the gap"), (TERMINOLOGY_SOURCE, "A variant gap counts only when"))),
    # ---- Image metrics (auxiliary) ----
    GlossaryEntry(TOPIC_IMAGE, "PSNR, SSIM, LPIPS",
                  "Image-quality scores of SplatAD's built-in evaluation of the rendered camera images, read from the training-time eval on "
                  "the test and guard frames. They are a guard only (the radar claim is judged on occupancy), because below the grass the "
                  "ground-truth pixels are grass.",
                  None, _refs((RUN_QUALITY, "IMAGE_METRIC_NAMES"), (EXTENSION_POINTS, "psnr"), (PLAN, "image metrics cannot show the radar gain"))),
    # ---- Bookkeeping ----
    GlossaryEntry(TOPIC_BOOKKEEPING, "Transform id T<n>",
                  "The number under which a conversion, calibration or metric function is registered with `@transform`; each has a unit test "
                  "that covers it.",
                  None, _refs(("src/viking_splatad/registry.py", "transform"), (CLAUDE, "Every conversion/calibration function"))),
    GlossaryEntry(TOPIC_BOOKKEEPING, "Summary (scripts/20)",
                  "The per-session summary that groups the runs by arm and describes each metric over the repeats.",
                  None, _refs(("scripts/20_summarize_gaussian_occupancy.py", "summarize"))),
    GlossaryEntry(TOPIC_BOOKKEEPING, "Session comparison (scripts/21)",
                  "The report that puts the sessions side by side, arm by arm, and writes this terminology file.",
                  None, _refs(("scripts/21_compare_sessions.py", "render_markdown"))),
    GlossaryEntry(TOPIC_BOOKKEEPING, "Figure manifest (scripts/26)",
                  "The `figures_manifest.json` written beside each session's figures: input hashes, the numbers quoted in captions and each "
                  "figure's status.",
                  None, _refs(("scripts/26_occupancy_figures.py", "figures_manifest.json"))),
)


# Code map of the document's last section: (what it defines, source). Every source is checked by the tests.
CODE_REFERENCES: Tuple[Tuple[str, WhereRef], ...] = (
    ("Centre-of-cell rule: voxelising cuboids into a class-id grid", WhereRef("src/viking_splatad/labels/occupancy.py", "T31")),
    ("Cuboid detection rate (P1cuboid)", WhereRef(DETECTION, "T45")),
    ("Neighbourhood point precision (P1cuboid)", WhereRef(DETECTION, "T46")),
    ("Three-sigma Gaussian occupancy", WhereRef(VOXEL_OCCUPANCY, "T57")),
    ("Peak-alpha occupancy, the binding estimator", WhereRef(VOXEL_OCCUPANCY, "T81")),
    ("Pooled recall, precision and IoU", WhereRef(SCENE_METRICS, "T94")),
    ("Per-object recall, precision and IoU, averaged", WhereRef(SCENE_METRICS, "T95")),
    ("Margin of a class", WhereRef(DETECTION, "median_half_footprint")),
    ("Other-class mask and class-group masks", WhereRef(SCENE_METRICS, "class_group_masks")),
    ("Grown box", WhereRef(DETECTION, "grow_cuboid_horizontally")),
    ("Scored area of one class", WhereRef(DETECTION, "visible_neighbourhood_mask")),
    ("Occupancy values on and next to obstacles", WhereRef(FIGURE_METRICS, "occupancy_samples")),
    ("Threshold curves", WhereRef(FIGURE_METRICS, "threshold_curves")),
)


def glossary_by_topic() -> Dict[str, List[GlossaryEntry]]:
    """Group the glossary entries by topic, in glossary order.

    Returns:
        Topic name -> its entries (every topic of ``TOPIC_ORDER`` is a key, possibly with an empty list).

    Raises:
        ValueError: An entry names a topic that is not in ``TOPIC_ORDER``.
    """
    grouped: Dict[str, List[GlossaryEntry]] = {topic: [] for topic in TOPIC_ORDER}
    for entry in GLOSSARY:
        if entry.topic not in grouped:
            raise ValueError(f"glossary entry {entry.term!r} has unknown topic {entry.topic!r}")
        grouped[entry.topic].append(entry)
    return grouped


def where_markdown(references: Tuple[WhereRef, ...]) -> str:
    """Render the sources of an entry as ``Where:`` text.

    Args:
        references: The entry's ``where``.

    Returns:
        ``Where: `path` (`anchor`); ...``.
    """
    return "Where: " + "; ".join(f"`{reference.path}` (`{reference.anchor}`)" for reference in references)


def glossary_markdown() -> str:
    """Render the whole glossary: one level-3 heading per topic, one bullet per term.

    Returns:
        Markdown (no trailing newline).
    """
    lines: List[str] = []
    for topic, entries in glossary_by_topic().items():
        # An empty topic prints nothing (the tests require every topic to have an entry).
        if not entries:
            continue
        lines += [f"### {topic}", ""]
        for entry in entries:
            lines.append(f"- **{entry.term}.** {entry.definition}")
            if entry.formula:
                lines.append(f"  - Formula: `{entry.formula}`")
            lines.append(f"  - {where_markdown(entry.where)}")
        lines.append("")
    return "\n".join(lines).rstrip("\n")
