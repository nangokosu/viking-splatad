"""Plain-language definitions that open the evaluation reports, rendered from data.

Every number in the text (cell edge, margins, range, field of view, repeat count, cut-offs, cell counts)
is a placeholder filled by :func:`definition_values` from a scripts/20 summary, the figure settings and
the figure's cell counts. Nothing is typed into the text. Scripts/20, scripts/21 and the stand-alone
terminology document call the same functions, so the reports can never disagree.

Checked against the code: ``p2_cell_outcomes``, ``occupancy_samples``, ``threshold_curves``, T81, T94, T95.
"""

import math
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

# Median-range stop rule of SplatAD (section 3.3): a lidar point sits at the first Gaussian where the
# transmittance prod(1 - alpha) falls below this value. Quoted from the paper, not from our config.
SPLATAD_MEDIAN_RANGE_TRANSMITTANCE = 0.5

# Date on which every quoted passage of the references below was checked against the source.
REFERENCE_CHECK_DATE = "2026-11-25"

# References: title, authors, arXiv id, URL, section, date checked. Named constants so the text and the
# terminology document quote identical strings.
SPLATAD_REFERENCE = {
    "title": "SplatAD: Real-Time Lidar and Camera Rendering with 3D Gaussian Splatting for Autonomous Driving",
    "authors": "Hess et al.",
    "arxiv_id": "arXiv:2411.16816",
    "url": "https://arxiv.org/abs/2411.16816",
    "section": "Eqs. 4-5 (section 3.2), median-range rule (section 3.3)",
    "date_checked": REFERENCE_CHECK_DATE,
}
KERBL_REFERENCE = {
    "title": "3D Gaussian Splatting for Real-Time Radiance Field Rendering",
    "authors": "Kerbl et al.",
    "arxiv_id": "arXiv:2308.04079",
    "url": "https://arxiv.org/abs/2308.04079",
    "section": "section 2.3, Eqs. 1-3 (ACM TOG 42(4), 2023)",
    "date_checked": REFERENCE_CHECK_DATE,
}
GAUSSIANFORMER_REFERENCE = {
    "title": "GaussianFormer: Scene as Gaussians for Vision-Based 3D Semantic Occupancy Prediction",
    "authors": "Huang et al.",
    "arxiv_id": "arXiv:2405.17429",
    "url": "https://arxiv.org/abs/2405.17429",
    "section": "section 3.3, Gaussian-to-Voxel Splatting (general reference only)",
    "date_checked": REFERENCE_CHECK_DATE,
}

# Plain arm label by SplatAD training method (CLAUDE.md "Current scope"); the definitions use these labels.
ARM_LABEL_BY_METHOD = {"splatad": "stock", "splatad-radar-init": "radar init",
                       "splatad-radar-init-lidar": "radar init + loss"}
# Plain arm label by the figure suite's arm key (manifest ``arms`` and ``outcome_counts_by_arm``).
ARM_LABEL_BY_FIGURE_KEY = {"stock": "stock", "radar_init": "radar init", "radar_init_loss": "radar init + loss"}
# Shown in place of a value whose source (the threshold-curves figure) is missing.
NOT_AVAILABLE_TEXT = "not available"
# Fallback texts of the curve placeholders when the curve grid is unknown.
CURVE_RANGE_FALLBACK_TEXT = "over a range that is not available (the curve figure failed)"
CURVE_POINTS_FALLBACK_TEXT = "number of values not available"
# Separator of the min-max range of unequal repeat counts (en dash).
REPEAT_RANGE_SEPARATOR = "–"

# Obstacle classes whose margin the text quotes (keys of provenance["margin_m_by_class"]).
MARGIN_CLASS_NAMES = ("tree_trunk", "rock", "bush_or_small_tree")
# Required provenance fields.
PROVENANCE_FIELDS = ("voxel_size_m", "near_band_m", "max_range_m", "vertical_half_fov_rad", "margin_m_by_class")
CURVE_SETTING_FIELDS = ("min", "max", "points")
SCENE_CELL_COUNT_FIELDS = ("on_obstacle", "next_to_obstacle")
# Names of the figures that have a "how to read" text.
FIGURE_NAMES = ("arms_scene_3d", "threshold_curves_scene")
# Outcome counts of one drawn run that the worked example needs, and the arm whose run is the example.
OUTCOME_COUNT_FIELDS = ("hit", "false_alarm", "missed", "context")
EXAMPLE_ARM_KEY = "stock"
# Decimal places of the worked-example recall and precision (3 places is enough to tell arms apart in prose).
EXAMPLE_DECIMALS = 3


def _require_fields(container: Mapping, field_names: Sequence[str], container_label: str) -> None:
    """Raise a clear KeyError when a mapping lacks required fields.

    Args:
        container: Mapping to check.
        field_names: Keys that must be present.
        container_label: Name of the mapping for the error message.

    Raises:
        KeyError: Listing every missing field.
    """
    # Collect all gaps at once so the caller fixes them in one pass.
    missing_fields = [field_name for field_name in field_names if field_name not in container]
    if missing_fields:
        raise KeyError(f"{container_label} is missing required field(s): {', '.join(missing_fields)}")


def repeat_count_text(counts_by_label: Mapping[str, int]) -> str:
    """Repeat count of the arms as text: the shared number, or a range with the per-arm counts.

    Args:
        counts_by_label: Arm label -> number of repeats (not empty).

    Returns:
        "5" when every arm agrees, otherwise e.g. "4–5 (stock 5, radar init 4, radar init + loss 5)".

    Raises:
        ValueError: No arms.
    """
    if not counts_by_label:
        raise ValueError("no arms to count repeats of")
    distinct_counts = set(counts_by_label.values())
    # All arms agree: a single number keeps "each arm is trained N times" true.
    if len(distinct_counts) == 1:
        return str(next(iter(distinct_counts)))
    # Otherwise state the range, then every arm's own count so nothing is hidden.
    per_arm_text = ", ".join(f"{label} {count}" for label, count in counts_by_label.items())
    return f"{min(distinct_counts)}{REPEAT_RANGE_SEPARATOR}{max(distinct_counts)} ({per_arm_text})"


def threshold_rule_sentence(binding_threshold: str) -> str:
    """Sentence relating our binding cut-off to SplatAD's median-range stop rule.

    Args:
        binding_threshold: First P2 threshold as text.

    Returns:
        The "same quantity, same cut-off" wording when the cut-off equals SplatAD's, otherwise a sentence
        naming both values and saying whether our filled cell is stricter or looser.
    """
    binding_value = float(binding_threshold)
    # The occupancy 1 - prod(1 - alpha) passes the cut-off exactly when the transmittance prod(1 - alpha)
    # falls below 1 - cut-off, so a larger cut-off than SplatAD's is the stricter rule.
    if math.isclose(binding_value, SPLATAD_MEDIAN_RANGE_TRANSMITTANCE):
        return (f"Our binding cut-off is that same {binding_threshold}. So \"the cell is filled\" means \"SplatAD's own "
                "renderer would consider a ray through these Gaussians stopped\".")
    comparison_word = "stricter" if binding_value > SPLATAD_MEDIAN_RANGE_TRANSMITTANCE else "looser"
    return (f"SplatAD's rule uses {SPLATAD_MEDIAN_RANGE_TRANSMITTANCE:g}; our binding cut-off is {binding_threshold}, "
            f"so a filled cell is {comparison_word} than SplatAD's stop rule.")


def definition_values(summary: Mapping, curve_settings: Optional[Mapping], scene_cell_counts: Mapping,
                      outcome_counts_by_arm: Mapping,
                      curve_repeat_counts: Optional[Mapping[str, int]] = None) -> Dict[str, Any]:
    """Collect every placeholder value of the definitions text from the report inputs.

    Args:
        summary: A scripts/20 summary (schema ``.../gaussian-occupancy-summary/2``) with ``session``,
            ``p2_thresholds`` (the first is the binding one), ``arms`` ({profile: {"runs": [...]}}) and
            ``provenance``.
        curve_settings: Threshold-curve grid, ``{"min", "max", "points"}``, or None when the curve figure
            failed (the curve placeholders then read "not available").
        scene_cell_counts: Whole-scene cells, ``{"on_obstacle", "next_to_obstacle"}`` (manifest ``scene_counts``).
        outcome_counts_by_arm: ``{arm_key: {"hit", "false_alarm", "missed", "context"}}``, the drawn run's
            whole-scene cell counts at the binding cut-off. Arm keys: stock, radar_init, radar_init_loss.
            The worked example uses the ``stock`` entry.
        curve_repeat_counts: Optional ``{figure arm key: repeats behind the curve band}`` (manifest
            ``campaign_run_count``); defaults to the summary's run counts.

    Returns:
        Flat dict of placeholder name to value.

    Raises:
        KeyError: A required field is missing.
        ValueError: No thresholds, no arms, hit + missed differing from the on-obstacle cell count, or an
            empty recall or precision denominator. Unequal repeat counts do not raise: ``repeat_count`` is then text.
    """
    # Boundary validation: every field is checked here so the text functions can trust the values.
    _require_fields(summary, ("session", "p2_thresholds", "arms", "provenance"), "summary")
    _require_fields(summary["provenance"], PROVENANCE_FIELDS, "summary['provenance']")
    _require_fields(summary["provenance"]["margin_m_by_class"], MARGIN_CLASS_NAMES,
                    "summary['provenance']['margin_m_by_class']")
    if curve_settings is not None:
        _require_fields(curve_settings, CURVE_SETTING_FIELDS, "curve_settings")
    _require_fields(scene_cell_counts, SCENE_CELL_COUNT_FIELDS, "scene_cell_counts")
    _require_fields(outcome_counts_by_arm, (EXAMPLE_ARM_KEY,), "outcome_counts_by_arm")
    _require_fields(outcome_counts_by_arm[EXAMPLE_ARM_KEY], OUTCOME_COUNT_FIELDS,
                    f"outcome_counts_by_arm['{EXAMPLE_ARM_KEY}']")
    provenance = summary["provenance"]
    thresholds = [str(threshold) for threshold in summary["p2_thresholds"]]
    if not thresholds:
        raise ValueError("summary['p2_thresholds'] is empty")

    # Repeat count: the shared number, or a range with the per-arm counts when the arms differ.
    if not summary["arms"]:
        raise ValueError("summary['arms'] is empty")
    repeat_counts_by_label = {
        ARM_LABEL_BY_METHOD.get(arm.get("train_method"), profile): len(arm["runs"])
        for profile, arm in summary["arms"].items()}
    # The shared count stays a number (table spec "d"); unequal counts become text.
    distinct_repeat_counts = set(repeat_counts_by_label.values())
    repeat_count = (next(iter(distinct_repeat_counts)) if len(distinct_repeat_counts) == 1
                    else repeat_count_text(repeat_counts_by_label))
    # The curve band's repeat count comes from the figure suite when given (it counts the campaign runs).
    curve_repeat_count = repeat_count_text(repeat_counts_by_label)
    if curve_repeat_counts:
        curve_repeat_count = repeat_count_text({ARM_LABEL_BY_FIGURE_KEY.get(arm_key, arm_key): run_count
                                                for arm_key, run_count in curve_repeat_counts.items()})
    # Curve placeholders: real values, or fallbacks when the figure that holds the grid failed.
    if curve_settings is None:
        curve_values = {"curve_min": NOT_AVAILABLE_TEXT, "curve_max": NOT_AVAILABLE_TEXT,
                        "curve_points": NOT_AVAILABLE_TEXT, "curve_range_text": CURVE_RANGE_FALLBACK_TEXT,
                        "curve_points_text": CURVE_POINTS_FALLBACK_TEXT}
    else:
        curve_values = {"curve_min": curve_settings["min"], "curve_max": curve_settings["max"],
                        "curve_points": curve_settings["points"],
                        "curve_range_text": f"from {curve_settings['min']:g} to {curve_settings['max']:g}",
                        "curve_points_text": f"{curve_settings['points']} evenly spaced values"}

    # Worked example from the stock arm's drawn run. Every obstacle cell is either a hit or missed, so
    # hit + missed must equal the on-obstacle cell count; a mismatch means the counts come from different data.
    example_counts = outcome_counts_by_arm[EXAMPLE_ARM_KEY]
    obstacle_cell_count = example_counts["hit"] + example_counts["missed"]
    if obstacle_cell_count != scene_cell_counts["on_obstacle"]:
        raise ValueError(f"hit + missed = {obstacle_cell_count} but on_obstacle = {scene_cell_counts['on_obstacle']}")
    filled_scored_count = example_counts["hit"] + example_counts["false_alarm"]
    if obstacle_cell_count == 0 or filled_scored_count == 0:
        raise ValueError("worked example needs hit + missed > 0 and hit + false_alarm > 0")
    # recall = hit / (hit + missed); precision = hit / (hit + false alarm).
    example_recall = example_counts["hit"] / obstacle_cell_count
    example_precision = example_counts["hit"] / filled_scored_count
    # IoU = hit / (hit + missed + false alarm) = TP / (TP + FP + FN), as pooled_hidden_metrics (T94) computes it.
    example_iou = example_counts["hit"] / (obstacle_cell_count + example_counts["false_alarm"])

    margins = provenance["margin_m_by_class"]
    return {
        "session": summary["session"],
        "voxel_size": provenance["voxel_size_m"],
        "margin_tree_trunk": margins["tree_trunk"],
        "margin_rock": margins["rock"],
        "margin_bush_or_small_tree": margins["bush_or_small_tree"],
        "max_range_m": provenance["max_range_m"],
        # Stored in radians by the run reports; the text quotes degrees.
        "vertical_half_fov_deg": float(np.degrees(provenance["vertical_half_fov_rad"])),
        "near_band_m": provenance["near_band_m"],
        "repeat_count": repeat_count,
        "curve_repeat_count": curve_repeat_count,
        "threshold": thresholds[0],
        "threshold_rule_sentence": threshold_rule_sentence(thresholds[0]),
        "p2_thresholds": ", ".join(thresholds),
        **curve_values,
        "n_on": scene_cell_counts["on_obstacle"],
        "n_off": scene_cell_counts["next_to_obstacle"],
        "example_hit": example_counts["hit"],
        "example_missed": example_counts["missed"],
        "example_false_alarm": example_counts["false_alarm"],
        "example_recall": f"{example_recall:.{EXAMPLE_DECIMALS}f}",
        "example_precision": f"{example_precision:.{EXAMPLE_DECIMALS}f}",
        "example_iou": f"{example_iou:.{EXAMPLE_DECIMALS}f}",
    }


def accumulated_opacity(alphas: Sequence[float]) -> float:
    """Total opacity a ray collects from Gaussians: 1 - prod(1 - alpha).

    The product is commutative, so the result does not depend on the Gaussians' order. It lies in [0, 1]
    for alphas in [0, 1] and equals 0 for no Gaussians.

    Args:
        alphas: Per-Gaussian opacity values in [0, 1].

    Returns:
        The accumulated opacity.
    """
    # Transmittance = probability the ray passes every Gaussian; blocking = its complement.
    transmittance = float(np.prod(1.0 - np.asarray(alphas, dtype=float)))
    return 1.0 - transmittance


def front_to_back_weights(alphas: Sequence[float]) -> np.ndarray:
    """SplatAD / 3DGS front-to-back blending weights w_i = alpha_i * prod_{j<i}(1 - alpha_j).

    Telescoping: with T_k = prod_{i<=k}(1 - alpha_i) and T_0 = 1, w_k = T_{k-1} - T_k, so
    sum_k w_k = T_0 - T_n = 1 - prod(1 - alpha) = accumulated_opacity(alphas).

    Args:
        alphas: Opacities in front-to-back order, values in [0, 1].

    Returns:
        Array of the same length with the blending weights.
    """
    alpha_array = np.asarray(alphas, dtype=float)
    # Exclusive cumulative product: light left in front of each Gaussian (1 for the first).
    transmittance_before = np.concatenate([[1.0], np.cumprod(1.0 - alpha_array)[:-1]])
    return alpha_array * transmittance_before


def format_reference(reference: Mapping[str, str]) -> str:
    """Render one reference constant as a markdown bullet body.

    Args:
        reference: One of the ``*_REFERENCE`` constants.

    Returns:
        ``authors, *title*, arXiv id, URL, section; checked <date>``.
    """
    return (f"{reference['authors']}, *{reference['title']}*, {reference['arxiv_id']}, {reference['url']}, "
            f"{reference['section']}; passages checked {reference['date_checked']}")


# Points 1-7 of the definitions. Placeholders are str.format fields; literal subscript braces are doubled.
_DEFINITIONS_TEMPLATE = """\
## Definitions

Plain words first, the formula last. Every number is read from the run reports, never typed in.

1. **Cells.** The scene is cut into small cubes ("cells") of edge {voxel_size:g} m. Everything below is about cells.
2. **Ground truth, "on an obstacle" and "next to an obstacle".** People drew a box around every tree trunk, rock and bush ("obstacles"). The boxes are used only for scoring, never for training.
   - **On an obstacle (G):** a cell whose **centre** lies inside an obstacle box.
   - **The scored area around a class (N_c):** for each class (trunk, rock, bush) separately:
     1. **Margin:** m_c = half of the class's typical footprint width, i.e. the median over all of the session's boxes of that class of (length + width) / 4. Values for session {session}: trunk {margin_tree_trunk:.3f} m, rock {margin_rock:.3f} m, bush {margin_bush_or_small_tree:.3f} m.
     2. **Grow:** widen every box of the class by m_c on each of its four sides, along the box's own axes, leaving its height unchanged. A typical object's grown box is therefore about twice its own width.
     3. **Collect:** N_c = the cells whose centre lies inside a grown box.
   - **Only space the lidar could have covered is scored:** a box counts only when its centre is within the lidar's range ({max_range_m:g} m) and vertical field of view (±{vertical_half_fov_deg:.1f}°) of at least one recorded sweep. The same range and field-of-view test is applied to every cell of N_c.
   - **Obstacles as one group:** N = N_trunk ∪ N_rock ∪ N_bush, minus any cell whose centre lies in a box of a different, non-obstacle annotated class. Those cells count neither for nor against.
   - **Next to an obstacle:** N ∖ G, the cells in the grown area that are not inside any obstacle box.
   - **Scored:** only cells in N; everything else in the scene is ignored.
   - **Caveat:** the boxes are rectangular and a cell counts by its centre only. So for a trunk thinner than one cell ({voxel_size:g} m), whether the trunk gets 0, 1 or 2 cells across depends on where the grid lines fall.
3. **What the model contains.** A trained SplatAD model is millions of small fuzzy blobs (3D Gaussians). Each blob has an opacity from 0 (see-through) to 1 (solid).
4. **Occupancy of a cell** is how solid the model says the cell is, from 0 (empty) to 1 (solid).
   - Each blob adds its opacity to the cell its centre lies in, and a fading share to nearby cells.
   - Blobs combine like stacked tinted glass: light passes through a blob of opacity α with probability (1 − α). So occupancy = 1 − ∏_k(1 − α_k), which grows with every blob but never passes 1.
   - **Why opacities are combined this way (the source of the definition).** The rule is not new. It reads, at a cell, the same accumulation SplatAD uses to render a lidar ray. The design choice is recorded in PLAN §5 (P2) and `docs/research/evaluation_caveats.md`, "Blocking probability". Quotes are verbatim from the sources below.
     1. **How SplatAD blends Gaussians along a ray.** SplatAD ({splatad_reference}) models the scene as "translucent 3D Gaussians with learnable occupancy o∈(0,1)" (§3.1).
        - **Eq. 4 (§3.2):** [F_p^rgb, F_p] = Σ_i [f_i^rgb, f_i] α_i ∏_{{j=1}}^{{i−1}}(1 − α_j). Each Gaussian's own opacity α_i is multiplied by ∏_{{j<i}}(1 − α_j), the light still left in front of it (the transmittance).
        - **Eq. 5 (§3.2):** α_i = sqrt(|Σ_i^I| / |Σ_i^I + s·I|) · o_i · exp(−½ Δ_iᵀ (Σ_i^I + s·I)⁻¹ Δ_i). The Gaussian's α_i is its learned opacity o_i times its falloff at the ray, projected onto the sensor plane.
        - **§3.3:** "For each lidar point, we α-blend the Gaussians' features 𝐟_i using Eqs. 4, 5 and 6." So the lidar uses exactly these weights.
     2. **SplatAD's own 0.5 rule, which is our definition.** SplatAD places a rendered lidar point at the median range, "the rolling shutter corrected range of the first Gaussian that satisfies ∏_{{j=1}}^{{i}}(1−α_j) < 0.5 during the α-blending in Eq. 4" (§3.3). It is used at inference: "While we use the expected range for training, the median range is used during inference as it, in contrast to the expected range, does not yield depths between Gaussians." (§3.3). In words: SplatAD says a ray has hit something once **1 − ∏(1 − α) passes 0.5**. Our occupancy is that same quantity. {threshold_rule_sentence}
     3. **The lineage.** 3D Gaussian Splatting ({kerbl_reference}) gives the same blending, Eq. 3: C = Σ_{{i∈N}} c_i α_i ∏_{{j=1}}^{{i−1}}(1 − α_j). It states: "Point-based α-blending and NeRF-style volumetric rendering share essentially the same image formation model." (§2.3). So ∏(1 − α_j) is the transmittance T_i of volume rendering (its Eqs. 1–2).
     4. **Why 1 − ∏(1 − α), as one number per cell.** The blending weights along a ray always add up to Σ_i α_i ∏_{{j<i}}(1 − α_j) = 1 − ∏_i(1 − α_i). Proof by telescoping: with T_k = ∏_{{i≤k}}(1 − α_i) and T_0 = 1, each weight equals T_{{k−1}} − T_k, so the sum is T_0 − T_n = 1 − T_n. A unit test checks this identity. The product does not depend on the order of the Gaussians, so no particular ray direction is needed. The result is the total opacity a ray collects from these Gaussians, between 0 and 1.
   - **How ours differs from SplatAD's α (stated here, not hidden):**
     - SplatAD evaluates each Gaussian **projected** onto the image or lidar plane, with a small blur term s·I that matches the pixel size or beam divergence (Eq. 5). We evaluate the **3D** Gaussian at a point: α_k(x) = σ(o_k) · exp(−½ (x − μ_k)ᵀ Σ_k⁻¹ (x − μ_k)), at the cell centre, or at its own peak for the cell holding its mean (T81).
     - We combine every Gaussian touching the cell, rather than those along one beam.
     - So occupancy is a blocking probability at one point of the cell, in the spirit of SplatAD's accumulation. It is not the fraction of the cell that is filled. Like the renderer, it assumes each Gaussian blocks light independently.
   - **General reference, not the source of our rule.** Using 3D Gaussians to predict 3D occupancy has been done before ({gaussianformer_reference}). It adds up the Gaussians' contributions at voxel positions. Ours combines opacities with the renderer's rule above instead, so that the result stays in [0, 1] and matches SplatAD's stop rule.
   - It is read from the blobs directly, not from a rendered image, so something hidden behind grass still counts.
5. **Filled, recall and precision.** Pick a cut-off t; a cell with occupancy ≥ t counts as *filled*. Let O_t be the set of filled cells.
   - **(a) What is scored and what is not.** Only the cells in N are scored: the cells **on** an obstacle (G) and the cells **next to** an obstacle (N ∖ G, the ring, {n_off} cells). "Not ground truth" here means only that ring, not the whole scene. Every other cell is ignored, even if the model fills it, so the ring is what makes false alarms countable: a filled cell far from every obstacle is never counted. Cells on another labelled class are excluded from both metrics.
   - **(b) The 2 × 2 table.** Each scored cell is either filled or empty, and either on an obstacle or next to one:

     | | on an obstacle (G) | next to an obstacle (N ∖ G) |
     |---|---|---|
     | **model filled** | hit | false alarm |
     | **model left empty** | missed | correct empty |

     "Correct empty" enters neither metric.
   - **(c) Recall** = hit / (hit + missed) = |O_t ∩ G| / |G|. *Did it rebuild the obstacles?* The denominator is every obstacle cell ({n_on}); it depends only on the ground truth. So recall cannot be gamed by leaving things out, but it ignores false alarms.
   - **(d) Precision** = hit / (hit + false alarm) = |O_t ∩ G| / |O_t ∩ N|. *When it fills something, is it real?* The denominator is every filled scored cell; it depends on the model. Filling everything near obstacles raises recall and lowers precision. This is why both are read together, and why the reading rule asks for higher recall *without lower precision*.
   - **IoU** (intersection over union) = hit / (hit + missed + false alarm) = |O_t ∩ G| / |(O_t ∪ G) ∩ N|. In words: the overlap between what the model filled and the obstacles, divided by everything either of them covers in the scored area. It puts both kinds of error in one number: a miss and a false alarm each lower it. Correct-empty cells do not enter it. It is always ≤ min(recall, precision). It is a summary only: the reading rule uses recall and precision.
   - **(e) Worked example.** The stock arm's drawn run in {session} at t = {threshold}, with its real counts: {example_hit} hits, {example_missed} missed, {example_false_alarm} false alarms → recall {example_recall}, precision {example_precision}, IoU {example_iou}. These are one run; the tables give mean ± std over the repeats.
   - **(f) Pooled vs mean over objects.** *Pooled* counts all cells together, so big objects weigh most. *Mean over objects* scores each object k alone, with its own cells G_k and its own ring N_k, and then averages, so every object weighs the same. An occupied cell inside several rings counts against each of them.
   - **(g) Why precision stays below 1.** "Next to an obstacle" is not "empty space". The ring also holds things nobody labelled: grass (tall in June), ground and small plants, which really are solid. It also catches the edges of a trunk's own Gaussians, because a trunk is thinner than one cell. So precision stays below 1 even for a correct model. This is a likely cause, not a tested finding.
   - **(h) How the threshold changes both.** Raising t moves cells from filled to empty, so recall can only fall; precision usually rises.
6. **Hidden cells (reference for the later round).** We replay every recorded lidar beam.
   - A cell is *hidden* if no beam reached it, none passed through it, and at least one was stopped before it (by grass, for example).
   - *H_near* keeps the hidden cells within {near_band_m:g} m of the driven path.
   - Hidden-cell metrics are the formulas of point 5, restricted to H_near. The figures will take the same form as the whole-scene ones.
   - Stock SplatAD is expected to leave hidden cells mostly empty, since no lidar or camera signal reached them. That gap is what radar is meant to close.
7. **Statistics.** Each arm is trained {repeat_count} times. A table cell shows mean ± std over the repeats. Gap = difference of means. "sep" means the two arms' [min, max] ranges do not overlap. A gap is read only when it is larger than the spread within the arms.
"""

# "How to read" text per figure, same placeholder rules as above.
_FIGURE_READING_TEMPLATES = {
    "arms_scene_3d": """\
**How to read this figure**
- **GT panel:** the obstacle boxes, coloured by class, for orientation.
- **Arm panels:** green = hit (filled, on an obstacle); red = false alarm (filled, next to an obstacle); grey = missed (empty, on an obstacle); faint grey = the scored area, the ring included. Correct-empty cells are not drawn. Cut-off t = {threshold}. The terms are defined in point 5.
- **Difference panel:** hits only radar init + loss has, hits only stock has, and hits both have.
- One run per arm (the median repeat), so it illustrates and does not decide anything.
""",
    "threshold_curves_scene": """\
**How to read this figure**
- **What is plotted:** recall (left) and precision (right) recomputed at each cut-off t {curve_range_text} ({curve_points_text}). Solid lines are pooled and dashed lines are the mean over objects. The line is the median repeat. The band spans, at each t, the lowest and highest value of both the pooled and the mean-over-objects curve over the {curve_repeat_count} repeats of the arm; it is therefore wider than either line's own spread, so "bands do not overlap" is a conservative reading. Dotted lines mark the table cut-offs {p2_thresholds}.
- **Definitions:** see point 5. Only recall and precision are plotted; IoU is in the tables. Recall = hit / (hit + missed) has all {n_on} obstacle cells as its denominator; precision = hit / (hit + false alarm) has every filled scored cell as its denominator.
- **How the curves move:** raising t fills fewer cells, so recall can only fall and precision usually rises.
- **When one arm beats another:** its curve is higher and the bands do not overlap.
- **Why t runs {curve_range_text}:** the extremes are left out because they say nothing about the model. At t = 0 every cell counts as filled (recall 1, precision = the share of obstacle cells). At t ≈ 1 precision rests on a handful of cells and jumps.
""",
}


def definitions_markdown(values: Mapping) -> str:
    """Render points 1-7 of the definitions as markdown.

    Args:
        values: Output of :func:`definition_values`.

    Returns:
        Markdown starting with a ``## Definitions`` heading, references included.

    Raises:
        KeyError: A placeholder value is missing from ``values``.
    """
    return _DEFINITIONS_TEMPLATE.format(
        splatad_reference=format_reference(SPLATAD_REFERENCE),
        kerbl_reference=format_reference(KERBL_REFERENCE),
        gaussianformer_reference=format_reference(GAUSSIANFORMER_REFERENCE),
        **values,
    )


def figure_reading_markdown(figure_name: str, values: Mapping) -> str:
    """Render the "how to read" text of one figure.

    Args:
        figure_name: One of ``FIGURE_NAMES``.
        values: Output of :func:`definition_values`.

    Returns:
        Markdown text to print under the figure.

    Raises:
        ValueError: Unknown figure name.
        KeyError: A placeholder value is missing from ``values``.
    """
    if figure_name not in _FIGURE_READING_TEMPLATES:
        raise ValueError(f"unknown figure {figure_name!r}; expected one of {FIGURE_NAMES}")
    return _FIGURE_READING_TEMPLATES[figure_name].format(**values)
