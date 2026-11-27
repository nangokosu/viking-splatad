"""Tests for the shared evaluation definitions text (synthetic summary, known-answer values)."""

import copy
import string

import numpy as np
import pytest

from viking_splatad.eval import evaluation_terminology as terminology

# Synthetic inputs; the expected strings are derived in the tests from these same numbers.
HALF_FOV_DEGREES = 15.0
# A farthest-single-return value that differs from the scored range; the text must never print it.
FARTHEST_RETURN_M = 91.5
SYNTHETIC_SUMMARY = {
    "session": "synthetic_session",
    "p2_thresholds": ["0.5", "0.8"],
    "arms": {"arm_a": {"runs": [{}, {}, {}]}, "arm_b": {"runs": [{}, {}, {}]}},
    "provenance": {
        "voxel_size_m": 0.25,
        "scored_range_m": 70.0,
        "scored_range_quantile": 0.9999,
        "vertical_half_fov_rad": float(np.radians(HALF_FOV_DEGREES)),
        "max_range_m": FARTHEST_RETURN_M,
        "margin_m_by_class": {"tree_trunk": 0.123, "rock": 0.234, "bush_or_small_tree": 0.345},
    },
}
SYNTHETIC_CURVE = {"min": 0.2, "max": 0.9, "points": 15}
SYNTHETIC_CELLS = {"on_obstacle": 1234, "next_to_obstacle": 5678}
# Known-answer outcome counts: hit + missed == on_obstacle.
SYNTHETIC_OUTCOMES = {
    "stock": {"hit": 234, "missed": 1000, "false_alarm": 702, "context": 40},
    "radar_init": {"hit": 300, "missed": 934, "false_alarm": 800, "context": 40},
    "radar_init_loss": {"hit": 400, "missed": 834, "false_alarm": 900, "context": 40},
}
RANDOM_SEED = 7
RANDOM_TRIAL_COUNT = 20
MAX_ALPHA_COUNT = 30


def build_values():
    """Return placeholder values from the synthetic inputs."""
    return terminology.definition_values(SYNTHETIC_SUMMARY, SYNTHETIC_CURVE, SYNTHETIC_CELLS, SYNTHETIC_OUTCOMES)


def test_values_converted_and_read():
    """Degrees come from radians; repeat count is the runs per arm; the first threshold binds."""
    values = build_values()
    assert values["vertical_half_fov_deg"] == pytest.approx(HALF_FOV_DEGREES)
    assert values["repeat_count"] == len(SYNTHETIC_SUMMARY["arms"]["arm_a"]["runs"])
    assert values["threshold"] == SYNTHETIC_SUMMARY["p2_thresholds"][0]


def test_definitions_fill_every_placeholder():
    """No format field survives and the data values appear in the text."""
    text = terminology.definitions_markdown(build_values())
    # No format field name may be left over; literal subscript braces (escaped in the template) must survive.
    template_text = terminology._DEFINITIONS_TEMPLATE
    field_names = {field_name for _, field_name, _, _ in string.Formatter().parse(template_text) if field_name}
    assert field_names
    for field_name in field_names:
        assert "{" + field_name + "}" not in text
    assert "∏_{j<i}(1 − α_j)" in text and "T_{k−1}" in text
    # Real symbols, not ASCII stand-ins.
    for symbol in ("∩", "∪", "∏", "Σ", "≥", "±", "α", "σ", "μ", "Σ_k⁻¹", "°"):
        assert symbol in text
    # Verbatim paper quotes.
    for quote in ("translucent 3D Gaussians with learnable occupancy o∈(0,1)",
                  "For each lidar point, we α-blend the Gaussians' features 𝐟_i using Eqs. 4, 5 and 6.",
                  "the rolling shutter corrected range of the first Gaussian that satisfies ∏_{j=1}^{i}(1−α_j) < 0.5 during the α-blending in Eq. 4",
                  "does not yield depths between Gaussians.",
                  "Point-based α-blending and NeRF-style volumetric rendering share essentially the same image formation model."):
        assert quote in text
    for expected in ("0.25 m", "0.123 m", "0.234 m", "0.345 m", "70 m (the distance within which 99.99% of the session's lidar returns lie)", "±15.0°",
                     "trained 3 times", "arXiv:2411.16816", "arXiv:2308.04079", "arXiv:2405.17429",
                     terminology.REFERENCE_CHECK_DATE, "telescoping"):
        assert expected in text
    # Points 1-6 all present, in order.
    point_positions = [text.index(f"\n{point_number}. **") for point_number in range(1, 7)]
    assert point_positions == sorted(point_positions)


def test_missing_provenance_field_raises():
    """Dropping any provenance field gives a KeyError naming it."""
    for field_name in terminology.PROVENANCE_FIELDS:
        broken_summary = copy.deepcopy(SYNTHETIC_SUMMARY)
        del broken_summary["provenance"][field_name]
        with pytest.raises(KeyError, match=field_name):
            terminology.definition_values(broken_summary, SYNTHETIC_CURVE, SYNTHETIC_CELLS, SYNTHETIC_OUTCOMES)


def test_missing_margin_class_curve_and_cell_fields_raise():
    """Missing margin class, curve setting or cell count raises."""
    broken_summary = copy.deepcopy(SYNTHETIC_SUMMARY)
    del broken_summary["provenance"]["margin_m_by_class"]["rock"]
    with pytest.raises(KeyError, match="rock"):
        terminology.definition_values(broken_summary, SYNTHETIC_CURVE, SYNTHETIC_CELLS, SYNTHETIC_OUTCOMES)
    with pytest.raises(KeyError, match="points"):
        terminology.definition_values(SYNTHETIC_SUMMARY, {"min": 0.2, "max": 0.9}, SYNTHETIC_CELLS, SYNTHETIC_OUTCOMES)
    with pytest.raises(KeyError, match="next_to_obstacle"):
        terminology.definition_values(SYNTHETIC_SUMMARY, SYNTHETIC_CURVE, {"on_obstacle": 1}, SYNTHETIC_OUTCOMES)


def test_unequal_repeat_counts_give_a_text_not_an_error():
    """Arms with different numbers of runs give "min-max (per-arm counts)" instead of raising."""
    unequal_summary = copy.deepcopy(SYNTHETIC_SUMMARY)
    unequal_summary["arms"]["arm_b"]["runs"].append({})
    values = terminology.definition_values(unequal_summary, SYNTHETIC_CURVE, SYNTHETIC_CELLS, SYNTHETIC_OUTCOMES)
    assert values["repeat_count"] == "3–4 (arm_a 3, arm_b 4)"
    assert "trained 3–4 (arm_a 3, arm_b 4) times" in terminology.definitions_markdown(values)


def test_threshold_sentence_follows_the_binding_cut_off():
    """Equal to SplatAD's 0.5: the same-cut-off wording; otherwise both values and stricter or looser."""
    equal_text = terminology.definitions_markdown(build_values())
    assert "binding cut-off reuses that value, 0.5" in equal_text
    assert "does not mean \"SplatAD's renderer would consider a ray stopped\"" in equal_text
    for cut_off, comparison_word in (("0.8", "stricter"), ("0.3", "looser")):
        changed_summary = copy.deepcopy(SYNTHETIC_SUMMARY)
        changed_summary["p2_thresholds"] = [cut_off, "0.9"]
        text = terminology.definitions_markdown(
            terminology.definition_values(changed_summary, SYNTHETIC_CURVE, SYNTHETIC_CELLS, SYNTHETIC_OUTCOMES))
        assert "binding cut-off reuses that value" not in text
        assert (f"SplatAD's rule uses {terminology.SPLATAD_MEDIAN_RANGE_TRANSMITTANCE:g}; our binding cut-off is {cut_off}, "
                f"so our cut-off is {comparison_word} than SplatAD's value") in text


def test_curve_settings_none_renders_fallbacks_and_session_is_named():
    """curve_settings=None fills the curve placeholders with "not available" text; the session name appears."""
    values = terminology.definition_values(SYNTHETIC_SUMMARY, None, SYNTHETIC_CELLS, SYNTHETIC_OUTCOMES)
    text = terminology.figure_reading_markdown("threshold_curves_scene", values)
    assert "not available" in text and "{" not in text
    definitions = terminology.definitions_markdown(values)
    assert "Values for session synthetic_session: trunk" in definitions
    assert "drawn run in synthetic_session at t" in definitions


def test_curve_band_text_says_what_the_band_spans():
    """Each line has its own band (same curve over the repeats), and curve_repeat_counts feed the count."""
    values = terminology.definition_values(SYNTHETIC_SUMMARY, SYNTHETIC_CURVE, SYNTHETIC_CELLS, SYNTHETIC_OUTCOMES,
                                           {"stock": 5, "radar_init": 4})
    text = terminology.figure_reading_markdown("threshold_curves_scene", values)
    assert "the lowest and highest value of that same curve (pooled or mean over objects)" in text
    assert "over the 4–5 (stock 5, radar init 4) repeats of the arm" in text
def test_telescoping_identity_and_order_independence():
    """sum_i a_i prod_{j<i}(1-a_j) == 1 - prod(1-a), computed independently by a plain loop."""
    random_generator = np.random.default_rng(RANDOM_SEED)
    for _ in range(RANDOM_TRIAL_COUNT):
        alpha_count = int(random_generator.integers(1, MAX_ALPHA_COUNT))
        alphas = random_generator.uniform(0.0, 1.0, size=alpha_count)
        # Reference weights from the definition, with an explicit running transmittance.
        running_transmittance = 1.0
        expected_weights = []
        for alpha in alphas:
            expected_weights.append(alpha * running_transmittance)
            running_transmittance *= 1.0 - alpha
        weights = terminology.front_to_back_weights(alphas)
        assert np.allclose(weights, expected_weights)
        assert np.isclose(weights.sum(), terminology.accumulated_opacity(alphas))
        assert np.isclose(terminology.accumulated_opacity(alphas), 1.0 - running_transmittance)
        shuffled_alphas = random_generator.permutation(alphas)
        assert np.isclose(terminology.accumulated_opacity(shuffled_alphas), terminology.accumulated_opacity(alphas))


def test_accumulated_opacity_known_values():
    """Two half-opaque blobs give 0.75; no blobs give 0."""
    assert terminology.accumulated_opacity([0.5, 0.5]) == pytest.approx(1.0 - 0.5 * 0.5)
    assert terminology.accumulated_opacity([]) == 0.0


def test_figure_reading_contains_values_and_unknown_raises():
    """Each figure's text carries its values with no unfilled fields; unknown names raise."""
    values = build_values()
    expected_by_figure = {
        "arms_scene_3d": ["t = 0.5", "green = hit", "red = false alarm", "grey = missed", "Correct-empty cells are not drawn"],
        "threshold_curves_scene": ["point 5", "all 1234 obstacle cells", "from 0.2 to 0.9", "15 evenly spaced", "0.5, 0.8", "the 3 repeats"],
    }
    assert set(expected_by_figure) == set(terminology.FIGURE_NAMES)
    for figure_name, expected_fragments in expected_by_figure.items():
        text = terminology.figure_reading_markdown(figure_name, values)
        figure_fields = {name for _, name, _, _ in
                         string.Formatter().parse(terminology._FIGURE_READING_TEMPLATES[figure_name]) if name}
        assert all("{" + name + "}" not in text for name in figure_fields)
        for fragment in expected_fragments:
            assert fragment in text
    with pytest.raises(ValueError, match="unknown figure"):
        terminology.figure_reading_markdown("no_such_figure", values)


def test_example_values_computed_from_counts():
    """Worked-example recall and precision come from the stock counts, rounded by the named constant."""
    values = build_values()
    stock = SYNTHETIC_OUTCOMES["stock"]
    decimals = terminology.EXAMPLE_DECIMALS
    assert values["example_recall"] == f"{stock['hit'] / (stock['hit'] + stock['missed']):.{decimals}f}"
    assert values["example_precision"] == f"{stock['hit'] / (stock['hit'] + stock['false_alarm']):.{decimals}f}"
    assert (values["example_hit"], values["example_missed"], values["example_false_alarm"]) == (234, 1000, 702)
    text = terminology.definitions_markdown(values)
    assert f"recall {values['example_recall']}, precision {values['example_precision']}" in text


def test_outcome_count_mismatch_and_missing_fields_raise():
    """hit + missed != on_obstacle raises; a missing stock entry or count field raises."""
    broken_outcomes = copy.deepcopy(SYNTHETIC_OUTCOMES)
    broken_outcomes["stock"]["missed"] += 1
    with pytest.raises(ValueError, match="hit \\+ missed"):
        terminology.definition_values(SYNTHETIC_SUMMARY, SYNTHETIC_CURVE, SYNTHETIC_CELLS, broken_outcomes)
    with pytest.raises(KeyError, match="stock"):
        terminology.definition_values(SYNTHETIC_SUMMARY, SYNTHETIC_CURVE, SYNTHETIC_CELLS, {})
    del broken_outcomes["stock"]["context"]
    with pytest.raises(KeyError, match="context"):
        terminology.definition_values(SYNTHETIC_SUMMARY, SYNTHETIC_CURVE, SYNTHETIC_CELLS, broken_outcomes)


def test_point_five_names_the_two_by_two_terms():
    """Point 5 defines the four outcomes, both denominators and both counts from the data."""
    text = terminology.definitions_markdown(build_values())
    point_five = text[text.index("\n5. **"):text.index("\n6. **")]
    for term in ("hit", "missed", "false alarm", "correct empty", "hit / (hit + missed)",
                 "hit / (hit + false alarm)", "5678 cells", "(1234)", "without lower precision"):
        assert term in point_five
    assert "occupancy_boxplots" not in terminology.FIGURE_NAMES


def test_example_iou_from_counts_and_bounded_by_recall_and_precision():
    """IoU = hit / (hit + missed + false alarm); it is at most min(recall, precision); text shows it."""
    values = build_values()
    stock = SYNTHETIC_OUTCOMES["stock"]
    expected_iou = stock["hit"] / (stock["hit"] + stock["missed"] + stock["false_alarm"])
    assert values["example_iou"] == f"{expected_iou:.{terminology.EXAMPLE_DECIMALS}f}"
    assert expected_iou <= min(float(values["example_recall"]), float(values["example_precision"]))
    text = terminology.definitions_markdown(values)
    assert f"IoU {values['example_iou']}." in text
    assert "|(O_t ∪ G) ∩ N|" in text
    assert "IoU is in the tables" in terminology.figure_reading_markdown("threshold_curves_scene", values)


def test_pooling_example_values_follow_the_two_definitions():
    """Pooled = sum of hits / sum of cells; mean = average of per-object ratios; computed here from first principles."""
    example_objects = (("big", 50, 10), ("small", 5, 4))
    values = terminology.pooling_example_values(example_objects)
    # Pooled: (10 + 4) / (50 + 5) = 14 / 55. Mean: (10/50 + 4/5) / 2 = (0.2 + 0.8) / 2 = 0.5.
    assert values["pooling_example_pooled"] == f"{14 / 55:.{terminology.EXAMPLE_DECIMALS}f}"
    assert values["pooling_example_mean"] == f"{0.5:.{terminology.EXAMPLE_DECIMALS}f}"
    assert values["pooling_example_per_object"] == "0.200 and 0.800"
    with pytest.raises(ValueError):
        terminology.pooling_example_values((("empty", 0, 0),))


def test_point_5f_defines_pooled_and_mean_and_the_curve_guide_names_solid_and_dashed():
    """Point 5(f) carries both formulas, the undefined-object rule and the example; the guide maps line styles."""
    values = terminology.definition_values(SYNTHETIC_SUMMARY, SYNTHETIC_CURVE, SYNTHETIC_CELLS, SYNTHETIC_OUTCOMES)
    definitions = terminology.definitions_markdown(values)
    assert "Σ_k |O_t ∩ G_k| / Σ_k |G_k|" in definitions and "(1/K) Σ_k recall_k" in definitions
    assert "has no precision and is left out of the precision mean" in definitions
    assert values["pooling_example_pooled"] in definitions and values["pooling_example_mean"] in definitions
    assert "made-up numbers" in definitions
    guide = terminology.figure_reading_markdown("threshold_curves_scene", values)
    assert "solid = *pooled*" in guide and "dashed = *mean over objects*" in guide and "point 5(f)" in guide


def test_definitions_print_the_scored_range_not_the_farthest_return():
    """The range text quotes scored_range_m and its quantile in percent, never max_range_m; no hidden-cell wording remains."""
    values = build_values()
    text = terminology.definitions_markdown(values)
    assert (f"within {SYNTHETIC_SUMMARY['provenance']['scored_range_m']:g} m (the distance within which "
            f"{SYNTHETIC_SUMMARY['provenance']['scored_range_quantile'] * 100:g}% of the session's lidar returns lie)") in text
    assert f"{FARTHEST_RETURN_M:g}" not in text and "max_range_m" not in values
    # The word "hidden" and the hidden-cell names are gone from the definitions and both figure guides.
    guides = " ".join(terminology.figure_reading_markdown(name, values) for name in terminology.FIGURE_NAMES)
    for forbidden in ("hidden", "Hidden", "H_near", "near band", "near_band"):
        assert forbidden not in text and forbidden not in guides


def test_definitions_state_the_cell_edge_origin_and_the_neighbour_rule():
    """The cell edge is half the median trunk width; a filled cell on another obstacle is a false alarm of k."""
    text = terminology.definitions_markdown(build_values())
    assert "half the median trunk width" in text and "about two cells across" in text
    assert "a filled cell on a different obstacle (for example a bush beside the trunk) counts as a false alarm of k" in text
    # The corrected occupancy wording: home-cell peaks count for more than one ray collects, a more generous call, the same for every arm.
    assert "count for more than one ray through the cell would collect from them" in text
    assert "blocking probability at one point of the cell" not in text
    assert "The same rule is applied to every arm" in text


def test_figure_guide_names_the_representative_metric():
    """The 3D figure guide says the drawn run is the median repeat of whole-scene pooled recall at the binding cut-off."""
    metric_key = "obstacles/P2@0.5/recall/pooled"
    values = terminology.definition_values(SYNTHETIC_SUMMARY, SYNTHETIC_CURVE, SYNTHETIC_CELLS, SYNTHETIC_OUTCOMES,
                                           representative_metric=metric_key)
    text = terminology.figure_reading_markdown("arms_scene_3d", values)
    assert "median repeat of whole-scene pooled recall at the binding cut-off t = 0.5" in text and metric_key in text
    assert "without losing precision" in terminology.definitions_markdown(values)
