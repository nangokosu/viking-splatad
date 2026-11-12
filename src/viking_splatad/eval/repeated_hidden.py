"""Repeated hidden-trunk summaries: describe between-arm gaps against within-arm spread."""

import numpy as np

from viking_splatad.eval.hidden_geometry import EVIDENCE_NAMES, exclusive_evidence_labels, occupied_cell_metrics


def scalar_repeat_summary(values):
    """Return sample mean/std/range only when every repeat supplies a defined value.

    Args:
        values: Ratios or None from matched repeats; at least two are required.
    Returns:
        Descriptive statistics, never a significance claim. Undefined values
        cannot be silently dropped to improve the apparent effect.
    """
    if len(values) < 2:
        raise ValueError("at least two independent runs per arm are required")
    if any(value is None for value in values):
        return {"values": values, "mean": None, "sample_std": None, "minimum": None, "maximum": None, "range": None}
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("repeat values must be finite or explicitly undefined")
    return {"values": values, "mean": float(array.mean()), "sample_std": float(array.std(ddof=1)),
            "minimum": float(array.min()), "maximum": float(array.max()), "range": float(np.ptp(array))}


def consistent_cell_outcomes(baseline_masks, radar_masks, truth, evidence):
    """Find repeatedly recovered/lost cells and unstable baseline cells consistently preserved.

    Args:
        baseline_masks, radar_masks: [repeat,cell] occupied masks; truth/evidence:
        common GT and recorded observations. No arm selects or filters targets.
    Returns:
        Shared masks used by report and viewer plus counts per GT evidence stratum.
        Added off-trunk cells are measured in the same frozen neighbourhood.
    """
    baseline_masks, radar_masks = np.asarray(baseline_masks, dtype=bool), np.asarray(radar_masks, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    if (baseline_masks.ndim != 2 or baseline_masks.shape != radar_masks.shape or baseline_masks.shape[0] < 2
            or baseline_masks.shape[1:] != truth.shape):
        raise ValueError("matched cells and at least two runs per arm required")
    baseline_any, baseline_always = baseline_masks.any(axis=0), baseline_masks.all(axis=0)
    radar_any, radar_always = radar_masks.any(axis=0), radar_masks.all(axis=0)
    masks = {"B_always": baseline_always, "L_always": radar_always,
             "consistent_recovery": truth & ~baseline_any & radar_always,
             "consistent_loss": truth & baseline_always & ~radar_any,
             "intermittent_B_preserved": truth & baseline_any & ~baseline_always & radar_always,
             "consistent_new_off_trunk": ~truth & ~baseline_any & radar_always}
    labels = exclusive_evidence_labels(evidence)
    strata = {}
    for group_index, group in enumerate(EVIDENCE_NAMES):
        selected = truth & (labels == group_index)
        strata[group] = {"gt_cells": int(selected.sum()), **{
            key: int((mask & selected).sum()) for key, mask in masks.items() if key != "consistent_new_off_trunk"}}
    return masks, {"strata": strata, "consistent_new_off_trunk_cells": int(masks["consistent_new_off_trunk"].sum())}


def summarize_comparisons(reports, cell_arrays):
    """Validate matching completed comparison products and summarize independent repeats.

    Args:
        reports: Parsed script 17 comparison JSONs; cell_arrays: their NPZ dicts.
    Returns:
        JSON-ready descriptive summary and P1/P2 consistency masks for matched views.
        Report metrics are recomputed from cells and must match exactly.
    """
    if len(reports) < 2 or len(reports) != len(cell_arrays):
        raise ValueError("at least two matched comparisons and their cell arrays required")
    reference = reports[0]
    if reference.get("schema") != "viking-splatad/hidden-trunk-pilot/1":
        raise ValueError("supported hidden-trunk comparison schema required")
    shared_keys = ("schema", "session", "source_revision", "membership_sha256", "labels_sha256", "threshold", "voxel_size_m", "margin_m", "scored_trunks")
    for report in reports:
        if any(report[key] != reference[key] for key in shared_keys):
            raise ValueError("repeat comparisons must share source, data split, labels and scoring definitions")
    for arm in ("B", "L"):
        if len({report["arms"][arm]["run_id"] for report in reports}) != len(reports):
            raise ValueError("each repeat must be a distinct run in each arm")
    for arrays in cell_arrays:
        if any(not np.array_equal(arrays[key], cell_arrays[0][key]) for key in ("indices", "centres", "truth", "evidence", "threshold")):
            raise ValueError("repeat cell coordinates, GT and recorded observations must be identical")
        if float(arrays["threshold"]) != reference["threshold"]:
            raise ValueError("cell and report thresholds disagree")
    truth, evidence = cell_arrays[0]["truth"], cell_arrays[0]["evidence"]
    metrics, stacks = {}, {}
    for representation in ("P1", "P2"):
        stacks[representation] = {}
        for arm in ("B", "L"):
            masks = []
            for report, arrays in zip(reports, cell_arrays):
                values = arrays[arm] if representation == "P2" else arrays[arm + "_P1"]
                if values.shape != truth.shape or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
                    raise ValueError("finite matching cell probabilities required")
                occupied = values >= reference["threshold"] if representation == "P2" else values.astype(bool)
                actual = occupied_cell_metrics(occupied, truth, evidence)
                if actual != report["arms"][arm][representation]:
                    raise ValueError("reported metrics do not match the archived cells")
                masks.append(occupied)
            stacks[representation][arm] = np.stack(masks)
        paths = [(name, None) for name in ("precision", "recall", "iou")]
        paths += [("recall", group) for group in EVIDENCE_NAMES]
        metrics[representation] = {}
        for metric_name, group in paths:
            values = {arm: [report["arms"][arm][representation][metric_name] if group is None else
                            report["arms"][arm][representation]["strata"][group][metric_name] for report in reports] for arm in ("B", "L")}
            baseline, radar = scalar_repeat_summary(values["B"]), scalar_repeat_summary(values["L"])
            differences = [None if baseline_value is None or radar_value is None else radar_value - baseline_value
                           for baseline_value, radar_value in zip(values["B"], values["L"])]
            delta = scalar_repeat_summary(differences)
            key = metric_name if group is None else group + "_" + metric_name
            metrics[representation][key] = {"B": baseline, "L": radar, "paired_L_minus_B": delta,
                "absolute_mean_gap_exceeds_both_within_arm_ranges": None if delta["mean"] is None or baseline["range"] is None or radar["range"] is None
                else abs(delta["mean"]) > max(baseline["range"], radar["range"])}
    consistency, views = {}, {}
    for representation, arms in stacks.items():
        masks, consistency[representation] = consistent_cell_outcomes(arms["B"], arms["L"], truth, evidence)
        views[representation] = {"masks": masks, "B": arms["B"], "L": arms["L"]}
    return {"schema": "viking-splatad/repeated-hidden-trunks/1", "repeats_per_arm": len(reports),
            "session": reference["session"], "runs": {arm: [report["arms"][arm]["run_id"] for report in reports] for arm in ("B", "L")},
            "all_alignment_reviews_complete": all(report["alignment_reviewed"] for report in reports),
            "metrics": metrics, "consistent_cells": consistency,
            "caution": "Descriptive initial repeat check, not a significance test. Interpret only after label/frame review; neighbourhood FP is not a global phantom-tree count."}, views
