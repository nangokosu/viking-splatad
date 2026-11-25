"""Metric data prepared for the figure suite: per-object tiles, threshold curves, occupancy samples.

Pure numpy, no I/O and no drawing. Every ratio comes from ``eval.hidden_metrics`` (T94/T95), the code
scripts/19 scores with; this module only slices a run's ``gaussian_occupancy_cells.npz`` arrays and
re-packs those results for plotting. Notation: O occupied (occupancy >= threshold, inclusive), G truth,
N neighbourhood, H the band's hidden flag.
"""

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from viking_splatad.eval.hidden_metrics import (per_object_hidden_metrics, pooled_hidden_metrics,
                                                select_object_lists)
from viking_splatad.eval.hidden_settings import class_group_names_ordered, parse_hidden_settings
from viking_splatad.eval.p2_cells import HIDDEN_KEY_BY_BAND, NEIGHBOURHOOD_KEY_PREFIX, TRUTH_KEY_PREFIX

# Band name of the whole-scene baseline (hidden=None in the T94/T95 functions).
BAND_ALL = "all"
# Archive keys read here besides the shared constants of p2_cells / hidden_metrics.
OCCUPANCY_KEY, CENTRES_KEY, VOXEL_SIZE_KEY = "occupancy", "centres", "voxel_size"
# A voxel edge is split into two equal halves to get from a cell centre to its face.
HALVES_PER_VOXEL_EDGE = 2.0


def class_ids_for_name(name: str, eval_config: Mapping[str, Any]) -> Tuple[int, ...]:
    """Class ids of a report entry (a class group or a single scored class), as scripts/19 resolves them.

    Args:
        name: Group name (e.g. "obstacles") or scored class name (e.g. "tree_trunk").
        eval_config: The ``eval`` section of configs/eval.yaml.

    Returns:
        The entry's class ids.

    Raises:
        KeyError: If the name is neither a group nor a scored class.
    """
    # Step 1: the same merge as scripts/19 class_entries: groups first, then one-id entries per scored class.
    settings = parse_hidden_settings(dict(eval_config))
    scored_class_ids = dict(eval_config["scored_class_ids"])
    ids_by_name = {**settings.class_groups, **{class_name: (class_id,) for class_name, class_id in scored_class_ids.items()}}
    # Step 2: restrict to the names scripts/19 reports.
    reported_names = class_group_names_ordered(settings, scored_class_ids)
    if name not in reported_names:
        raise KeyError(f"{name!r} is not a reported entry; choose from {reported_names}")
    return tuple(int(class_id) for class_id in ids_by_name[name])


def _band_hidden(cells: Mapping[str, np.ndarray], band: str) -> Optional[np.ndarray]:
    """The hidden flag of a band.

    Args:
        cells: Cells archive arrays.
        band: "near" (H_near), "full" (H) or "all" (no restriction).

    Returns:
        (V,) bool array, or None for "all".

    Raises:
        ValueError: If the band is unknown.
    """
    if band == BAND_ALL:
        return None
    if band not in HIDDEN_KEY_BY_BAND:
        raise ValueError(f"band {band!r} is not one of {sorted(HIDDEN_KEY_BY_BAND) + [BAND_ALL]}")
    return np.asarray(cells[HIDDEN_KEY_BY_BAND[band]], dtype=bool)


def _class_masks(cells: Mapping[str, np.ndarray], class_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Neighbourhood N and truth G of a report entry.

    Args:
        cells: Cells archive arrays.
        class_name: Group or class name.

    Returns:
        (neighbourhood, truth), (V,) bool each.
    """
    return (np.asarray(cells[f"{NEIGHBOURHOOD_KEY_PREFIX}{class_name}"], dtype=bool),
            np.asarray(cells[f"{TRUTH_KEY_PREFIX}{class_name}"], dtype=bool))


def _nan_if_none(value: Optional[float]) -> float:
    """Map an undefined metric (None) to NaN so it fits a float array.

    Args:
        value: Metric value or None.

    Returns:
        The float value, or NaN.
    """
    return float("nan") if value is None else float(value)


def object_tiles(cells: Mapping[str, np.ndarray], class_name: str, class_ids: Sequence[int],
                 threshold: float, band: str) -> List[Dict[str, Any]]:
    """One record per hidden-space object of the given classes, with its metrics and geometry.

    Objects and order are those scripts/19 scores (``select_object_lists``); per-object values are the
    ``per_object`` entries of T95 at O = occupancy >= threshold on the band, and the baseline recall
    is the same call with hidden=None.

    Args:
        cells: Arrays of ``gaussian_occupancy_cells.npz``.
        class_name: Group or class name naming the ``neighbourhood_/truth_`` arrays.
        class_ids: Class ids whose objects are kept.
        threshold: Occupancy cut (inclusive).
        band: "near", "full" or "all".

    Returns:
        List of dicts: box_id, class_id, cells (grown-box cell indices), inside_cells (the subset whose centre
        lies inside the object's own box, the cells G_k is made of), recall_H, precision_H (float or None),
        baseline_recall (float or None), hidden_gt_cell_count (|G_k n H|), centre (3,) mean of inside-cell
        centres (NaN if none), extent_min / extent_max (3,) bounding box of the cells' centres, widened by
        half a voxel when ``voxel_size`` is in the archive.
    """
    # Step 1: the entry's objects in CSR form, plus their ids in the same order.
    object_class_ids = np.asarray(cells["object_class_ids"])
    offsets, object_cells, object_inside = select_object_lists(
        object_class_ids, np.asarray(cells["object_cell_offsets"]), np.asarray(cells["object_cells"]),
        np.asarray(cells["object_inside"]), class_ids)
    kept_objects = np.flatnonzero(np.isin(object_class_ids, list(class_ids)))
    # Step 2: T95 on the band and on the whole scene, same occupied set.
    neighbourhood, truth = _class_masks(cells, class_name)
    occupied = np.asarray(cells[OCCUPANCY_KEY]) >= float(threshold)
    band_result = per_object_hidden_metrics(occupied, truth, neighbourhood, _band_hidden(cells, band),
                                            offsets, object_cells, object_inside)
    baseline_result = per_object_hidden_metrics(occupied, truth, neighbourhood, None, offsets, object_cells, object_inside)
    # Step 3: half voxel margin (the box edge lies half an edge beyond the outermost centre).
    centres = np.asarray(cells[CENTRES_KEY], dtype=float)
    half_voxel = float(cells[VOXEL_SIZE_KEY]) / HALVES_PER_VOXEL_EDGE if VOXEL_SIZE_KEY in cells else 0.0
    tiles = []
    for tile_position, object_row in enumerate(kept_objects):
        block = slice(offsets[tile_position], offsets[tile_position + 1])
        object_cell_indices = object_cells[block]
        cell_centres = centres[object_cell_indices]
        inside_centres = cell_centres[object_inside[block]]
        entry, baseline_entry = band_result["per_object"][tile_position], baseline_result["per_object"][tile_position]
        has_cells = len(cell_centres) > 0
        tiles.append({
            "box_id": int(cells["object_box_ids"][object_row]),
            "class_id": int(object_class_ids[object_row]),
            "cells": object_cell_indices,
            "inside_cells": object_cell_indices[object_inside[block]],
            "recall_H": entry["recall"],
            "precision_H": entry["precision"],
            "baseline_recall": baseline_entry["recall"],
            "hidden_gt_cell_count": entry["gt_cells"],
            "centre": inside_centres.mean(axis=0) if len(inside_centres) else np.full(centres.shape[1], np.nan),
            "extent_min": cell_centres.min(axis=0) - half_voxel if has_cells else np.full(centres.shape[1], np.nan),
            "extent_max": cell_centres.max(axis=0) + half_voxel if has_cells else np.full(centres.shape[1], np.nan),
        })
    return tiles


def threshold_curves(cells: Mapping[str, np.ndarray], class_name: str, thresholds: Sequence[float],
                     band: str, class_ids: Optional[Sequence[int]] = None) -> Dict[str, np.ndarray]:
    """Pooled and per-object-mean recall and precision at each occupancy threshold.

    Calls T94 (``pooled_hidden_metrics``) and T95 (``per_object_hidden_metrics``) once per threshold with
    O = occupancy >= threshold (inclusive, as scripts/19).

    Args:
        cells: Arrays of ``gaussian_occupancy_cells.npz``.
        class_name: Group or class name naming the ``neighbourhood_/truth_`` arrays.
        thresholds: Occupancy cuts (inclusive).
        band: "near", "full" or "all".
        class_ids: Class ids whose objects enter the per-object means (pass ``class_ids_for_name`` to match
            scripts/19); None keeps every object of the archive, which also averages objects of other classes.

    Returns:
        {"thresholds", "pooled_recall", "pooled_precision", "mean_recall", "mean_precision"}: float arrays
        aligned with ``thresholds`` (undefined -> NaN).
    """
    # Step 1: masks and the entry's objects (all objects when no class ids are given).
    neighbourhood, truth = _class_masks(cells, class_name)
    hidden = _band_hidden(cells, band)
    object_class_ids = np.asarray(cells["object_class_ids"])
    selected_class_ids = np.unique(object_class_ids) if class_ids is None else class_ids
    offsets, object_cells, object_inside = select_object_lists(
        object_class_ids, np.asarray(cells["object_cell_offsets"]), np.asarray(cells["object_cells"]),
        np.asarray(cells["object_inside"]), selected_class_ids)
    occupancy = np.asarray(cells[OCCUPANCY_KEY])
    curves = {"thresholds": np.asarray(thresholds, dtype=float), "pooled_recall": [], "pooled_precision": [],
              "mean_recall": [], "mean_precision": []}
    # Step 2: one T94 + T95 evaluation per threshold.
    for threshold in thresholds:
        occupied = occupancy >= float(threshold)
        pooled = pooled_hidden_metrics(occupied, truth, neighbourhood, hidden)
        per_object = per_object_hidden_metrics(occupied, truth, neighbourhood, hidden, offsets, object_cells, object_inside)
        curves["pooled_recall"].append(_nan_if_none(pooled["recall"]))
        curves["pooled_precision"].append(_nan_if_none(pooled["precision"]))
        curves["mean_recall"].append(_nan_if_none(per_object["mean_recall"]))
        curves["mean_precision"].append(_nan_if_none(per_object["mean_precision"]))
    return {name: np.asarray(values, dtype=float) for name, values in curves.items()}


def occupancy_samples(cells: Mapping[str, np.ndarray], class_name: str, band: str) -> Dict[str, np.ndarray]:
    """Occupancy values of the truth cells and of the off-truth neighbourhood cells on a band.

    truth = occupancy[G & H]; off_truth = occupancy[(N & H) \\ G], with H all cells for band "all".

    Args:
        cells: Arrays of ``gaussian_occupancy_cells.npz``.
        class_name: Group or class name.
        band: "near", "full" or "all".

    Returns:
        {"truth": (n,) float array, "off_truth": (m,) float array}.
    """
    neighbourhood, truth = _class_masks(cells, class_name)
    hidden = _band_hidden(cells, band)
    # Step 1: restrict to the band (every cell when hidden is None).
    in_band = np.ones_like(truth) if hidden is None else hidden
    occupancy = np.asarray(cells[OCCUPANCY_KEY], dtype=float)
    # Step 2: split the band's neighbourhood into truth and the rest.
    return {"truth": occupancy[truth & in_band], "off_truth": occupancy[neighbourhood & in_band & ~truth]}
