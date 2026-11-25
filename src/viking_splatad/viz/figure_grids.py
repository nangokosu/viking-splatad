"""Pure-numpy grid helpers for the Gaussian-occupancy figure suite (no plotting, no I/O)."""

import numpy as np

from viking_splatad.eval.hidden_geometry import EVIDENCE_NAMES, exclusive_evidence_labels

# Ordered stratum names; the label of a cell is the position of its name in this tuple.
OUTCOME_STRATUM_NAMES = ("hit_seen", "hit_hidden", "false_alarm", "missed_seen", "missed_hidden")
# Label of a cell that belongs to no outcome (a sentinel outside the valid label range).
NO_OUTCOME_LABEL = -1
# Column positions of the (x, y) axes inside a (V, 3) index / coordinate array.
HORIZONTAL_AXES = (0, 1)
# Evidence name that marks a cell as "seen"; every other name counts as hidden.
SEEN_EVIDENCE_NAME = "seen"


def column_max_grid(cell_indices, values, mask, grid_origin, voxel_size):
    """Bird's-eye grid holding the maximum of ``values`` over the masked cells of each (x, y) column.

    Convention (same as ``eval.detection.voxel_centers``): voxel ``index`` spans
    ``[origin + index * voxel_size, origin + (index + 1) * voxel_size]``, so pixel edges coincide
    with voxel edges.

    Args:
        cell_indices: (V, 3) integer voxel indices.
        values: (V,) values to reduce.
        mask: (V,) bool; only these cells contribute and they alone define the pixel range.
        grid_origin: (3,) grid minimum corner (m).
        voxel_size: Voxel edge (m).

    Returns:
        ``(image, extent)``: ``image`` is (rows, cols) float indexed [y_row, x_col] (row 0 = smallest y,
        for ``imshow(origin="lower")``), NaN where no masked cell exists; ``extent`` is
        (x_min, x_max, y_min, y_max) in metres of the pixel edges.

    Raises:
        ValueError: if the mask selects no cell.
    """
    cell_indices = np.asarray(cell_indices)
    values = np.asarray(values, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    # Step 1: an empty selection has no pixel range.
    if not mask.any():
        raise ValueError("empty mask: no cell to draw")
    # Step 2: horizontal indices of the masked cells and the index range they span.
    horizontal_indices = cell_indices[mask][:, list(HORIZONTAL_AXES)]
    index_minimum = horizontal_indices.min(axis=0)
    index_maximum = horizontal_indices.max(axis=0)
    # Inclusive range: the pixel count along an axis is (max - min) + one voxel.
    pixel_counts = index_maximum - index_minimum + np.ones_like(index_minimum)
    column_count, row_count = int(pixel_counts[0]), int(pixel_counts[1])
    # Step 3: max-reduction per pixel. np.maximum.at is an unbuffered scatter-max, O(V);
    # starting from -inf makes the first value always win, NaN is restored afterwards.
    image = np.full((row_count, column_count), -np.inf)
    pixel_columns = horizontal_indices[:, 0] - index_minimum[0]
    pixel_rows = horizontal_indices[:, 1] - index_minimum[1]
    np.maximum.at(image, (pixel_rows, pixel_columns), values[mask])
    image[np.isneginf(image)] = np.nan
    # Step 4: pixel-edge extent: minimum corner of the first voxel to maximum corner of the last.
    origin_horizontal = np.asarray(grid_origin, dtype=float)[list(HORIZONTAL_AXES)]
    edge_low = origin_horizontal + index_minimum * voxel_size
    edge_high = origin_horizontal + (index_maximum + np.ones_like(index_maximum)) * voxel_size
    extent = (float(edge_low[0]), float(edge_high[0]), float(edge_low[1]), float(edge_high[1]))
    return image, extent


def opacity_weighted_density(means_xy, opacities, extent, bin_size):
    """Opacity-weighted 2D histogram of Gaussian means on square bins covering ``extent``.

    Args:
        means_xy: (N, 2) Gaussian mean positions (m).
        opacities: (N,) weights in [0, 1].
        extent: (x_min, x_max, y_min, y_max) in metres.
        bin_size: Bin edge (m).

    Returns:
        ``(density, edges_x, edges_y)``; ``density`` is indexed [y, x] and sums to the opacity of the
        points inside the (half-open, last bin closed) extent; points outside are dropped.
    """
    means_xy = np.asarray(means_xy, dtype=float)
    opacities = np.asarray(opacities, dtype=float)
    x_min, x_max, y_min, y_max = extent
    # Step 1: bin count per axis = ceil(span / bin_size), so the bins cover the whole extent.
    bin_count_x = int(np.ceil((x_max - x_min) / bin_size))
    bin_count_y = int(np.ceil((y_max - y_min) / bin_size))
    edges_x = x_min + np.arange(bin_count_x + 1) * bin_size
    edges_y = y_min + np.arange(bin_count_y + 1) * bin_size
    # Step 2: weighted histogram; np.histogram2d drops points outside the edges. It returns [x, y], so transpose.
    density_xy, _, _ = np.histogram2d(
        means_xy[:, 0], means_xy[:, 1], bins=(edges_x, edges_y), weights=opacities)
    return density_xy.T, edges_x, edges_y


def arm_difference_sets(hit_first, hit_second):
    """Partition the union of two arms' hit masks into only-first, only-second and both.

    Args:
        hit_first, hit_second: (V,) bool masks.

    Returns:
        Dict ``only_first``, ``only_second``, ``both`` of (V,) bool; pairwise disjoint, union = first | second.
    """
    hit_first = np.asarray(hit_first, dtype=bool)
    hit_second = np.asarray(hit_second, dtype=bool)
    return {"only_first": hit_first & ~hit_second,
            "only_second": hit_second & ~hit_first,
            "both": hit_first & hit_second}


def outcome_strata(outcomes, evidence):
    """Label every cell by outcome and, for hits/misses, by whether the cell was seen.

    Args:
        outcomes: dict of disjoint (V,) bool masks ``hit``, ``false_alarm``, ``missed``.
        evidence: (V, 4) bool evidence flags; "hidden" = exclusive evidence label is not "seen".

    Returns:
        (V,) int index into ``OUTCOME_STRATUM_NAMES``, or ``NO_OUTCOME_LABEL`` for cells in no outcome.
    """
    # Step 1: hidden = any exclusive evidence label other than "seen" (seen index looked up by name).
    seen_label = EVIDENCE_NAMES.index(SEEN_EVIDENCE_NAME)
    is_hidden = exclusive_evidence_labels(evidence) != seen_label
    # Step 2: one mask per stratum, in OUTCOME_STRATUM_NAMES order.
    stratum_masks = {
        "hit_seen": outcomes["hit"] & ~is_hidden,
        "hit_hidden": outcomes["hit"] & is_hidden,
        "false_alarm": outcomes["false_alarm"],
        "missed_seen": outcomes["missed"] & ~is_hidden,
        "missed_hidden": outcomes["missed"] & is_hidden,
    }
    # Step 3: write labels; outcomes are disjoint, so no cell is overwritten.
    labels = np.full(is_hidden.shape, NO_OUTCOME_LABEL, dtype=int)
    for stratum_index, stratum_name in enumerate(OUTCOME_STRATUM_NAMES):
        labels[np.asarray(stratum_masks[stratum_name], dtype=bool)] = stratum_index
    return labels
