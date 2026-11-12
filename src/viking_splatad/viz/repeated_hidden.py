"""Matched pictures for repeat-consistent recovery, loss, and off-trunk occupancy."""

from pathlib import Path

import cv2
import numpy as np

from viking_splatad.eval.hidden_geometry import exclusive_evidence_labels
from viking_splatad.eval.repeated_hidden import consistent_cell_outcomes
from viking_splatad.viz.hidden_trunks import PANEL_PIXELS, STRATUM_COLORS, TP_BGR, FP_BGR
from viking_splatad.viz.radar_bev import bev_pixels


def write_consistency_map(path, centres, truth, evidence, baseline_masks, radar_masks):
    """Draw shared XY cell projections using exactly the repeat metric's masks.

    Args:
        path: PNG destination; centres/truth/evidence: common script 17 cells;
        baseline_masks, radar_masks: [repeat,cell] bool masks used by the report.
    Returns:
        PNG path; counts remain 3D even though the drawing pools height in XY.
    """
    masks, counts = consistent_cell_outcomes(baseline_masks, radar_masks, truth, evidence)
    bounds = np.stack([centres[:, :2].min(axis=0), centres[:, :2].max(axis=0)])
    # A minimal representable display extent handles degenerate analytic grids without
    # changing any physical count; real evaluation neighbourhoods have XY span.
    bounds[1] = np.where(bounds[1] > bounds[0], bounds[1], np.nextafter(bounds[0], np.inf))
    panels = []
    titles = ["GT cells: recorded evidence", "B: occupied in every repeat", "L: occupied in every repeat",
              "Consistent recovery / loss", "Intermittent B, preserved in L", "Consistent new off-trunk cells"]
    for panel_index, title in enumerate(titles):
        image = np.full((PANEL_PIXELS, PANEL_PIXELS, 3), 255, dtype=np.uint8)
        context_pixels, _ = bev_pixels(centres[truth, :2], bounds, (PANEL_PIXELS, PANEL_PIXELS))
        image[tuple(context_pixels.T)] = (220, 220, 220)
        if panel_index == 0:
            selected = truth
            colors = np.asarray(STRATUM_COLORS)[exclusive_evidence_labels(evidence[selected])]
        elif panel_index in (1, 2):
            selected = masks["B_always" if panel_index == 1 else "L_always"]
            colors = np.where(truth[selected, None], TP_BGR, FP_BGR)
        elif panel_index == 3:
            selected = masks["consistent_recovery"] | masks["consistent_loss"]
            colors = np.where(masks["consistent_recovery"][selected, None], TP_BGR, FP_BGR)
        else:
            selected = masks["intermittent_B_preserved" if panel_index == 4 else "consistent_new_off_trunk"]
            colors = np.tile((220, 100, 20) if panel_index == 4 else FP_BGR, (int(selected.sum()), 1))
        pixels, keep = bev_pixels(centres[selected, :2], bounds, (PANEL_PIXELS, PANEL_PIXELS))
        image[tuple(pixels.T)] = colors[keep]
        caption = np.full((PANEL_PIXELS // 8, PANEL_PIXELS, 3), 255, dtype=np.uint8)
        cv2.putText(caption, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 1)
        cv2.putText(caption, f"{int(selected.sum())} cells; XY projection, height pooled", (8, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 0, 0), 1)
        panels.append(np.vstack([caption, image]))
    output = np.vstack([np.hstack(panels[:3]), np.hstack(panels[3:])])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), output):
        raise OSError(f"failed to draw {path}")
    path.with_suffix(".txt").write_text(
        "Repeat consistency, not a significance test. Counts use the shared 3D cells; XY projection pools height.\n"
        "Green: on-GT occupancy/recovery. Red: off-GT occupancy/loss. Blue: intermittently present in B, always in L.\n"
        "Evidence: seen blue, blocked orange, no-return purple, passed-through cyan, unobserved gray.\n"
        f"Consistent new off-trunk cells: {counts['consistent_new_off_trunk_cells']}.\n")
    return path
