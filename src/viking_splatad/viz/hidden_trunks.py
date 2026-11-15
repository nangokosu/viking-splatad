"""Metric-matched observation/P2/visibility pictures; real arrays only, no synthetic imagery."""

from pathlib import Path

import cv2
import numpy as np

from viking_splatad.eval.hidden_geometry import exclusive_evidence_labels, occupied_cell_metrics, EVIDENCE_NAMES
from viking_splatad.labels.cuboids import compute_cuboid_geometry
from viking_splatad.viz.radar_bev import bev_pixels

# Same display resolution as the existing alignment viewer. Colors are BGR,
# solely for presentation: TP green, off-trunk occupancy red, real returns gray.
PANEL_PIXELS = 512
TP_BGR, FP_BGR, REAL_BGR = (40, 150, 20), (40, 40, 220), (110, 110, 110)
STRATUM_COLORS = ((220, 110, 20), (0, 150, 230), (200, 40, 150), (160, 180, 20), (160, 160, 160))


def write_box_comparison(path, cuboid, centres, truth, baseline, radar, evidence, real_points, margin, p1_masks):
    """Draw one box's actual observations, both P2 masks, change and visibility.

    Args:
        path: PNG output; cuboid: scored GT; centres/truth/B/L/evidence: same
        cells used by the metric; real_points: pooled real returns in map frame;
        margin: frozen GT-derived growth; p1_masks: matching B/L ungated P1 masks.
    Returns:
        PNG path. Panels project box-local XZ (not a visibility rendering).
        Caption counts call the scoring module itself, preserving metric meaning.
    """
    geometry = compute_cuboid_geometry(cuboid)
    local = (centres - geometry.center) @ geometry.rotation_map_box
    actual = (real_points - geometry.center) @ geometry.rotation_map_box
    half = geometry.half_extents
    extent = half + np.array([margin, margin, 0.])
    selected = np.all(np.abs(local) <= extent, axis=1)
    real_selected = np.all(np.abs(actual) <= extent, axis=1)
    # One voxel-row of padding is a display margin, derived from the box size.
    bounds = np.stack([-extent[[0, 2]], extent[[0, 2]]])
    pictures = []
    titles = ["Observed LiDAR returns", "Baseline: rendered LiDAR", "Radar loss: rendered LiDAR",
              "Baseline: Gaussian occupancy", "Radar loss: Gaussian occupancy",
              "Added blue / removed red", "Recorded GT-cell evidence"]
    local_truth = truth & selected
    for panel_index, title in enumerate(titles):
        panel = np.full((PANEL_PIXELS, PANEL_PIXELS, 3), 255, dtype=np.uint8)
        if panel_index == 0:
            pixels, _ = bev_pixels(actual[real_selected][:, [0, 2]], bounds, (PANEL_PIXELS, PANEL_PIXELS))
            panel[tuple(pixels.T)] = REAL_BGR
        else:
            if panel_index in (1, 2, 3, 4):
                occupied = {1: p1_masks["B"], 2: p1_masks["L"], 3: baseline, 4: radar}[panel_index]
                drawn = selected & occupied
                colors = np.where(truth[drawn, None], TP_BGR, FP_BGR)
            elif panel_index == 5:
                drawn = selected & (baseline != radar)
                colors = np.where(radar[drawn, None], (220, 100, 20), FP_BGR)
            else:
                drawn = local_truth
                colors = np.asarray(STRATUM_COLORS)[exclusive_evidence_labels(evidence[drawn])]
            pixels, keep = bev_pixels(local[drawn][:, [0, 2]], bounds, (PANEL_PIXELS, PANEL_PIXELS))
            panel[tuple(pixels.T)] = colors[keep]
        # GT outline uses the box-local axes, including a rotated map cuboid.
        corners = np.array([[-half[0], -half[2]], [half[0], half[2]]])
        outline, _ = bev_pixels(corners, bounds, (PANEL_PIXELS, PANEL_PIXELS))
        cv2.rectangle(panel, tuple(outline[0, ::-1]), tuple(outline[1, ::-1]), (0, 0, 0), 1)
        banner = np.full((PANEL_PIXELS // 8, PANEL_PIXELS, 3), 255, dtype=np.uint8)
        cv2.putText(banner, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
        if panel_index in (1, 2, 3, 4):
            metrics = occupied_cell_metrics(occupied[selected], truth[selected], evidence[selected])
            caption = f"TP {metrics['true_positive']} FP {metrics['false_positive']} FN {metrics['false_negative']}"
            cv2.putText(banner, caption, (8, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        pictures.append(np.vstack([banner, panel]))
    # Four columns keep the seven panels readable in a single review image.
    pictures.append(np.full_like(pictures[0], 255))
    picture = np.vstack([np.hstack(pictures[:4]), np.hstack(pictures[4:])])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), picture):
        raise OSError(f"failed to write {path}")
    path.with_suffix(".txt").write_text(
        f"Box {cuboid.box_id}; box-local XZ projection, all box depths pooled. Black: GT outline.\n"
        "P2: green inside GT, red off GT within frozen neighbourhood. Pixel overlap is not the metric.\n"
        "Evidence seen=blue, blocked=orange, no-return=purple, passed-through=cyan, unobserved=gray.\n"
        "Blocked means a nearer recorded return, not independently labeled grass.\n"
        f"Evidence summary order: {EVIDENCE_NAMES}. Raw multi-evidence flags are preserved in cells.npz.\n")
    return path
