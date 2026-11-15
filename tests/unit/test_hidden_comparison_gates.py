"""Reject unmatched arms before expensive scientific evaluation; verify metric picture output."""

import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from viking_splatad.labels.cuboids import Cuboid
from viking_splatad.viz.hidden_trunks import PANEL_PIXELS, write_box_comparison


def test_comparison_refuses_changed_seed_or_split(tmp_path):
    """Successful jobs alone cannot authorize a scientific comparison across seeds/splits."""
    source = Path(__file__).resolve().parents[2] / "scripts/17_compare_hidden_trunks.py"
    specification = importlib.util.spec_from_file_location("compare_hidden", source)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    baseline, radar = tmp_path / "B", tmp_path / "L"
    baseline.mkdir()
    radar.mkdir()
    shared = {"end_utc": "complete", "split_render_check": "match", "overfitting_plot": "written", "SESSION": "june_tall_grass",
              "TRAIN_SEED": "42", "MAX_NUM_ITERATIONS": "20001", "TRAIN_LIDAR_ONLY": "True", "ADD_MISSING_POINTS": "True"}
    for directory, method in ((baseline, "splatad"), (radar, "splatad-radar-lidar")):
        information = dict(shared, TRAIN_METHOD=method, RADAR_LIDAR_AUTO_WEIGHT="True")
        (directory / "job_info.txt").write_text("\n".join(f"{key}: {value}" for key, value in information.items()))
        (directory / "split.json").write_text(json.dumps({"dataset": {"name": "june"}, "membership_sha256": "same"}))
    (radar / "radar_diagnostics.jsonl").write_text(json.dumps({"event": "stock_initialization", "radar_gaussians_added": 0}) + "\n")
    module.matching_runs(baseline, radar)
    information = (radar / "job_info.txt").read_text().replace("TRAIN_SEED: 42", "TRAIN_SEED: 43")
    (radar / "job_info.txt").write_text(information)
    with pytest.raises(ValueError, match="TRAIN_SEED"):
        module.matching_runs(baseline, radar)


def test_metric_matched_picture_has_observations_both_predictions_and_evidence(tmp_path):
    """A known small box yields seven panels plus one blank, with metric captions and key."""
    box = Cuboid("analytic", 1, "tree_trunk", (0., 0., 0.), (2., 2., 4.), (0., 0., 0., 1.))
    centres = np.array([[0., 0., 0.], [1.5, 0., 0.]])
    truth, baseline, radar = np.array([True, False]), np.array([False, False]), np.array([True, True])
    evidence = np.array([[False, True, False, False], [False, False, False, False]])
    path = write_box_comparison(tmp_path / "box.png", box, centres, truth, baseline, radar, evidence,
                                np.array([[0., 0., 1.]]), 1., {"B": baseline, "L": baseline})
    image = cv2.imread(str(path))
    assert image.shape == (2 * (PANEL_PIXELS + PANEL_PIXELS // 8), 4 * PANEL_PIXELS, 3)
    assert "nearer recorded return" in path.with_suffix(".txt").read_text()
    assert np.any(image != 255)
