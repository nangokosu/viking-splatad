"""End-to-end tests of scripts/05_extract_cuboids.py and scripts/06_validate_contract.py."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from tests.unit.test_writer import build_synthetic_dataset
from viking_splatad.dataset.writer import write_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXTRACT_SCRIPT = PROJECT_ROOT / "scripts" / "05_extract_cuboids.py"
VALIDATE_SCRIPT = PROJECT_ROOT / "scripts" / "06_validate_contract.py"
MAY_CONFIG = PROJECT_ROOT / "configs" / "sessions" / "may_short_grass.yaml"

# Expected per-class box counts of the real May sample (read from the label file by hand).
EXPECTED_MAY_CLASS_COUNTS = {1: 190, 2: 167, 3: 21, 4: 48, 5: 5, 6: 11, 7: 21, 8: 4}
EXPECTED_MAY_TOTAL = 467

# Synthetic geometry (meters). The rock is an axis-aligned cube at the origin; the trunk is a
# 1 x 1 x 3 box 10 m away. Voxel 0.5 m aligns every face with a voxel boundary (see test).
VOXEL_SIZE = 0.5
ROCK_EDGE = 2.0
TRUNK_FOOTPRINT = 1.0
TRUNK_HEIGHT = 3.0
TRUNK_OFFSET_X = 10.0


def make_annotation(annotation_id, class_id, center, dimensions):
    """Build one axis-aligned (identity rotation) annotation dict."""
    return {
        "id": annotation_id, "category_id": class_id, "type": "cuboid",
        "position": dict(zip("xyz", center)), "dimensions": dict(zip("xyz", dimensions)),
        "rotation": {"qx": 0, "qy": 0, "qz": 0, "qw": 1},
    }


def write_synthetic_session(tmp_path):
    """Write a synthetic labels JSON and session YAML; return (config_path, gt_dir)."""
    document = {"dataset": {
        "task_attributes": {"categories": [{"id": 1, "name": "Tree_trunk"}, {"id": 3, "name": "Rock"}]},
        "samples": [{"name": "map_2024_05.ply", "labels": {"ground-truth": {"attributes": {"annotations": [
            make_annotation(1, 3, (0, 0, 0), (ROCK_EDGE,) * 3),
            make_annotation(2, 1, (TRUNK_OFFSET_X, 0, 0), (TRUNK_FOOTPRINT, TRUNK_FOOTPRINT, TRUNK_HEIGHT)),
        ]}}}}],
    }}
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(json.dumps(document))
    gt_dir = tmp_path / "gt"
    # Start from the real May config (the loader requires every top-level section), then
    # point the session paths at the synthetic files.
    config = yaml.safe_load(MAY_CONFIG.read_text())
    config["session"].update({"name": "synthetic", "labels_json": str(labels_path),
                              "labels_sample_key": "2024_05", "gt_dir": str(gt_dir)})
    config_path = tmp_path / "session.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return config_path, gt_dir


def run_script(script, *arguments):
    """Run a script with the test interpreter; return the CompletedProcess."""
    return subprocess.run([sys.executable, str(script), *map(str, arguments)],
                          capture_output=True, text=True)


def test_extract_synthetic_counts_and_files(tmp_path):
    """Analytic voxel counts: rock 4x4x4 = 64, trunk 2x2x6 = 24 at 0.5 m voxels."""
    config_path, gt_dir = write_synthetic_session(tmp_path)
    result = run_script(EXTRACT_SCRIPT, "--config", config_path, "--voxel-size", VOXEL_SIZE)
    assert result.returncode == 0, result.stderr
    for file_name in ("cuboids.json", "occupancy.npz", "summary.md"):
        assert (gt_dir / file_name).exists()
    # Expected counts from first principles: edge / voxel voxels per axis, centres never on faces.
    rock_voxels = int(ROCK_EDGE / VOXEL_SIZE) ** 3
    trunk_voxels = int(TRUNK_FOOTPRINT / VOXEL_SIZE) ** 2 * int(TRUNK_HEIGHT / VOXEL_SIZE)
    saved = np.load(gt_dir / "occupancy.npz")
    assert saved["class_ids"].dtype == np.uint8
    assert float(saved["voxel_size"]) == VOXEL_SIZE
    assert np.count_nonzero(saved["class_ids"] == 3) == rock_voxels
    assert np.count_nonzero(saved["class_ids"] == 1) == trunk_voxels
    boxes = json.loads((gt_dir / "cuboids.json").read_text())
    assert len(boxes) == 2 and len(boxes[0]["corners_xyz"]) == 8
    assert boxes[1]["class_name"] == "tree_trunk"
    summary = (gt_dir / "summary.md").read_text()
    assert "UNVERIFIED" in summary and str(trunk_voxels) in summary and "rock" in summary


def test_extract_default_voxel_size_is_fraction_of_trunk_diameter(tmp_path):
    """Without --voxel-size the voxel is 0.5 x the (1.0 m) trunk diameter."""
    config_path, gt_dir = write_synthetic_session(tmp_path)
    assert run_script(EXTRACT_SCRIPT, "--config", config_path).returncode == 0
    voxel_size = float(np.load(gt_dir / "occupancy.npz")["voxel_size"])
    assert voxel_size == pytest.approx(0.5 * TRUNK_FOOTPRINT)


def test_extract_exceeding_memory_budget_fails_before_allocating(tmp_path):
    """A tiny budget gives a clear error, non-zero exit, and no outputs."""
    config_path, gt_dir = write_synthetic_session(tmp_path)
    result = run_script(EXTRACT_SCRIPT, "--config", config_path, "--voxel-size", VOXEL_SIZE,
                        "--max-grid-mb", 0.001)
    assert result.returncode != 0
    assert "max-grid-mb" in result.stderr
    assert not (gt_dir / "occupancy.npz").exists()


def test_extract_real_may_labels(tmp_path):
    """Real May file: 467 boxes with the known per-class counts."""
    config = yaml.safe_load(MAY_CONFIG.read_text())
    labels_path = PROJECT_ROOT / config["session"]["labels_json"]
    if not labels_path.exists():
        pytest.skip("real May labels not downloaded")
    config["session"]["labels_json"] = str(labels_path)
    config["session"]["gt_dir"] = str(tmp_path / "gt")
    config_path = tmp_path / "may.yaml"
    config_path.write_text(yaml.safe_dump(config))
    result = run_script(EXTRACT_SCRIPT, "--config", config_path)
    assert result.returncode == 0, result.stderr
    boxes = json.loads((tmp_path / "gt" / "cuboids.json").read_text())
    assert len(boxes) == EXPECTED_MAY_TOTAL
    class_ids = np.array([box["class_id"] for box in boxes])
    for class_id, expected_count in EXPECTED_MAY_CLASS_COUNTS.items():
        assert np.count_nonzero(class_ids == class_id) == expected_count


def test_validate_valid_dataset_exits_zero(tmp_path):
    """A valid synthetic dataset passes and the report file is written."""
    write_dataset(tmp_path / "ds", build_synthetic_dataset())
    report_path = tmp_path / "report.md"
    result = run_script(VALIDATE_SCRIPT, "--dataset", tmp_path / "ds", "--report", report_path)
    assert result.returncode == 0, result.stdout
    assert "PASS" in report_path.read_text()


def test_validate_corrupted_dataset_exits_one_naming_check(tmp_path):
    """A NaN pose fails poses_finite_and_count and is named in the output."""
    dataset = build_synthetic_dataset()
    dataset.streams["cam_front"].poses_world_sensor[2, 0, 3] = np.nan
    write_dataset(tmp_path / "ds", dataset)
    result = run_script(VALIDATE_SCRIPT, "--dataset", tmp_path / "ds")
    assert result.returncode == 1
    assert "poses_finite_and_count" in result.stdout and "FAIL" in result.stdout
