"""Tests of the framework-free parts of scripts/08_render_lidar_pointcloud.py (no nerfstudio needed)."""

import importlib.util
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RENDER_LIDAR_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "08_render_lidar_pointcloud.py"


def load_render_lidar_module():
    """Import scripts/08_render_lidar_pointcloud.py (scripts/ is not a package).

    Returns:
        The loaded module.
    """
    module_spec = importlib.util.spec_from_file_location("render_lidar_pointcloud", RENDER_LIDAR_SCRIPT_PATH)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def test_pose_source_splits_like_ns_render():
    """'train+test' gives both splits in order; a single split gives one."""
    render_module = load_render_lidar_module()
    assert render_module.parse_pose_source("train+test") == ["train", "test"]
    assert render_module.parse_pose_source("test") == ["test"]


@pytest.mark.parametrize("bad_pose_source", ["val", "train+train", "train+val", ""])
def test_unsupported_or_repeated_splits_raise(bad_pose_source):
    """Only train and test exist in eval_setup(test_mode='test'); repeats would overwrite outputs."""
    render_module = load_render_lidar_module()
    with pytest.raises(ValueError):
        render_module.parse_pose_source(bad_pose_source)


def test_single_lidar_name_from_manifest():
    """The one sensor of kind 'lidar' is returned; cameras are ignored."""
    render_module = load_render_lidar_module()
    manifest = {"sensors": [{"name": "ids_camera", "kind": "camera"}, {"name": "ouster", "kind": "lidar"}]}
    assert render_module.single_lidar_sensor_name(manifest) == "ouster"


def test_two_lidars_in_manifest_raise():
    """Two lidar streams would make the single-stream time match ambiguous."""
    render_module = load_render_lidar_module()
    manifest = {"sensors": [{"name": "lidar_a", "kind": "lidar"}, {"name": "lidar_b", "kind": "lidar"}]}
    with pytest.raises(ValueError, match="exactly one lidar"):
        render_module.single_lidar_sensor_name(manifest)


def test_mean_metrics_average_each_key_over_its_own_scans():
    """Chamfer present in 1 of 2 scans: its mean is that one value; depth averages both (1+3)/2 = 2."""
    render_module = load_render_lidar_module()
    per_scan_metrics = {"a": {"depth_median_l2": 1.0, "chamfer_distance": 0.5},
                        "b": {"depth_median_l2": 3.0}}
    assert render_module.mean_metrics(per_scan_metrics) == {"chamfer_distance": 0.5, "depth_median_l2": 2.0}


def test_glossary_documents_every_lidar_npz_key():
    """Each npz key appears as a `key` table cell in the run-folder glossary, and the key list has no repeats."""
    render_module = load_render_lidar_module()
    glossary_text = (PROJECT_ROOT / "runs" / "RUN_FOLDER_GLOSSARY.md").read_text()
    assert len(set(render_module.LIDAR_NPZ_KEYS)) == len(render_module.LIDAR_NPZ_KEYS)
    undocumented_keys = [npz_key for npz_key in render_module.LIDAR_NPZ_KEYS
                         if f"| `{npz_key}` |" not in glossary_text]
    assert undocumented_keys == []

