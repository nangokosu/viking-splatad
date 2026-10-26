"""Tests of the framework-free parts of scripts/08_render_lidar_pointcloud.py (no nerfstudio needed)."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from viking_splatad.splatad.viking_conversions import load_lidar_sweep

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


def glossary_table_row(glossary_text: str, npz_key: str) -> str:
    """Return the glossary's table row for one npz key.

    Args:
        glossary_text: Contents of runs/RUN_FOLDER_GLOSSARY.md.
        npz_key: Key whose `| `key` |` row to find.

    Returns:
        The full table row line.
    """
    row_prefix = f"| `{npz_key}` |"
    matching_rows = [line for line in glossary_text.splitlines() if line.startswith(row_prefix)]
    # Exactly one row per key, or the lookup is ambiguous.
    assert len(matching_rows) == 1, f"{npz_key}: {len(matching_rows)} glossary rows"
    return matching_rows[0]


def test_glossary_point_rows_state_their_column_layout():
    """gt_points / pred_points rows give the script's column names in order and the matching width.

    gt_points is the input sweep's own rows, so its width is len(INPUT_SWEEP_COLUMNS) with the
    optional trailing channel, or one less without it ("5 or 6"); pred_points is always
    len(RENDERED_POINT_COLUMNS).
    """
    render_module = load_render_lidar_module()
    glossary_text = (PROJECT_ROOT / "runs" / "RUN_FOLDER_GLOSSARY.md").read_text()
    full_sweep_width = len(render_module.INPUT_SWEEP_COLUMNS)
    # Expected widths, derived from the script's column tuples.
    expected_width_text = {
        "gt_points": f"[N, {full_sweep_width - 1} or {full_sweep_width}]",
        "pred_points": f"[M, {len(render_module.RENDERED_POINT_COLUMNS)}]",
    }
    expected_columns_text = {
        "gt_points": "(" + ", ".join(render_module.INPUT_SWEEP_COLUMNS) + ")",
        "pred_points": ", ".join(render_module.RENDERED_POINT_COLUMNS[:3]),
    }
    for npz_key, width_text in expected_width_text.items():
        row_text = glossary_table_row(glossary_text, npz_key)
        assert width_text in row_text, f"{npz_key}: width {width_text} missing from {row_text!r}"
        assert expected_columns_text[npz_key] in row_text, f"{npz_key}: columns missing from {row_text!r}"


def test_input_sweep_keeps_channel_column_as_last(tmp_path):
    """A 6-column sweep loads with all INPUT_SWEEP_COLUMNS, channel ids unchanged, so gt_points is [N, 6].

    Synthetic sweep: 4 points at ranges 1..4 m along x, channels 0..3. With the ego filters
    disabled (radius 0, zero box) every finite row survives, so the loaded array must equal the input.
    """
    render_module = load_render_lidar_module()
    point_count = 4
    column_count = len(render_module.INPUT_SWEEP_COLUMNS)
    channel_column_index = render_module.INPUT_SWEEP_COLUMNS.index("channel")
    sweep = np.zeros((point_count, column_count), dtype=np.float32)
    # Ranges 1..point_count m on +x; channel id = row number.
    sweep[:, 0] = np.arange(1, point_count + 1, dtype=np.float32)
    sweep[:, channel_column_index] = np.arange(point_count, dtype=np.float32)
    sweep_path = tmp_path / "000000_0.npy"
    np.save(sweep_path, sweep)
    loaded_sweep = load_lidar_sweep(sweep_path, min_lidar_radius=0.0)
    # The whole row, channel included, is kept untouched.
    np.testing.assert_array_equal(loaded_sweep, sweep)
