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


class RecordingDataparserOutputs:
    """Stand-in for nerfstudio's DataparserOutputs: writes the same JSON keys as
    ``save_dataparser_transform`` (base_dataparser.py) from a known w2m and scale."""

    def __init__(self, dataparser_transform, dataparser_scale, time_offset):
        """Store the values the writer serialises.

        Args:
            dataparser_transform: (3, 4) w2m.
            dataparser_scale: Dataparser scale.
            time_offset: Dataparser time offset (s).
        """
        self.dataparser_transform = dataparser_transform
        self.dataparser_scale = dataparser_scale
        self.time_offset = time_offset

    def save_dataparser_transform(self, path):
        """Write ``{"transform", "scale", "time-offset"}`` to ``path``, as nerfstudio does.

        Args:
            path: Destination JSON path.
        """
        import json  # pylint: disable=import-outside-toplevel

        with open(path, "w", encoding="utf-8") as transforms_file:
            json.dump({"transform": self.dataparser_transform.tolist(), "scale": float(self.dataparser_scale),
                       "time-offset": self.time_offset}, transforms_file)


def test_dataparser_transforms_land_in_the_train_dir_script_10_reads(tmp_path):
    """08 writes dataparser_transforms.json next to config.yml, under the name 10 opens (job 222867 failed
    because ns-train never writes it for SplatAD), and the known w2m and scale survive the round trip."""
    import json  # pylint: disable=import-outside-toplevel

    render_module = load_render_lidar_module()
    export_spec = importlib.util.spec_from_file_location(
        "export_gaussians", PROJECT_ROOT / "scripts" / "10_export_gaussians.py")
    export_module = importlib.util.module_from_spec(export_spec)
    export_spec.loader.exec_module(export_module)
    # The two scripts must agree on the file name.
    assert render_module.DATAPARSER_TRANSFORMS_FILE_NAME == export_module.DATAPARSER_TRANSFORMS_FILE_NAME
    # A 90° yaw w2m with a shift, built from first principles: R = [[0,-1,0],[1,0,0],[0,0,1]], t = (2,-3,0.5).
    yaw_radians = np.pi / 2
    known_transform = np.array([[np.cos(yaw_radians), -np.sin(yaw_radians), 0.0, 2.0],
                                [np.sin(yaw_radians), np.cos(yaw_radians), 0.0, -3.0],
                                [0.0, 0.0, 1.0, 0.5]])
    train_dir = tmp_path / "train" / "session" / "splatad" / "run"
    train_dir.mkdir(parents=True)
    written_path = render_module.save_dataparser_transforms(
        RecordingDataparserOutputs(known_transform, 0.25, 12.5), train_dir)
    # 10 is called with --train-dir "$(dirname "$TRAIN_CONFIG")", i.e. config.yml's parent.
    assert written_path == train_dir / export_module.DATAPARSER_TRANSFORMS_FILE_NAME
    with open(written_path, encoding="utf-8") as transforms_file:
        saved_transforms = json.load(transforms_file)
    np.testing.assert_allclose(np.asarray(saved_transforms["transform"]), known_transform)
    assert saved_transforms["scale"] == 0.25


def test_sbatch_passes_config_parent_as_the_export_train_dir():
    """08 gets config.yml and writes into its parent; 10 must be pointed at that same parent."""
    sbatch_text = (PROJECT_ROOT / "scripts" / "slurm" / "splatad_run.sbatch").read_text()
    lidar_call_text = sbatch_text[sbatch_text.index("08_render_lidar_pointcloud.py"):]
    assert '--load-config "$TRAIN_CONFIG"' in lidar_call_text
    assert '--train-dir "$(dirname "$TRAIN_CONFIG")"' in sbatch_text


class RecordingStreamline:
    """Stand-in for nerfstudio's ``streamline_ad_config``: turns add_missing_points off, as render.py:1263 does."""

    def __init__(self):
        """Start with no recorded call."""
        self.call_count = 0

    def __call__(self, config):
        """Apply the nerfstudio edit that drops no-return rays, and count the call.

        Args:
            config: The config to edit in place.

        Returns:
            The same config.
        """
        self.call_count += 1
        if getattr(config.pipeline.datamanager.dataparser, "add_missing_points", None):
            config.pipeline.datamanager.dataparser.add_missing_points = False
        return config


def make_render_config(**dataparser_fields):
    """A config with only the attributes prepare_render_config touches.

    Args:
        **dataparser_fields: Dataparser attributes (e.g. add_missing_points, max_eval_frames).

    Returns:
        A nested SimpleNamespace shaped like config.pipeline.datamanager.dataparser.
    """
    from types import SimpleNamespace

    dataparser = SimpleNamespace(**dataparser_fields)
    return SimpleNamespace(pipeline=SimpleNamespace(datamanager=SimpleNamespace(dataparser=dataparser)))


@pytest.mark.parametrize("trained_add_missing_points", [True, False])
def test_render_config_keeps_the_trained_add_missing_points(trained_add_missing_points):
    """streamline_ad_config turns no-returns off; the render keeps whatever training used, and drops the eval cap."""
    render_module = load_render_lidar_module()
    streamline = RecordingStreamline()
    config = make_render_config(add_missing_points=trained_add_missing_points, max_eval_frames=7)
    edited_config = render_module.prepare_render_config(config, streamline)
    dataparser = edited_config.pipeline.datamanager.dataparser
    assert streamline.call_count == 1
    assert dataparser.add_missing_points is trained_add_missing_points
    assert dataparser.max_eval_frames is None


def test_render_config_without_the_flag_stays_without_it():
    """A dataparser with no add_missing_points field does not gain one."""
    render_module = load_render_lidar_module()
    config = make_render_config(max_eval_frames=7)
    edited_config = render_module.prepare_render_config(config, RecordingStreamline())
    assert not hasattr(edited_config.pipeline.datamanager.dataparser, "add_missing_points")


def test_ray_return_masks_separate_returns_no_returns_and_empty_cells():
    """Depths 0 (empty), 5 (return), threshold (return, inclusive), 2*threshold (no-return).

    Expected masks from the definitions: raster_did_return = 0 < depth <= threshold, and
    pred_points_did_return = that mask at the kept rays, in flattened raster order.
    """
    render_module = load_render_lidar_module()
    distance_threshold = 1e3
    near_return_depth = distance_threshold / 200.0
    no_return_depth = 2.0 * distance_threshold
    # Raster [C=1, H=2, W=3].
    raster_depth = np.array([[[0.0, near_return_depth, distance_threshold],
                              [no_return_depth, near_return_depth, 0.0]]])
    expected_raster_did_return = np.array([[[False, True, True], [False, True, False]]])
    # Keep every non-empty ray except the second near return (index 4), as a ray-drop gate might.
    kept_ray_mask = np.array([False, True, True, True, False, False])
    raster_did_return, pred_points_did_return = render_module.ray_return_masks(
        raster_depth, distance_threshold, kept_ray_mask
    )
    np.testing.assert_array_equal(raster_did_return, expected_raster_did_return)
    np.testing.assert_array_equal(pred_points_did_return,
                                  expected_raster_did_return.reshape(-1)[kept_ray_mask])
    np.testing.assert_array_equal(pred_points_did_return, [True, True, False])
