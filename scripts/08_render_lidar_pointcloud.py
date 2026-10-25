"""Render SplatAD lidar point clouds at every input lidar pose of a trained run (GPU node only).

`ns-render dataset --render-point-clouds` cannot do this for SplatAD: it calls
``get_outputs_for_lidar`` (ADModel's ray path, "Unknown sensor type" for SplatAD) and only writes
PNG plots (render.py:1140-1171). This script uses SplatAD's own lidar path instead, the same one
``SplatADPipeline.get_average_eval_image_metrics(dump_img_to_disk=True)`` uses
(splatad_pipeline.py:340-378), and keeps the arrays for evaluation on the Mac.

Steps:
    1. Load the trained pipeline from config.yml + its latest checkpoint (``eval_setup``).
    2. For each split of --pose-source, build one (Lidars, batch) pair per input scan:
         * test: ``datamanager.fixed_indices_eval_lidar_dataloader``;
         * train: the cached train scans with ``is_eval`` forced to True (otherwise
           ``_lidar_to_raster_pts`` shuffles the points and keeps one per raster cell, so some
           returns would get no ray), and ``cam_idx`` offset by the train camera count exactly as
           ``next_train_lidar`` does during training.
    3. Render each scan (``get_lidar_outputs``), turn it into points
       (``filter_lidar_pred_and_gt(output_point_cloud=True)``) and compute SplatAD's built-in
       per-scan lidar metrics (``get_image_metrics_and_images``).
    4. Name each scan after the input sweep it re-renders (time match, rendered_lidar.py) and
       write ``<output>/<split>/lidar/<idx>_<ts_ns>.npz`` + ``<output>/<split>/lidar_metrics.json``.

Per-scan npz keys (float32 unless noted; "lidar frame" = the sweep's own sensor frame, the frame
of the input ``.npy`` sweeps):
    pred_points          [M, 5] rendered points (x, y, z, intensity, t), lidar frame; rays with
                         ray-drop probability <= 0.5 and a valid GT ray only (SplatAD's rule).
    pred_points_median   [M, 5] same rays, median depth instead of expected depth.
    gt_points            [N, 5] input points that returned (x, y, z, intensity, t), lidar frame.
    raster_pts           [C, H, W, 5] GT ray raster (azimuth deg, elevation deg, depth m, t,
                         intensity) as built before rendering; depth 0 = no GT ray in the cell.
    pred_depth, pred_median_depth, pred_intensity, pred_ray_drop_prob
                         [C, H, W, 1] rendered values on the same raster.
    lidar_to_world       [3, 4] scan pose in the dataparser frame.
    dataparser_transform [3, 4] w2m (map frame -> dataparser frame).
    lidar_to_map         [4, 4] float64 scan pose in the map frame (T50): p_map = lidar_to_map @ p.
    absolute_time_s      [] float64 scan time in the poses.npz time base.
    sweep_index          [] int64 index of the input sweep (poses.npz row).

Usage (inside the SplatAD environment, on a GPU node; see scripts/slurm/splatad_run.sbatch):
    python scripts/08_render_lidar_pointcloud.py --load-config <run>/config.yml \
        --output-path <run>/renders --pose-source train+test --dataset-dir <data>/<session>
"""

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

# Splits this script renders; ``test`` is the dataparser's held-out split (eval_setup test_mode="test").
SUPPORTED_SPLITS = ("train", "test")
# Divisor turning the sweep period into the time-match tolerance: consecutive sweeps are one
# period apart, so a scan more than half a period from a sweep could belong to its neighbour.
HALF_PERIOD_DIVISOR = 2.0
# poses.npz key separator, as in dataset/writer.py (POSE_KEY_SEPARATOR); key = <sensor>__times.
POSES_TIMES_SUFFIX = "__times"


def parse_pose_source(pose_source: str) -> List[str]:
    """Split a ``train+test``-style pose source into its splits.

    Args:
        pose_source: Splits joined by "+", same syntax as ``ns-render dataset --pose-source``.

    Returns:
        Split names in the given order.

    Raises:
        ValueError: If a split is not in SUPPORTED_SPLITS or is repeated.
    """
    split_names = pose_source.split("+")
    unsupported = [split_name for split_name in split_names if split_name not in SUPPORTED_SPLITS]
    if unsupported or len(set(split_names)) != len(split_names):
        raise ValueError(f"pose source {pose_source!r}: splits must be distinct members of {SUPPORTED_SPLITS}")
    return split_names


def single_lidar_sensor_name(manifest: Dict[str, Any]) -> str:
    """Name of the dataset's only lidar stream (sweep dir name and poses.npz key prefix).

    Args:
        manifest: Parsed ``manifest.json``.

    Returns:
        The lidar sensor name.

    Raises:
        ValueError: If the manifest lists no lidar or more than one (the time match below
            assumes all rendered scans come from one stream).
    """
    lidar_names = [sensor["name"] for sensor in manifest["sensors"] if sensor["kind"] == "lidar"]
    if len(lidar_names) != 1:
        raise ValueError(f"expected exactly one lidar in the manifest, found {lidar_names}")
    return lidar_names[0]


def mean_metrics(per_scan_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Mean of every metric over the scans that report it.

    Chamfer metrics are missing for a scan with no predicted or no GT point (splatad.py:1501-1508),
    so each key is averaged over its own scan count, as ``get_average_eval_image_metrics`` does.

    Args:
        per_scan_metrics: Scan stem -> {metric name: value}.

    Returns:
        Metric name -> mean value.
    """
    metric_names = sorted({name for scan_metrics in per_scan_metrics.values() for name in scan_metrics})
    return {
        metric_name: float(np.mean([scan_metrics[metric_name] for scan_metrics in per_scan_metrics.values()
                                    if metric_name in scan_metrics]))
        for metric_name in metric_names
    }


def iterate_split_scans(datamanager: Any, split_name: str) -> Iterator[Tuple[Any, Dict[str, Any]]]:
    """Yield (single-scan Lidars, batch) for every scan of a split, ready for rendering.

    Args:
        datamanager: The loaded ``FullImageLidarDatamanager``.
        split_name: "train" or "test".

    Yields:
        (Lidars with one scan on the model device, batch dict built by ``_add_metadata``).
    """
    if split_name == "test":
        # Upstream eval path: is_eval=True rasters, cam_idx offset by the eval camera count.
        yield from datamanager.fixed_indices_eval_lidar_dataloader
        return
    # Train: copy of the training scans so the datamanager's own Lidars stay untouched.
    train_lidars = deepcopy(datamanager.train_lidar_dataset.lidars).to(datamanager.device)
    train_camera_count = len(datamanager.train_dataset)
    for scan_index, cached_scan in enumerate(datamanager.cached_lidar_train):
        scan_lidar = train_lidars[scan_index : scan_index + 1]
        if scan_lidar.metadata is None:
            scan_lidar.metadata = {}
        scan_lidar.metadata["lidar_idx"] = scan_index
        batch = cached_scan.copy()
        # Every GT return gets its own ray (batched raster), no shuffle.
        batch["is_eval"] = True
        datamanager._add_metadata(scan_lidar, batch, train_camera_count)  # pylint: disable=protected-access
        yield scan_lidar, batch


def render_scan(model: Any, scan_lidar: Any, batch: Dict[str, Any]) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """Render one scan and collect its arrays and SplatAD's per-scan lidar metrics.

    Args:
        model: The loaded SplatAD model (eval mode).
        scan_lidar: Single-scan Lidars with raster metadata.
        batch: Matching batch from ``iterate_split_scans``.

    Returns:
        (arrays for the npz without pose/time keys, metric name -> value).
    """
    import torch  # pylint: disable=import-outside-toplevel

    # Step 1: keep the GT raster as built: get_lidar_outputs shifts its time channel in place
    # (rolling-shutter centring, splatad.py:1122) through a view of the same tensor.
    raster_before_render = batch["raster_pts"].detach().clone()
    with torch.no_grad():
        # Step 2: render on the GT rays, then SplatAD's own point conversion
        # (xyz = depth * ray direction + velocity * t, kept where ray-drop prob <= 0.5).
        outputs = model.get_lidar_outputs(scan_lidar)
        prediction, ground_truth = model.filter_lidar_pred_and_gt(outputs, batch, output_point_cloud=True)
        kept_ray_mask = (prediction["ray_drop"].sigmoid() <= 0.5) * ground_truth["valid"]
        kept_intensity = outputs["intensity"].flatten()[kept_ray_mask]
        kept_time = batch["raster_pts"][..., 3].flatten()[kept_ray_mask]
        # Step 3: (x, y, z, intensity, t) rows, as splatad_pipeline.py:365-370 writes them.
        pred_points = torch.cat([prediction["point_cloud"], kept_intensity[:, None], kept_time[:, None]], dim=-1)
        pred_points_median = torch.cat(
            [prediction["median_point_cloud"], kept_intensity[:, None], kept_time[:, None]], dim=-1
        )
        gt_points = batch["lidar"][batch["lidar_pts_did_return"]]
        # Step 4: built-in metrics (depth L2, intensity RMSE, ray-drop accuracy, Chamfer).
        metrics, _ = model.get_image_metrics_and_images(outputs, batch)

    def to_float32(tensor: Any) -> np.ndarray:
        """Detached CPU float32 copy of a tensor."""
        return tensor.detach().cpu().numpy().astype(np.float32)

    arrays = {
        "pred_points": to_float32(pred_points),
        "pred_points_median": to_float32(pred_points_median),
        "gt_points": to_float32(gt_points),
        "raster_pts": to_float32(raster_before_render),
        "pred_depth": to_float32(outputs["depth"]),
        "pred_median_depth": to_float32(outputs["median_depth"]),
        "pred_intensity": to_float32(outputs["intensity"]),
        "pred_ray_drop_prob": to_float32(outputs["ray_drop_prob"]),
    }
    return arrays, {metric_name: float(value) for metric_name, value in metrics.items()}


def render_split(pipeline: Any, split_name: str, output_path: Path, sweep_stems: List[str],
                 frame_times_seconds: np.ndarray, tolerance_seconds: float) -> int:
    """Render, name and save every scan of one split.

    Args:
        pipeline: Loaded SplatAD pipeline.
        split_name: "train" or "test".
        output_path: Render root; files go to ``<output_path>/<split>/lidar/``.
        sweep_stems: Input sweep stems in frame-index order (``sorted_sweep_stems``).
        frame_times_seconds: poses.npz ``<lidar>__times`` (same order as sweep_stems).
        tolerance_seconds: Time-match tolerance (half the sweep period).

    Returns:
        Number of scans written.
    """
    from viking_splatad.splatad.rendered_lidar import (  # pylint: disable=import-outside-toplevel
        lidar_pose_in_map_frame,
        match_scan_times_to_frame_indices,
    )

    datamanager = pipeline.datamanager
    dataparser_outputs = datamanager.train_dataparser_outputs
    # 3x4 w2m and the time origin are shared by both splits (computed once, before the split).
    dataparser_transform = dataparser_outputs.dataparser_transform.detach().cpu().numpy().astype(np.float64)
    time_offset_seconds = float(dataparser_outputs.time_offset)
    scan_directory = output_path / split_name / "lidar"
    scan_directory.mkdir(parents=True, exist_ok=True)
    per_scan_metrics: Dict[str, Dict[str, float]] = {}
    for scan_lidar, batch in iterate_split_scans(datamanager, split_name):
        # Step 1: identify the input sweep this scan re-renders.
        absolute_time_seconds = float(scan_lidar.times.reshape(-1)[0].item()) + time_offset_seconds
        sweep_index = int(match_scan_times_to_frame_indices(
            np.array([absolute_time_seconds]), frame_times_seconds, tolerance_seconds)[0])
        sweep_stem = sweep_stems[sweep_index]
        if sweep_stem in per_scan_metrics:
            raise ValueError(f"{split_name}: two scans matched sweep {sweep_stem}")
        # Step 2: render and convert.
        arrays, per_scan_metrics[sweep_stem] = render_scan(pipeline.model, scan_lidar, batch)
        # Step 3: poses (dataparser frame and map frame, T50) and identity, then save.
        lidar_to_world = scan_lidar.lidar_to_worlds[0].detach().cpu().numpy().astype(np.float64)
        np.savez_compressed(
            scan_directory / f"{sweep_stem}.npz",
            **arrays,
            lidar_to_world=lidar_to_world.astype(np.float32),
            dataparser_transform=dataparser_transform.astype(np.float32),
            lidar_to_map=lidar_pose_in_map_frame(lidar_to_world, dataparser_transform),
            absolute_time_s=np.float64(absolute_time_seconds),
            sweep_index=np.int64(sweep_index),
        )
    # Step 4: per-scan and mean metrics for the split.
    with open(output_path / split_name / "lidar_metrics.json", "w", encoding="utf-8") as metrics_file:
        json.dump({"mean": mean_metrics(per_scan_metrics), "per_scan": per_scan_metrics}, metrics_file, indent=1)
    print(f"{split_name}: wrote {len(per_scan_metrics)} lidar scans to {scan_directory}")
    return len(per_scan_metrics)


def build_argument_parser() -> argparse.ArgumentParser:
    """Command-line interface.

    Returns:
        The parser.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--load-config", type=Path, required=True, help="Run's config.yml (ns-train output).")
    parser.add_argument("--output-path", type=Path, required=True, help="Render root (same as ns-render's).")
    parser.add_argument("--pose-source", default="train+test", help="Splits joined by '+': train, test.")
    parser.add_argument("--dataset-dir", type=Path, required=True,
                        help="data/<session> holding manifest.json, poses.npz and lidar/<sensor>/.")
    return parser


def main(argument_list: Optional[List[str]] = None) -> int:
    """Render the requested splits.

    Args:
        argument_list: CLI arguments (None = sys.argv).

    Returns:
        Process exit code (0 on success).
    """
    arguments = build_argument_parser().parse_args(argument_list)
    split_names = parse_pose_source(arguments.pose_source)
    # Step 1: input sweep names, times and the match tolerance, all from the dataset itself.
    from viking_splatad.splatad.rendered_lidar import sorted_sweep_stems  # pylint: disable=import-outside-toplevel

    with open(arguments.dataset_dir / "manifest.json", encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)
    lidar_name = single_lidar_sensor_name(manifest)
    sweep_stems = sorted_sweep_stems(arguments.dataset_dir / "lidar" / lidar_name)
    with np.load(arguments.dataset_dir / "poses.npz") as poses:
        frame_times_seconds = np.asarray(poses[f"{lidar_name}{POSES_TIMES_SUFFIX}"], dtype=np.float64)
    if len(frame_times_seconds) != len(sweep_stems):
        raise ValueError(f"{len(sweep_stems)} sweeps but {len(frame_times_seconds)} poses.npz times")
    tolerance_seconds = float(manifest["lidar"]["sweep_period"]) / HALF_PERIOD_DIVISOR
    # Step 2: load the model (nerfstudio imported here so --help and tests work without it).
    from nerfstudio.scripts.render import streamline_ad_config  # pylint: disable=import-outside-toplevel
    from nerfstudio.utils.eval_utils import eval_setup  # pylint: disable=import-outside-toplevel

    def prepare_render_config(config: Any) -> Any:
        """Same config edits as ``ns-render dataset`` (render.py:916-932) that apply here."""
        config = streamline_ad_config(config)
        # No eval-frame cap: every held-out scan is rendered (a cap also reseeds torch, 123).
        config.pipeline.datamanager.dataparser.max_eval_frames = None
        return config

    _, pipeline, checkpoint_path, step = eval_setup(
        arguments.load_config, test_mode="test", update_config_callback=prepare_render_config
    )
    print(f"loaded {checkpoint_path} (step {step})")
    pipeline.eval()
    # Step 3: render every split.
    for split_name in split_names:
        render_split(pipeline, split_name, arguments.output_path, sweep_stems, frame_times_seconds,
                     tolerance_seconds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
