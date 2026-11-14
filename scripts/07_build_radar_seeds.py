#!/usr/bin/env python3
"""Build the June train-only, label-free radar Gaussian seed artifact.

This command reads only the intermediate sensor dataset and committed reference split.
Ground-truth paths are never opened. The output directory must not already exist.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from viking_splatad.config import find_project_root, load_session_config
from viking_splatad.dataset.run_split import (
    TRAIN_SPLIT_NAME,
    build_run_split_with_cores,
    nearest_reference_positions,
    split_digest,
)
from viking_splatad.dataset.writer import FRAME_FILE_PATTERN
from viking_splatad.extract.pose import Trajectory
from viking_splatad.radar_init.accumulate import accumulate_radar_points, corrected_radar_times
from viking_splatad.radar_init.artifact import sha256_file, write_radar_seed_artifact
from viking_splatad.radar_init.seeds import (
    NEUTRAL_GREY_RGB,
    build_seed_gaussians,
    paint_visible_seed_colors,
)
from viking_splatad.radar_init.select import select_radar_candidates
from viking_splatad.splatad.viking_conversions import load_lidar_sweep, open_dataset_view

NANOSECONDS_PER_SECOND = 1_000_000_000


def _split_frames(config: Dict, dataset_view) -> list:
    """Rebuild and verify the committed session split.

    Args:
        config: Loaded session configuration.
        dataset_view: Pixel-free intermediate dataset view.

    Returns:
        Exact split frame records.
    """
    split_config = config["split"]
    frames, _, _, _ = build_run_split_with_cores(
        dataset_view,
        0.0,
        1.0,
        split_config["train_fraction"],
        test_block_count=split_config.get("test_block_count"),
        test_block_core_fraction=split_config.get("test_block_core_fraction", 1.0),
        test_split_method=split_config.get("method"),
        test_render_fraction=split_config.get("test_render_fraction", 1.0),
    )
    project_root = find_project_root()
    reference_path = project_root / "configs" / "reference_splits" / f"{config['session']['name']}.json"
    reference_record = json.loads(reference_path.read_text())
    if split_digest(frames) != reference_record["split_digest"]:
        raise ValueError("dataset does not match the committed reference split")
    return frames


def _radar_files_and_times(dataset_root: Path, manifest: Dict) -> tuple[List[Path], np.ndarray]:
    """Return ordered radar files and session-reference seconds from their names.

    Args:
        dataset_root: Intermediate session directory.
        manifest: Parsed intermediate manifest.

    Returns:
        ``(paths, times_s)`` ordered by frame index.
    """
    indexed_paths = []
    for radar_path in (dataset_root / "sidecar" / "radar").glob("*.npy"):
        name_match = FRAME_FILE_PATTERN.match(radar_path.name)
        if name_match is not None:
            indexed_paths.append((int(name_match.group(1)), int(name_match.group(2)), radar_path))
    indexed_paths.sort()
    if not indexed_paths:
        raise FileNotFoundError(f"no radar sidecar files under {dataset_root}")
    reference_ns = int(manifest["pipeline"]["reference_ns"])
    times_s = np.array([
        (timestamp_ns - reference_ns) / NANOSECONDS_PER_SECOND
        for _, timestamp_ns, _ in indexed_paths
    ], dtype=np.float64)
    return [radar_path for _, _, radar_path in indexed_paths], times_s


def _base_trajectory(dataset_view, lidar_name: str, transform_base_lidar: np.ndarray) -> Trajectory:
    """Recover ``T_map_base`` samples from stored lidar poses and calibration.

    Args:
        dataset_view: Intermediate dataset view.
        lidar_name: Lidar stream used as the trajectory reference.
        transform_base_lidar: Calibration ``T_base_lidar``.

    Returns:
        Base trajectory covering the stored lidar times.
    """
    map_from_base = dataset_view.sensor_poses[lidar_name] @ np.linalg.inv(transform_base_lidar)
    return Trajectory(
        dataset_view.sensor_times[lidar_name],
        map_from_base[:, :3, 3],
        Rotation.from_matrix(map_from_base[:, :3, :3]).as_quat(),
    )


def _train_lidar_points(dataset_view, train_files: set[str], lidar_name: str) -> np.ndarray:
    """Accumulate only training LiDAR returns in the map frame.

    Args:
        dataset_view: Intermediate dataset view.
        train_files: Dataset-relative paths belonging to train.
        lidar_name: Lidar stream name.

    Returns:
        Concatenated finite map-frame points ``[N,3]``.
    """
    lidar_model = dataset_view.manifest["lidar"]
    minimum_distance = lidar_model["min_lidar_dist"]
    if isinstance(minimum_distance, list):
        minimum_radius = 0.0
        ego_half_extents = tuple(minimum_distance)
    else:
        minimum_radius = float(minimum_distance)
        ego_half_extents = (0.0, 0.0, 0.0)
    point_blocks = []
    for frame_index, sweep_path in enumerate(dataset_view.sensor_frame_paths[lidar_name]):
        if sweep_path.relative_to(dataset_view.root).as_posix() not in train_files:
            continue
        sweep = load_lidar_sweep(sweep_path, minimum_radius, ego_half_extents)
        pose = dataset_view.sensor_poses[lidar_name][frame_index]
        point_blocks.append(sweep[:, :3] @ pose[:3, :3].T + pose[:3, 3])
    if not point_blocks:
        raise ValueError("reference split contains no train LiDAR returns")
    return np.concatenate(point_blocks)


def _paint_from_train_cameras(
    candidates,
    dataset_view,
    train_files: set[str],
    camera_name: str,
    lidar_name: str,
    azimuth_resolution_rad: float,
    elevation_resolution_rad: float,
    range_resolution_m: float,
) -> np.ndarray:
    """Paint candidates from nearest train cameras, retaining grey when occluded.

    Args:
        candidates: T35 candidate voxels.
        dataset_view: Intermediate dataset view.
        train_files: Dataset-relative paths belonging to train.
        camera_name: Camera stream used for color.
        lidar_name: Lidar stream providing the nearest train occlusion sweep.
        azimuth_resolution_rad: Hugin azimuth resolution.
        elevation_resolution_rad: Hugin elevation resolution.
        range_resolution_m: Hugin range resolution and occlusion margin.

    Returns:
        uint8 RGB colors, one per candidate.
    """
    camera_paths = dataset_view.sensor_frame_paths[camera_name]
    train_camera_indices = np.array([
        frame_index
        for frame_index, camera_path in enumerate(camera_paths)
        if camera_path.relative_to(dataset_view.root).as_posix() in train_files
    ], dtype=np.int64)
    if train_camera_indices.size == 0:
        raise ValueError("reference split contains no train camera frames")
    lidar_paths = dataset_view.sensor_frame_paths[lidar_name]
    train_lidar_indices = np.array([
        frame_index
        for frame_index, lidar_path in enumerate(lidar_paths)
        if lidar_path.relative_to(dataset_view.root).as_posix() in train_files
    ], dtype=np.int64)
    if train_lidar_indices.size == 0:
        raise ValueError("reference split contains no train LiDAR sweeps")
    lidar_model = dataset_view.manifest["lidar"]
    minimum_distance = lidar_model["min_lidar_dist"]
    if isinstance(minimum_distance, list):
        minimum_radius = 0.0
        ego_half_extents = tuple(minimum_distance)
    else:
        minimum_radius = float(minimum_distance)
        ego_half_extents = (0.0, 0.0, 0.0)
    candidate_times = candidates.time_spans_s.mean(axis=1)
    train_camera_times = dataset_view.sensor_times[camera_name][train_camera_indices]
    nearest_positions = nearest_reference_positions(train_camera_times, candidate_times)
    selected_camera_indices = train_camera_indices[nearest_positions]
    colors = np.repeat(NEUTRAL_GREY_RGB[None], len(candidate_times), axis=0)
    intrinsics = dataset_view.manifest["cameras"][camera_name]
    camera_matrix = np.array([
        [intrinsics["fx"], 0.0, intrinsics["cx"]],
        [0.0, intrinsics["fy"], intrinsics["cy"]],
        [0.0, 0.0, 1.0],
    ])
    footprint_pixels = max(
        intrinsics["fx"] * np.tan(azimuth_resolution_rad),
        intrinsics["fy"] * np.tan(elevation_resolution_rad),
    )
    for camera_index in np.unique(selected_camera_indices):
        candidate_mask = selected_camera_indices == camera_index
        image_bgr = cv2.imread(str(camera_paths[camera_index]), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError(f"cannot read train image {camera_paths[camera_index]}")
        nearest_lidar_position = nearest_reference_positions(
            dataset_view.sensor_times[lidar_name][train_lidar_indices],
            np.array([dataset_view.sensor_times[camera_name][camera_index]]),
        )[0]
        lidar_index = train_lidar_indices[nearest_lidar_position]
        lidar_sweep = load_lidar_sweep(
            lidar_paths[lidar_index], minimum_radius, ego_half_extents
        )
        lidar_pose = dataset_view.sensor_poses[lidar_name][lidar_index]
        occluder_points_map = (
            lidar_sweep[:, :3] @ lidar_pose[:3, :3].T + lidar_pose[:3, 3]
        )
        camera_colors, painted = paint_visible_seed_colors(
            candidates.means_map[candidate_mask],
            dataset_view.sensor_poses[camera_name][camera_index],
            camera_matrix,
            cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
            occluder_points_map,
            footprint_pixels,
            range_resolution_m,
        )
        selected_rows = np.flatnonzero(candidate_mask)
        colors[selected_rows[painted]] = camera_colors[painted]
    return colors


def main(argument_list: Optional[List[str]] = None) -> int:
    """Build a radar seed artifact without touching GT or submitting work.

    Args:
        argument_list: CLI arguments, defaulting to ``sys.argv[1:]``.

    Returns:
        Zero after writing and validating the artifact.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="June session YAML")
    parser.add_argument("--output", type=Path, default=None, help="new artifact directory")
    arguments = parser.parse_args(argument_list)
    config = load_session_config(arguments.config)
    if config["session"]["name"] != "june_tall_grass":
        raise ValueError("radar initialization is currently scoped to june_tall_grass")
    dataset_root = config["session"]["output_dir"]
    output_directory = arguments.output or dataset_root / "radar_inputs"
    dataset_view = open_dataset_view(dataset_root)
    frames = _split_frames(config, dataset_view)
    train_files = {frame.file for frame in frames if frame.split == TRAIN_SPLIT_NAME}
    manifest = dataset_view.manifest
    lidar_name = config["lidar"]["name"]
    camera_name = config["camera"]["name"]
    radar_config = config["radar"]
    radar_paths, radar_times = _radar_files_and_times(dataset_root, manifest)
    clock = manifest["clock"]
    radar_clock_offset_s = (
        clock["sensor_offsets_seconds"][radar_config["name"]]
        + clock["radar_drift_offset"]
    )
    corrected_times = corrected_radar_times(
        radar_times, radar_clock_offset_s, clock["radar_drift_rate"]
    )
    lidar_train_times = np.array([
        frame.time_s for frame in frames
        if frame.kind == "lidar" and frame.split == TRAIN_SPLIT_NAME
    ])
    all_lidar_times = dataset_view.sensor_times[lidar_name]
    nearest_lidar_positions = nearest_reference_positions(all_lidar_times, corrected_times)
    train_scan_mask = np.isin(all_lidar_times[nearest_lidar_positions], lidar_train_times)
    base_to_lidar_key = f"{config['frames']['base']}->{config['frames']['lidar']}"
    base_to_radar_key = manifest["pipeline"]["radar"]["extrinsic_key"]
    accumulated = accumulate_radar_points(
        [np.load(radar_path, allow_pickle=False) for radar_path in radar_paths],
        radar_times,
        train_scan_mask,
        _base_trajectory(
            dataset_view, lidar_name, np.asarray(manifest["extrinsics"][base_to_lidar_key])
        ),
        np.asarray(manifest["extrinsics"][base_to_radar_key]),
        radar_clock_offset_s,
        clock["radar_drift_rate"],
    )
    train_lidar_points = _train_lidar_points(dataset_view, train_files, lidar_name)
    range_resolution_m = float(radar_config["range_resolution_m"])
    azimuth_resolution_rad = np.deg2rad(radar_config["azimuth_resolution_degrees"])
    elevation_resolution_rad = np.deg2rad(radar_config["elevation_resolution_degrees"])
    candidates = select_radar_candidates(
        accumulated,
        train_lidar_points,
        range_resolution_m,
        azimuth_resolution_rad,
        elevation_resolution_rad,
        radar_config["doppler_sigma_multiplier"],
        radar_config["support_percentile"],
        radar_config["canopy_height_percentile"],
    )
    colors = _paint_from_train_cameras(
        candidates, dataset_view, train_files, camera_name, lidar_name,
        azimuth_resolution_rad, elevation_resolution_rad, range_resolution_m,
    )
    seeds = build_seed_gaussians(
        candidates, colors, range_resolution_m, azimuth_resolution_rad, elevation_resolution_rad
    )
    parameter_record = {
        parameter_name: radar_config[parameter_name]
        for parameter_name in (
            "range_resolution_m", "azimuth_resolution_degrees",
            "elevation_resolution_degrees", "doppler_sigma_multiplier",
            "support_percentile", "canopy_height_percentile",
        )
    }
    write_radar_seed_artifact(
        output_directory,
        seeds,
        config["session"]["name"],
        {
            "manifest_sha256": sha256_file(dataset_root / "manifest.json"),
            "poses_sha256": sha256_file(dataset_root / "poses.npz"),
            "split_digest": split_digest(frames),
            "config_sha256": sha256_file(arguments.config),
        },
        parameter_record,
    )
    print(f"wrote {len(seeds.means)} radar seeds to {output_directory}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
