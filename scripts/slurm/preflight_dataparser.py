"""CPU preflight of the SplatAD inputs, run INSIDE the viking-splatad image before any GPU job.

Builds ``VikingHillDataParser`` exactly as ``ns-train splatad viking-data`` would, then:
1. prints a summary of what SplatAD will see (counts, image size, intrinsics, pose extent,
   viewing directions, split sizes);
2. writes lidar-on-image overlays for a spread of camera frames, projecting each lidar sweep
   through the *dataparser outputs* (nerfstudio ``Cameras``/``Lidars`` after the w2m recentring).
   Trunks in the lidar must land on trunks in the image; if they do not, the conversion
   (pose convention, intrinsics, extrinsics, timing) is wrong before training even starts.

Usage (Mac, CPU only):
    docker run --rm --platform linux/amd64 --entrypoint python \
        -v $PWD/data/intermediate:/data:ro -v $PWD/scripts:/scripts:ro -v $PWD/reports:/reports \
        viking-splatad:cu118 /scripts/slurm/preflight_dataparser.py \
        --data /data --session-name may_chunks0_1_smoke --out /reports/may_chunks0_1_smoke/preflight
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from viking_splatad.align.spatial_check import project_lidar_to_image
from viking_splatad.extract.geometry import apply_opencv_to_nerfstudio
from viking_splatad.splatad.viking_dataparser import VikingHillDataParserConfig


def to_homogeneous_4x4(pose_3x4: np.ndarray) -> np.ndarray:
    """Append the [0,0,0,1] row to a 3x4 rigid pose.

    Args:
        pose_3x4: [3,4] rotation | translation.

    Returns:
        [4,4] float64 homogeneous matrix.
    """
    pose_4x4 = np.eye(4, dtype=np.float64)
    pose_4x4[:3, :4] = pose_3x4
    return pose_4x4


def camera_from_lidar_transform(camera_to_world_nerfstudio: np.ndarray, lidar_to_world: np.ndarray) -> np.ndarray:
    """Compose T_camera_lidar (OpenCV camera axes) from the two sensor-to-world poses.

    The dataparser stores cameras in nerfstudio axes (x right, y up, z back). T9b is an
    involution, so applying it again returns OpenCV axes (x right, y down, z forward), the
    convention T25's pinhole projection expects. Then
        T_camera_lidar = inv(T_world_camera_cv) @ T_world_lidar.

    Args:
        camera_to_world_nerfstudio: [3,4] camera pose from ``Cameras.camera_to_worlds``.
        lidar_to_world: [3,4] lidar pose from ``Lidars.lidar_to_worlds`` (same world frame).

    Returns:
        [4,4] float64 transform mapping lidar-frame points into the OpenCV camera frame.
    """
    # Undo the OpenCV -> nerfstudio flip (involution) to get OpenCV camera axes.
    camera_to_world_opencv = to_homogeneous_4x4(apply_opencv_to_nerfstudio(camera_to_world_nerfstudio))
    # world <- lidar, then camera <- world.
    return np.linalg.inv(camera_to_world_opencv) @ to_homogeneous_4x4(lidar_to_world)


def draw_depth_coloured_points(image_bgr: np.ndarray, pixels: np.ndarray, depths: np.ndarray) -> np.ndarray:
    """Draw projected points coloured by depth (near = red, far = blue) onto a copy of the image.

    Depth is normalised by its own 5th..95th percentile range so the colour map adapts to the
    scene instead of a hardcoded metre range.

    Args:
        image_bgr: [H,W,3] uint8 image.
        pixels: [N,2] pixel coordinates of valid points.
        depths: [N] camera-frame depth (Z) of the same points.

    Returns:
        [H,W,3] uint8 overlay.
    """
    overlay = image_bgr.copy()
    if len(depths) == 0:
        return overlay
    # Robust range: percentiles ignore a few extreme returns.
    near_depth, far_depth = np.percentile(depths, [5.0, 95.0])
    depth_span = max(far_depth - near_depth, np.finfo(np.float64).eps)
    normalised_depth = np.clip((depths - near_depth) / depth_span, 0.0, 1.0)
    # JET maps 0 -> blue, 255 -> red; invert so near points are red.
    colour_indices = (255.0 * (1.0 - normalised_depth)).astype(np.uint8)
    point_colours = cv2.applyColorMap(colour_indices.reshape(-1, 1), cv2.COLORMAP_JET).reshape(-1, 3)
    # Marker radius scales with image width so overlays look the same at any resolution;
    # 1/400 of the width is a visual choice (≈3 px at 1280 px wide), not a measured value.
    marker_radius = max(1, int(round(image_bgr.shape[1] / 400)))
    rounded_pixels = np.round(pixels).astype(int)
    for point_index in range(len(rounded_pixels)):
        cv2.circle(overlay, tuple(rounded_pixels[point_index]), marker_radius,
                   tuple(int(channel) for channel in point_colours[point_index]), -1)
    return overlay


def summarise_split(split_name: str, outputs) -> None:
    """Print what SplatAD will see for one split.

    Args:
        split_name: "train" or "test".
        outputs: ``DataparserOutputs`` for that split.
    """
    cameras = outputs.cameras
    lidars = outputs.metadata["lidars"]
    point_clouds = outputs.metadata["point_clouds"]
    camera_positions = cameras.camera_to_worlds[:, :3, 3].numpy()
    # nerfstudio cameras look along -z, so the viewing direction in world is -(third column).
    viewing_directions = -cameras.camera_to_worlds[:, :3, 2].numpy()
    points_per_sweep = np.array([len(cloud) for cloud in point_clouds])
    print(f"## split={split_name}")
    print(f"cameras: {len(cameras)}  lidar sweeps: {len(lidars)}  images listed: {len(outputs.image_filenames)}")
    print(f"image size (H x W): {int(cameras.height[0])} x {int(cameras.width[0])}")
    print(f"intrinsics fx fy cx cy: {float(cameras.fx[0]):.2f} {float(cameras.fy[0]):.2f} "
          f"{float(cameras.cx[0]):.2f} {float(cameras.cy[0]):.2f}")
    # None = images were undistorted at extraction, so SplatAD renders a pure pinhole.
    distortion = cameras.distortion_params
    print(f"distortion (neurad order): {None if distortion is None else distortion[0].numpy().round(5).tolist()}")
    print(f"camera position min: {camera_positions.min(0).round(2)}  max: {camera_positions.max(0).round(2)}")
    print(f"mean viewing direction (world): {viewing_directions.mean(0).round(3)}")
    print(f"camera time range [s]: {float(cameras.times.min()):.2f} .. {float(cameras.times.max()):.2f}")
    print(f"lidar time range [s]:  {float(lidars.times.min()):.2f} .. {float(lidars.times.max()):.2f}")
    print(f"points per sweep: min {points_per_sweep.min()} median {int(np.median(points_per_sweep))} "
          f"max {points_per_sweep.max()}  (columns: {point_clouds[0].shape[1]})")
    print(f"scene box aabb: {outputs.scene_box.aabb.numpy().round(1).tolist()}")


def write_overlays(outputs, frame_count: int, output_dir: Path) -> List[Path]:
    """Project the nearest-in-time lidar sweep onto evenly spaced camera frames.

    Args:
        outputs: train-split ``DataparserOutputs``.
        frame_count: number of camera frames, spread evenly from first to last.
        output_dir: where the overlay PNGs go.

    Returns:
        Paths of the written overlays.
    """
    cameras = outputs.cameras
    lidars = outputs.metadata["lidars"]
    point_clouds = outputs.metadata["point_clouds"]
    lidar_times = lidars.times.squeeze(-1).numpy()
    output_dir.mkdir(parents=True, exist_ok=True)
    written_paths = []
    # Evenly spaced indices cover start, middle and end of the drive.
    camera_indices = np.unique(np.linspace(0, len(cameras) - 1, frame_count).round().astype(int))
    for camera_index in camera_indices:
        camera_time = float(cameras.times[camera_index])
        # Nearest sweep in time: both time axes share the dataparser's min-time origin.
        lidar_index = int(np.argmin(np.abs(lidar_times - camera_time)))
        time_gap_seconds = float(lidar_times[lidar_index] - camera_time)
        T_camera_lidar = camera_from_lidar_transform(
            cameras.camera_to_worlds[camera_index].numpy().astype(np.float64),
            lidars.lidar_to_worlds[lidar_index].numpy().astype(np.float64),
        )
        # Pinhole intrinsics of this camera (distortion ignored: overlay is a coarse check).
        camera_matrix = np.array([
            [float(cameras.fx[camera_index]), 0.0, float(cameras.cx[camera_index])],
            [0.0, float(cameras.fy[camera_index]), float(cameras.cy[camera_index])],
            [0.0, 0.0, 1.0],
        ])
        image_bgr = cv2.imread(str(outputs.image_filenames[camera_index]), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"cannot read {outputs.image_filenames[camera_index]}")
        points_lidar = point_clouds[lidar_index][:, :3].numpy().astype(np.float64)
        pixels, valid = project_lidar_to_image(
            points_lidar, T_camera_lidar, camera_matrix, image_bgr.shape[1], image_bgr.shape[0])
        # Depth for colouring: Z in the OpenCV camera frame.
        points_camera_z = points_lidar @ T_camera_lidar[2, :3] + T_camera_lidar[2, 3]
        overlay = draw_depth_coloured_points(image_bgr, pixels[valid], points_camera_z[valid])
        overlay_path = output_dir / f"overlay_cam{camera_index:05d}_lidar{lidar_index:05d}.png"
        cv2.imwrite(str(overlay_path), overlay)
        print(f"overlay {overlay_path.name}: t_cam={camera_time:.3f}s dt_lidar={time_gap_seconds:+.3f}s "
              f"points in image {int(valid.sum())}/{len(valid)}")
        written_paths.append(overlay_path)
    return written_paths


def main(argument_list: Optional[List[str]] = None) -> int:
    """Run the preflight.

    Args:
        argument_list: CLI args (defaults to sys.argv).

    Returns:
        0 on success (exceptions propagate as a non-zero exit).
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True, help="Directory holding the session dirs.")
    parser.add_argument("--session-name", required=True, help="Session dir name under --data.")
    parser.add_argument("--out", type=Path, required=True, help="Directory for overlay PNGs.")
    parser.add_argument("--frames", type=int, default=5,
                        help="Number of overlay frames (default 5: start, quartiles, end).")
    arguments = parser.parse_args(argument_list)

    # Same config path as `ns-train splatad viking-data --data ... --session-name ...`.
    dataparser = VikingHillDataParserConfig(data=arguments.data, session_name=arguments.session_name).setup()
    train_outputs = dataparser.get_dataparser_outputs(split="train")
    summarise_split("train", train_outputs)
    summarise_split("test", dataparser.get_dataparser_outputs(split="test"))
    write_overlays(train_outputs, arguments.frames, arguments.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
