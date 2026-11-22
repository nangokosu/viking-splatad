"""Choose the corridor width from renders: holes in the train-camera view vs the kept extent.

Input: a run whose ``supersplat.ply`` is still a MAP-frame export with the old lidar-box crop (no
``supersplat_frame.json``), so it holds the background the corridor would remove. For each range
percentile P, the production corridor (T87 ``train_lidar_corridor_map(range_percentile=P)`` and
``corridor_crop_mask``) is applied to the PLY's Gaussians, and the kept set is rendered from the run's
median train camera with ``render_ply_view.composite_front_to_back`` (same image model and projection).

Per width: Gaussians kept, the kept extent (what a viewer frames), the hole fraction (pixels whose
accumulated alpha < HOLE_ALPHA) and the median / p90 absolute error vs the photo. Also the uncropped
PLY as the reference.

    scripts/test_in_image.sh python docs/research/supersplat_export/corridor_width_render_sweep.py --run-dir runs/<run>
"""

import argparse
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT_PATH = HERE / "corridor_width_render_sweep_output.json"
# Range percentiles tried as corridor widths (95 = the first default; 100 = farthest return).
RANGE_PERCENTILES = (95, 99, 99.9, 100)
# A pixel is a hole when the Gaussians cover less than half of it (alpha < 0.5): the background
# (black in a viewer) then dominates its colour.
HOLE_ALPHA = 0.5


def load_module(name: str, path: Path):
    """Import a script file as a module (file names start with digits or are not on the path).

    Args:
        name: Module name to register.
        path: File path.

    Returns:
        The module.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    """Render the PLY under each corridor width and save the comparison JSON and a panel PNG.

    Returns:
        None.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--downscale", type=int, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    arguments = parser.parse_args()
    from gsplat.cuda._torch_impl import _fully_fused_projection
    from viking_splatad.export.crop import corridor_crop_mask, train_lidar_corridor_map
    from viking_splatad.export.supersplat_ply import read_supersplat_ply
    from viking_splatad.export.viewer_frame import VIEWER_FRAME_FILE_NAME
    from viking_splatad.splatad.viking_conversions import open_dataset_view

    render = load_module("render_ply_view", HERE / "render_ply_view.py")
    check = render.load_script_23()
    downscale = arguments.downscale or render.DEFAULT_DOWNSCALE
    # Step 0: only a map-frame, box-cropped PLY still has the background to measure.
    if (arguments.run_dir / "renders" / VIEWER_FRAME_FILE_NAME).exists():
        raise SystemExit("the PLY is a viewer-frame export; this needs the old map-frame export")
    # Step 1: camera of the median train frame (as render_ply_view).
    dataset_dir = check.resolve_dataset_dir(arguments)
    frame_row = check.pick_median_frame(arguments.run_dir / "split_frames.csv", check.TRAIN_SPLIT)
    view = open_dataset_view(dataset_dir)
    frame_index, frame_path = check.locate_dataset_frame(view, frame_row)
    manifest_camera = view.manifest["cameras"][check.CAMERA_SENSOR]
    image_height = int(manifest_camera["height"]) // downscale
    image_width = int(manifest_camera["width"]) // downscale
    camera_matrix = check.camera_matrix_from_intrinsics(manifest_camera)
    camera_matrix[:2] /= downscale
    view_matrix = np.linalg.inv(view.sensor_poses[check.CAMERA_SENSOR][frame_index]).astype(np.float32)
    photo = cv2.resize(check.read_rgb_unit(frame_path), (image_width, image_height), interpolation=cv2.INTER_AREA)
    # Step 2: PLY, projected once (the masks only select Gaussians).
    fields = read_supersplat_ply(arguments.run_dir / check.PLY_RELATIVE_PATH)
    means, covariances, opacities, colours = render.ply_covariances(fields)
    zero_velocity = torch.zeros((1, 3))
    radii, means2d, depths, conics, _, _ = _fully_fused_projection(
        torch.from_numpy(means), torch.from_numpy(covariances), None, torch.from_numpy(view_matrix)[None],
        torch.from_numpy(camera_matrix.astype(np.float32))[None], image_width, image_height,
        zero_velocity, zero_velocity, torch.zeros(1), eps2d=render.PROJECTION_EPS2D,
        near_plane=render.NEAR_PLANE_METRES)
    radii, means2d, depths, conics = (tensor[0].numpy() for tensor in (radii, means2d, depths, conics))

    def render_subset(keep: np.ndarray, label: str) -> tuple:
        """Render the kept Gaussians (others get radius 0 = culled) and summarise.

        Args:
            keep: (N,) bool mask.
            label: Name of the variant.

        Returns:
            (summary dict, rendered image [H,W,3]).
        """
        subset_radii = np.where(keep[:, None], radii, 0)
        image, alpha = render.composite_front_to_back(means2d, conics, subset_radii, depths, opacities, colours,
                                                      (image_height, image_width))
        absolute_error = np.abs(np.clip(image, 0.0, 1.0) - photo)
        summary = {
            "variant": label, "gaussians_kept": int(keep.sum()),
            "kept_extent_min": means[keep].min(axis=0).round(1).tolist(),
            "kept_extent_max": means[keep].max(axis=0).round(1).tolist(),
            "hole_fraction": float(np.mean(alpha < HOLE_ALPHA)),
            "abs_error_median": float(np.median(absolute_error)),
            "abs_error_p90": float(np.percentile(absolute_error, 90)),
        }
        print(json.dumps(summary))
        return summary, np.clip(image, 0.0, 1.0)

    # Step 3: reference (the whole box-cropped PLY), then each corridor width.
    min_opacity = float(opacities.min())  # the PLY was already cut at mcmc_min_opacity
    results, panels = [], [photo]
    summary, image = render_subset(np.ones(len(means), dtype=bool), "box (current PLY)")
    results.append(summary)
    panels.append(image)
    for percentile in RANGE_PERCENTILES:
        corridor = train_lidar_corridor_map(arguments.run_dir, dataset_dir, range_percentile=percentile)
        keep = corridor_crop_mask(means, opacities, corridor.path_xy, corridor.width_m, corridor.z_lower,
                                  corridor.z_upper, min_opacity)
        summary, image = render_subset(keep, f"corridor p{percentile}")
        summary.update(width_m=corridor.width_m, z_lower=corridor.z_lower, z_upper=corridor.z_upper)
        results.append(summary)
        panels.append(image)
    image_path = arguments.output.with_suffix(".png")
    cv2.imwrite(str(image_path), check.rgb_unit_to_bgr_uint8(np.vstack([np.hstack(panels[:3]), np.hstack(panels[3:])])))
    report = {"run_dir": arguments.run_dir.as_posix(), "frame_index": frame_index, "hole_alpha": HOLE_ALPHA,
              "downscale": downscale, "variants": results, "image": image_path.as_posix(),
              "panel_order": "row 1: photo, box, p95; row 2: p99, p99.9, p100"}
    arguments.output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
