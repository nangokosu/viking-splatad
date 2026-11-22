"""Render a run's ``supersplat.ply`` from one of its own camera frames, as a 3DGS viewer would.

Why: the colour check (``scripts/23_check_supersplat_colours.py``) samples only Gaussian centres, so it
cannot tell whether the PLY *splats* into a good picture. This renders the PLY with the standard 3DGS
image model and shows it next to the photo, to separate "the PLY content is bad" from "the viewer is
looking at it from a bad place".

Image model (3DGS, Kerbl et al. 2023), per pixel, Gaussians sorted front to back:
    C = sum_i c_i a_i prod_{j<i} (1 - a_j),   a_i = min(o_i exp(-0.5 d_i^T Q_i d_i), A_MAX)
with o_i the activated opacity, d_i the pixel offset from the projected centre and Q_i the conic
(inverse 2D covariance). Gaussians with a_i < 1/255 are skipped, as in the INRIA/gsplat rasterisers.

Ported, not rewritten: the PLY quaternion/scale -> 3D covariance (``_quat_scale_to_covar_preci``)
and the perspective projection with the 0.3 px dilation (``_fully_fused_projection``) come from
gsplat's pure-torch reference (the SplatAD fork pinned in the image). Written here, because it
cannot be ported on a CPU: gsplat's tile intersection is a CUDA kernel and its torch ``accumulate``
needs nerfacc, whose ``pack_info`` raises "Only support cuda inputs" (checked in the image). So the
pixel/Gaussian pairing and the segmented front-to-back product are implemented below.

Camera, frame and dataset come from script 23's helpers (median TRAIN camera frame).

Run in the image (torch + gsplat):
    scripts/test_in_image.sh python docs/research/supersplat_export/render_ply_view.py --run-dir runs/<run>
"""

import argparse
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_23_PATH = PROJECT_ROOT / "scripts" / "23_check_supersplat_colours.py"
DEFAULT_OUTPUT_PATH = Path(__file__).with_name("render_ply_view_output.json")
# The rendered picture goes to the scratch-free research folder next to its JSON.
DEFAULT_IMAGE_SUFFIX = "_render.png"

# Render at 1/DOWNSCALE of the manifest resolution: the CPU compositing below scales with the
# pixel count, and the question (does the PLY splat into a recognisable scene?) needs no full size.
DEFAULT_DOWNSCALE = 4
# 3DGS rasteriser constants (INRIA diff-gaussian-rasterization and gsplat): minimum alpha that is
# composited (1/255, one 8-bit level) and the cap that keeps transmittance > 0.
MIN_COMPOSITED_ALPHA = 1.0 / 255.0
MAX_ALPHA = 0.999
# gsplat's default 2D dilation (pixels^2) and near plane (metres), as passed by its rasterization().
PROJECTION_EPS2D = 0.3
NEAR_PLANE_METRES = 0.01
# Gaussians per compositing chunk: bounds the pair arrays (each Gaussian spans a few pixels).
GAUSSIANS_PER_CHUNK = 200_000


def load_script_23():
    """Import ``scripts/23_check_supersplat_colours.py`` as a module (its name starts with a digit).

    Returns:
        The module.
    """
    spec = importlib.util.spec_from_file_location("check_supersplat_colours", SCRIPT_23_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ply_covariances(fields: dict) -> tuple:
    """Means, 3D covariances, activated opacities and degree-0 colours of a SuperSplat PLY.

    The covariance is built exactly as a 3DGS viewer reads the file: rot_0..3 = (w,x,y,z) normalised,
    scale = exp(scale_k), Sigma = R diag(scale^2) R^T (gsplat's ``_quat_scale_to_covar_preci``).

    Args:
        fields: Columns from ``read_supersplat_ply``.

    Returns:
        (means [N,3] float32, covariances [N,3,3] float32, opacities [N] float32, colours [N,3] float32).
    """
    from gsplat.cuda._torch_impl import _quat_scale_to_covar_preci
    from viking_splatad.export.spherical_harmonics import SH_C0, SH_COLOUR_OFFSET

    means = np.stack([fields[axis] for axis in "xyz"], axis=1).astype(np.float32)
    quaternions = torch.from_numpy(np.stack([fields[f"rot_{index}"] for index in range(4)], axis=1).astype(np.float32))
    scales = torch.from_numpy(np.exp(np.stack([fields[f"scale_{index}"] for index in range(3)], axis=1)).astype(np.float32))
    covariances, _ = _quat_scale_to_covar_preci(quaternions, scales, compute_covar=True, compute_preci=False)
    # opacity is a logit; colour = 0.5 + C0 * f_dc (degree 0), clipped like a viewer's display.
    opacities = (1.0 / (1.0 + np.exp(-fields["opacity"].astype(np.float64)))).astype(np.float32)
    colours = np.clip(SH_COLOUR_OFFSET + SH_C0 * np.stack([fields[f"f_dc_{channel}"] for channel in range(3)], axis=1),
                      0.0, 1.0).astype(np.float32)
    return means, covariances.numpy(), opacities, colours


def composite_front_to_back(means2d: np.ndarray, conics: np.ndarray, radii: np.ndarray, depths: np.ndarray,
                            opacities: np.ndarray, colours: np.ndarray, image_size_hw: tuple) -> tuple:
    """Front-to-back alpha compositing of projected Gaussians (the 3DGS image model, CPU).

    Each Gaussian covers the pixels of its 3-sigma box (gsplat's radii). Gaussians are processed in
    depth order, in chunks; inside a chunk the pairs are sorted by (pixel, depth) and the
    transmittance before each pair is the running per-pixel transmittance times the exclusive
    product of (1 - a) of the chunk's earlier pairs on that pixel, computed as an exclusive segmented
    cumulative sum of log(1 - a). Cost O(pairs log pairs).

    Args:
        means2d: [N,2] pixel coordinates of the centres (pixel centres at +0.5, gsplat convention).
        conics: [N,3] upper triangle (a, b, c) of the inverse 2D covariance.
        radii: [N,2] int box half-sizes in pixels; 0 means culled.
        depths: [N] camera-frame depths.
        opacities: [N] activated opacities.
        colours: [N,3] RGB in [0,1].
        image_size_hw: (height, width).

    Returns:
        (image [H,W,3] float32, accumulated alpha [H,W] float32).
    """
    image_height, image_width = image_size_hw
    pixel_count = image_height * image_width
    colour_sum = np.zeros((pixel_count, 3), dtype=np.float64)
    transmittance = np.ones(pixel_count, dtype=np.float64)
    # Step 1: keep projected Gaussians only, sorted front to back.
    kept = np.flatnonzero((radii > 0).all(axis=1))
    kept = kept[np.argsort(depths[kept], kind="stable")]
    for chunk_start in range(0, len(kept), GAUSSIANS_PER_CHUNK):
        chunk = kept[chunk_start:chunk_start + GAUSSIANS_PER_CHUNK]
        # Step 2: integer pixel box per Gaussian, clipped to the image.
        column_low = np.clip(np.floor(means2d[chunk, 0] - radii[chunk, 0]), 0, image_width - 1).astype(np.int64)
        column_high = np.clip(np.ceil(means2d[chunk, 0] + radii[chunk, 0]), 0, image_width - 1).astype(np.int64)
        row_low = np.clip(np.floor(means2d[chunk, 1] - radii[chunk, 1]), 0, image_height - 1).astype(np.int64)
        row_high = np.clip(np.ceil(means2d[chunk, 1] + radii[chunk, 1]), 0, image_height - 1).astype(np.int64)
        box_width = column_high - column_low + 1
        box_height = row_high - row_low + 1
        pair_counts = box_width * box_height
        # Step 3: enumerate (Gaussian, pixel) pairs: position inside the box -> (column, row).
        pair_gaussian = np.repeat(np.arange(len(chunk)), pair_counts)
        pair_offset = np.arange(int(pair_counts.sum())) - np.repeat(np.cumsum(pair_counts) - pair_counts, pair_counts)
        pair_column = column_low[pair_gaussian] + pair_offset % box_width[pair_gaussian]
        pair_row = row_low[pair_gaussian] + pair_offset // box_width[pair_gaussian]
        # Step 4: alpha = o exp(-sigma), sigma = 0.5 (a dx^2 + c dy^2) + b dx dy at the pixel centre.
        global_index = chunk[pair_gaussian]
        delta_x = means2d[global_index, 0] - (pair_column + 0.5)
        delta_y = means2d[global_index, 1] - (pair_row + 0.5)
        conic = conics[global_index]
        sigma = 0.5 * (conic[:, 0] * delta_x ** 2 + conic[:, 2] * delta_y ** 2) + conic[:, 1] * delta_x * delta_y
        alpha = np.minimum(opacities[global_index] * np.exp(-sigma), MAX_ALPHA)
        significant = (sigma >= 0.0) & (alpha >= MIN_COMPOSITED_ALPHA)
        pixel = (pair_row * image_width + pair_column)[significant]
        alpha = alpha[significant].astype(np.float64)
        depth_rank = pair_gaussian[significant]  # chunk order is depth order
        colour = colours[global_index[significant]]
        # Step 5: sort by (pixel, depth); exclusive segmented sum of log(1 - a) per pixel.
        order = np.lexsort((depth_rank, pixel))
        pixel, alpha, colour = pixel[order], alpha[order], colour[order]
        log_keep = np.log1p(-alpha)
        inclusive = np.cumsum(log_keep)
        segment_start = np.r_[True, pixel[1:] != pixel[:-1]]
        start_index = np.maximum.accumulate(np.where(segment_start, np.arange(len(pixel)), 0))
        exclusive_in_segment = inclusive - log_keep - (inclusive[start_index] - log_keep[start_index])
        weight = transmittance[pixel] * np.exp(exclusive_in_segment) * alpha
        # Step 6: accumulate colour; update each touched pixel's transmittance by its segment product.
        np.add.at(colour_sum, pixel, weight[:, None] * colour)
        segment_log_total = np.zeros(pixel_count, dtype=np.float64)
        np.add.at(segment_log_total, pixel, log_keep)
        transmittance *= np.exp(segment_log_total)
    image = colour_sum.reshape(image_height, image_width, 3).astype(np.float32)
    accumulated_alpha = (1.0 - transmittance).reshape(image_height, image_width).astype(np.float32)
    return image, accumulated_alpha


def main() -> None:
    """Render the PLY from the run's median train camera frame; write PNG (photo | render | alpha) + JSON.

    Returns:
        None.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--downscale", type=int, default=DEFAULT_DOWNSCALE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    arguments = parser.parse_args()
    from gsplat.cuda._torch_impl import _fully_fused_projection
    from viking_splatad.export.supersplat_ply import read_supersplat_ply
    from viking_splatad.export.viewer_frame import VIEWER_FRAME_FILE_NAME, read_viewer_frame_record
    from viking_splatad.splatad.viking_conversions import open_dataset_view

    check = load_script_23()
    # Step 1: dataset (verified against split.json), median train frame, its pose and intrinsics.
    dataset_dir = check.resolve_dataset_dir(arguments)
    frame_row = check.pick_median_frame(arguments.run_dir / "split_frames.csv", check.TRAIN_SPLIT)
    view = open_dataset_view(dataset_dir)
    frame_index, frame_path = check.locate_dataset_frame(view, frame_row)
    manifest_camera = view.manifest["cameras"][check.CAMERA_SENSOR]
    image_height = int(manifest_camera["height"]) // arguments.downscale
    image_width = int(manifest_camera["width"]) // arguments.downscale
    camera_matrix = check.camera_matrix_from_intrinsics(manifest_camera)
    # Intrinsics scale with the image: fx, fy, cx, cy all divide by the downscale (rows 0-1).
    camera_matrix[:2] /= arguments.downscale
    pose_map_camera = view.sensor_poses[check.CAMERA_SENSOR][frame_index]
    # A PLY written in the SuperSplat viewer frame (T88 sidecar present): p_viewer = M (p_map - o), so the
    # camera pose moves the same way, T_viewer_camera = [M | -M o] T_map_camera. Rendering the viewer-frame
    # PLY from that pose tests its rotated quaternions too, not only its centres.
    frame_record_path = arguments.run_dir / "renders" / VIEWER_FRAME_FILE_NAME
    frame_name = "map"
    if frame_record_path.exists():
        record = read_viewer_frame_record(frame_record_path)
        viewer_from_map = np.eye(4)
        viewer_from_map[:3, :3] = record["rotation"]
        viewer_from_map[:3, 3] = -record["rotation"] @ record["origin_map"]
        pose_map_camera = viewer_from_map @ pose_map_camera
        frame_name = "supersplat_viewer"
    view_matrix = np.linalg.inv(pose_map_camera).astype(np.float32)  # PLY frame -> OpenCV camera
    # Step 2: PLY -> covariances (as a viewer reads it), then gsplat's projection.
    fields = read_supersplat_ply(arguments.run_dir / check.PLY_RELATIVE_PATH)
    means, covariances, opacities, colours = ply_covariances(fields)
    camera_count = 1
    zero_velocity = torch.zeros((camera_count, 3))
    radii, means2d, depths, conics, _, _ = _fully_fused_projection(
        torch.from_numpy(means), torch.from_numpy(covariances), None, torch.from_numpy(view_matrix)[None],
        torch.from_numpy(camera_matrix.astype(np.float32))[None], image_width, image_height,
        zero_velocity, zero_velocity, torch.zeros(camera_count), eps2d=PROJECTION_EPS2D, near_plane=NEAR_PLANE_METRES)
    radii, means2d, depths, conics = (tensor[0].numpy() for tensor in (radii, means2d, depths, conics))
    # Step 3: composite; compare with the photo at the same size.
    image, accumulated_alpha = composite_front_to_back(means2d, conics, radii, depths, opacities, colours,
                                                       (image_height, image_width))
    photo = cv2.resize(check.read_rgb_unit(frame_path), (image_width, image_height), interpolation=cv2.INTER_AREA)
    absolute_error = np.abs(image - photo)
    alpha_panel = np.repeat(accumulated_alpha[..., None], 3, axis=2)
    panel = np.hstack([photo, np.clip(image, 0.0, 1.0), alpha_panel])
    image_path = arguments.output.with_name(arguments.output.stem + "_" + arguments.run_dir.name + DEFAULT_IMAGE_SUFFIX)
    cv2.imwrite(str(image_path), check.rgb_unit_to_bgr_uint8(panel))
    projected = (radii > 0).all(axis=1)
    report = {
        "run_dir": arguments.run_dir.as_posix(), "frame": {"frame_index": frame_index, "file": frame_row["file"]},
        "image_size_hw": [image_height, image_width], "downscale": arguments.downscale, "ply_frame": frame_name,
        "gaussians_total": int(len(means)), "gaussians_projected": int(projected.sum()),
        "projected_radius_px_percentiles": {str(q): float(np.percentile(radii[projected].max(axis=1), q))
                                            for q in (50, 90, 99)},
        "accumulated_alpha_percentiles": {str(q): float(np.percentile(accumulated_alpha, q)) for q in (5, 25, 50)},
        "abs_error_vs_photo": {"median": float(np.median(absolute_error)),
                               "p90": float(np.percentile(absolute_error, 90))},
        "image": image_path.as_posix(),
    }
    arguments.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
