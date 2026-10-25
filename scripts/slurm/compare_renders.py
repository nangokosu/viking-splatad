"""Compare a fetched SplatAD run's rendered camera images with the real frames (Mac, CPU only).

For each split under ``<run_dir>/renders/{train,test}/{rgb,gt-rgb}`` (written by ``ns-render dataset
--rendered-output-names rgb gt-rgb`` in splatad_run.sbatch):
1. prints per-split PSNR of rendered ``rgb`` against ``gt-rgb`` (mean / min / max);
2. checks that ``gt-rgb`` is the extracted frame, by PSNR against the same-named raw image;
3. writes ``<out>/<split>_render_vs_gt.jpg``: start, middle and end frames, render left, real right.

This is a smoke check, not an evaluation: PSNR here only flags gross failure, and the sheets are for
looking at alignment (trunks, edges, car) between render and real frame.

Usage:
    uv run python scripts/slurm/compare_renders.py runs/<run_id> \
        --raw-camera-dir data/intermediate/<session>/camera/ids_camera
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

# Splits that `ns-render dataset --pose-source train+test` writes.
RENDER_SPLITS = ("test", "train")


def peak_signal_to_noise_ratio(rendered_image: np.ndarray, reference_image: np.ndarray) -> float:
    """PSNR of two same-shape uint8 images.

    PSNR = 10 log10(MAX^2 / MSE), with MAX = 255 for uint8. Identical images (MSE = 0) give +inf.

    Args:
        rendered_image: [H,W,C] uint8.
        reference_image: [H,W,C] uint8, same shape.

    Returns:
        PSNR in dB.
    """
    if rendered_image.shape != reference_image.shape:
        raise ValueError(f"shape mismatch {rendered_image.shape} vs {reference_image.shape}")
    # Promote to float64 so the squared difference of uint8 values does not wrap.
    mean_squared_error = float(np.mean((rendered_image.astype(np.float64) - reference_image.astype(np.float64)) ** 2))
    if mean_squared_error == 0.0:
        return float("inf")
    max_pixel_value = float(np.iinfo(np.uint8).max)
    return 10.0 * float(np.log10(max_pixel_value ** 2 / mean_squared_error))


def read_image(image_path: Path) -> np.ndarray:
    """Read an image as BGR uint8, failing loudly on a missing or unreadable file.

    Args:
        image_path: image file.

    Returns:
        [H,W,3] uint8 array.
    """
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"cannot read {image_path}")
    return image


def start_middle_end_indices(frame_count: int) -> List[int]:
    """Indices of the first, middle and last frame (deduplicated for fewer than 3 frames).

    Args:
        frame_count: number of frames (> 0).

    Returns:
        Sorted unique indices.
    """
    return sorted({0, frame_count // 2, frame_count - 1})


def compare_split(split_dir: Path, raw_camera_dir: Optional[Path], sheet_path: Path) -> List[float]:
    """Print PSNR stats for one split and write its comparison sheet.

    Args:
        split_dir: ``renders/<split>`` holding ``rgb/`` and ``gt-rgb/`` with matching file names.
        raw_camera_dir: directory of extracted frames with the same file names, or None to skip.
        sheet_path: output JPEG path.

    Returns:
        Per-frame PSNR values, in file-name (= time) order.
    """
    rendered_paths = sorted((split_dir / "rgb").glob("*.jpg"))
    if not rendered_paths:
        raise FileNotFoundError(f"no rendered frames in {split_dir / 'rgb'}")
    # Per-frame PSNR of render vs the real frame the model was supervised with.
    psnr_values = [peak_signal_to_noise_ratio(read_image(rendered_path),
                                              read_image(split_dir / "gt-rgb" / rendered_path.name))
                   for rendered_path in rendered_paths]
    psnr_array = np.array(psnr_values)
    print(f"{split_dir.name}: frames {len(psnr_array)}  PSNR mean {psnr_array.mean():.2f}  "
          f"min {psnr_array.min():.2f}  max {psnr_array.max():.2f} dB")
    # gt-rgb should be the extracted frame up to JPEG re-encoding (high PSNR, ~50 dB);
    # a low value means the dataparser fed SplatAD a different image than we extracted.
    if raw_camera_dir is not None:
        first_name = rendered_paths[0].name
        gt_versus_raw = peak_signal_to_noise_ratio(read_image(split_dir / "gt-rgb" / first_name),
                                                   read_image(raw_camera_dir / first_name))
        print(f"{split_dir.name}: gt-rgb vs raw extracted frame {first_name}: {gt_versus_raw:.2f} dB")
    # Sheet rows: render | real, each at half resolution to keep the file viewable.
    sheet_rows = []
    for frame_index in start_middle_end_indices(len(rendered_paths)):
        rendered_image = read_image(rendered_paths[frame_index])
        real_image = read_image(split_dir / "gt-rgb" / rendered_paths[frame_index].name)
        half_size = (rendered_image.shape[1] // 2, rendered_image.shape[0] // 2)
        sheet_rows.append(np.hstack([cv2.resize(rendered_image, half_size, interpolation=cv2.INTER_AREA),
                                     cv2.resize(real_image, half_size, interpolation=cv2.INTER_AREA)]))
        print(f"  sheet row: {rendered_paths[frame_index].name}  PSNR {psnr_values[frame_index]:.2f} dB")
    sheet_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(sheet_path), np.vstack(sheet_rows))
    return psnr_values


def main(argument_list: Optional[List[str]] = None) -> int:
    """Compare every rendered split of a fetched run.

    Args:
        argument_list: CLI args (defaults to sys.argv).

    Returns:
        0 on success (exceptions propagate as a non-zero exit).
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, help="Fetched run directory (runs/<run_id>).")
    parser.add_argument("--raw-camera-dir", type=Path, default=None,
                        help="Extracted camera frames, to check gt-rgb against them.")
    parser.add_argument("--out", type=Path, default=None, help="Sheet directory (default <run_dir>/compare).")
    arguments = parser.parse_args(argument_list)
    output_dir = arguments.out if arguments.out is not None else arguments.run_dir / "compare"
    for split_name in RENDER_SPLITS:
        split_dir = arguments.run_dir / "renders" / split_name
        if split_dir.is_dir():
            compare_split(split_dir, arguments.raw_camera_dir, output_dir / f"{split_name}_render_vs_gt.jpg")
    print(f"sheets in {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
