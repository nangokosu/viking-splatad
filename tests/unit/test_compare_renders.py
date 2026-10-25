"""Tests of scripts/slurm/compare_renders.py with synthetic images (expected values from the PSNR definition)."""

import importlib.util
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMPARE_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "slurm" / "compare_renders.py"


def load_compare_module():
    """Import scripts/slurm/compare_renders.py (scripts/ is not a package).

    Returns:
        The loaded module.
    """
    module_spec = importlib.util.spec_from_file_location("compare_renders", COMPARE_SCRIPT_PATH)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def test_psnr_of_constant_offset_matches_definition():
    """Every pixel off by d gives MSE = d^2, so PSNR = 10 log10(255^2 / d^2) = 20 log10(255 / d)."""
    compare_module = load_compare_module()
    pixel_offset = 5
    reference_image = np.full((4, 6, 3), 100, dtype=np.uint8)
    rendered_image = reference_image + np.uint8(pixel_offset)
    expected_psnr = 20.0 * math.log10(255.0 / pixel_offset)
    assert compare_module.peak_signal_to_noise_ratio(rendered_image, reference_image) == pytest.approx(expected_psnr)


def test_psnr_does_not_wrap_uint8_differences():
    """0 vs 255 everywhere: MSE = 255^2, so PSNR = 0 dB (a wrapped uint8 subtraction would give 1 and ~48 dB)."""
    compare_module = load_compare_module()
    black_image = np.zeros((2, 2, 3), dtype=np.uint8)
    white_image = np.full((2, 2, 3), 255, dtype=np.uint8)
    assert compare_module.peak_signal_to_noise_ratio(black_image, white_image) == pytest.approx(0.0)


def test_psnr_identical_is_infinite_and_shape_mismatch_raises():
    """MSE = 0 gives +inf; different shapes are an error, not a broadcast."""
    compare_module = load_compare_module()
    image = np.full((3, 3, 3), 7, dtype=np.uint8)
    assert compare_module.peak_signal_to_noise_ratio(image, image) == math.inf
    with pytest.raises(ValueError):
        compare_module.peak_signal_to_noise_ratio(image, np.zeros((3, 4, 3), dtype=np.uint8))


def test_start_middle_end_indices():
    """First, middle (n // 2) and last index, deduplicated for short splits."""
    compare_module = load_compare_module()
    assert compare_module.start_middle_end_indices(15) == [0, 7, 14]
    assert compare_module.start_middle_end_indices(2) == [0, 1]
    assert compare_module.start_middle_end_indices(1) == [0]


def test_compare_split_on_synthetic_run(tmp_path):
    """A synthetic split with a known per-frame offset: PSNR per frame and the sheet size follow from the inputs."""
    compare_module = load_compare_module()
    frame_height, frame_width = 8, 12
    pixel_offsets = [1, 2, 4, 8]
    split_dir = tmp_path / "renders" / "test"
    raw_camera_dir = tmp_path / "raw"
    for directory in (split_dir / "rgb", split_dir / "gt-rgb", raw_camera_dir):
        directory.mkdir(parents=True)
    for frame_index, pixel_offset in enumerate(pixel_offsets):
        frame_name = f"{frame_index:06d}.jpg"
        real_image = np.full((frame_height, frame_width, 3), 120, dtype=np.uint8)
        # JPEG (the format ns-render writes) at quality 100 stores a flat-colour image exactly,
        # so each frame's PSNR is the analytic 20 log10(255 / offset).
        cv2.imwrite(str(split_dir / "gt-rgb" / frame_name), real_image, [cv2.IMWRITE_JPEG_QUALITY, 100])
        cv2.imwrite(str(raw_camera_dir / frame_name), real_image, [cv2.IMWRITE_JPEG_QUALITY, 100])
        cv2.imwrite(str(split_dir / "rgb" / frame_name), real_image + np.uint8(pixel_offset),
                    [cv2.IMWRITE_JPEG_QUALITY, 100])
    sheet_path = tmp_path / "compare" / "test_render_vs_gt.jpg"
    psnr_values = compare_module.compare_split(split_dir, raw_camera_dir, sheet_path)
    # Constant offset d on every pixel: MSE = d^2, PSNR = 20 log10(255 / d).
    expected_values = [20.0 * math.log10(255.0 / pixel_offset) for pixel_offset in pixel_offsets]
    assert psnr_values == pytest.approx(expected_values)
    # PSNR falls as the offset grows (offsets double, so each step lowers PSNR by ~6 dB).
    assert all(earlier > later for earlier, later in zip(psnr_values, psnr_values[1:]))
    # Sheet: 3 rows (start, middle, end of 4 frames) of two half-size images side by side.
    sheet_image = cv2.imread(str(sheet_path))
    assert sheet_image.shape[:2] == (3 * (frame_height // 2), 2 * (frame_width // 2))
