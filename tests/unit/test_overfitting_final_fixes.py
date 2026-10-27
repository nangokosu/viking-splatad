"""Final-review fixes of the overfitting monitor: finite-only plot limits and small source checks."""

import csv
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PLOT_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "11_plot_overfitting_curves.py"
SPLIT_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "09_extract_run_split.py"
TRAIN_TAG = "Train Metrics Dict/psnr"
TEST_TAG = "Eval Images Metrics Dict (all images)/psnr"


def _plot_dependencies_missing() -> bool:
    """True when torch, tensorboard or matplotlib is absent (Mac venv); they exist only in the image."""
    return any(importlib.util.find_spec(module_name) is None
               for module_name in ("torch", "tensorboard", "matplotlib"))


requires_plot_stack = pytest.mark.skipif(
    _plot_dependencies_missing(), reason="needs torch + tensorboard + matplotlib (image / cluster env)")


def _load_plot_module():
    """Import scripts/11_plot_overfitting_curves.py (its name starts with a digit)."""
    spec = importlib.util.spec_from_file_location("plot_overfitting_curves_final", PLOT_SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@requires_plot_stack
def test_panel_y_limits_ignore_non_finite_values():
    """NaN/inf in train or test values give finite limits equal to those of the finite values alone."""
    module = _load_plot_module()
    steps = np.arange(1, 11)
    finite_values = 10.0 + steps.astype(float)
    dirty_values = finite_values.copy()
    dirty_values[2], dirty_values[5] = np.nan, np.inf
    test_values = np.array([12.0, np.inf, 15.0])
    lower, upper = module.panel_y_limits([(steps, dirty_values)], 1, test_values)
    assert math.isfinite(lower) and math.isfinite(upper) and lower < upper
    # Reference: the same call with the bad samples removed by hand.
    keep = np.isfinite(dirty_values)
    expected = module.panel_y_limits([(steps[keep], dirty_values[keep])], 1, np.array([12.0, 15.0]))
    assert (lower, upper) == pytest.approx(expected)


@requires_plot_stack
def test_panel_y_limits_all_non_finite_falls_back_to_finite_limits():
    """With nothing finite left the zero-range path still returns finite, distinct limits."""
    module = _load_plot_module()
    steps = np.arange(1, 4)
    lower, upper = module.panel_y_limits([(steps, np.array([np.nan, np.inf, -np.inf]))], 1)
    assert math.isfinite(lower) and math.isfinite(upper) and lower < upper


@requires_plot_stack
def test_plot_exits_zero_with_non_finite_logged_values(tmp_path):
    """A NaN and an inf train sample plus an inf test value do not stop the plot; CSV keeps them."""
    from torch.utils.tensorboard import SummaryWriter

    train_dir, output_dir = tmp_path / "train", tmp_path / "out"
    train_dir.mkdir()
    (train_dir / "config.yml").write_text(
        "pipeline:\n  resolution_schedule: 30\n  num_downscales: 2\n  steps_per_eval_all_images: 20\n")
    writer = SummaryWriter(log_dir=str(train_dir))
    for step in range(1, 101):
        value = float("nan") if step == 10 else float("inf") if step == 50 else 10.0 + step
        writer.add_scalar(TRAIN_TAG, value, step)
    for step in range(20, 101, 20):
        writer.add_scalar(TEST_TAG, float("inf") if step == 60 else 8.0 + step, step)
    writer.close()
    exit_code = _load_plot_module().main(["--train-dir", str(train_dir), "--output-dir", str(output_dir)])
    assert exit_code == 0
    assert (output_dir / "overfitting_curves.png").stat().st_size > 0
    with open(output_dir / "overfitting_curves.csv", newline="") as csv_handle:
        csv_rows = list(csv.DictReader(csv_handle))
    # The CSV keeps the raw values as Python's float text.
    written_values = {row["value"] for row in csv_rows}
    assert "nan" in written_values and "inf" in written_values


def test_split_script_parses_eval_flag_case_insensitively():
    """Script 09 lower-cases config.yml's eval_whole_test_blocks before comparing with 'true'."""
    source_text = SPLIT_SCRIPT_PATH.read_text()
    assert '.lower() == "true"' in source_text


def test_new_split_script_lines_fit_the_line_limit():
    """The lines this change added to script 09 stay within 100 characters."""
    source_lines = SPLIT_SCRIPT_PATH.read_text().splitlines()
    touched = [line for line in source_lines
               if "eval_whole_test_blocks" in line or "counts_of_sensor" in line
               or "eval_during_training_s" in line]
    assert touched and all(len(line) <= 100 for line in touched)


def test_plot_script_names_its_styling_and_fallback_constants():
    """No bare 20.0 fallback, dpi=120 or fontsize literals remain in script 11's logic."""
    source_text = PLOT_SCRIPT_PATH.read_text()
    assert "else 20.0" not in source_text
    assert "dpi=120" not in source_text
    assert "fontsize=7" not in source_text and "fontsize=10" not in source_text
