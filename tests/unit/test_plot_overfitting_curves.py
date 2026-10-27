"""Tests for scripts/11_plot_overfitting_curves.py and the moved ``read_config_scalar``."""

import csv
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from viking_splatad.io.nerfstudio_config import read_config_scalar

PLOT_SCRIPT_PATH = Path(__file__).resolve().parent.parent.parent / "scripts" / "11_plot_overfitting_curves.py"
TRAIN_TAG = "Train Metrics Dict/psnr"
TEST_TAG = "Eval Images Metrics Dict (all images)/psnr"
LOSS_TAG = "Train Loss"


def _plot_dependencies_missing() -> bool:
    """True when torch, tensorboard or matplotlib is absent (Mac venv); they exist only in the image."""
    return any(importlib.util.find_spec(module_name) is None
               for module_name in ("torch", "tensorboard", "matplotlib"))


requires_plot_stack = pytest.mark.skipif(
    _plot_dependencies_missing(), reason="needs torch + tensorboard + matplotlib (image / cluster env)")


def test_read_config_scalar_moved(tmp_path):
    """The moved reader returns the raw scalar, None for YAML null and for an absent field."""
    config_path = tmp_path / "config.yml"
    config_path.write_text("pipeline:\n  num_downscales: 2\n  test_block_count: null\n")
    assert read_config_scalar(config_path, "num_downscales") == "2"
    assert read_config_scalar(config_path, "test_block_count") is None
    assert read_config_scalar(config_path, "absent_field") is None


def _write_train_dir(train_dir: Path) -> None:
    """Create ``train_dir`` with the minimal config.yml the script reads.

    Args:
        train_dir: Directory to create.
    """
    train_dir.mkdir(parents=True)
    (train_dir / "config.yml").write_text(
        "pipeline:\n  resolution_schedule: 30\n  num_downscales: 2\n"
        "  steps_per_eval_all_images: 20\n")


def _write_events(train_dir: Path, scalars) -> None:
    """Write one TensorBoard events file holding the given scalars.

    Args:
        train_dir: Directory for the events file.
        scalars: Iterable of (tag, step, value).
    """
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(log_dir=str(train_dir))
    for tag, step, value in scalars:
        writer.add_scalar(tag, value, step)
    writer.close()


def _load_plot_module():
    """Import scripts/11_plot_overfitting_curves.py (its name starts with a digit)."""
    spec = importlib.util.spec_from_file_location("plot_overfitting_curves", PLOT_SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_csv(csv_path: Path):
    """Read the curves CSV into a list of dict rows."""
    with open(csv_path, newline="") as csv_handle:
        return list(csv.DictReader(csv_handle))


@requires_plot_stack
def test_plot_writes_png_and_csv(tmp_path):
    """5 test points and 100 train points give a PNG and a CSV with those row counts."""
    train_dir, output_dir = tmp_path / "train", tmp_path / "out"
    _write_train_dir(train_dir)
    train_steps = list(range(1, 101))
    test_steps = list(range(20, 101, 20))
    _write_events(train_dir, [(TRAIN_TAG, step, 10.0 + step) for step in train_steps]
                  + [(TEST_TAG, step, 8.0 + step) for step in test_steps])
    exit_code = _load_plot_module().main(["--train-dir", str(train_dir), "--output-dir", str(output_dir)])
    assert exit_code == 0
    png_path = output_dir / "overfitting_curves.png"
    assert png_path.is_file() and png_path.stat().st_size > 0
    csv_rows = _read_csv(output_dir / "overfitting_curves.csv")
    assert sum(row["metric"] == "psnr" and row["split"] == "test" for row in csv_rows) == len(test_steps)
    assert sum(row["metric"] == "psnr" and row["split"] == "train" for row in csv_rows) == len(train_steps)


@requires_plot_stack
def test_plot_fails_without_test_points(tmp_path, capsys):
    """Train-only logs exit 1 with 'no test points' and write nothing."""
    train_dir, output_dir = tmp_path / "train", tmp_path / "out"
    _write_train_dir(train_dir)
    _write_events(train_dir, [(TRAIN_TAG, step, float(step)) for step in range(1, 11)])
    exit_code = _load_plot_module().main(["--train-dir", str(train_dir), "--output-dir", str(output_dir)])
    assert exit_code == 1
    assert "no test points" in capsys.readouterr().err
    assert not output_dir.exists() or not list(output_dir.iterdir())


@requires_plot_stack
def test_plot_merges_two_event_files(tmp_path):
    """Test points split over two events files in one directory are merged by step."""
    train_dir, output_dir = tmp_path / "train", tmp_path / "out"
    _write_train_dir(train_dir)
    _write_events(train_dir, [(TRAIN_TAG, step, float(step)) for step in range(1, 91)]
                  + [(TEST_TAG, 20, 1.0), (TEST_TAG, 40, 2.0)])
    _write_events(train_dir, [(TEST_TAG, 60, 3.0), (TEST_TAG, 80, 4.0)])
    exit_code = _load_plot_module().main(["--train-dir", str(train_dir), "--output-dir", str(output_dir)])
    assert exit_code == 0
    test_steps = [int(row["step"]) for row in _read_csv(output_dir / "overfitting_curves.csv")
                  if row["metric"] == "psnr" and row["split"] == "test"]
    assert test_steps == [20, 40, 60, 80]


@requires_plot_stack
def test_all_panels_share_run_x_range_and_skip_start_up_spike(tmp_path):
    """Every drawn axes spans 0..max step; a step-0 spike does not set the y-limits; test-only titles say so."""
    module = _load_plot_module()
    train_steps = np.arange(1, 101)
    train_values = np.where(train_steps < 5, 1000.0, 10.0 + 0.01 * train_steps)
    series = {
        TRAIN_TAG: (train_steps, train_values),
        TEST_TAG: (np.array([20, 40]), np.array([9.0, 9.5])),
        "Eval Images Metrics Dict (all images)/ssim": (np.array([40]), np.array([0.6])),
    }
    panels = module.pair_metric_tags(series.keys())
    figure = module.draw_figure("run", series, panels, 20, 30)
    drawn_axes = [axes for axes in figure.axes if axes.axison]
    assert {tuple(axes.get_xlim()) for axes in drawn_axes} == {(0, 100)}
    psnr_axes = next(axes for axes in drawn_axes if axes.get_title() == "psnr")
    assert psnr_axes.get_ylim()[1] < 100  # warm-up at step 30 (train extends past it) drops the 1000 spike
    assert "ssim (no train value logged)" in {axes.get_title() for axes in drawn_axes}


@requires_plot_stack
def test_panel_y_limits_pure():
    """Limits cover the values at steps >= warm-up and all test values, plus a 5% margin; constant data is widened."""
    module = _load_plot_module()
    steps, values = np.array([0, 10, 20]), np.array([100.0, 2.0, 4.0])
    lower, upper = module.panel_y_limits([(steps, values)], 10, np.array([1.0]))
    # kept values: 2, 4 and the test 1 -> range [1, 4], margin 0.05 * 3
    assert (lower, upper) == pytest.approx((1.0 - 0.15, 4.0 + 0.15))
    flat_lower, flat_upper = module.panel_y_limits([(steps, np.full(3, 7.0))], 0)
    assert flat_lower < 7.0 < flat_upper
