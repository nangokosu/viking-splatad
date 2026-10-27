"""Tests for the overfitting-monitor wiring: the sbatch plot step and the split record's eval stretches."""

import json
from pathlib import Path

from tests.unit.test_block_cores import make_core_run
from tests.unit.test_block_split import SBATCH_PATH
from tests.unit.test_run_split import run_script

PLOT_CALL = "11_plot_overfitting_curves.py"


def test_sbatch_plots_after_training_and_gates_end_utc():
    """The plot runs between training and rendering, tolerates failure, and gates end_utc."""
    sbatch_text = SBATCH_PATH.read_text()
    # Step 1: the plot call sits after the training timing and before ns-render.
    plot_position = sbatch_text.index(PLOT_CALL)
    assert sbatch_text.index("train_elapsed_s") < plot_position < sbatch_text.index("\nns-render")
    # Step 2: both outcomes are recorded in job_info.
    assert "overfitting_plot: written" in sbatch_text
    assert "overfitting_plot: FAILED" in sbatch_text
    # Step 3: the gate comes after the split check and before end_utc is written.
    gate_position = sbatch_text.index("grep -q '^overfitting_plot: written$'")
    assert sbatch_text.index("split_render_check: match") < gate_position
    assert gate_position < sbatch_text.index('echo "end_utc:')


def _record_with_config_field(tmp_path: Path, extra_config_line: str) -> dict:
    """Run script 09 on a core run whose config.yml gets one extra line; return split.json."""
    run_dir, dataset_dir, _ = make_core_run(tmp_path, 0.5, 0.5)
    config_path = run_dir / "train" / "synthetic" / "splatad" / "run-c" / "config.yml"
    if extra_config_line:
        config_path.write_text(config_path.read_text() + extra_config_line)
    assert run_script(run_dir, dataset_dir) == 0
    return json.loads((run_dir / "split.json").read_text())


def test_split_record_val_counts_whole_blocks(tmp_path):
    """eval_whole_test_blocks: true -> training evaluated the blocks: val = test + guard."""
    record = _record_with_config_field(tmp_path, "  eval_whole_test_blocks: true\n")
    assert record["selection"]["eval_whole_test_blocks"] is True
    assert record["selection"]["eval_during_training_s"] == record["selection"]["test_blocks_s"]
    for sensor_counts in record["counts"].values():
        assert sensor_counts["val"] == sensor_counts["test"] + sensor_counts["guard"]


def test_split_record_old_run_val_is_cores(tmp_path):
    """No field (old run) -> val = test, eval stretches = cores, membership unchanged."""
    new_record = _record_with_config_field(tmp_path / "new", "  eval_whole_test_blocks: true\n")
    old_record = _record_with_config_field(tmp_path / "old", "")
    assert old_record["selection"]["eval_whole_test_blocks"] is False
    assert old_record["selection"]["eval_during_training_s"] == old_record["selection"]["test_cores_s"]
    for sensor_counts in old_record["counts"].values():
        assert sensor_counts["val"] == sensor_counts["test"]
    assert old_record["membership_sha256"] == new_record["membership_sha256"]
