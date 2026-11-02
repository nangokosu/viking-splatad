"""Tests that scripts/13 launches the PyVista pictures after the report and survives their failure.

Uses the synthetic session and run of test_eval_semantic_occupancy.py and a fake subprocess runner,
so neither uv nor PyVista is needed.
"""

import json
import subprocess
from pathlib import Path

import yaml

from tests.unit.test_eval_semantic_occupancy import (
    EVAL_SETTINGS,
    load_eval_script,
    write_fake_run,
    write_fake_session,
)


class FakeRunner:
    """Stand-in for subprocess.run that records its call and answers with a chosen outcome."""

    def __init__(self, returncode=0, stderr="", error=None):
        self.returncode, self.stderr, self.error = returncode, stderr, error
        self.calls = []
        self.report_existed_at_call = None

    def __call__(self, command, cwd, **keyword_arguments):
        self.calls.append((command, cwd))
        # The metrics must already be on disk when the pictures start.
        self.report_existed_at_call = any(Path(cwd).glob("runs/*/eval/semantic_occupancy.json"))
        if self.error is not None:
            raise self.error
        return subprocess.CompletedProcess(command, self.returncode, stdout="", stderr=self.stderr)


def prepare_session(tmp_path, monkeypatch):
    """Fake session, run and eval config in tmp_path; the script then treats tmp_path as the repo root.

    Returns:
        (script module, run folder, eval config path).
    """
    eval_script = load_eval_script()
    manifest_sha256 = write_fake_session(tmp_path, "may_x", "2024_05", "may_x_smoke")
    run_dir = write_fake_run(tmp_path, "run-may", "may_x_smoke", manifest_sha256)
    eval_config_path = tmp_path / "eval.yaml"
    eval_config_path.write_text(yaml.safe_dump({"eval": EVAL_SETTINGS}))
    monkeypatch.setattr(eval_script, "find_project_root", lambda: tmp_path)
    return eval_script, run_dir, eval_config_path


def read_report(run_dir):
    """The run's semantic_occupancy.json."""
    return json.loads((run_dir / "eval" / "semantic_occupancy.json").read_text())


def test_viz_runs_after_the_report_from_the_repo_root(tmp_path, monkeypatch):
    """One launch per run, cwd = repo root, report already written, outcome recorded in the report."""
    eval_script, run_dir, eval_config_path = prepare_session(tmp_path, monkeypatch)
    runner = FakeRunner()
    assert eval_script.main(["--eval-config", str(eval_config_path), str(run_dir)], run_command=runner) == 0
    assert len(runner.calls) == 1 and runner.report_existed_at_call
    command, working_directory = runner.calls[0]
    assert working_directory == tmp_path
    assert str(run_dir.resolve()) in command
    # What the evaluation used is what the viewer is told: output folder, config and threshold.
    assert command[command.index("--output-dir") + 1] == str((run_dir / "eval" / "viz").resolve())
    assert command[command.index("--eval-config") + 1] == str(eval_config_path.resolve())
    assert float(command[command.index("--accumulation-threshold") + 1]) == EVAL_SETTINGS["render_accumulation_threshold"]
    assert read_report(run_dir)["viz"].startswith("written: <repo>")
    # No absolute path (it would carry the home directory) reaches the recorded report.
    assert str(tmp_path) not in (run_dir / "eval" / "semantic_occupancy.json").read_text()


def test_no_viz_flag_skips_the_launch(tmp_path, monkeypatch):
    """--no-viz: no subprocess, and the report says so."""
    eval_script, run_dir, eval_config_path = prepare_session(tmp_path, monkeypatch)
    runner = FakeRunner()
    assert eval_script.main(["--eval-config", str(eval_config_path), "--no-viz", str(run_dir)], run_command=runner) == 0
    assert runner.calls == []
    assert read_report(run_dir)["viz"] == "skipped: --no-viz"


def test_viz_failure_warns_and_leaves_the_metrics_and_exit_code_alone(tmp_path, monkeypatch, capsys):
    """A failing viewer and a missing launcher both: exit 0, 'failed: ...' recorded, fix command printed,
    metrics identical to a run without viz."""
    eval_script, run_dir, eval_config_path = prepare_session(tmp_path, monkeypatch)
    assert eval_script.main(["--eval-config", str(eval_config_path), "--no-viz", str(run_dir)]) == 0
    baseline_point_sets = read_report(run_dir)["point_sets"]
    for runner in (FakeRunner(returncode=2, stderr="ModuleNotFoundError: pyvista\n"),
                   FakeRunner(error=FileNotFoundError("uv"))):
        assert eval_script.main(["--eval-config", str(eval_config_path), str(run_dir)], run_command=runner) == 0
        report = read_report(run_dir)
        assert report["viz"].startswith("failed")
        assert report["point_sets"] == baseline_point_sets
        assert eval_script.VIZ_ENVIRONMENT_FIX_COMMAND in capsys.readouterr().err
    assert "ModuleNotFoundError: pyvista" in eval_script.launch_pose_viz(
        run_dir, None, run_dir / "eval", eval_config_path, 0.5, tmp_path,
        FakeRunner(returncode=2, stderr="ModuleNotFoundError: pyvista\n"))


def test_viz_timeout_is_a_reported_failure(tmp_path, monkeypatch, capsys):
    """A viewer that never finishes ends as 'failed: no result within ...', not a hang or crash."""
    eval_script, run_dir, eval_config_path = prepare_session(tmp_path, monkeypatch)
    timed_out = FakeRunner(error=subprocess.TimeoutExpired("uv", eval_script.VIZ_TIMEOUT_SECONDS))
    assert eval_script.main(["--eval-config", str(eval_config_path), str(run_dir)], run_command=timed_out) == 0
    assert read_report(run_dir)["viz"].startswith("failed: no result within")
    assert eval_script.VIZ_ENVIRONMENT_FIX_COMMAND in capsys.readouterr().err


def test_undecodable_viewer_output_is_a_reported_failure_not_a_refusal(tmp_path, monkeypatch):
    """A UnicodeDecodeError (a ValueError) from the subprocess must not make main() exit 1 as 'refused'."""
    eval_script, run_dir, eval_config_path = prepare_session(tmp_path, monkeypatch)
    undecodable = FakeRunner(error=UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte"))
    assert eval_script.main(["--eval-config", str(eval_config_path), str(run_dir)], run_command=undecodable) == 0
    assert read_report(run_dir)["viz"].startswith("failed")


def test_recorded_messages_replace_the_repo_root(tmp_path):
    """The repo root in a message becomes <repo>; text without it is unchanged."""
    eval_script = load_eval_script()
    assert eval_script.repo_relative(f"boom at {tmp_path}/x.py", tmp_path) == "boom at <repo>/x.py"
    assert eval_script.repo_relative("plain", tmp_path) == "plain"
