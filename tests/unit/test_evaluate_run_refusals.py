"""scripts/13 ``evaluate_run`` must refuse every input whose renders, split record, dataset or GT do not agree.

Each case damages one thing in an otherwise valid fake run (the control test proves the undamaged
run is accepted) and asserts the refusal and its reason. Scoring a run whose renders do not belong
to the recorded split would silently report numbers for the wrong frames.
"""

import json
import shutil

import pytest

from tests.unit.test_eval_semantic_occupancy import (
    EVAL_SETTINGS,
    load_eval_script,
    write_fake_run,
    write_fake_session,
)

TEST_STEM = "000005_1000"
TRAIN_STEM = "000004_900"


def make_valid_run(tmp_path):
    """Valid fake session + run; returns (eval script module, run dir, split.json path, split record)."""
    eval_script = load_eval_script()
    manifest_sha256 = write_fake_session(tmp_path, "may_x", "2024_05", "may_x_smoke")
    run_dir = write_fake_run(tmp_path, "run-may", "may_x_smoke", manifest_sha256)
    split_path = run_dir / "split.json"
    return eval_script, run_dir, split_path, json.loads(split_path.read_text())


def rewrite_split(split_path, split_record):
    """Persist a damaged split record."""
    split_path.write_text(json.dumps(split_record))


def evaluate(eval_script, run_dir, tmp_path, rerender_label=None):
    """Call evaluate_run with the standard settings."""
    return eval_script.evaluate_run(run_dir, rerender_label, EVAL_SETTINGS, None, tmp_path)


def test_control_valid_run_is_accepted(tmp_path):
    """The undamaged run scores, so each refusal below is caused by its one fault."""
    eval_script, run_dir, _, _ = make_valid_run(tmp_path)
    _, report = evaluate(eval_script, run_dir, tmp_path)
    assert report["sweep_count"] == 1


@pytest.mark.parametrize("render_check", [{"lidar/test": {"match": False}}, {}, {"lidar/test": {}},
                                          {"camera/test": {"match": True}}])
def test_render_check_not_matching_or_absent_is_refused(tmp_path, render_check):
    """match False, a missing key, an entry without 'match', or only another key's match: all refused."""
    eval_script, run_dir, split_path, split_record = make_valid_run(tmp_path)
    split_record["render_check"] = render_check
    rewrite_split(split_path, split_record)
    with pytest.raises(ValueError, match="do not match"):
        evaluate(eval_script, run_dir, tmp_path)


def test_wrong_manifest_sha256_is_refused(tmp_path):
    """A split.json hash that is not the dataset manifest's sha256 means a different dataset."""
    eval_script, run_dir, split_path, split_record = make_valid_run(tmp_path)
    split_record["dataset"]["manifest_sha256"] = "0" * 64
    rewrite_split(split_path, split_record)
    with pytest.raises(ValueError, match="manifest_sha256"):
        evaluate(eval_script, run_dir, tmp_path)


def test_manifest_changed_after_the_run_is_refused(tmp_path):
    """One appended byte in the local manifest changes its hash, so it no longer is the run's dataset."""
    eval_script, run_dir, _, _ = make_valid_run(tmp_path)
    manifest_path = tmp_path / "data" / "intermediate" / "may_x_smoke" / "manifest.json"
    manifest_path.write_text(manifest_path.read_text() + " ")
    with pytest.raises(ValueError, match="manifest_sha256"):
        evaluate(eval_script, run_dir, tmp_path)


def test_split_test_frame_that_differs_from_rendered_stem_is_refused(tmp_path):
    """split.json names test frame 000006_1100 but 000005_1000 was rendered: missing + unexpected."""
    eval_script, run_dir, split_path, split_record = make_valid_run(tmp_path)
    test_frame = next(frame for frame in split_record["frames"] if frame["split"] == "test")
    test_frame["file"] = "lidar/ouster/000006_1100.npy"
    rewrite_split(split_path, split_record)
    with pytest.raises(ValueError, match="differ"):
        evaluate(eval_script, run_dir, tmp_path)


def test_extra_split_test_frame_without_render_is_refused(tmp_path):
    """A second test frame in split.json with no render file is refused (missing render)."""
    eval_script, run_dir, split_path, split_record = make_valid_run(tmp_path)
    split_record["frames"].append(
        {"kind": "lidar", "split": "test", "frame_index": 6, "file": "lidar/ouster/000006_1100.npy"})
    rewrite_split(split_path, split_record)
    with pytest.raises(ValueError, match="missing"):
        evaluate(eval_script, run_dir, tmp_path)


def test_train_frame_rendered_into_the_test_folder_is_refused(tmp_path):
    """A render of a TRAIN frame in renders/test/lidar is a leak into the scored set, refused as unexpected."""
    eval_script, run_dir, _, _ = make_valid_run(tmp_path)
    render_dir = run_dir / "renders" / "test" / "lidar"
    shutil.copy(render_dir / f"{TEST_STEM}.npz", render_dir / f"{TRAIN_STEM}.npz")
    with pytest.raises(ValueError, match="unexpected"):
        evaluate(eval_script, run_dir, tmp_path)


def test_train_frame_relabelled_test_without_render_is_refused(tmp_path):
    """Moving the train frame into the test split in split.json (renders unchanged) is refused."""
    eval_script, run_dir, split_path, split_record = make_valid_run(tmp_path)
    for frame in split_record["frames"]:
        frame["split"] = "test"
    rewrite_split(split_path, split_record)
    with pytest.raises(ValueError, match="missing"):
        evaluate(eval_script, run_dir, tmp_path)


def test_missing_split_record_is_refused(tmp_path):
    """No split.json: scripts/09 has not documented the run."""
    eval_script, run_dir, split_path, _ = make_valid_run(tmp_path)
    split_path.unlink()
    with pytest.raises(ValueError, match="split.json"):
        evaluate(eval_script, run_dir, tmp_path)


def test_missing_render_folder_is_refused(tmp_path):
    """A run without renders/test/lidar is refused."""
    eval_script, run_dir, _, _ = make_valid_run(tmp_path)
    shutil.rmtree(run_dir / "renders")
    with pytest.raises(ValueError, match="no lidar renders"):
        evaluate(eval_script, run_dir, tmp_path)


def test_missing_ground_truth_is_refused(tmp_path):
    """Without data/gt/<session>/occupancy.npz nothing can be scored."""
    eval_script, run_dir, _, _ = make_valid_run(tmp_path)
    (tmp_path / "data" / "gt" / "may_x" / "occupancy.npz").unlink()
    with pytest.raises(ValueError, match="GT missing"):
        evaluate(eval_script, run_dir, tmp_path)


def make_rerender(run_dir, job_info_text, selected_lidar_indices):
    """Copy the run's renders into rerenders/job1 with the given job_info.txt and selection.json."""
    rerender_dir = run_dir / "rerenders" / "job1"
    shutil.copytree(run_dir / "renders", rerender_dir / "renders")
    (rerender_dir / "job_info.txt").write_text(job_info_text)
    (rerender_dir / "selection.json").write_text(json.dumps({"lidar": selected_lidar_indices}))


def test_rerender_is_accepted_only_with_its_match_line_and_selection(tmp_path):
    """Control + two faults: no 'render_check: match' line, and a selection naming another frame."""
    eval_script, run_dir, _, _ = make_valid_run(tmp_path)
    make_rerender(run_dir, "render_check: match\n", [5])
    _, report = evaluate(eval_script, run_dir, tmp_path, "job1")
    assert report["rerender_label"] == "job1" and report["sweep_count"] == 1

    (run_dir / "rerenders" / "job1" / "job_info.txt").write_text("render_check: MISMATCH\n")
    with pytest.raises(ValueError, match="render_check: match"):
        evaluate(eval_script, run_dir, tmp_path, "job1")

    # Selection names frame 4 (the train sweep), whose render is absent: refused, not scored.
    make_rerender_selection = run_dir / "rerenders" / "job1" / "selection.json"
    (run_dir / "rerenders" / "job1" / "job_info.txt").write_text("render_check: match\n")
    make_rerender_selection.write_text(json.dumps({"lidar": [4]}))
    with pytest.raises(ValueError):
        evaluate(eval_script, run_dir, tmp_path, "job1")
