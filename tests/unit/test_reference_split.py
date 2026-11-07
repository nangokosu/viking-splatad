"""Tests of the committed reference split: record building, comparison, the check script and the gate.

Not a registered transform (no T-number): the record only restates the split of
``viking_splatad.dataset.run_split`` (T58, T59, T71, T72, which have their own covering tests).
Expected values are computed here from first principles (the LINSPACE rule and the nearest-frame rule),
never taken from the code under test.
"""

import dataclasses
import importlib.util
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import List

import numpy as np
import pytest

from tests.unit.test_run_split import write_synthetic_dataset
from tests.unit.test_slurm_submit import load_submit_module
from viking_splatad.config import load_session_config
from viking_splatad.dataset.reference_split import (
    REFERENCE_RECORD_SCHEMA,
    REFERENCE_SETTINGS_KEYS,
    build_reference_record,
    compare_with_reference,
)
from viking_splatad.dataset.run_split import (
    GUARD_SPLIT_NAME,
    SHARED_INTERLEAVED_SPLIT_METHOD,
    TEST_SPLIT_NAME,
    TRAIN_SPLIT_NAME,
    SplitFrame,
    build_run_split_with_cores,
)
from viking_splatad.splatad.viking_conversions import open_dataset_view

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECK_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "16_check_reference_split.py"
COMMITTED_RECORD_PATH = PROJECT_ROOT / "configs" / "reference_splits" / "june_tall_grass.json"
JUNE_SESSION_CONFIG_PATH = PROJECT_ROOT / "configs" / "sessions" / "june_tall_grass.yaml"

# Synthetic recording: lidar sweeps every 0.1 s, camera frames 0.03 s after each sweep (so the camera
# frame nearest sweep k is camera frame k).
SWEEP_COUNT = 100
SWEEP_PERIOD_S = 0.1
CAMERA_OFFSET_S = 0.03
TRAIN_FRACTION = 0.9
SENSOR_KINDS = (("cam", "camera"), ("lidar", "lidar"))
SETTINGS = {
    "session": "synthetic", "dataset_start_fraction": 0.0, "dataset_end_fraction": 1.0,
    "train_fraction": TRAIN_FRACTION, "test_split_method": SHARED_INTERLEAVED_SPLIT_METHOD,
    "test_render_fraction": 1.0, "chunk_files": ["a.mcap", "b.mcap"],
}
# A SHA-256 digest is 256 bits = 64 hex characters.
SHA256_HEX_LENGTH = 256 // 4
SOURCE = {"run_id": "run_x", "git_sha": "abc123"}


def load_check_module():
    """Import scripts/16_check_reference_split.py as a module (its name starts with a digit)."""
    module_spec = importlib.util.spec_from_file_location("check_reference_split", CHECK_SCRIPT_PATH)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def build_synthetic_frames(tmp_path: Path, render_fraction: float = 1.0) -> List[SplitFrame]:
    """Real split of the synthetic recording: 100 lidar sweeps and 100 camera frames.

    Args:
        tmp_path: Directory to write the synthetic dataset into.
        render_fraction: T72 fraction of the held-out instants that stay test (the rest are guard).

    Returns:
        All frames with their split.
    """
    sweep_times = np.arange(SWEEP_COUNT) * SWEEP_PERIOD_S
    write_synthetic_dataset(tmp_path / "dataset", {("cam", "camera"): sweep_times + CAMERA_OFFSET_S,
                                                   ("lidar", "lidar"): sweep_times}, TRAIN_FRACTION)
    view = open_dataset_view(tmp_path / "dataset")
    frames, _, _, _ = build_run_split_with_cores(
        view, 0.0, 1.0, TRAIN_FRACTION, test_split_method=SHARED_INTERLEAVED_SPLIT_METHOD,
        test_render_fraction=render_fraction)
    return frames


def lidar_stamps_of(frames: List[SplitFrame], split_name: str) -> List[int]:
    """Sorted lidar timestamps of one split."""
    return sorted(frame.timestamp_ns for frame in frames if frame.kind == "lidar" and frame.split == split_name)


def test_synthetic_split_matches_first_principles_and_record_lists_only_held_out(tmp_path):
    """100 sweeps at f = 0.9: LINSPACE trains ceil(90) = 90 at floor(k*99/89), so 10 are held out.

    The camera holds out the frame nearest each held-out sweep, i.e. the same position (offset 0.03 s
    is below half the period). The record lists exactly those stamps and counts 90 train.
    """
    frames = build_synthetic_frames(tmp_path)
    train_count = math.ceil(SWEEP_COUNT * TRAIN_FRACTION)
    train_positions = {(step * (SWEEP_COUNT - 1)) // (train_count - 1) for step in range(train_count)}
    held_out_positions = sorted(set(range(SWEEP_COUNT)) - train_positions)
    assert len(held_out_positions) == SWEEP_COUNT - train_count
    record = build_reference_record(frames, SETTINGS, SOURCE)
    sweep_stamp = lambda position: round(position * SWEEP_PERIOD_S * 1e9)  # noqa: E731 (known-answer helper)
    camera_stamp = lambda position: round((position * SWEEP_PERIOD_S + CAMERA_OFFSET_S) * 1e9)  # noqa: E731
    assert record["sensors"]["lidar"]["test_timestamps_ns"] == [sweep_stamp(p) for p in held_out_positions]
    assert record["sensors"]["cam"]["test_timestamps_ns"] == [camera_stamp(p) for p in held_out_positions]
    assert record["sensors"]["lidar"]["guard_timestamps_ns"] == []
    assert record["sensors"]["lidar"]["counts"] == {
        "frames_in_window": SWEEP_COUNT, TRAIN_SPLIT_NAME: train_count,
        TEST_SPLIT_NAME: SWEEP_COUNT - train_count, GUARD_SPLIT_NAME: 0}
    assert record["schema"] == REFERENCE_RECORD_SCHEMA


def test_identical_split_has_no_differences_and_survives_a_json_roundtrip(tmp_path):
    """Rebuilt frames equal the record, also after json.dumps/loads (what the committed file goes through)."""
    frames = build_synthetic_frames(tmp_path)
    record = build_reference_record(frames, SETTINGS, SOURCE)
    assert compare_with_reference(frames, record) == []
    assert compare_with_reference(frames, json.loads(json.dumps(record))) == []


def test_guard_frames_are_recorded_and_compared(tmp_path):
    """Render fraction 0.5 of the 10 held-out instants: 5 test and 5 guard per sensor, all recorded."""
    frames = build_synthetic_frames(tmp_path, render_fraction=0.5)
    record = build_reference_record(frames, {**SETTINGS, "test_render_fraction": 0.5}, SOURCE)
    held_out_count = SWEEP_COUNT - math.ceil(SWEEP_COUNT * TRAIN_FRACTION)
    assert len(record["sensors"]["lidar"]["test_timestamps_ns"]) == held_out_count // 2
    assert len(record["sensors"]["lidar"]["guard_timestamps_ns"]) == held_out_count - held_out_count // 2
    assert compare_with_reference(frames, record) == []
    # A guard frame that became test is reported on both sides.
    guard_stamp = record["sensors"]["lidar"]["guard_timestamps_ns"][0]
    flipped_frames = [dataclasses.replace(frame, split=TEST_SPLIT_NAME)
                      if frame.kind == "lidar" and frame.timestamp_ns == guard_stamp else frame
                      for frame in frames]
    differences = compare_with_reference(flipped_frames, record)
    assert any("unexpected test frames" in line and str(guard_stamp) in line for line in differences)
    assert any("guard frames of the record not guard" in line and str(guard_stamp) in line for line in differences)


def test_a_shifted_timestamp_is_detected(tmp_path):
    """A held-out sweep stamped 1 ns later: the record's stamp is absent, the new stamp unexpected."""
    frames = build_synthetic_frames(tmp_path)
    record = build_reference_record(frames, SETTINGS, SOURCE)
    shifted_stamp = lidar_stamps_of(frames, TEST_SPLIT_NAME)[0]
    shifted_frames = [dataclasses.replace(frame, timestamp_ns=frame.timestamp_ns + 1)
                      if frame.kind == "lidar" and frame.timestamp_ns == shifted_stamp else frame
                      for frame in frames]
    differences = compare_with_reference(shifted_frames, record)
    assert any("absent from the dataset" in line and str(shifted_stamp) in line for line in differences)
    assert any("unexpected test frames" in line and str(shifted_stamp + 1) in line for line in differences)


def test_a_missing_frame_is_detected(tmp_path):
    """A held-out sweep missing from the dataset: window and test counts drop by 1, the stamp is absent."""
    frames = build_synthetic_frames(tmp_path)
    record = build_reference_record(frames, SETTINGS, SOURCE)
    missing_stamp = lidar_stamps_of(frames, TEST_SPLIT_NAME)[0]
    reduced_frames = [frame for frame in frames if not (frame.kind == "lidar" and frame.timestamp_ns == missing_stamp)]
    differences = compare_with_reference(reduced_frames, record)
    held_out_count = SWEEP_COUNT - math.ceil(SWEEP_COUNT * TRAIN_FRACTION)
    assert f"lidar: frames_in_window count {SWEEP_COUNT - 1}, record {SWEEP_COUNT}" in differences
    assert f"lidar: test count {held_out_count - 1}, record {held_out_count}" in differences
    assert any("absent from the dataset" in line and str(missing_stamp) in line for line in differences)
    # The camera is untouched.
    assert not any(line.startswith("cam:") for line in differences)


def test_an_extra_test_frame_is_detected(tmp_path):
    """One training sweep moved to test: test +1, train -1, and the stamp is reported unexpected."""
    frames = build_synthetic_frames(tmp_path)
    record = build_reference_record(frames, SETTINGS, SOURCE)
    moved_stamp = lidar_stamps_of(frames, TRAIN_SPLIT_NAME)[0]
    moved_frames = [dataclasses.replace(frame, split=TEST_SPLIT_NAME)
                    if frame.kind == "lidar" and frame.timestamp_ns == moved_stamp else frame
                    for frame in frames]
    differences = compare_with_reference(moved_frames, record)
    held_out_count = SWEEP_COUNT - math.ceil(SWEEP_COUNT * TRAIN_FRACTION)
    train_count = SWEEP_COUNT - held_out_count
    assert f"lidar: test count {held_out_count + 1}, record {held_out_count}" in differences
    assert f"lidar: train count {train_count - 1}, record {train_count}" in differences
    assert any("unexpected test frames" in line and str(moved_stamp) in line for line in differences)


def test_a_held_out_frame_trained_on_is_detected(tmp_path):
    """A frame the record holds out but the rebuilt split trains on is reported as not test."""
    frames = build_synthetic_frames(tmp_path)
    record = build_reference_record(frames, SETTINGS, SOURCE)
    held_out_stamp = lidar_stamps_of(frames, TEST_SPLIT_NAME)[0]
    trained_frames = [dataclasses.replace(frame, split=TRAIN_SPLIT_NAME)
                      if frame.kind == "lidar" and frame.timestamp_ns == held_out_stamp else frame
                      for frame in frames]
    differences = compare_with_reference(trained_frames, record)
    assert any("test frames of the record not test in the rebuilt split" in line and str(held_out_stamp) in line
               for line in differences)


def test_sensor_missing_on_either_side_and_wrong_schema_are_reported(tmp_path):
    """A sensor only in the record, only in the rebuilt split, and an unknown schema string."""
    frames = build_synthetic_frames(tmp_path)
    record = build_reference_record(frames, SETTINGS, SOURCE)
    camera_free_frames = [frame for frame in frames if frame.kind != "camera"]
    assert "cam: in the record but not in the rebuilt split" in compare_with_reference(camera_free_frames, record)
    camera_free_record = build_reference_record(camera_free_frames, SETTINGS, SOURCE)
    assert "cam: in the rebuilt split but not in the record" in compare_with_reference(frames, camera_free_record)
    assert compare_with_reference(frames, {**record, "schema": "other/1"})[0].startswith("record schema")


def test_frame_index_and_file_name_changes_alone_are_not_differences(tmp_path):
    """A re-extraction that renumbers and renames files but keeps the stamps gives the same split."""
    frames = build_synthetic_frames(tmp_path)
    record = build_reference_record(frames, SETTINGS, SOURCE)
    renumbered_frames = [dataclasses.replace(frame, frame_index=frame.frame_index + 7,
                                             file=f"renamed/{frame.timestamp_ns}.bin") for frame in frames]
    assert compare_with_reference(renumbered_frames, record) == []
    # The informational digest does change (it hashes file names), which is why it is not compared.
    assert build_reference_record(renumbered_frames, SETTINGS, SOURCE)["split_digest"] != record["split_digest"]


def test_incomplete_header_is_refused(tmp_path):
    """A record must say which settings and which run produced it."""
    frames = build_synthetic_frames(tmp_path)
    with pytest.raises(ValueError):
        build_reference_record(frames, {key: SETTINGS[key] for key in REFERENCE_SETTINGS_KEYS[1:]}, SOURCE)
    with pytest.raises(ValueError):
        build_reference_record(frames, SETTINGS, {"run_id": "only"})


def test_committed_june_record_is_well_formed_and_matches_the_session_yaml():
    """Schema, counts equal to the lists, stamps sorted and unique, settings equal to the session YAML."""
    record = json.loads(COMMITTED_RECORD_PATH.read_text())
    assert record["schema"] == REFERENCE_RECORD_SCHEMA
    assert set(record["settings"]) == set(REFERENCE_SETTINGS_KEYS)
    assert record["source"]["run_id"] and record["source"]["git_sha"]
    assert len(record["split_digest"]) == SHA256_HEX_LENGTH
    for sensor_name, sensor_record in record["sensors"].items():
        counts = sensor_record["counts"]
        test_stamps, guard_stamps = sensor_record["test_timestamps_ns"], sensor_record["guard_timestamps_ns"]
        assert counts[TEST_SPLIT_NAME] == len(test_stamps), sensor_name
        assert counts[GUARD_SPLIT_NAME] == len(guard_stamps), sensor_name
        # Train is the windowed rest (train fraction < 1, so the splits are disjoint).
        assert counts[TRAIN_SPLIT_NAME] == counts["frames_in_window"] - len(test_stamps) - len(guard_stamps)
        assert test_stamps == sorted(set(test_stamps)) and guard_stamps == sorted(set(guard_stamps))
        assert not set(test_stamps) & set(guard_stamps)
        # Held out is ~1 - f of the window (LINSPACE rounds the train count up).
        assert counts[TEST_SPLIT_NAME] + counts[GUARD_SPLIT_NAME] == (
            counts["frames_in_window"] - math.ceil(counts["frames_in_window"] * record["settings"]["train_fraction"]))
    session_config = load_session_config(JUNE_SESSION_CONFIG_PATH)
    split_section = session_config["split"]
    assert record["settings"]["session"] == session_config["session"]["name"]
    assert record["settings"]["train_fraction"] == split_section["train_fraction"]
    assert record["settings"]["test_split_method"] == split_section["method"]
    assert record["settings"]["test_render_fraction"] == split_section["test_render_fraction"]
    assert record["settings"]["chunk_files"] == session_config["window"]["chunk_files"]


def prepare_check_environment(tmp_path: Path):
    """A synthetic dataset, a session YAML copy and the record built from them, in tmp_path.

    The synthetic dataset is sized for the June YAML's split (train 0.9, shared_interleaved, render
    fraction from the YAML), so the YAML copy is the real one.

    Args:
        tmp_path: Working directory.

    Returns:
        (check module, session name, dataset dir, session config dir, reference dir, frames).
    """
    check_module = load_check_module()
    session_config_directory = tmp_path / "sessions"
    session_config_directory.mkdir()
    shutil.copy(JUNE_SESSION_CONFIG_PATH, session_config_directory / "june_tall_grass.yaml")
    build_synthetic_frames(tmp_path)
    # The dataset must have been built with the YAML's own fraction and method: rebuild with them.
    settings = check_module.session_split_settings("june_tall_grass", session_config_directory)
    dataset_directory = tmp_path / "dataset"
    frames = check_module.rebuild_frames(dataset_directory, settings)
    return check_module, "june_tall_grass", dataset_directory, session_config_directory, tmp_path / "reference", frames, settings


def test_check_script_exit_codes(tmp_path, capsys):
    """No record -> 2; matching record -> 0; a record with one held-out stamp changed -> 1 with the stamp printed."""
    check_module, session, dataset_dir, session_dir, reference_dir, frames, settings = prepare_check_environment(tmp_path)
    assert check_module.check_session(session, dataset_dir, reference_dir, session_dir) == check_module.EXIT_NO_REFERENCE
    reference_dir.mkdir()
    record = build_reference_record(frames, settings, SOURCE)
    record_path = reference_dir / f"{session}.json"
    record_path.write_text(json.dumps(record))
    assert check_module.check_session(session, dataset_dir, reference_dir, session_dir) == check_module.EXIT_IDENTICAL
    # Change one recorded test stamp of the lidar: the check must fail and name it.
    changed_stamp = record["sensors"]["lidar"]["test_timestamps_ns"][0]
    record["sensors"]["lidar"]["test_timestamps_ns"][0] = changed_stamp + 1
    record_path.write_text(json.dumps(record))
    capsys.readouterr()
    assert check_module.check_session(session, dataset_dir, reference_dir, session_dir) == check_module.EXIT_DIFFERENT
    assert str(changed_stamp + 1) in capsys.readouterr().out


def test_check_script_flags_different_chunk_files(tmp_path, capsys):
    """A record made from other bag chunks is a difference even when every frame stamp agrees."""
    check_module, session, dataset_dir, session_dir, reference_dir, frames, settings = prepare_check_environment(tmp_path)
    reference_dir.mkdir()
    other_chunks_settings = {**settings, "chunk_files": ["tall_grass__ros2_01.mcap"]}
    (reference_dir / f"{session}.json").write_text(json.dumps(build_reference_record(frames, other_chunks_settings, SOURCE)))
    assert check_module.check_session(session, dataset_dir, reference_dir, session_dir) == check_module.EXIT_DIFFERENT
    assert "chunk_files" in capsys.readouterr().out


def fake_gate_commands(issued_commands: List[List[str]], reference_exit: int):
    """run_command stand-in: every gate command passes except the reference check, which exits ``reference_exit``."""
    def fake_run_command(command, capture_output=True, input_text=None):
        """Record the command; git status is clean."""
        issued_commands.append(list(command))
        is_reference_check = any("16_check_reference_split.py" in part for part in command)
        return subprocess.CompletedProcess(command, reference_exit if is_reference_check else 0, stdout="", stderr="")
    return fake_run_command


def test_gate_runs_the_reference_check_for_a_session_with_a_record(monkeypatch):
    """june_tall_grass has a committed record: the gate runs scripts/16 on its dataset and fails when it exits 1."""
    submit_module = load_submit_module()
    issued_commands: List[List[str]] = []
    monkeypatch.setattr(submit_module, "run_command", fake_gate_commands(issued_commands, 0))
    submit_module.run_gates("june_tall_grass")
    reference_commands = [command for command in issued_commands
                          if any("16_check_reference_split.py" in part for part in command)]
    assert len(reference_commands) == 1 and "june_tall_grass" in reference_commands[0]
    monkeypatch.setattr(submit_module, "run_command", fake_gate_commands([], 1))
    with pytest.raises(submit_module.GateFailure, match="reference split"):
        submit_module.run_gates("june_tall_grass")


def test_gate_skips_the_reference_check_for_a_session_without_a_record(monkeypatch):
    """A session with no record (e.g. the may_chunks0_1_smoke dataset) is gated by the old three gates only."""
    submit_module = load_submit_module()
    issued_commands: List[List[str]] = []
    # may_short_grass gained a record (2026-11-06), so the example is a dataset that has none.
    session_without_record = "may_chunks0_1_smoke"
    assert not (submit_module.REFERENCE_SPLIT_DIRECTORY / f"{session_without_record}.json").exists()
    monkeypatch.setattr(submit_module, "run_command", fake_gate_commands(issued_commands, 1))
    submit_module.run_gates(session_without_record)
    assert not any("16_check_reference_split.py" in part for command in issued_commands for part in command)
