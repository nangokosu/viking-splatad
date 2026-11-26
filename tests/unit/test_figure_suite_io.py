"""Tests of the run loading, consistency checks, hashing, failure bookkeeping and manifests of the figure suite
(viz/figure_suite_io.py; pure numpy, no PyVista or matplotlib)."""

import json
from pathlib import Path

import numpy as np
import pytest

from viking_splatad.eval.run_context import file_sha256
from viking_splatad.viz import figure_suite_io as suite_io

ARMS = ("stock", "radar_init", "radar_init_loss")
CELL_COUNT = 6
CAMPAIGN_PROFILES = {"stock": "profile_stock", "radar_init": "profile_init", "radar_init_loss": "profile_loss"}
METRIC_KEY = "obstacles/P2@0.5/recall_Hnear/mean"


def tiny_cells(occupancy_values):
    """A minimal cells archive (6 cells) with the keys the checks read.

    Args:
        occupancy_values: (6,) occupancy per cell (model dependent, must be ignored by the checks).

    Returns:
        dict of arrays.
    """
    return {
        "cell_indices": np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [2, 0, 0], [2, 1, 0]]),
        "centres": np.zeros((CELL_COUNT, 3)),
        "occupancy": np.asarray(occupancy_values, dtype=float),
        "threshold": np.array(0.5),
        "p2_thresholds": np.array([0.5, 0.8]),
        "evidence": np.arange(CELL_COUNT),
        "hidden_near": np.ones(CELL_COUNT, dtype=bool),
        "neighbourhood_obstacles": np.array([True, True, True, True, False, False]),
        "truth_obstacles": np.array([True, True, False, False, False, False]),
    }


def test_parse_run_overrides_valid_and_errors():
    """arm=dir parses; a missing '=', an unknown arm and a repeated arm raise ValueError."""
    parsed = suite_io.parse_run_overrides(["stock=runs/a", "radar_init=runs/b=c"], ARMS)
    assert parsed == {"stock": Path("runs/a"), "radar_init": Path("runs/b=c")}
    with pytest.raises(ValueError, match="arm=<dir>"):
        suite_io.parse_run_overrides(["runs/a"], ARMS)
    with pytest.raises(ValueError, match="unknown arm"):
        suite_io.parse_run_overrides(["radar=runs/a"], ARMS)
    with pytest.raises(ValueError, match="twice"):
        suite_io.parse_run_overrides(["stock=runs/a", "stock=runs/b"], ARMS)


def test_choose_drawn_run_median_or_override():
    """Without an override the median repeat is drawn (3rd of 5); an override wins."""
    metric_by_run = {Path(f"run{index}"): value for index, value in enumerate([0.5, 0.1, 0.9, 0.3, 0.7])}
    # Sorted values 0.1 0.3 0.5 0.7 0.9: the median 0.5 belongs to run0.
    assert suite_io.choose_drawn_run(metric_by_run, None) == Path("run0")
    assert suite_io.choose_drawn_run(metric_by_run, Path("elsewhere")) == Path("elsewhere")


@pytest.mark.parametrize("only_value", [0.4, None], ids=["defined metric", "undefined metric"])
def test_choose_drawn_run_single_run_is_drawn_whatever_its_metric(only_value):
    """One run per arm (the 30k presentation set): that run is drawn, even when its metric is undefined."""
    only_run = Path("only_run")
    assert suite_io.choose_drawn_run({only_run: only_value}, None) == only_run


def test_check_runs_consistent_detects_mismatch():
    """A differing hidden_near or truth array raises, naming the key; identical archives pass."""
    reference = tiny_cells(np.zeros(CELL_COUNT))
    identical = tiny_cells(np.ones(CELL_COUNT))  # occupancy is model-dependent and must be ignored
    suite_io.check_runs_consistent({"run_a": reference, "run_b": identical})
    for key in ("hidden_near", "truth_obstacles"):
        changed = tiny_cells(np.zeros(CELL_COUNT))
        changed[key] = ~changed[key]
        with pytest.raises(ValueError, match=key):
            suite_io.check_runs_consistent({"run_a": reference, "run_b": changed})


@pytest.mark.parametrize("key, changed_value", [("threshold", np.array(0.8)),
                                                ("evidence", np.arange(CELL_COUNT)[::-1].copy())])
def test_check_runs_consistent_compares_threshold_and_evidence(key, changed_value):
    """Regression: the archived threshold and the evidence labels are compared across runs too (they were not)."""
    reference = tiny_cells(np.zeros(CELL_COUNT))
    changed = tiny_cells(np.zeros(CELL_COUNT))
    changed[key] = changed_value
    with pytest.raises(ValueError, match=key):
        suite_io.check_runs_consistent({"run_a": reference, "run_b": changed})


def test_check_binding_threshold_checks_every_run():
    """Regression: a run (not only the first) archived at a cut other than p2_thresholds[0] is refused."""
    first = tiny_cells(np.zeros(CELL_COUNT))
    second = tiny_cells(np.zeros(CELL_COUNT))
    assert suite_io.check_binding_threshold({"run_a": first, "run_b": second}) == 0.5
    second["threshold"] = np.array(0.8)
    with pytest.raises(ValueError, match="run_b"):
        suite_io.check_binding_threshold({"run_a": first, "run_b": second})
    with pytest.raises(ValueError, match="no run"):
        suite_io.check_binding_threshold({})


def test_hash_recorder_scopes(tmp_path):
    """The digest is cached across scopes; each scope lists only the files recorded in it."""
    first_file = tmp_path / "a.bin"
    first_file.write_bytes(b"abc")
    recorder = suite_io.HashRecorder()
    recorder.record(first_file)
    assert list(recorder.scope.values()) == [file_sha256(first_file)]
    recorder.begin_scope()
    assert recorder.scope == {}
    assert len(recorder.cache) == 1


def make_selection_inputs(tmp_path):
    """Run folders and cells for a selection: 3 stock repeats, 2 radar_init repeats, no radar_init_loss run.

    Args:
        tmp_path: Pytest temporary folder.

    Returns:
        (runs_by_arm, cells_by_dir, metric_by_dir).
    """
    runs_by_arm = {
        "stock": [tmp_path / f"stock-{index}" for index in range(3)],
        "radar_init": [tmp_path / f"init-{index}" for index in range(2)],
        "radar_init_loss": []}
    all_dirs = [run_dir for run_dirs in runs_by_arm.values() for run_dir in run_dirs]
    cells_by_dir = {run_dir: tiny_cells(np.full(CELL_COUNT, float(position))) for position, run_dir in enumerate(all_dirs)}
    metric_by_dir = {run_dir: float(position) for position, run_dir in enumerate(all_dirs)}
    return runs_by_arm, cells_by_dir, metric_by_dir


def select(tmp_path, overrides, profile_by_run=None, cells_extra=None):
    """Run ``select_arm_runs`` on in-memory runs.

    Args:
        tmp_path: Pytest temporary folder.
        overrides: arm -> override folder.
        profile_by_run: Ledger profile by folder name.
        cells_extra: Extra folder -> cells entries (override folders outside the campaign).

    Returns:
        The ``RunSelection``.
    """
    runs_by_arm, cells_by_dir, metric_by_dir = make_selection_inputs(tmp_path)
    cells_by_dir = {**cells_by_dir, **(cells_extra or {})}
    return suite_io.select_arm_runs(
        ARMS, CAMPAIGN_PROFILES, runs_by_arm, overrides, profile_by_run or {}, METRIC_KEY,
        cells_loader=lambda run_dir: cells_by_dir[run_dir],
        metric_reader=lambda run_dir, metric_key: metric_by_dir[run_dir])


def test_select_arm_runs_median_repeat_and_manifest_info(tmp_path):
    """Without overrides the median repeat is drawn; the manifest entry counts the repeats and records the profile."""
    selection = select(tmp_path, overrides={"radar_init_loss": tmp_path / "init-0"})
    # stock metric values 0, 1, 2 -> median is stock-1.
    assert selection.drawn["stock"].run_dir == tmp_path / "stock-1"
    assert selection.arms_info["stock"]["drawn_by"] == "median repeat"
    assert selection.arms_info["stock"]["campaign_run_count"] == 3
    assert selection.arms_info["stock"]["profile"] == CAMPAIGN_PROFILES["stock"]
    assert selection.binding_threshold == 0.5


def test_select_arm_runs_cross_arm_override_uses_the_overrides_arm_and_ledger_profile(tmp_path):
    """Regression: a run of another arm drawn for an arm carries THAT arm, and its own ledger profile in the manifest."""
    cross_run = tmp_path / "stock-2"  # a stock campaign run, drawn for radar_init
    selection = select(tmp_path, overrides={"radar_init": cross_run, "radar_init_loss": tmp_path / "init-0"},
                       profile_by_run={"stock-2": "profile_stock", "init-0": "profile_init"})
    drawn = selection.drawn["radar_init"]
    assert drawn.run_dir == cross_run and drawn.arm == "radar_init"
    assert selection.drawn["stock"].arm == "stock"
    # The manifest shows the profile of the run actually drawn, not the arm's campaign profile.
    assert selection.arms_info["radar_init"]["profile"] == "profile_stock"
    assert selection.arms_info["radar_init"]["campaign_profile"] == CAMPAIGN_PROFILES["radar_init"]
    assert selection.arms_info["radar_init"]["drawn_by"] == "override"
    # The repeats of the arm stay its own campaign runs.
    assert [run.run_dir for run in selection.repeats["radar_init"]] == [tmp_path / "init-0", tmp_path / "init-1"]


def test_select_arm_runs_override_stands_in_for_an_arm_without_campaign_runs(tmp_path):
    """An arm with no campaign runs is allowed with an override; it has no band and says so; without one it is refused."""
    selection = select(tmp_path, overrides={"radar_init_loss": tmp_path / "init-1"},
                       profile_by_run={"init-1": "profile_init"})
    info = selection.arms_info["radar_init_loss"]
    assert selection.repeats["radar_init_loss"] == []
    assert info["campaign_band"] is False and "no spread band" in info["campaign_band_note"]
    assert selection.drawn["radar_init_loss"].arm == "radar_init_loss"
    with pytest.raises(ValueError, match=r"radar_init_loss.*--run"):
        select(tmp_path, overrides={})


def test_select_arm_runs_checks_threshold_and_evidence_of_every_loaded_run(tmp_path):
    """Regression: an override archived at another threshold, or with other evidence, is refused."""
    for key, changed_value in (("threshold", np.array(0.8)), ("evidence", np.arange(CELL_COUNT)[::-1].copy())):
        outside_run = tmp_path / "outside"
        outside_cells = tiny_cells(np.zeros(CELL_COUNT))
        outside_cells[key] = changed_value
        with pytest.raises(ValueError, match=key):
            select(tmp_path, overrides={"radar_init_loss": outside_run}, cells_extra={outside_run: outside_cells})


def test_recorder_records_written_skipped_and_failed(tmp_path, capsys):
    """attempt records written / skipped (FigureSkipped) / failed; only a failure makes any_failed true."""
    recorder = suite_io.FigureRecorder()

    def skip_figure():
        """Draw function with nothing to draw."""
        raise suite_io.FigureSkipped("nothing to draw")

    def fail_figure():
        """Draw function that raises."""
        raise RuntimeError("exploded")

    assert recorder.attempt("ok", tmp_path / "ok.png", lambda: None) is True
    assert recorder.attempt("skipped", tmp_path / "skipped.png", skip_figure) is False
    assert recorder.figures["skipped"]["status"] == suite_io.STATUS_SKIPPED
    assert recorder.figures["skipped"]["error"] == "nothing to draw"
    assert not recorder.any_failed()
    assert recorder.attempt("failed", tmp_path / "failed.png", fail_figure) is False
    assert recorder.figures["failed"]["status"] == suite_io.STATUS_FAILED and "exploded" in recorder.figures["failed"]["error"]
    assert recorder.any_failed()
    assert recorder.figures["ok"]["status"] == suite_io.STATUS_WRITTEN
    assert "figure failed FAILED" in capsys.readouterr().err


def test_recorder_guard_records_only_failures(tmp_path):
    """guard returns the result and adds no 'written' entry, so a preparation step is not a figure of its own."""
    recorder = suite_io.FigureRecorder()
    assert recorder.guard("prep", tmp_path, lambda: 42) == (True, 42)
    assert recorder.figures == {}
    succeeded, result = recorder.guard("prep", tmp_path, lambda: 1 / 0)
    assert (succeeded, result) == (False, None) and recorder.figures["prep"]["status"] == suite_io.STATUS_FAILED


def test_json_number_list_maps_non_finite_to_none():
    """NaN and infinities become None; finite values stay floats in order."""
    assert suite_io.json_number_list(np.array([0.25, np.nan, np.inf, 1.0])) == [0.25, None, None, 1.0]


def test_build_and_write_manifest_roundtrip(tmp_path):
    """The manifest carries the schema, the thresholds as floats and the reported numbers, and reads back as JSON."""
    manifest = suite_io.build_manifest(
        session="s", scope_label="arms", git={"head_sha": "abc"}, class_name="obstacles", binding_threshold=0.5,
        p2_thresholds=np.array([0.5, 0.8]), representative_metric=METRIC_KEY, arms_info={"stock": {}},
        figures={"f": {"status": "written"}}, hashes={"file": "digest"}, reported_numbers={"n": [1.0, None]})
    manifest_path = suite_io.write_manifest(tmp_path / "out", manifest)
    loaded = json.loads(manifest_path.read_text())
    assert loaded["schema"] == suite_io.MANIFEST_SCHEMA and loaded["p2_thresholds"] == [0.5, 0.8]
    assert loaded["reported_numbers"] == {"n": [1.0, None]} and loaded["input_sha256"] == {"file": "digest"}
