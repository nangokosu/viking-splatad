"""Tests for figure settings parsing and run resolution (viz/figure_runs.py)."""

import copy
import importlib.util
from pathlib import Path

import pytest
import yaml

from viking_splatad.viz.figure_runs import load_figure_settings, resolve_arm_runs

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_YAML_PATH = PROJECT_ROOT / "configs" / "eval.yaml"
SLURM_YAML_PATH = PROJECT_ROOT / "configs" / "slurm.yaml"
SESSION_NAME = "may_short_grass"
OTHER_SESSION_NAME = "june_tall_grass"


def _real_eval_section():
    """Return the committed ``eval`` section."""
    return yaml.safe_load(EVAL_YAML_PATH.read_text())["eval"]


def _row(run_id, session, profile):
    """Return a synthetic ledger row."""
    return {"run_id": run_id, "session": session, "profile": profile}


def _settings():
    """Return settings parsed from the real config."""
    return load_figure_settings(_real_eval_section())


def _make_runs(tmp_path, run_ids):
    """Create run folders under tmp_path and return it."""
    for run_id in run_ids:
        (tmp_path / run_id).mkdir()
    return tmp_path


def test_real_config_parses_and_profiles_exist():
    """The committed block parses and every profile name is a profile in configs/slurm.yaml."""
    settings = _settings()
    slurm_profiles = set(yaml.safe_load(SLURM_YAML_PATH.read_text())["profiles"])
    for profile_by_arm in settings.campaign_profiles.values():
        assert set(profile_by_arm.values()) <= slurm_profiles
    assert settings.arm_order == ("stock", "radar_init", "radar_init_loss")


def test_arm_mismatch_rejected():
    """A profile map missing an arm of arm_order raises ValueError."""
    section = copy.deepcopy(_real_eval_section())
    del section["figures"]["campaign_profiles"][SESSION_NAME]["stock"]
    with pytest.raises(ValueError, match="campaign_profiles"):
        load_figure_settings(section)


@pytest.mark.parametrize("bad_colour", [[1, 2], [1, 2, 256], [1, 2, -1], [1, 2, 3, 4]])
def test_bad_colour_rejected(bad_colour):
    """Colours must be 3 channels within 0-255."""
    section = copy.deepcopy(_real_eval_section())
    section["figures"]["colours"]["hit"] = bad_colour
    with pytest.raises(ValueError, match="colours.hit"):
        load_figure_settings(section)


def test_non_positive_int_rejected():
    """dpi must be a positive int."""
    section = copy.deepcopy(_real_eval_section())
    section["figures"]["dpi"] = 0
    with pytest.raises(ValueError, match="dpi"):
        load_figure_settings(section)


def test_resolve_arm_runs_filters_session_and_sorts(tmp_path):
    """Only rows of the session and profile count, absent folders are dropped, order is by run id."""
    rows = [
        _row("run-b", SESSION_NAME, "p_stock"),
        _row("run-a", SESSION_NAME, "p_stock"),
        _row("run-c", OTHER_SESSION_NAME, "p_stock"),
        _row("run-gone", SESSION_NAME, "p_stock"),
        _row("run-x", SESSION_NAME, "p_init"),
    ]
    runs_dir = _make_runs(tmp_path, ["run-a", "run-b", "run-c", "run-x"])
    resolved = resolve_arm_runs(SESSION_NAME, {"stock": "p_stock", "radar_init": "p_init", "empty": "p_none"},
                                rows, runs_dir)
    assert resolved["stock"] == [runs_dir / "run-a", runs_dir / "run-b"]
    assert resolved["radar_init"] == [runs_dir / "run-x"]
    assert resolved["empty"] == []


def test_real_ledger_campaign_counts():
    """With the real ledger, ledger-listed runs resolve consistently (skips when runs/ is absent)."""
    ledger_path = PROJECT_ROOT / "runs" / "submitted.tsv"
    if not ledger_path.is_file():
        pytest.skip("no local runs/submitted.tsv to check")
    spec = importlib.util.spec_from_file_location("submit_run_for_figures",
                                                  PROJECT_ROOT / "scripts" / "slurm" / "submit_run.py")
    submit_run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(submit_run)
    ledger_rows = submit_run.read_ledger(ledger_path)
    settings = _settings()
    resolved = resolve_arm_runs(OTHER_SESSION_NAME, settings.campaign_profiles[OTHER_SESSION_NAME],
                                ledger_rows, ledger_path.parent)
    assert set(resolved) == set(settings.arm_order)
    for run_folders in resolved.values():
        assert all(folder.is_dir() for folder in run_folders)


def test_median_repeat_odd_count_takes_middle_value():
    """Five repeats: the third smallest value is the median (exact for odd n)."""
    from viking_splatad.viz.figure_runs import median_repeat
    values = [0.40, 0.10, 0.30, 0.50, 0.20]
    metric_by_run = {Path(f"runs/run_{index_letter}"): value
                     for index_letter, value in zip("abcde", values)}
    expected_value = sorted(values)[(len(values) - 1) // 2]
    chosen_run = median_repeat(metric_by_run)
    assert metric_by_run[chosen_run] == expected_value


def test_median_repeat_skips_undefined_and_breaks_ties_by_name():
    """None values are dropped; equal values are ordered by folder name (lower median for even n)."""
    from viking_splatad.viz.figure_runs import median_repeat
    metric_by_run = {Path("runs/run_b"): 0.2, Path("runs/run_a"): 0.2, Path("runs/run_c"): None,
                     Path("runs/run_d"): 0.9}
    # Defined, sorted by (value, name): run_a 0.2, run_b 0.2, run_d 0.9 -> position (3 - 1) // 2 = 1.
    assert median_repeat(metric_by_run) == Path("runs/run_b")


def test_median_repeat_raises_without_defined_values():
    """An arm whose metric is undefined in every repeat cannot be represented."""
    from viking_splatad.viz.figure_runs import median_repeat
    with pytest.raises(ValueError):
        median_repeat({Path("runs/run_a"): None})


def test_real_representative_metric_is_a_flat_metric_key():
    """The configured key follows flatten_hidden_metrics' H_near grammar and names a per-object mean."""
    settings = load_figure_settings(_real_eval_section())
    # Key grammar (p2_repeats.flatten_hidden_metrics): <class>/P2@<threshold key>/<metric>_Hnear/<pooled|mean|n_defined>.
    class_name, test_name, metric_name, segment = settings.representative_metric.split("/")
    assert test_name.startswith("P2@") and metric_name.endswith("_Hnear") and segment == "mean"


def test_threshold_curve_points_parsed_and_grid_contains_p2_thresholds():
    """The committed point count parses, and its [min, max] grid contains every p2 threshold exactly."""
    import numpy as np
    section = _real_eval_section()
    settings = load_figure_settings(section)
    assert settings.threshold_curve_points == section["figures"]["threshold_curve_points"]
    grid = np.linspace(settings.threshold_curve_min, settings.threshold_curve_max, settings.threshold_curve_points)
    for p2_threshold in section["p2_thresholds"]:
        assert np.isclose(grid, p2_threshold).any()


@pytest.mark.parametrize("bad_value", [0, 1, -3, 2.5, True, None])
def test_bad_threshold_curve_points_rejected(bad_value):
    """The point count must be an int of at least 2 (the grid needs both ends)."""
    section = copy.deepcopy(_real_eval_section())
    section["figures"]["threshold_curve_points"] = bad_value
    with pytest.raises(ValueError, match="threshold_curve_points"):
        load_figure_settings(section)


def test_missing_threshold_curve_points_rejected():
    """The key is required in the config."""
    section = copy.deepcopy(_real_eval_section())
    del section["figures"]["threshold_curve_points"]
    with pytest.raises(ValueError, match="threshold_curve_points"):
        load_figure_settings(section)


def test_curve_range_settings_parsed():
    """The grid ends parse to the committed values."""
    section = _real_eval_section()
    settings = load_figure_settings(section)
    assert settings.threshold_curve_min == section["figures"]["threshold_curve_min"]
    assert settings.threshold_curve_max == section["figures"]["threshold_curve_max"]


@pytest.mark.parametrize("key, bad_value", [("threshold_curve_min", 0.95), ("threshold_curve_max", 0.1)])
def test_bad_curve_range_settings_rejected(key, bad_value):
    """An inverted grid range is refused, naming the entry."""
    section = copy.deepcopy(_real_eval_section())
    section["figures"][key] = bad_value
    with pytest.raises(ValueError, match="threshold_curve"):
        load_figure_settings(section)


PRESENTATION_PROFILES = {
    "may_short_grass": {"stock": "may_scene_a_30k", "radar_init": "may_scene_a_radar_init_30k",
                        "radar_init_loss": "may_scene_a_radar_init_loss_30k"},
    "june_tall_grass": {"stock": "june_scene_a_cam_30k", "radar_init": "june_scene_a_radar_init_cam_30k",
                        "radar_init_loss": "june_scene_a_radar_init_loss_cam_30k"},
}


def test_presentation_settings_parsed_from_the_real_config():
    """The 30k profile map and its footer note are read as written in configs/eval.yaml."""
    settings = _settings()
    assert settings.presentation_profiles == PRESENTATION_PROFILES
    assert settings.presentation_footer_note.startswith("One 30k run per arm")
    # The campaign settings are untouched by the new keys.
    assert settings.footer_note != settings.presentation_footer_note


def test_presentation_profiles_exist_in_the_slurm_config():
    """Every presentation profile names a profile of configs/slurm.yaml."""
    slurm_profiles = yaml.safe_load(SLURM_YAML_PATH.read_text())["profiles"]
    for profile_by_arm in _settings().presentation_profiles.values():
        for profile in profile_by_arm.values():
            assert profile in slurm_profiles


def test_presentation_map_missing_an_arm_rejected():
    """A presentation map lacking one arm is refused, naming the entry."""
    section = copy.deepcopy(_real_eval_section())
    del section["figures"]["presentation_profiles"][SESSION_NAME]["radar_init"]
    with pytest.raises(ValueError, match="presentation_profiles"):
        load_figure_settings(section)


@pytest.mark.parametrize("missing_key", ["presentation_profiles", "presentation_footer_note"])
def test_missing_presentation_key_rejected(missing_key):
    """Both new keys are required."""
    section = copy.deepcopy(_real_eval_section())
    del section["figures"][missing_key]
    with pytest.raises(ValueError, match=missing_key):
        load_figure_settings(section)


def test_resolve_arm_runs_with_presentation_map_selects_only_30k_rows(tmp_path):
    """A ledger mixing 20k and 30k rows resolves the 30k runs only when given the presentation map."""
    settings = _settings()
    session_profiles = settings.presentation_profiles[SESSION_NAME]
    campaign_profiles = settings.campaign_profiles[SESSION_NAME]
    rows = []
    run_ids = []
    for arm in settings.arm_order:
        # Two 20k repeats and one 30k run per arm.
        for suffix, profile in (("20k-a", campaign_profiles[arm]), ("20k-b", campaign_profiles[arm]),
                                ("30k", session_profiles[arm])):
            run_id = f"{SESSION_NAME}-{arm}-{suffix}"
            rows.append(_row(run_id, SESSION_NAME, profile))
            run_ids.append(run_id)
    runs_dir = _make_runs(tmp_path, run_ids)
    presentation_runs = resolve_arm_runs(SESSION_NAME, session_profiles, rows, runs_dir)
    campaign_runs = resolve_arm_runs(SESSION_NAME, campaign_profiles, rows, runs_dir)
    for arm in settings.arm_order:
        assert presentation_runs[arm] == [runs_dir / f"{SESSION_NAME}-{arm}-30k"]
        assert campaign_runs[arm] == [runs_dir / f"{SESSION_NAME}-{arm}-20k-a",
                                      runs_dir / f"{SESSION_NAME}-{arm}-20k-b"]
