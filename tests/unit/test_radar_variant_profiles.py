"""Radar-variant Slurm profiles: config shape, option resolver, job environment, split sharing.

Expectations are derived from the real ``configs/slurm.yaml`` and ``configs/sessions/*.yaml``.
"""

import copy
import json
from pathlib import Path

import pytest
import yaml

from tests.unit.test_slurm_submit import load_submit_module

# Repository root: tests/unit/<this file> -> two levels up.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SESSION_CONFIG_DIRECTORY = REPOSITORY_ROOT / "configs" / "sessions"

# (session name, month prefix used in profile names, stock profile name)
SESSION_MONTHS = (("may_short_grass", "may", "may_scene_a"),
                  ("june_tall_grass", "june", "june_scene_a"))
RADAR_KEYS = ("radar_variant", "radar_anchor_profile")
SMOKE_OVERRIDDEN_KEYS = ("max_num_iterations", "steps_per_save")
WINDOW_KEYS = ("window_seconds", "window_start_seconds")

# Run profiles and smokes per month, in the naming the work order fixes.
VARIANT_SUFFIXES = (("radar_init", "init"), ("radar_init_loss", "init_loss"))
RUN_PROFILE_CASES = [(f"{month}_scene_a_{suffix}", month, stock, variant)
                     for _, month, stock in SESSION_MONTHS for suffix, variant in VARIANT_SUFFIXES]
SMOKE_PROFILE_CASES = [(f"{month}_{suffix}_smoke", month, stock, variant)
                       for _, month, stock in SESSION_MONTHS for suffix, variant in VARIANT_SUFFIXES]
ALL_RADAR_CASES = RUN_PROFILE_CASES + SMOKE_PROFILE_CASES
MONTH_TO_SESSION = {month: session for session, month, _ in SESSION_MONTHS}
STOCK_PROFILES = [stock for _, _, stock in SESSION_MONTHS]


@pytest.fixture(scope="module")
def submit_module():
    """The submit_run module, loaded once from its file path."""
    return load_submit_module()


@pytest.fixture(scope="module")
def slurm_config(submit_module):
    """The real parsed configs/slurm.yaml."""
    return submit_module.load_slurm_config(REPOSITORY_ROOT / "configs" / "slurm.yaml")


def load_session_yaml(session_name):
    """Parse a committed session YAML.

    Args:
        session_name: Session file stem.

    Returns:
        The parsed mapping.
    """
    return yaml.safe_load((SESSION_CONFIG_DIRECTORY / f"{session_name}.yaml").read_text())


def expected_anchor_profile(profile_name, month):
    """The anchor profile a variant profile must point at (its month's radar_init run profile)."""
    return f"{month}_scene_a_radar_init"


def resolve(submit_module, config, profile_name, month):
    """Run the resolver for a profile of the given month with the real session YAML."""
    session_config = load_session_yaml(MONTH_TO_SESSION[month])
    return submit_module.profile_radar_variant_options(config, session_config, profile_name)


# ---------------------------------------------------------------- 1, 2: profile shape

@pytest.mark.parametrize("profile_name,month,stock,variant", RUN_PROFILE_CASES)
def test_run_profile_equals_stock_profile_except_radar_keys(slurm_config, profile_name, month, stock, variant):
    """After dropping the two radar keys a radar run profile is identical to the stock profile."""
    profile = dict(slurm_config["profiles"][profile_name])
    assert profile.pop("radar_variant") == variant
    profile.pop("radar_anchor_profile", None)
    assert profile == slurm_config["profiles"][stock]


@pytest.mark.parametrize("profile_name,month,stock,variant", SMOKE_PROFILE_CASES)
def test_smoke_profile_equals_core_except_iterations_save_and_radar(slurm_config, profile_name, month,
                                                                     stock, variant):
    """A radar smoke is the month's core with the smoke profile's iteration/save settings, full window."""
    profiles = slurm_config["profiles"]
    smoke_profile = dict(profiles[profile_name])
    assert smoke_profile.pop("radar_variant") == variant
    # The smoke trains against the full-run artifact of the same month.
    assert smoke_profile.pop("radar_anchor_profile") == expected_anchor_profile(profile_name, month)
    for overridden_key in SMOKE_OVERRIDDEN_KEYS:
        assert smoke_profile.pop(overridden_key) == profiles["smoke"][overridden_key]
    expected = {key: value for key, value in profiles[stock].items() if key not in SMOKE_OVERRIDDEN_KEYS}
    assert smoke_profile == expected
    assert not any(window_key in profiles[profile_name] for window_key in WINDOW_KEYS)


def test_anchor_profile_of_loss_runs_and_train_lidar_only_flags(slurm_config):
    """init_loss run profiles anchor on the month's init profile; train_lidar_only matches the brief."""
    profiles = slurm_config["profiles"]
    for profile_name, month, _, variant in RUN_PROFILE_CASES:
        if variant == "init_loss":
            assert profiles[profile_name]["radar_anchor_profile"] == expected_anchor_profile(profile_name, month)
    assert "train_lidar_only" not in profiles["may_scene_a"]
    assert profiles["june_scene_a"]["train_lidar_only"] is True
    for profile_name, month, _, _ in ALL_RADAR_CASES:
        assert ("train_lidar_only" in profiles[profile_name]) == (month == "june")


# ---------------------------------------------------------------- 3: resolver

@pytest.mark.parametrize("month", [month for _, month, _ in SESSION_MONTHS])
def test_resolver_stock_profile_is_plain_splatad(submit_module, slurm_config, month):
    """A stock profile resolves to method splatad and no radar inputs."""
    stock = f"{month}_scene_a"
    options = resolve(submit_module, slurm_config, stock, month)
    assert options["method"] == "splatad"
    assert options["inputs_dir"] is None


@pytest.mark.parametrize("profile_name,month,stock,variant", ALL_RADAR_CASES)
def test_resolver_variant_options(submit_module, slurm_config, profile_name, month, stock, variant):
    """Resolver output per variant: method, artifact directory, weights and gate from the session YAML."""
    session_splatad = load_session_yaml(MONTH_TO_SESSION[month])["splatad"]
    options = resolve(submit_module, slurm_config, profile_name, month)
    # No session or profile override: the model uses stock's init_opacities (user, 2026-11-14).
    assert options["seed_opacity"] is None
    assert options["loss_weight"] == 0.0
    if variant == "init":
        assert options["method"] == "splatad-radar-init"
        # The artifact is named after the anchor profile (contract: "radar_inputs/<anchor_profile>").
        # The full init run is its own anchor; an init smoke anchors on the month's full init run,
        # so both resolve to the same directory.
        assert options["inputs_dir"] == f"radar_inputs/{month}_scene_a_radar_init"
        assert options["auto_weight"] is False
    else:
        assert options["method"] == "splatad-radar-init-lidar"
        # Smokes and run profiles alike read the month's full-run init artifact.
        assert options["inputs_dir"] == f"radar_inputs/{month}_scene_a_radar_init"
        assert options["auto_weight"] is True
        assert options["gate_quantile"] == session_splatad["radar_lidar_gate_quantile"]
        assert options["gate_chunk_size"] == session_splatad["radar_lidar_gate_chunk_size"]


# ---------------------------------------------------------------- 4: validation

def test_variant_with_loss_only_is_rejected(submit_module, slurm_config):
    """radar_variant and radar_loss_only are mutually exclusive."""
    config = copy.deepcopy(slurm_config)
    config["profiles"]["june_scene_a_radar_init"]["radar_loss_only"] = True
    with pytest.raises(ValueError):
        resolve(submit_module, config, "june_scene_a_radar_init", "june")


def test_unknown_variant_is_rejected(submit_module, slurm_config):
    """A radar_variant outside {init, init_loss} raises."""
    config = copy.deepcopy(slurm_config)
    config["profiles"]["june_scene_a_radar_init"]["radar_variant"] = "bogus"
    with pytest.raises(ValueError):
        resolve(submit_module, config, "june_scene_a_radar_init", "june")


def test_init_loss_does_not_require_lidar_only(submit_module, slurm_config):
    """May's init_loss profile (camera loss on) resolves without train_lidar_only."""
    profile = slurm_config["profiles"]["may_scene_a_radar_init_loss"]
    assert "train_lidar_only" not in profile
    options = resolve(submit_module, slurm_config, "may_scene_a_radar_init_loss", "may")
    assert options["method"] == "splatad-radar-init-lidar"


# ---------------------------------------------------------------- 5: job environment

def build_environment(submit_module, config, profile_name, month):
    """build_job_environment for a profile, passing the resolver output as radar_lidar_options."""
    options = resolve(submit_module, config, profile_name, month)
    return submit_module.build_job_environment(
        config["profiles"][profile_name], MONTH_TO_SESSION[month], "run-id", "a" * 40, "/stage",
        {"env_prefix": "/env"}, radar_lidar_options=options), options


@pytest.mark.parametrize("profile_name,month,stock,variant",
                         [case for case in ALL_RADAR_CASES if case[3] == "init_loss"])
def test_init_loss_environment_exports(submit_module, slurm_config, profile_name, month, stock, variant):
    """init_loss exports every loss variable plus the diagnostics path under the run directory."""
    environment, options = build_environment(submit_module, slurm_config, profile_name, month)
    assert environment["TRAIN_METHOD"] == "splatad-radar-init-lidar"
    assert environment["RADAR_INPUTS_DIR"] == options["inputs_dir"]
    # Seed opacity unset -> not exported; the model falls back to stock's init_opacities.
    assert "RADAR_SEED_OPACITY" not in environment
    assert environment["RADAR_LIDAR_AUTO_WEIGHT"] == "True"
    assert float(environment["RADAR_LIDAR_LOSS_WEIGHT"]) == options["loss_weight"]
    assert float(environment["RADAR_LIDAR_GATE_QUANTILE"]) == options["gate_quantile"]
    assert int(environment["RADAR_LIDAR_GATE_CHUNK_SIZE"]) == options["gate_chunk_size"]
    assert environment["RADAR_DIAGNOSTICS_PATH"].startswith("/stage/runs/run-id/")


@pytest.mark.parametrize("profile_name,month,stock,variant",
                         [case for case in ALL_RADAR_CASES if case[3] == "init"])
def test_init_environment_exports(submit_module, slurm_config, profile_name, month, stock, variant):
    """init exports the seed opacity and inputs directory but no diagnostics path."""
    environment, options = build_environment(submit_module, slurm_config, profile_name, month)
    assert environment["TRAIN_METHOD"] == "splatad-radar-init"
    # Seed opacity unset -> not exported; the model falls back to stock's init_opacities.
    assert "RADAR_SEED_OPACITY" not in environment
    assert environment["RADAR_INPUTS_DIR"] == options["inputs_dir"]
    assert "RADAR_DIAGNOSTICS_PATH" not in environment


# ---------------------------------------------------------------- 6: Neel's resolver regression

def test_loss_only_resolver_still_forces_radar_lidar(submit_module):
    """profile_radar_lidar_options keeps method splatad-radar-lidar and the train_lidar_only requirement."""
    baseline = {"method": "splatad", "loss_weight": 0., "gate_quantile": 0.95,
                "chunk_size": 4096, "inputs_dir": None}
    profile = {"radar_loss_only": True, "train_lidar_only": True}
    resolved = submit_module.profile_radar_lidar_options(profile, baseline, "/stage", "june", "L", "rev")
    assert resolved["method"] == "splatad-radar-lidar"
    assert resolved["auto_weight"] is True
    assert resolved["inputs_dir"] == "/stage/radar-inputs/june/L/rev"
    with pytest.raises(ValueError, match="train_lidar_only"):
        submit_module.profile_radar_lidar_options({"radar_loss_only": True}, baseline, "/stage", "june", "L", "rev")


# ---------------------------------------------------------------- 7: session YAMLs

class DuplicateKeyLoader(yaml.SafeLoader):
    """SafeLoader that raises on a repeated mapping key."""


def construct_mapping_without_duplicates(loader, node, deep=False):
    """Build a mapping, raising ValueError if two explicit keys are equal.

    Args:
        loader: The active YAML loader.
        node: The mapping node.
        deep: Passed to the standard constructor.

    Returns:
        The constructed mapping.
    """
    # Step 1: merge keys (<<) may legally repeat across anchors, so only plain keys are compared.
    seen_keys = set()
    for key_node, _ in node.value:
        if key_node.tag == "tag:yaml.org,2002:merge":
            continue
        key = loader.construct_object(key_node, deep=deep)
        if key in seen_keys:
            raise ValueError(f"duplicate key {key!r} at line {key_node.start_mark.line + 1}")
        seen_keys.add(key)
    # Step 2: defer to the standard behaviour.
    return yaml.SafeLoader.construct_mapping(loader, node, deep=deep)


DuplicateKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
                                   construct_mapping_without_duplicates)


@pytest.mark.parametrize("session_name", [session for session, _, _ in SESSION_MONTHS])
def test_session_yaml_splatad_section(session_name):
    """The splatad: section drops the per-method keys and the seed opacity (stock's is used)."""
    splatad_section = load_session_yaml(session_name)["splatad"]
    for removed_key in ("method", "radar_inputs_dir", "radar_lidar_loss_weight"):
        assert removed_key not in splatad_section
    assert "radar_seed_opacity" not in splatad_section


def test_no_yaml_file_has_duplicate_keys():
    """No committed YAML under configs/ repeats a mapping key."""
    yaml_paths = sorted((REPOSITORY_ROOT / "configs").rglob("*.yaml"))
    assert yaml_paths, "no YAML files found"
    for yaml_path in yaml_paths:
        yaml.load(yaml_path.read_text(), Loader=DuplicateKeyLoader)


# ---------------------------------------------------------------- 8: shared split settings

@pytest.mark.parametrize("session_name,month,stock", SESSION_MONTHS)
def test_variant_profiles_share_split_settings(submit_module, tmp_path, session_name, month, stock):
    """init, init_loss and stock profiles give identical split settings, so one artifact serves the session."""
    # Step 1: a synthetic dataset directory named after the session, with the manifest fields read.
    session_split = load_session_yaml(session_name)["split"]
    dataset = tmp_path / session_name
    dataset.mkdir()
    (dataset / "manifest.json").write_text(json.dumps(
        {"session": {"name": session_name, "train_fraction": session_split["train_fraction"]}}))
    # Step 2: resolve each profile and compare the split dictionaries.
    results = {profile: submit_module.profile_radar_split_settings(REPOSITORY_ROOT, dataset, session_name, profile)
               for profile in (stock, f"{month}_scene_a_radar_init", f"{month}_scene_a_radar_init_loss")}
    stock_session_config, stock_settings = results[stock]
    for session_config, settings in results.values():
        assert settings == stock_settings
        assert session_config == stock_session_config


# ---------------------------------------------------------------- 9: June radar init with the camera loss

# The camera-on June radar init profile and the lidar-only core it derives from.
JUNE_CAMERA_PROFILE = "june_scene_a_radar_init_cam"
JUNE_LIDAR_ONLY_PROFILE = "june_scene_a"


def test_june_camera_profile_differs_from_lidar_only_core_only_by_camera_loss_and_radar_keys(slurm_config):
    """The camera-on profile is the lidar-only June core with train_lidar_only off plus the init keys."""
    # Step 1: drop the radar keys and the one flag that must differ.
    profile = dict(slurm_config["profiles"][JUNE_CAMERA_PROFILE])
    assert profile.pop("radar_variant") == "init"
    assert profile.pop("radar_anchor_profile") == f"{JUNE_LIDAR_ONLY_PROFILE}_radar_init"
    assert profile.pop("train_lidar_only") is False
    # Step 2: everything left (iterations, evals, seed, time limit, render settings) equals the core.
    expected = dict(slurm_config["profiles"][JUNE_LIDAR_ONLY_PROFILE])
    expected.pop("train_lidar_only")
    assert profile == expected


def test_june_camera_profile_resolves_to_radar_init_and_exports_camera_loss(submit_module, slurm_config):
    """The profile trains splatad-radar-init on the June init artifact and does not export lidar-only."""
    # Step 1: the resolver picks the init method and the lidar-only profile's artifact.
    options = resolve(submit_module, slurm_config, JUNE_CAMERA_PROFILE, "june")
    assert options["method"] == "splatad-radar-init"
    assert options["inputs_dir"] == "radar_inputs/june_scene_a_radar_init"
    assert options["loss_weight"] == 0.0
    # Step 2: the profile's own flag is false, which is what submit exports as TRAIN_LIDAR_ONLY=False.
    assert slurm_config["profiles"][JUNE_CAMERA_PROFILE]["train_lidar_only"] is False
