"""Regression tests for the code-review findings on the Slurm wiring of the six radar runs.

Covers ``scripts/slurm/submit_run.py``, ``scripts/slurm/splatad_run.sbatch`` and
``scripts/07_build_radar_seeds.py``. Expected values come from the committed ``configs/`` and from
first principles, never from the code under test. Nothing touches ssh, git or the cluster.
"""

import ast
import copy
import importlib.util
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from tests.unit.test_radar_variant_sbatch import (
    RUN_SBATCH_PATH, method_environment, parse_records, run_selection)
from tests.unit.test_reference_split_submit_guard import (
    EXAMPLE_SESSION, EXAMPLE_SUBMIT_TIME, prepare_gated_submit, sbatch_was_issued,
    session_reference_settings)
from tests.unit.test_slurm_submit import load_submit_module

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SUBMIT_SCRIPT_PATH = REPOSITORY_ROOT / "scripts" / "slurm" / "submit_run.py"
BUILD_SCRIPT_PATH = REPOSITORY_ROOT / "scripts" / "07_build_radar_seeds.py"
SESSION_CONFIG_DIRECTORY = REPOSITORY_ROOT / "configs" / "sessions"
# Sessions of the six radar runs, each with one init and one init_loss run profile.
RADAR_SESSIONS = ("may_short_grass", "june_tall_grass")
# A radar-variant profile of EXAMPLE_SESSION (anchored on may_scene_a_radar_init) and a loss-only profile.
VARIANT_PROFILE = "may_radar_init_smoke"
VARIANT_ANCHOR_PROFILE = "may_scene_a_radar_init"
LOSS_ONLY_PROFILE = "june_loss_only"


@pytest.fixture(scope="module")
def submit_module():
    """The submit_run module, loaded once from its file path."""
    return load_submit_module()


def make_dataset_directory(root, session):
    """Create ``<root>/<session>`` holding the manifest fields the split resolution reads.

    Args:
        root: Intermediate-data root directory.
        session: Session (and dataset) name.

    Returns:
        The dataset directory.
    """
    session_split = yaml.safe_load((SESSION_CONFIG_DIRECTORY / f"{session}.yaml").read_text())["split"]
    dataset = root / session
    dataset.mkdir(parents=True)
    (dataset / "manifest.json").write_text(json.dumps(
        {"session": {"name": session, "train_fraction": session_split["train_fraction"]}}))
    return dataset


def make_code_root(tmp_path, slurm_config):
    """A code root holding the given Slurm config and the real session configs.

    Args:
        tmp_path: Pytest temporary directory.
        slurm_config: Parsed (possibly edited) ``slurm.yaml`` mapping.

    Returns:
        The code root; ``configs/slurm.yaml`` and ``configs/sessions/`` exist under it.
    """
    code_root = tmp_path / "code_root"
    (code_root / "configs").mkdir(parents=True)
    (code_root / "configs" / "slurm.yaml").write_text(yaml.safe_dump(slurm_config))
    (code_root / "configs" / "sessions").symlink_to(SESSION_CONFIG_DIRECTORY)
    return code_root


# ------------------------------------------------------------ fix 1: anchor split equality

def test_variant_with_split_different_from_anchor_is_refused(submit_module, tmp_path):
    """A variant whose train_fraction differs from its anchor's would leak held-out radar into seeds."""
    config = copy.deepcopy(submit_module.load_slurm_config())
    session = "may_short_grass"
    session_train_fraction = yaml.safe_load(
        (SESSION_CONFIG_DIRECTORY / f"{session}.yaml").read_text())["split"]["train_fraction"]
    # A valid fraction that differs from the session's: halfway between it and 1.
    config["profiles"]["may_scene_a_radar_init_loss"]["train_fraction"] = (session_train_fraction + 1) / 2
    code_root = make_code_root(tmp_path, config)
    dataset = make_dataset_directory(tmp_path / "data", session)
    with pytest.raises(submit_module.GateFailure, match="train_fraction"):
        submit_module.require_radar_anchor_split_matches(
            code_root, dataset, session, "may_scene_a_radar_init_loss")


@pytest.mark.parametrize("session", RADAR_SESSIONS)
def test_committed_radar_profiles_pass_the_anchor_split_check(submit_module, tmp_path, session):
    """Every committed radar-variant profile shares its anchor's split; the stock profile is its own anchor."""
    config = submit_module.load_slurm_config()
    month = session.split("_")[0]
    variant_profiles = [name for name, settings in config["profiles"].items()
                        if settings.get("radar_variant") and name.startswith((f"{month}_scene_a", f"{month}_radar"))]
    assert variant_profiles, "no committed radar profile found for the session"
    dataset = make_dataset_directory(tmp_path / "data", session)
    for profile in variant_profiles + [f"{month}_scene_a"]:
        submit_module.require_radar_anchor_split_matches(REPOSITORY_ROOT, dataset, session, profile)


def test_run_gates_checks_the_anchor_split_of_a_variant(submit_module, monkeypatch):
    """run_gates hands the variant profile to the split-equality helper."""
    recorded_calls = []
    monkeypatch.setattr(submit_module, "check_clean_tree", lambda: [])
    monkeypatch.setattr(submit_module, "run_command",
                        lambda command, capture_output=True, input_text=None: SimpleNamespace(returncode=0))
    monkeypatch.setattr(submit_module, "require_radar_anchor_split_matches",
                        lambda *arguments: recorded_calls.append(arguments), raising=False)
    submit_module.run_gates(EXAMPLE_SESSION, profile=VARIANT_PROFILE)
    assert len(recorded_calls) == 1
    assert recorded_calls[0][-2:] == (EXAMPLE_SESSION, VARIANT_PROFILE)


@pytest.mark.parametrize("skip_gates", [True, False])
def test_submit_run_checks_the_anchor_split_even_without_gates(monkeypatch, tmp_path, skip_gates):
    """A split mismatch refuses the submit, with the gates on or skipped, before any ssh."""
    submit_module, issued_commands = prepare_gated_submit(
        monkeypatch, tmp_path, session_reference_settings(EXAMPLE_SESSION))
    make_dataset_directory(tmp_path / "data", EXAMPLE_SESSION)
    (tmp_path / "data" / EXAMPLE_SESSION / "radar_inputs" / VARIANT_ANCHOR_PROFILE).mkdir(parents=True)
    monkeypatch.setattr(submit_module, "INTERMEDIATE_DATA_ROOT", tmp_path / "data")

    def refuse_split(*arguments):
        """Stand-in helper reporting a split mismatch."""
        raise submit_module.GateFailure("split differs: train_fraction")

    monkeypatch.setattr(submit_module, "require_radar_anchor_split_matches", refuse_split, raising=False)
    monkeypatch.setattr(submit_module, "run_gates", lambda *arguments, **keywords: None)
    with pytest.raises(submit_module.GateFailure, match="train_fraction"):
        submit_module.submit_run(submit_module.load_slurm_config(), EXAMPLE_SESSION, VARIANT_PROFILE,
                                 skip_gates=skip_gates)
    assert not sbatch_was_issued(issued_commands)


def test_script_07_checks_the_anchor_split_before_building(tmp_path, monkeypatch):
    """Script 07 refuses a profile whose split differs from its anchor's, without building."""
    script = load_build_script()
    helpers = load_submit_module()
    built = []

    def refuse_split(*arguments):
        """Stand-in helper reporting a split mismatch."""
        raise helpers.GateFailure("split differs: train_fraction")

    monkeypatch.setattr(helpers, "require_radar_anchor_split_matches", refuse_split, raising=False)
    monkeypatch.setattr(script, "load_submit_helpers", lambda code_root: helpers)
    monkeypatch.setattr(script, "build_radar_anchor_artifact", lambda *arguments, **keywords: built.append(1))
    dataset = make_dataset_directory(tmp_path, "may_short_grass")
    with pytest.raises(helpers.GateFailure, match="train_fraction"):
        script.main(["--dataset", str(dataset), "--session", "may_short_grass", "--profile", VARIANT_PROFILE])
    assert not built


# ------------------------------------------------------------ fix 2: stage validates the anchors only

STAGE_CONFIG = {"profiles": {
    "plain": {},
    "anchor_profile": {"radar_variant": "init"},
    "consumer_profile": {"radar_variant": "init_loss", "radar_anchor_profile": "anchor_profile"},
}}


def install_recording_contract(monkeypatch, submit_module, validated):
    """Replace the contract validator and the split resolver with recorders.

    Args:
        monkeypatch: Pytest monkeypatch.
        submit_module: The loaded submit module.
        validated: List receiving ``(artifact directory name, require_seed_colours)`` per validation.
    """
    from viking_splatad.contract import splatad_contract

    def fake_validate(root, session_name, dataset_root, expected_split_settings, require_seed_colours):
        """Record the call and report success."""
        validated.append((Path(root).name, require_seed_colours))
        return SimpleNamespace(passed=True, to_markdown=lambda: "")

    monkeypatch.setattr(splatad_contract, "validate_radar_inputs", fake_validate)
    monkeypatch.setattr(submit_module, "profile_radar_split_settings",
                        lambda code_root, dataset, session, profile: ({}, {"profile": profile}))


def test_stage_validates_exactly_the_anchor_artifacts_and_notes_other_folders(
        submit_module, monkeypatch, tmp_path, capsys):
    """Only anchors named by variant profiles are validated (colours required); strays are noted."""
    dataset = tmp_path / "may_short_grass"
    for folder_name in ("anchor_profile", "stray_folder", "loss_only_unpainted"):
        (dataset / "radar_inputs" / folder_name).mkdir(parents=True)
    validated = []
    install_recording_contract(monkeypatch, submit_module, validated)
    submit_module.require_valid_radar_inputs(dataset, "may_short_grass", STAGE_CONFIG)
    assert validated == [("anchor_profile", True)]
    printed = capsys.readouterr().out
    assert "stray_folder" in printed and "loss_only_unpainted" in printed


def test_stage_refuses_when_no_named_anchor_artifact_exists(submit_module, monkeypatch, tmp_path):
    """A radar_inputs folder holding none of the named anchors is a clear GateFailure, not a KeyError."""
    dataset = tmp_path / "may_short_grass"
    (dataset / "radar_inputs" / "stray_folder").mkdir(parents=True)
    install_recording_contract(monkeypatch, submit_module, [])
    with pytest.raises(submit_module.GateFailure, match="anchor_profile"):
        submit_module.require_valid_radar_inputs(dataset, "may_short_grass", STAGE_CONFIG)


def test_stage_without_a_radar_inputs_directory_checks_nothing(submit_module, monkeypatch, tmp_path):
    """No radar_inputs directory: nothing to validate."""
    dataset = tmp_path / "may_short_grass"
    dataset.mkdir()
    validated = []
    install_recording_contract(monkeypatch, submit_module, validated)
    submit_module.require_valid_radar_inputs(dataset, "may_short_grass", STAGE_CONFIG)
    assert validated == []


# ------------------------------------------------------------ fix 3 and 5: submit refusals

def prepare_variant_submit(monkeypatch, tmp_path, create_artifact):
    """A fake cluster plus a tmp intermediate root, with or without the variant's local artifact.

    Args:
        monkeypatch: Pytest monkeypatch.
        tmp_path: Pytest temporary directory.
        create_artifact: Whether ``<dataset>/radar_inputs/<anchor profile>`` exists.

    Returns:
        (submit module with fakes, issued commands).
    """
    prepared_module, issued_commands = prepare_gated_submit(
        monkeypatch, tmp_path, session_reference_settings(EXAMPLE_SESSION))
    dataset = make_dataset_directory(tmp_path / "data", EXAMPLE_SESSION)
    if create_artifact:
        (dataset / "radar_inputs" / VARIANT_ANCHOR_PROFILE).mkdir(parents=True)
    monkeypatch.setattr(prepared_module, "INTERMEDIATE_DATA_ROOT", tmp_path / "data")
    return prepared_module, issued_commands


def test_skip_gates_variant_submit_without_local_artifact_is_refused(monkeypatch, tmp_path):
    """--skip-gates must not queue a radar run whose artifact is not staged locally."""
    prepared_module, issued_commands = prepare_variant_submit(monkeypatch, tmp_path, create_artifact=False)
    with pytest.raises(prepared_module.GateFailure, match="radar_inputs"):
        prepared_module.submit_run(prepared_module.load_slurm_config(), EXAMPLE_SESSION, VARIANT_PROFILE,
                                   skip_gates=True, now_utc=EXAMPLE_SUBMIT_TIME)
    assert not sbatch_was_issued(issued_commands)


def test_skip_gates_variant_submit_with_local_artifact_is_submitted(monkeypatch, tmp_path):
    """Control: with the artifact present and the real split check passing, the job is submitted."""
    prepared_module, issued_commands = prepare_variant_submit(monkeypatch, tmp_path, create_artifact=True)
    prepared_module.submit_run(prepared_module.load_slurm_config(), EXAMPLE_SESSION, VARIANT_PROFILE,
                               skip_gates=True, now_utc=EXAMPLE_SUBMIT_TIME)
    assert sbatch_was_issued(issued_commands)


@pytest.mark.parametrize("profile, create_artifact, error_match", [
    (LOSS_ONLY_PROFILE, True, "loss-only"),
    (VARIANT_PROFILE, False, "radar_inputs"),
])
def test_cheap_refusals_happen_before_the_full_test_suite(monkeypatch, tmp_path, profile, create_artifact,
                                                          error_match):
    """A loss-only profile or a missing artifact is refused before run_gates (the full suite) runs."""
    prepared_module, issued_commands = prepare_variant_submit(monkeypatch, tmp_path, create_artifact)
    gate_calls = []
    monkeypatch.setattr(prepared_module, "run_gates", lambda *arguments, **keywords: gate_calls.append(1))
    with pytest.raises((ValueError, prepared_module.GateFailure), match=error_match):
        prepared_module.submit_run(prepared_module.load_slurm_config(), EXAMPLE_SESSION, profile,
                                   skip_gates=False, now_utc=EXAMPLE_SUBMIT_TIME)
    assert gate_calls == []
    assert not sbatch_was_issued(issued_commands)


# ------------------------------------------------------------ fix 4: dataset name vs session

def test_split_settings_refuse_a_dataset_not_named_after_the_session(submit_module, tmp_path):
    """The manifest and config lookups use dataset.parent / session, so the names must agree."""
    dataset = make_dataset_directory(tmp_path, "may_short_grass")
    with pytest.raises(ValueError, match="may_short_grass"):
        submit_module.profile_radar_split_settings(REPOSITORY_ROOT, dataset, "june_tall_grass", "june_scene_a")


# ------------------------------------------------------------ fix 6: stale keys in the variant resolver

@pytest.mark.parametrize("stale_key, stale_value", [("method", "splatad-radar-init"),
                                                    ("radar_inputs_dir", "radar_inputs")])
@pytest.mark.parametrize("profile_name", ["stock", "variant"])
def test_variant_resolver_refuses_stale_session_keys(submit_module, stale_key, stale_value, profile_name):
    """profile_radar_variant_options itself refuses the session keys that moved to the profile."""
    config = {"profiles": {"stock": {}, "variant": {"radar_variant": "init"}}}
    session_config = {"session": {"name": "synthetic"}, "split": {}, "splatad": {stale_key: stale_value}}
    with pytest.raises(ValueError, match=f"splatad.{stale_key}"):
        submit_module.profile_radar_variant_options(config, session_config, profile_name)


# ------------------------------------------------------------ fix 7: variant names are constants

def test_init_loss_variant_name_is_defined_once_as_a_constant():
    """The string literal ``init_loss`` occurs exactly once (its constant's definition)."""
    variant_literal = "init_loss"
    syntax_tree = ast.parse(SUBMIT_SCRIPT_PATH.read_text())
    occurrences = [node for node in ast.walk(syntax_tree)
                   if isinstance(node, ast.Constant) and node.value == variant_literal]
    assert len(occurrences) == 1


# ------------------------------------------------------------ fix 8: registration on resume

def test_every_registration_literal_equals_the_training_block_value():
    """Each literal NERFSTUDIO_METHOD_CONFIGS value in splatad_run.sbatch matches what block (d) produces."""
    training_value = parse_records(run_selection(method_environment("splatad-radar-init")))["CONFIGS"][0]
    assert training_value
    literals = re.findall(r'NERFSTUDIO_METHOD_CONFIGS="([^"$][^"]*)"', RUN_SBATCH_PATH.read_text())
    # One literal for the resume-capable section (b); the training block (d) builds its own copy.
    assert literals, "section (b) exports no literal registration"
    assert all(literal == training_value for literal in literals)


def test_registration_is_exported_in_the_environment_section_for_non_stock_methods():
    """Section (b), which also runs on resume, exports the registration for any non-stock TRAIN_METHOD."""
    script_text = RUN_SBATCH_PATH.read_text()
    environment_section = script_text[script_text.index("# ---- (b) environment"):
                                      script_text.index("# ---- (c) GPU sanity check")]
    assert "NERFSTUDIO_METHOD_CONFIGS=" in environment_section
    assert "TRAIN_METHOD" in environment_section


# ------------------------------------------------------------ fix 9: script 07

def load_build_script():
    """Load scripts/07_build_radar_seeds.py as a module.

    Returns:
        The loaded module.
    """
    module_spec = importlib.util.spec_from_file_location("build_radar_seeds_review", BUILD_SCRIPT_PATH)
    script = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(script)
    return script


def test_script_07_has_no_radar_inputs_literal():
    """The artifact directory name comes from the helpers' constant, not a duplicated literal."""
    syntax_tree = ast.parse(BUILD_SCRIPT_PATH.read_text())
    assert not [node for node in ast.walk(syntax_tree)
                if isinstance(node, ast.Constant) and node.value == "radar_inputs"]


def test_script_07_unknown_profile_is_a_clear_error(tmp_path, capsys):
    """An unknown --profile names the profile in an error and exits non-zero, not a raw KeyError."""
    script = load_build_script()
    dataset = make_dataset_directory(tmp_path, "may_short_grass")
    status = script.main(["--dataset", str(dataset), "--session", "may_short_grass",
                          "--profile", "no_such_profile"])
    assert status != 0
    assert "no_such_profile" in capsys.readouterr().err


@pytest.mark.parametrize("report_passed, expected_status", [(True, 0), (False, 1)])
def test_script_07_validates_the_built_artifact(tmp_path, monkeypatch, capsys, report_passed, expected_status):
    """After building, script 07 runs the contract with the paint flag; a failing report returns 1."""
    from viking_splatad.contract import splatad_contract
    validations = []

    def fake_validate(root, session_name, dataset_root, expected_split_settings, require_seed_colours):
        """Record the call and return a canned report."""
        validations.append((Path(root), session_name, require_seed_colours))
        return SimpleNamespace(passed=report_passed, to_markdown=lambda: "REPORT-MARKDOWN")

    # The script imports the name at load time, so patch the source module before loading it.
    monkeypatch.setattr(splatad_contract, "validate_radar_inputs", fake_validate)
    script = load_build_script()
    metadata = {"painting": {"painted_fraction": 0.0}}
    monkeypatch.setattr(script, "build_radar_anchor_artifact",
                        lambda *arguments, **keywords: _CountingMetadata(metadata))
    dataset = make_dataset_directory(tmp_path, "may_short_grass")
    status = script.main(["--dataset", str(dataset), "--session", "may_short_grass",
                          "--profile", VARIANT_PROFILE])
    assert status == expected_status
    assert validations == [(dataset / "radar_inputs" / VARIANT_ANCHOR_PROFILE, "may_short_grass", True)]
    assert ("REPORT-MARKDOWN" in capsys.readouterr().out) == (not report_passed)


class _CountingMetadata(dict):
    """Fake builder result: any key script 07 prints reads as zero."""

    def __missing__(self, key):
        """Return a JSON-serialisable placeholder for an unrequested key."""
        return 0
