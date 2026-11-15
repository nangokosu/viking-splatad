"""Deployment regressions for the radar-variant method resolution and the batch script's method wiring.

The training method is chosen by the run profile (`radar_variant`), not by the session YAML:
- resolution through `submit_run.profile_radar_variant_options` and the stale-session-key refusal;
- the text of `splatad_run.sbatch` (one ns-train command; seed methods add seed flags only).
The executed behaviour of the batch script is covered by `test_radar_variant_sbatch.py`.
"""

import importlib.util
import re
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUBMIT_SCRIPT = PROJECT_ROOT / "scripts" / "slurm" / "submit_run.py"
BATCH_SCRIPT = PROJECT_ROOT / "scripts" / "slurm" / "splatad_run.sbatch"
# Seed opacity written into the synthetic session YAML; the resolver must hand it back unchanged.
SYNTHETIC_SEED_OPACITY = 0.25
SYNTHETIC_SESSION_NAME = "synthetic"


def _load_submit_module():
    """Load ``submit_run.py`` without requiring it to be a Python package.

    Returns:
        The executed module object.
    """
    module_spec = importlib.util.spec_from_file_location("radar_submit_run", SUBMIT_SCRIPT)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def _synthetic_session_config(**extra_radar_settings) -> dict:
    """Build the subset of a session YAML consumed by variant resolution.

    Args:
        **extra_radar_settings: Extra keys for the ``splatad:`` section (e.g. stale keys).

    Returns:
        Parsed-session-config-shaped dict with a seed opacity.
    """
    return {
        "session": {"name": SYNTHETIC_SESSION_NAME},
        "split": {},
        "splatad": {"radar_seed_opacity": SYNTHETIC_SEED_OPACITY, **extra_radar_settings},
    }


def _resolve(profile_settings: dict) -> dict:
    """Resolve one synthetic profile through the real resolver.

    Args:
        profile_settings: The profile's key/value settings.

    Returns:
        The resolver's options mapping.
    """
    submit_module = _load_submit_module()
    config = {"profiles": {"synthetic_profile": profile_settings}}
    return submit_module.profile_radar_variant_options(
        config, _synthetic_session_config(), "synthetic_profile")


def test_method_resolution_supports_stock_and_initialization_only():
    """Stock resolves to splatad without an artifact; `init` to the init method with artifact and opacity."""
    stock_options = _resolve({})
    assert stock_options["method"] == "splatad"
    assert stock_options["inputs_dir"] is None
    assert stock_options["loss_weight"] == 0.0 and stock_options["auto_weight"] is False
    init_options = _resolve({"radar_variant": "init", "radar_anchor_profile": "anchor_profile"})
    assert init_options["method"] == "splatad-radar-init"
    assert init_options["inputs_dir"] == "radar_inputs/anchor_profile"
    assert init_options["seed_opacity"] == SYNTHETIC_SEED_OPACITY
    # Initialization alone never switches the opacity loss on.
    assert init_options["loss_weight"] == 0.0 and init_options["auto_weight"] is False
    # The anchor profile defaults to the profile itself.
    default_anchor_options = _resolve({"radar_variant": "init"})
    assert default_anchor_options["inputs_dir"] == "radar_inputs/synthetic_profile"


def test_method_resolution_rejects_loss_branch_and_absolute_artifact_path(tmp_path):
    """Bad variants, unsafe anchor paths and stale session keys are refused, never silently resolved."""
    # An unknown variant name (including the removed loss-only method name) is refused.
    with pytest.raises(ValueError, match="radar_variant"):
        _resolve({"radar_variant": "splatad-radar-lidar"})
    # An absolute or traversing anchor profile cannot select a local path.
    with pytest.raises(ValueError, match="radar_anchor_profile"):
        _resolve({"radar_variant": "init", "radar_anchor_profile": "/personal/radar_inputs"})
    with pytest.raises(ValueError, match="radar_anchor_profile"):
        _resolve({"radar_variant": "init", "radar_anchor_profile": "../escape"})
    # The variant and the loss-only profile flag are mutually exclusive.
    with pytest.raises(ValueError, match="mutually exclusive"):
        _resolve({"radar_variant": "init", "radar_loss_only": True})
    # A stale session YAML still carrying a moved key is refused by session_radar_lidar_options.
    submit_module = _load_submit_module()
    config_directory = tmp_path / "sessions"
    config_directory.mkdir()
    for stale_key, stale_value in (("method", "splatad-radar-init"), ("radar_inputs_dir", "radar_inputs")):
        session_config = _synthetic_session_config(**{stale_key: stale_value})
        (config_directory / f"{SYNTHETIC_SESSION_NAME}.yaml").write_text(yaml.safe_dump(session_config))
        with pytest.raises(ValueError, match=f"splatad.{stale_key}"):
            submit_module.session_radar_lidar_options(
                SYNTHETIC_SESSION_NAME, tmp_path / "data", config_directory)


def test_batch_script_keeps_literal_stock_command_and_separate_radar_command():
    """Stock stays stock (one command, empty registration); radar init adds seed flags, never loss flags."""
    batch_text = BATCH_SCRIPT.read_text()
    # One ns-train command serves every method; the method is only a suffix.
    train_lines = [line for line in batch_text.splitlines() if line.startswith("ns-train")]
    assert train_lines == ['ns-train splatad"$TRAIN_METHOD_SUFFIX" \\']
    # In the method case block, the init method sets the seed capability and not the loss one.
    case_block = batch_text[batch_text.index('case "$TRAIN_METHOD" in'):]
    case_block = case_block[:case_block.index("esac")]
    init_branch = re.search(r"^\s*splatad-radar-init\) (.*?) ;;$", case_block, re.MULTILINE)
    assert init_branch is not None
    assert "USES_RADAR_SEEDS=1" in init_branch.group(1)
    assert "USES_RADAR_LOSS" not in init_branch.group(1)
    # The stock branch exports an empty registration.
    stock_branch_start = batch_text.index('if [ "$TRAIN_METHOD" = splatad ]; then\n  # Stock')
    stock_branch = batch_text[stock_branch_start:batch_text.index("else", stock_branch_start)]
    assert 'export NERFSTUDIO_METHOD_CONFIGS=""' in stock_branch
    # Seed flags sit under the seed capability; loss flags only under the loss capability.
    seed_block = batch_text[batch_text.index('if [ "$USES_RADAR_SEEDS" = 1 ]'):
                            batch_text.index('if [ "$USES_RADAR_LOSS" = 1 ]')]
    assert '--pipeline.model.radar-seed-opacity "$RADAR_SEED_OPACITY"' in seed_block
    assert "radar-lidar-loss" not in seed_block
