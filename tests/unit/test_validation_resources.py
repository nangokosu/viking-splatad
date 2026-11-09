"""Validation may use another configured sm_86 allocation without changing training GPUs."""

import pytest

from tests.unit.test_slurm_submit import load_submit_module


def test_validation_alternative_is_explicit_and_default_allocation_is_preserved():
    """The A2000 is opt-in for validation; existing render/training resource defaults stay intact."""
    helpers = load_submit_module()
    config = helpers.load_slurm_config()
    assert helpers.validation_settings(config, None) == config["render_only"]
    assert helpers.validation_settings(config, "validation_gpu") == config["validation_gpu"]
    # NVIDIA's published compute capability table lists RTX A2000 under 8.6.
    assert helpers.gres_type_from_spec(config["validation_gpu"]["gres"]) == "rtx_a2000"
    assert helpers.resolve_job_gres(config, config["profiles"]["june_scene_a"]) == config["gres"]


def test_validation_override_cannot_change_an_optimization_allocation(monkeypatch):
    """A reviewed preflight cannot accidentally direct training onto a validation-only GPU."""
    helpers = load_submit_module()
    monkeypatch.setattr(helpers, "check_clean_tree", lambda: [])
    with pytest.raises(ValueError, match="cannot be used for training"):
        helpers.submit_remote_run(helpers.load_slurm_config(), "june_tall_grass", "june_scene_a",
                                  False, reviewed_preflight=True, resource_key="validation_gpu")


def test_unknown_validation_resource_section_is_rejected():
    """Arbitrary configuration sections cannot be selected as validation allocations."""
    helpers = load_submit_module()
    with pytest.raises(ValueError, match="validation resources"):
        helpers.validation_settings(helpers.load_slurm_config(), "profiles")
