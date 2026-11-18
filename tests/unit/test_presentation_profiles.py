"""Presentation (30k-iteration) Slurm profiles: each is its 20k base with only iterations and time changed.

Expectations are derived from the real ``configs/slurm.yaml`` and ``configs/sessions/*.yaml``.
"""

from pathlib import Path

import pytest
import yaml

from tests.unit.test_slurm_submit import load_submit_module

# Repository root: tests/unit/<this file> -> two levels up.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SESSION_CONFIG_DIRECTORY = REPOSITORY_ROOT / "configs" / "sessions"
# The only keys a presentation profile may change relative to its 20k base.
OVERRIDDEN_KEYS = ("max_num_iterations", "time_limit")
# Suffix that names a presentation profile after its base.
PRESENTATION_SUFFIX = "_30k"
# SplatAD's stock iteration count (last step 30000): the learning-rate schedules' max_steps + 1.
STOCK_SPLATAD_ITERATIONS = 30001

# (session, 20k base profile) for the six variants; June uses the camera-loss (*_cam) arms.
PRESENTATION_BASES = [
    ("may_short_grass", "may_scene_a"),
    ("may_short_grass", "may_scene_a_radar_init"),
    ("may_short_grass", "may_scene_a_radar_init_loss"),
    ("june_tall_grass", "june_scene_a_cam"),
    ("june_tall_grass", "june_scene_a_radar_init_cam"),
    ("june_tall_grass", "june_scene_a_radar_init_loss_cam"),
]


@pytest.fixture(scope="module")
def submit_module():
    """The submit_run module, loaded once from its file path."""
    return load_submit_module()


@pytest.fixture(scope="module")
def slurm_config(submit_module):
    """The real parsed configs/slurm.yaml."""
    return submit_module.load_slurm_config(REPOSITORY_ROOT / "configs" / "slurm.yaml")


@pytest.mark.parametrize("session_name,base_profile", PRESENTATION_BASES)
def test_presentation_profile_equals_base_except_iterations_and_time(submit_module, slurm_config, session_name, base_profile):
    """Dropping iterations and time_limit leaves exactly the 20k base profile; iterations are stock 30001."""
    profiles = slurm_config["profiles"]
    presentation = dict(profiles[base_profile + PRESENTATION_SUFFIX])
    assert presentation["max_num_iterations"] == STOCK_SPLATAD_ITERATIONS
    # The longer run needs at least the base's time cap.
    assert (submit_module.parse_slurm_time_limit_seconds(presentation["time_limit"])
            >= submit_module.parse_slurm_time_limit_seconds(profiles[base_profile]["time_limit"]))
    base_without_overrides = {key: value for key, value in profiles[base_profile].items()
                              if key not in OVERRIDDEN_KEYS}
    # A radar profile without radar_anchor_profile anchors on itself (configs/slurm.yaml); the 30k
    # copy must name the base explicitly, so apply that default before comparing.
    if "radar_variant" in base_without_overrides:
        base_without_overrides.setdefault("radar_anchor_profile", base_profile)
    presentation_without_overrides = {key: value for key, value in presentation.items()
                                      if key not in OVERRIDDEN_KEYS}
    assert presentation_without_overrides == base_without_overrides


@pytest.mark.parametrize("session_name,base_profile", PRESENTATION_BASES)
def test_presentation_profile_resolves_like_base(submit_module, slurm_config, session_name, base_profile):
    """Method, radar artifact, weights and split options resolve identically to the 20k base."""
    session_config = yaml.safe_load((SESSION_CONFIG_DIRECTORY / f"{session_name}.yaml").read_text())
    base_options = submit_module.profile_radar_variant_options(slurm_config, session_config, base_profile)
    presentation_options = submit_module.profile_radar_variant_options(
        slurm_config, session_config, base_profile + PRESENTATION_SUFFIX)
    assert presentation_options == base_options
