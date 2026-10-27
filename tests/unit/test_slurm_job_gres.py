"""Per-job GPU type (resolve_job_gres): a profile or render_only may name its own gres.

The environment is compiled for sm_86 on the default A6000; the A4500 is also sm_86, so render-only
jobs and the smoke_a4500 profile run there (user's choice 2026-10-27). Build-env and resume keep the
default. Covered here: the resolver, the committed config's choices, and that a submission sizes the
job from the nodes of its own GPU type and passes that gres to sbatch.
"""

import datetime
import math
import subprocess
from typing import Dict, List

import pytest

from tests.unit.test_slurm_submit import load_submit_module, make_fake_cluster

# Synthetic `sinfo -O NodeHost,Memory,CPUs,Gres` output (MEMORY in MB) with an A6000 node and two
# A4500 nodes of different sizes, so a correct sizing must take the A4500 minimum, not the A6000 node.
MIXED_SINFO_TEXT = """\
HOSTNAMES           MEMORY              CPUS                GRES
gpunode2            125000              24                  gpu:rtx_a6000:1
gpunode15           29000               4                   gpu:rtx_a4500:1
gpunode16           31000               8                   gpu:rtx_a4500:1
"""
A4500_MINIMUM_MEMORY_MB = 29000
A4500_MINIMUM_CPUS = 4
A6000_MEMORY_MB = 125000
A6000_CPUS = 24


def assert_sbatch_sized_for(sbatch_remote_text: str, gres: str, mem_fraction: float, node_memory_mb: int,
                            node_cpus: int) -> None:
    """Check an sbatch line's --gres, --mem = floor(mem_fraction x node MB / 1024) GiB and -c.

    Args:
        sbatch_remote_text: The remote sbatch command text.
        gres: Expected --gres value.
        mem_fraction: The config's mem_fraction.
        node_memory_mb: Smallest RAM (MB) among the nodes of that GPU type.
        node_cpus: Smallest CPU count among them.
    """
    assert f"--gres={gres}" in sbatch_remote_text
    assert f"--mem={math.floor(mem_fraction * node_memory_mb / 1024)}G" in sbatch_remote_text
    assert f"-c {node_cpus}" in sbatch_remote_text


def test_section_gres_overrides_the_default():
    """A section's own gres wins over the top-level one."""
    submit_module = load_submit_module()
    config = {"gres": "gpu:rtx_a6000:1"}
    assert submit_module.resolve_job_gres(config, {"gres": "gpu:rtx_a4500:1"}) == "gpu:rtx_a4500:1"


@pytest.mark.parametrize("section_settings", [None, {}, {"gres": ""}, {"time_limit": "0-01:00:00"}])
def test_missing_section_gres_falls_back_to_the_default(section_settings):
    """No section, no key or an empty value -> the top-level gres."""
    submit_module = load_submit_module()
    assert submit_module.resolve_job_gres({"gres": "gpu:rtx_a6000:1"}, section_settings) == "gpu:rtx_a6000:1"


def test_malformed_section_gres_is_refused():
    """A gres without the gpu:<type>:<count> form fails before any ssh."""
    submit_module = load_submit_module()
    with pytest.raises(ValueError, match="gpu:<type>:<count>"):
        submit_module.resolve_job_gres({"gres": "gpu:rtx_a6000:1"}, {"gres": "gpu:1"})


def test_committed_config_gpu_choices():
    """Default (build-env, resume, full runs) A6000; render-only and smoke_a4500 on the A4500."""
    submit_module = load_submit_module()
    config = submit_module.load_slurm_config()
    profiles = config["profiles"]
    assert config["gres"] == "gpu:rtx_a6000:1"
    assert submit_module.resolve_job_gres(config, config["render_only"]) == "gpu:rtx_a4500:1"
    assert submit_module.resolve_job_gres(config, profiles["smoke_a4500"]) == "gpu:rtx_a4500:1"
    # Full runs and the plain smoke keep the default.
    for profile_name in ("smoke", "may_scene_a", "june_scene_a"):
        assert submit_module.resolve_job_gres(config, profiles[profile_name]) == "gpu:rtx_a6000:1"


def test_smoke_a4500_is_the_smoke_profile_plus_its_gres():
    """The YAML merge copies every smoke key; only gres is added."""
    profiles = load_submit_module().load_slurm_config()["profiles"]
    a4500_settings = dict(profiles["smoke_a4500"])
    assert a4500_settings.pop("gres") == "gpu:rtx_a4500:1"
    assert a4500_settings == profiles["smoke"]


def test_submit_run_sizes_from_the_profiles_gpu_nodes(monkeypatch, tmp_path):
    """smoke_a4500: sbatch gets the A4500 gres and the A4500 nodes' minimum CPUs and RAM."""
    submit_module = load_submit_module()
    issued_commands: List[List[str]] = []
    written_files: Dict[str, str] = {}
    base_fake = make_fake_cluster("b" * 40, "/w/example/user/viking-splatad",
                                  "env_prefix=/w/example/user/viking-splatad/envs/splatad-x\nbuilt_utc=t\n",
                                  issued_commands, written_files)

    def fake_run_command(command, capture_output=True, input_text=None):
        """Serve the mixed-GPU sinfo table; every other command goes to the shared fake."""
        if command[0] == "ssh" and command[-1].startswith("sinfo"):
            issued_commands.append(list(command))
            return subprocess.CompletedProcess(command, 0, stdout=MIXED_SINFO_TEXT, stderr="")
        return base_fake(command, capture_output, input_text)

    monkeypatch.setattr(submit_module, "run_command", fake_run_command)
    monkeypatch.setattr(submit_module, "read_committed_file", lambda git_sha, relative_path: b"x")
    monkeypatch.setattr(submit_module, "SUBMITTED_LEDGER_PATH", tmp_path / "submitted.tsv")
    monkeypatch.setattr(submit_module, "dataset_time_span", lambda session_dir: (5.0, 205.0))
    monkeypatch.delenv("VIKING_SLURM_USER", raising=False)
    config = submit_module.load_slurm_config()
    submit_module.submit_run(config, "may_short_grass", "smoke_a4500", skip_gates=True,
                             now_utc=datetime.datetime(2026, 10, 27, tzinfo=datetime.timezone.utc))
    sbatch_remote_text = next(command[-1] for command in issued_commands
                              if command[0] == "ssh" and " sbatch " in f" {command[-1]} ")
    # --mem = floor(mem_fraction x minimum A4500 MB / 1024) GiB, -c = minimum A4500 CPUs.
    assert_sbatch_sized_for(sbatch_remote_text, "gpu:rtx_a4500:1", config["mem_fraction"],
                            A4500_MINIMUM_MEMORY_MB, A4500_MINIMUM_CPUS)


def test_submit_render_uses_render_only_gres_and_its_nodes(monkeypatch, tmp_path):
    """render: sbatch gets render_only.gres (A4500) and the A4500 nodes' minimum CPUs and RAM."""
    # Imported here: test_render_only_job imports MIXED_SINFO_TEXT from this module.
    from tests.unit.test_render_only_job import (RUN_ID, SUCCESSFUL_JOB_INFO_TEXT, _prepare_submit_module,
                                                 _sbatch_commands, make_fake_render_cluster, synthetic_split_record)
    issued_commands: List[List[str]] = []
    submit_module = _prepare_submit_module(monkeypatch, tmp_path, make_fake_render_cluster(
        SUCCESSFUL_JOB_INFO_TEXT, synthetic_split_record(), True, issued_commands, {}))
    config = submit_module.load_slurm_config()
    submit_module.submit_render(config, RUN_ID, "3,5-6", "5", skip_gates=True)
    assert_sbatch_sized_for(_sbatch_commands(issued_commands)[0], config["render_only"]["gres"],
                            config["mem_fraction"], A4500_MINIMUM_MEMORY_MB, A4500_MINIMUM_CPUS)


def test_resume_keeps_the_default_gres_and_its_nodes(monkeypatch, tmp_path):
    """resume: the top-level gres (A6000) and the A6000 node's size, not render_only's A4500."""
    from tests.unit.test_render_memory_and_resume import FAILED_IN_RENDER_JOB_INFO_TEXT, _fake_resume_cluster
    from tests.unit.test_render_only_job import RUN_ID, _prepare_submit_module, _sbatch_commands
    issued_commands: List[List[str]] = []
    submit_module = _prepare_submit_module(monkeypatch, tmp_path, _fake_resume_cluster(
        FAILED_IN_RENDER_JOB_INFO_TEXT, True, "", issued_commands, {}))
    config = submit_module.load_slurm_config()
    submit_module.resume_run(config, RUN_ID, skip_gates=True)
    assert_sbatch_sized_for(_sbatch_commands(issued_commands)[0], config["gres"], config["mem_fraction"],
                            A6000_MEMORY_MB, A6000_CPUS)


def test_full_profiles_evaluate_every_400_steps():
    """20001 iterations, minimum 50 evals: k = (20001 - 1) // 50 = 400 (T63), 50 eval points."""
    submit_module = load_submit_module()
    profiles = submit_module.load_slurm_config()["profiles"]
    for profile_name in ("may_scene_a", "june_scene_a"):
        assert profiles[profile_name]["max_num_iterations"] == 20001
        assert submit_module.profile_eval_cadence(profiles[profile_name]) == (50, (20001 - 1) // 50)
