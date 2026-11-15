"""CPU validation preserves training allocation policy and excludes preflight queue records."""

import subprocess

import pytest

from tests.unit.test_slurm_submit import load_submit_module


def test_cpu_validation_is_bounded_by_live_capacity(monkeypatch):
    """Select the smaller of the measured validation budget and every CPU node's capacity."""
    helpers = load_submit_module()
    config = helpers.load_slurm_config()
    source_cpus, source_memory = 4, 25  # Synthetic measured render allocation.
    node_cpus, node_megabytes = 2, 16384  # Synthetic smaller CPU node.
    monkeypatch.setattr(helpers, "query_node_resources", lambda *arguments: (source_cpus, source_memory))
    table = f"HOSTNAMES MEMORY CPUS\nnode-a {node_megabytes} {node_cpus}\nnode-b 32768 8\n"
    monkeypatch.setattr(helpers, "run_checked", lambda *arguments: subprocess.CompletedProcess([], 0, table))
    allocation = helpers.query_validation_allocation(config, "cluster", config["validation_cpu"])
    expected_memory = int(config["mem_fraction"] * node_megabytes / helpers.MEGABYTES_PER_GIGABYTE)
    assert allocation == ("cpunodes", None, min(source_cpus, node_cpus), min(source_memory, expected_memory))


def test_cpu_batch_has_no_gpu_request():
    """A CPU validation submission cannot reserve a GPU implicitly."""
    helpers = load_submit_module()
    command = helpers.build_sbatch_command("cpunodes", None, 4, 25, "0-00:15:00", "checks",
                                           "/stage/log-%j.out", "/stage/job.sbatch")
    assert "--partition=cpunodes" in command
    assert not any(argument.startswith("--gres") for argument in command)


def test_training_rejects_cpu_validation_override(monkeypatch):
    """Training keeps its configured GPU even when a CPU preflight was reviewed."""
    helpers = load_submit_module()
    monkeypatch.setattr(helpers, "check_clean_tree", lambda: [])
    with pytest.raises(ValueError, match="cannot be used for training"):
        helpers.submit_remote_run(helpers.load_slurm_config(), "june_tall_grass", "june_scene_a",
                                  False, reviewed_preflight=True, resource_key="validation_cpu")


def test_preflight_job_is_not_a_training_run():
    """Shared job.sbatch basenames cannot turn source SHAs into training run IDs."""
    helpers = load_submit_module()
    preflight = "/stage/preflights/session/profile/source-sha/job.sbatch"
    assert helpers.job_script_owner(preflight) is None
    assert helpers.parse_tracked_jobs(f"123 PENDING Dependency N/A {preflight}") == []
    assert helpers.job_script_owner("/stage/runs/actual-run/job.sbatch") == ("actual-run", "train")
