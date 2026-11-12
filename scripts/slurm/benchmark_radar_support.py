"""GPU support benchmark and allocation recommendations; never calibrates training loss weights."""

import argparse
import datetime
import importlib.util
import json
import os
from pathlib import Path
import resource
import time

import numpy as np
import torch

from viking_splatad.splatad.radar_lidar_loss import compute_radar_lidar_loss
from viking_splatad.splatad.radar_support_index import RadarSupportIndex
from viking_splatad.splatad.training_budget import measured_training_budgets


def main(argument_list=None):
    """Measure full-scene support twice on CUDA and atomically publish validated time budgets."""
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--baseline-run", required=True)
    parser.add_argument("--git-sha", required=True)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args(argument_list)
    if not os.environ.get("SLURM_JOB_ID") or root.name != arguments.git_sha:
        raise ValueError("benchmark must run under Slurm from its exact committed snapshot")
    specification = importlib.util.spec_from_file_location("submit_helpers", root / "scripts/slurm/submit_run.py")
    helpers = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(helpers)
    config = helpers.load_slurm_config(root / "configs/slurm.yaml")
    if helpers.RUN_ID_PATTERN.fullmatch(arguments.baseline_run) is None or helpers.RUN_ID_PATTERN.fullmatch(arguments.session) is None:
        raise ValueError("unsafe benchmark session/run identifier")
    torch.set_num_threads(int(os.environ["SLURM_CPUS_PER_TASK"]))
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 6):
        raise ValueError("benchmark requires a compatible sm_86 training GPU")
    baseline = arguments.stage_root / "runs" / arguments.baseline_run
    info = helpers.parse_job_info_text((baseline / "job_info.txt").read_text())
    if not info.get("end_utc") or info.get("split_render_check") != "match" or info.get("overfitting_plot") != "written":
        raise ValueError("benchmark baseline must have verified successful completion")
    start, end = (datetime.datetime.fromisoformat(info[key].replace("Z", "+00:00")) for key in ("start_utc", "end_utc"))
    baseline_seconds = (end - start).total_seconds()
    # Trained B positions probe worst-case full-scene support workload. They are
    # never used for the loss-only model's step-zero weight or initialization.
    with np.load(baseline / "renders/gaussians.npz", allow_pickle=False) as archive:
        means = torch.as_tensor(archive["means"], device="cuda")
    artifact = arguments.stage_root / "radar-inputs" / arguments.session / "june_loss_only" / arguments.git_sha
    with np.load(artifact / "radar_seeds.npz", allow_pickle=False) as archive:
        anchors = torch.as_tensor(archive["means"], dtype=means.dtype, device="cuda")
        covariances = torch.as_tensor(archive["covariances"], dtype=means.dtype, device="cuda")
    options = helpers.session_radar_lidar_options(arguments.session, arguments.stage_root / "data", root / "configs/sessions")
    torch.cuda.synchronize()
    started = time.monotonic()
    index = RadarSupportIndex(anchors, covariances, options["gate_quantile"], options["chunk_size"])
    torch.cuda.synchronize()
    index_seconds = time.monotonic() - started
    logits, static = torch.zeros(len(means), device="cuda"), torch.ones(len(means), device="cuda", dtype=torch.bool)
    passes = []
    # A cold and a warm pass expose startup variation; use the slower pass for budgets.
    for pass_index in range(2):
        torch.cuda.synchronize()
        started = time.monotonic()
        result = compute_radar_lidar_loss(means, logits, static, anchors, covariances, 1., options["gate_quantile"],
                                         options["chunk_size"], support_index=index)
        torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        if result.empty_support or not torch.isfinite(result.loss).item():
            raise ValueError("GPU support benchmark has no finite supported loss")
        measured = {"pass": pass_index, "seconds": elapsed, "candidate_pairs": index.last_candidate_pairs,
                    "supported_gaussians": result.supported_gaussian_count,
                    "supported_seed_fraction": result.supported_seed_fraction}
        passes.append(measured)
        print(json.dumps(measured), flush=True)
    profiles = {name: config["profiles"][name]["max_num_iterations"] for name in ("june_loss_only_smoke", "june_loss_only")}
    budgets = measured_training_budgets(baseline_seconds, max(record["seconds"] for record in passes), index_seconds,
                                         profiles, config["radar_runtime_budget"]["headroom_factor"], helpers.MAXIMUM_TIME_LIMIT_SECONDS)
    report = {"git_sha": arguments.git_sha, "benchmark_job_id": os.environ["SLURM_JOB_ID"],
              "gpu_forward_validated": True, "scope": "trained B geometry workload; not initial weight calibration",
              "gaussians": len(means), "anchors": len(anchors), "index_seconds": index_seconds,
              "passes": passes, "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "baseline_wall_seconds": baseline_seconds, "headroom_factor": config["radar_runtime_budget"]["headroom_factor"],
              "budgets": budgets}
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.out.with_suffix(".partial")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(arguments.out)
    print(json.dumps(report), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
