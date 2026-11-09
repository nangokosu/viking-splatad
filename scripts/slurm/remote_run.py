"""Validate workdisk datasets and launch training without downloading inputs to the Mac."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml


def file_digest(path: Path) -> str:
    """Return SHA256 of a provenance file; missing files fail rather than imply approval."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_remote(stage_root: Path, code_dir: Path, env_prefix: Path, session: str,
               profile_name: str, git_sha: str, run_id: str, preflight_only: bool) -> None:
    """Validate or train one profile using workdisk data and the pinned native env.

    Preflight runs full tests, contract, reference split, and overlay generation.
    Training requires the same provenance plus explicit review at submission time.
    Paths and names are supplied by the submit host; no raw data is returned there.
    """
    specification = importlib.util.spec_from_file_location("submit_helpers", code_dir / "scripts/slurm/submit_run.py")
    helpers = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(helpers)
    config = helpers.load_slurm_config(code_dir / "configs/slurm.yaml")
    profile = config["profiles"][profile_name]
    dataset_dir = stage_root / "data" / session
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"remote dataset is not ready: {dataset_dir}")
    settings_dir = code_dir / "configs/sessions"
    session_name, session_config = helpers.load_dataset_session_config(session, stage_root / "data", settings_dir)
    helpers.require_no_return_sidecar(dataset_dir, helpers.session_add_missing_points(session, stage_root / "data", settings_dir))
    preflight_dir = stage_root / "preflights" / session / profile_name / git_sha
    receipt_path = preflight_dir / "validation.json"
    provenance = {
        "git_sha": git_sha, "environment": env_prefix.name, "session": session,
        "profile": profile_name, "manifest_sha256": file_digest(dataset_dir / "manifest.json"),
        "poses_sha256": file_digest(dataset_dir / "poses.npz"),
        "session_config_sha256": file_digest(settings_dir / f"{session_name}.yaml"),
        "profile_sha256": hashlib.sha256(json.dumps(profile, sort_keys=True).encode()).hexdigest(),
        "validation_backend": "native-cluster",
    }
    # Derive window fractions here, where the actual timestamps exist.
    start_fraction, end_fraction = 0.0, 1.0
    if "window_start_seconds" in profile or "window_seconds" in profile:
        start_fraction, end_fraction = helpers.window_fractions(
            *helpers.dataset_time_span(dataset_dir), profile["window_start_seconds"], profile["window_seconds"],
        )
    if preflight_only:
        preflight_dir.mkdir(parents=True, exist_ok=True)
        # Remove any old approval before starting checks; failed reruns cannot reuse it.
        receipt_path.unlink(missing_ok=True)
        subprocess.run([
            sys.executable, str(code_dir / "scripts/slurm/prepare_test_inputs.py"),
            "--code-dir", str(code_dir), "--stage-root", str(stage_root), "--env-prefix", str(env_prefix),
        ], check=True)
        subprocess.run([sys.executable, "-m", "pytest", "-q", "-r", "a", "-p", "no:cacheprovider"], cwd=code_dir, check=True)
        subprocess.run([
            sys.executable, str(code_dir / "scripts/06_validate_contract.py"),
            "--dataset", str(dataset_dir), "--report", str(preflight_dir / "contract.md"),
        ], check=True)
        if (code_dir / "configs/reference_splits" / f"{session_name}.json").is_file():
            subprocess.run([
                sys.executable, str(code_dir / "scripts/16_check_reference_split.py"),
                "--session", session_name, "--dataset-dir", str(dataset_dir),
            ], check=True)
        subprocess.run([
            sys.executable, str(code_dir / "scripts/slurm/preflight_dataparser.py"),
            "--data", str(stage_root / "data"), "--session-name", session,
            "--out", str(preflight_dir / "overlays"),
            "--dataset-start-fraction", str(start_fraction), "--dataset-end-fraction", str(end_fraction),
        ], check=True)
        overlay_files = sorted(path for path in (preflight_dir / "overlays").rglob("*") if path.is_file())
        if not overlay_files:
            raise RuntimeError("preflight produced no overlays for review")
        receipt = dict(provenance, overlays={str(path.relative_to(preflight_dir)): file_digest(path) for path in overlay_files})
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        print(f"remote preflight complete: {preflight_dir}", flush=True)
        return
    receipt = json.loads(receipt_path.read_text())
    if any(receipt.get(key) != value for key, value in provenance.items()):
        raise ValueError("preflight provenance differs from the current code, profile, environment, or dataset")
    if not receipt.get("overlays"):
        raise ValueError("preflight has no reviewed overlays")
    for relative_path, digest in receipt["overlays"].items():
        overlay_path = (preflight_dir / relative_path).resolve()
        if not overlay_path.is_relative_to(preflight_dir.resolve()) or file_digest(overlay_path) != digest:
            raise ValueError("preflight overlays changed since validation")
    train_fraction, override, core_fraction = helpers.profile_split_fractions(profile, session, stage_root / "data", settings_dir)
    minimum_eval_points, eval_interval = helpers.profile_eval_cadence(profile)
    # build_job_environment otherwise tries to open local data for smoke windows.
    whole_recording_profile = {key: value for key, value in profile.items() if key not in {"window_start_seconds", "window_seconds"}}
    environment = helpers.build_job_environment(
        whole_recording_profile, session, run_id, git_sha, str(stage_root), {"env_prefix": str(env_prefix)},
        add_missing_points=helpers.session_add_missing_points(session, stage_root / "data", settings_dir),
        train_split_fraction=train_fraction, train_split_fraction_override=override,
        test_block_count=helpers.session_test_block_count(session, stage_root / "data", settings_dir),
        test_block_core_fraction=core_fraction,
        test_split_method=helpers.session_test_split_method(session, stage_root / "data", settings_dir),
        test_render_fraction=helpers.profile_test_render_fraction(profile, session, stage_root / "data", settings_dir),
        minimum_eval_points=minimum_eval_points, steps_per_eval_all_images=eval_interval,
        radar_lidar_options=helpers.session_radar_lidar_options(session, stage_root / "data", settings_dir),
    )
    environment.update(DATASET_START_FRACTION=repr(start_fraction), DATASET_END_FRACTION=repr(end_fraction))
    # Preserve the fully resolved variables in the standard job.sbatch location.
    # Run tracking/recovery must not have to reconstruct them from a profile that
    # may later change. Slurm has already copied the submitted wrapper at this point.
    run_dir = stage_root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    job_path = run_dir / "job.sbatch"
    if job_path.is_file():
        (run_dir / "submission.sbatch").write_text(job_path.read_text())
    resolved_job = run_dir / "job.sbatch.resolved"
    resolved_job.write_text(helpers.build_job_script(environment, str(code_dir / "scripts/slurm/splatad_run.sbatch")))
    resolved_job.replace(job_path)
    (run_dir / "validation.json").write_text(json.dumps(receipt, indent=2) + "\n")
    process_environment = dict(os.environ, **environment)
    os.execve("/bin/bash", ["bash", str(code_dir / "scripts/slurm/splatad_run.sbatch")], process_environment)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("stage-root", "code-dir", "env-prefix"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("session", "profile", "git-sha", "run-id"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    arguments = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("remote validation/training must run under Slurm")
    run_remote(arguments.stage_root, arguments.code_dir, arguments.env_prefix, arguments.session,
               arguments.profile, arguments.git_sha, arguments.run_id, arguments.preflight_only)
