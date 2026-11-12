"""Build calibrated train-only loss anchors on the cluster; never append model Gaussians."""

import argparse
import importlib.util
import json
from pathlib import Path

import yaml

from viking_splatad.radar_init.build import build_loss_only_artifact


def profile_settings(code_root: Path, dataset: Path, session: str, profile: str):
    """Resolve the exact Slurm profile/window split using the submission helper's own rules.

    Args:
        code_root: Committed snapshot; dataset: published session inputs.
        session, profile: Existing canonical identifiers.

    Returns:
        Session configuration, exact artifact split kwargs.
    """
    specification = importlib.util.spec_from_file_location("submit_helpers", code_root / "scripts/slurm/submit_run.py")
    helpers = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(helpers)
    config = helpers.load_slurm_config(code_root / "configs/slurm.yaml")
    selected = config["profiles"][profile]
    session_directory = code_root / "configs/sessions"
    _, session_config = helpers.load_dataset_session_config(session, dataset.parent, session_directory)
    train_fraction, _, core_fraction = helpers.profile_split_fractions(selected, session, dataset.parent, session_directory)
    start, end = 0., 1.
    if "window_seconds" in selected or "window_start_seconds" in selected:
        start, end = helpers.window_fractions(*helpers.dataset_time_span(dataset),
                                              selected["window_start_seconds"], selected["window_seconds"])
    return session_config, {"start_fraction": start, "end_fraction": end, "train_fraction": train_fraction,
        "test_block_count": helpers.session_test_block_count(session, dataset.parent, session_directory),
        "test_block_core_fraction": core_fraction,
        "test_split_method": helpers.session_test_split_method(session, dataset.parent, session_directory),
        "test_render_fraction": helpers.profile_test_render_fraction(selected, session, dataset.parent, session_directory)}


def main(argument_list=None):
    """Build an immutable artifact for one profile; return zero only after its full contract passes."""
    code_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, help="Optional independent measured report, validated against exact source IDs")
    parser.add_argument("--radar-config", type=Path, default=code_root / "configs/radar/loss_only.yaml")
    arguments = parser.parse_args(argument_list)
    session_config, settings = profile_settings(code_root, arguments.dataset, arguments.session, arguments.profile)
    options = yaml.safe_load(arguments.radar_config.read_text())
    calibration = json.loads(arguments.calibration.read_text()) if arguments.calibration else None
    metadata = build_loss_only_artifact(arguments.dataset, session_config, settings, options, arguments.out, calibration)
    print(json.dumps({key: metadata[key] for key in ("anchor_count", "training_radar_sweeps", "static_training_returns",
                                                    "voxel_size_m", "clock_calibration")}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
