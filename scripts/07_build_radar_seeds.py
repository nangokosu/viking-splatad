"""Build calibrated train-only radar anchors: unpainted loss-only anchors (cluster) or painted init anchors (Mac).

The radar variants (``radar_variant`` profiles) build on the Mac with camera-painted seed colours;
loss-only profiles build on the cluster without colours. The artifact is built for the profile's
anchor profile (``radar_anchor_profile``, default the profile itself) so that it matches the split
settings every consuming profile verifies. Never appends model Gaussians itself.
"""

import argparse
import importlib.util
import json
from pathlib import Path

import yaml

from viking_splatad.radar_init.build import build_radar_anchor_artifact


def load_submit_helpers(code_root: Path):
    """Load ``scripts/slurm/submit_run.py`` of the given snapshot as a module.

    Args:
        code_root: Repository (or committed snapshot) root.

    Returns:
        The loaded helper module.
    """
    specification = importlib.util.spec_from_file_location("submit_helpers", code_root / "scripts/slurm/submit_run.py")
    helpers = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(helpers)
    return helpers


def main(argument_list=None):
    """Build an immutable artifact for one profile; return zero only after its full contract passes."""
    code_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--out", type=Path, default=None,
                        help="Artifact directory; default <dataset>/radar_inputs/<anchor profile>")
    parser.add_argument("--calibration", type=Path, help="Optional independent measured report, validated against exact source IDs")
    parser.add_argument("--radar-config", type=Path, default=code_root / "configs/radar/loss_only.yaml")
    arguments = parser.parse_args(argument_list)
    # Step 1: resolve the anchor profile and whether it (or the requesting profile) is a radar variant.
    helpers = load_submit_helpers(code_root)
    profiles = helpers.load_slurm_config(code_root / "configs/slurm.yaml")["profiles"]
    anchor_profile = profiles[arguments.profile].get("radar_anchor_profile", arguments.profile)
    paint_seed_colours = ("radar_variant" in profiles[arguments.profile]
                          or "radar_variant" in profiles[anchor_profile])
    # Step 2: split settings of the ANCHOR profile, as the consuming profiles verify them.
    session_config, settings = helpers.profile_radar_split_settings(
        code_root, arguments.dataset, arguments.session, anchor_profile)
    out_directory = arguments.out or arguments.dataset / "radar_inputs" / anchor_profile
    options = yaml.safe_load(arguments.radar_config.read_text())
    calibration = json.loads(arguments.calibration.read_text()) if arguments.calibration else None
    metadata = build_radar_anchor_artifact(arguments.dataset, session_config, settings, options,
                                           out_directory, calibration, paint_seed_colours=paint_seed_colours)
    if paint_seed_colours:
        print(f"painted_fraction: {metadata['painting']['painted_fraction']}", flush=True)
    print(json.dumps({key: metadata[key] for key in ("anchor_count", "training_radar_sweeps", "static_training_returns",
                                                    "voxel_size_m", "clock_calibration")}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
