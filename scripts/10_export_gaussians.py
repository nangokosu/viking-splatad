"""Export a trained SplatAD run's Gaussians to a map-frame ``gaussians.npz`` (T56, P2 input).

The rendered lidar (``08``) shows only what a sensor sees first; this export holds every static
Gaussian of the model, so the P2 occupancy grid can be compared with the GT cuboid grid without
the visibility limit (docs/research/evaluation_caveats.md §5).

Steps:
    1. Pick the latest ``step-*.ckpt`` in ``<train dir>/nerfstudio_models`` (the one ``08`` renders).
    2. ``torch.load`` it on the CPU and convert the pipeline state dict to numpy.
    3. Read w2m and the scale from ``<train dir>/dataparser_transforms.json`` (written by
       ns-train, trainer.py:288-292), so no dataparser or dataset is needed.
    4. Decode and move the Gaussians to the map frame (T56) and write the npz
       (keys: viking_splatad.eval.gaussian_occupancy.GAUSSIANS_NPZ_KEYS, see
       runs/RUN_FOLDER_GLOSSARY.md).

Needs torch, so it runs in the SplatAD environment: as a job step (scripts/slurm/splatad_run.sbatch)
or in the Docker image on a fetched run. CPU only.

Usage:
    python scripts/10_export_gaussians.py --train-dir <run>/train/<session>/splatad/<run_id> \
        --output-path <run>/renders/gaussians.npz
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

# File ns-train writes next to config.yml with the dataparser's transform, scale and time offset.
DATAPARSER_TRANSFORMS_FILE_NAME = "dataparser_transforms.json"
# Checkpoint folder inside the train dir (nerfstudio trainer default).
CHECKPOINT_DIR_NAME = "nerfstudio_models"


def build_argument_parser() -> argparse.ArgumentParser:
    """Command-line interface.

    Returns:
        The parser.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train-dir", type=Path, required=True,
                        help="ns-train output dir holding config.yml, dataparser_transforms.json, nerfstudio_models/.")
    parser.add_argument("--output-path", type=Path, required=True, help="Destination gaussians.npz.")
    return parser


def main(argument_list: Optional[List[str]] = None) -> int:
    """Export the latest checkpoint's Gaussians.

    Args:
        argument_list: CLI arguments (None = sys.argv).

    Returns:
        Process exit code (0 on success).
    """
    from viking_splatad.eval.gaussian_occupancy import (  # pylint: disable=import-outside-toplevel
        ACTOR_SIZES_KEY,
        GAUSS_PARAM_NAMES,
        GAUSS_PARAMS_KEY_PREFIX,
        checkpoint_gaussians_to_map_frame,
        gauss_params_from_state_dict,
        latest_checkpoint_path,
        save_gaussians_npz,
    )

    arguments = build_argument_parser().parse_args(argument_list)
    # Step 1: the checkpoint 08 also loads (highest step).
    checkpoint_path = latest_checkpoint_path(arguments.train_dir / CHECKPOINT_DIR_NAME)
    # Step 2: torch only here, so --help and tests work without it. map_location as eval_utils.py:63.
    import torch  # pylint: disable=import-outside-toplevel

    loaded_state = torch.load(checkpoint_path, map_location="cpu")
    # Only the tensors T56 reads go to numpy (others, e.g. decoder weights, are not needed).
    needed_key_suffixes = tuple([GAUSS_PARAMS_KEY_PREFIX + param_name for param_name in GAUSS_PARAM_NAMES]
                                + [ACTOR_SIZES_KEY])
    pipeline_state = {state_key: value.detach().cpu().numpy()
                      for state_key, value in loaded_state["pipeline"].items()
                      if state_key.endswith(needed_key_suffixes)}
    gauss_params = gauss_params_from_state_dict(pipeline_state)
    # Step 3: the frame the means are stored in.
    with open(arguments.train_dir / DATAPARSER_TRANSFORMS_FILE_NAME, encoding="utf-8") as transforms_file:
        dataparser_transforms = json.load(transforms_file)
    dataparser_transform = np.asarray(dataparser_transforms["transform"], dtype=np.float64)
    # Step 4: decode, move to the map frame, save.
    gaussians = checkpoint_gaussians_to_map_frame(
        means=gauss_params["means"],
        log_scales=gauss_params["scales"],
        quaternions_wxyz=gauss_params["quats"],
        opacity_logits=gauss_params["opacities"],
        actor_ids=gauss_params["id"],
        actor_count=gauss_params["actor_count"],
        dataparser_transform=dataparser_transform,
        dataparser_scale=float(dataparser_transforms["scale"]),
    )
    checkpoint_step = int(loaded_state["step"])
    save_gaussians_npz(arguments.output_path, gaussians, dataparser_transform, checkpoint_step)
    print(f"wrote {len(gaussians.opacities)} Gaussians (step {checkpoint_step}, {checkpoint_path.name}) "
          f"to {arguments.output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
