"""Time `bake_gaussian_colours` on random Gaussians with a real run's RGB decoder.

Timing is machine-dependent: the JSON records the thread count and platform with it.

Run in the reference image (writes the JSON next to this script):
  VIKING_WRITABLE_DIR=docs/research/supersplat_export scripts/test_in_image.sh \
      python docs/research/supersplat_export/time_bake.py
"""
import argparse
import importlib.util
import json
import platform
import statistics
import time
from pathlib import Path

import torch

from viking_splatad.eval.gaussian_occupancy import latest_checkpoint_path
from viking_splatad.export.splatad_colours import (
    bake_gaussian_colours, camera_sensor_index, load_upstream_rgb_decoder, model_settings_from_config_yml)

DEFAULT_RUN_DIR = Path("runs/may_short_grass-may_scene_a-20261116-135432-17bb7a1")
DEFAULT_OUTPUT_PATH = Path(__file__).with_name("time_bake_output.json")
# Script 22 (export): its helpers resolve the train folder and load the checkpoint, so this measurement
# reads the run exactly as the export does.
EXPORT_SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "22_export_supersplat_ply.py"
# Random Gaussians baked per timing (order of the real export's size per chunk of work).
GAUSSIAN_COUNT = 100000
# View directions per Gaussian: 1 isolates the decoder cost per (Gaussian, direction) pair.
DIRECTION_COUNT = 1
# Chunk sizes compared (Gaussians decoded per decoder call).
CHUNK_SIZES = (4096, 65536)
# Seed of the random features and directions (any fixed value; it only makes the inputs repeatable).
SEED = 0
# Timed repeats per chunk size; the median is reported (robust to one slow outlier). An untimed warm-up
# bake runs first so that thread-pool start-up and first-call allocation are not counted.
TIMING_REPEAT_COUNT = 5


def load_export_script_module():
    """Import scripts/22 by path (its file name starts with a digit).

    Returns:
        The loaded module.
    """
    module_spec = importlib.util.spec_from_file_location("export_supersplat_ply_script", EXPORT_SCRIPT_PATH)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def time_one_bake(decoder, features_dc, features_rest, appearance_vector, directions, chunk_size) -> float:
    """Wall-clock seconds of one ``bake_gaussian_colours`` call.

    Args:
        decoder: Upstream RGB decoder.
        features_dc: (N, 3) albedo features.
        features_rest: (N, F) remaining features.
        appearance_vector: (A,) embedding row.
        directions: (M, 3) unit directions.
        chunk_size: Gaussians per decoder call.

    Returns:
        Elapsed seconds (``time.perf_counter``).
    """
    start_time = time.perf_counter()
    bake_gaussian_colours(decoder, features_dc, features_rest, appearance_vector, directions, chunk_size)
    return time.perf_counter() - start_time


def main() -> None:
    """Load the decoder, time each chunk size (warm-up, then repeats, median), print and write JSON."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    arguments = parser.parse_args()

    # Step 1: train folder and latest checkpoint exactly as script 22 finds them (job_info.txt, step-*.ckpt).
    export_module = load_export_script_module()
    train_dir = export_module.find_train_dir(arguments.run_dir)
    checkpoint_path = latest_checkpoint_path(train_dir / export_module.CHECKPOINT_DIR_NAME)
    settings = model_settings_from_config_yml(train_dir / export_module.CONFIG_FILE_NAME)
    tensors = export_module.load_checkpoint_tensors(checkpoint_path)
    # Step 2: real decoder and the first camera's appearance embedding (the export's choice).
    decoder, appearance_table = load_upstream_rgb_decoder(tensors["pipeline_state"], settings)
    appearance_vector = appearance_table[camera_sensor_index(settings["cameras"], settings["cameras"][0])]
    # Step 3: seeded random Gaussian features and unit view directions.
    generator = torch.Generator().manual_seed(SEED)
    features_dc = torch.rand(GAUSSIAN_COUNT, decoder.skip_dim, generator=generator)
    features_rest = torch.randn(GAUSSIAN_COUNT, settings["feature_dim"], generator=generator)
    directions = torch.nn.functional.normalize(torch.randn(DIRECTION_COUNT, 3, generator=generator), dim=-1)
    # Step 4: per chunk size one warm-up bake, then TIMING_REPEAT_COUNT timed bakes; report the median.
    timings = []
    for chunk_size in CHUNK_SIZES:
        bake_arguments = (decoder, features_dc, features_rest, appearance_vector, directions, chunk_size)
        time_one_bake(*bake_arguments)
        repeat_seconds = [time_one_bake(*bake_arguments) for _ in range(TIMING_REPEAT_COUNT)]
        median_seconds = statistics.median(repeat_seconds)
        timings.append({"chunk_size": chunk_size, "repeat_seconds": repeat_seconds,
                        "median_seconds": median_seconds,
                        "gaussians_per_second_at_median": GAUSSIAN_COUNT / median_seconds})
        print(f"chunk_size={chunk_size} gaussians={GAUSSIAN_COUNT} median_seconds={median_seconds:.2f} "
              f"of {TIMING_REPEAT_COUNT}")
    result = {
        "inputs": {"run_dir": arguments.run_dir.as_posix(), "checkpoint": checkpoint_path.name,
                   "gaussian_count": GAUSSIAN_COUNT, "direction_count": DIRECTION_COUNT, "seed": SEED,
                   "timing_repeat_count": TIMING_REPEAT_COUNT, "warm_up_bakes": 1},
        "machine": {"torch_threads": torch.get_num_threads(), "platform": platform.machine(),
                    "note": "timing is machine-dependent; inside the amd64 image on a Mac it is emulated"},
        "model_settings": dict(settings),
        "timings": timings,
    }
    arguments.output.write_text(json.dumps(result, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
