"""Write the exact train/test frames of one successful SplatAD run into its run folder (CPU, no torch).

SplatAD's split is deterministic (time window, then per-sensor LINSPACE or, with the dataparser's
``test_block_count``, middle test blocks shared by every sensor; no RNG; see
``viking_splatad.dataset.run_split``), so it is rebuilt from the run's recorded settings
(``job_info.txt``, ``config.yml``) and its dataset (``manifest.json``, ``poses.npz``, frame files),
then checked against the file names of the run's renders.

Only successful runs are documented. Nothing is written (and a stale record is removed) unless:
    * every render step finished (``lidar_render_elapsed_s`` in job_info.txt) and config.yml exists;
    * every split of RENDER_POSE_SOURCE has camera (rgb/) and lidar (lidar/) renders;
    * the train fraction agrees between job_info.txt (TRAIN_SPLIT_FRACTION), config.yml and the
      manifest (runs submitted before TRAIN_SPLIT_FRACTION existed: manifest only, because their
      config.yml holds the class default 0.5);
    * the test block count agrees between job_info.txt (TEST_BLOCK_COUNT) and config.yml
      (``test_block_count``; absent from both = the LINSPACE split of runs before T60);
    * the test block core fraction agrees between job_info.txt (TEST_BLOCK_CORE_FRACTION) and
      config.yml (``test_block_core_fraction``; absent from both = 1.0, runs before T62);
    * the rendered file names equal the rebuilt split.

Outputs, in ``<run dir>``:
    split.json        provenance (run, git SHA, seed, window, train fraction and its source,
                      test block count, block and core bounds, dataset hashes), per-sensor counts
                      (train / test / guard / val = frames evaluated during training: test + guard when
                      config.yml has eval_whole_test_blocks, else test), the eval stretches
                      (selection.eval_during_training_s), every frame with its split (guard = in a block, outside its core), the
                      membership digest and the render check. Deterministic: re-running on the
                      same inputs writes the same bytes.
    split_frames.csv  one row per frame: sensor, kind, split, frame_index, timestamp_ns, time_s, file.

Usage:
    # Locally, after `submit_run.py fetch <run_id>` (dataset default: data/intermediate/<SESSION>):
    uv run python scripts/09_extract_run_split.py --run-dir runs/<run_id>
    # On the cluster (the last step of splatad_run.sbatch, after every render):
    python scripts/09_extract_run_split.py --run-dir <run dir> --dataset-dir <stage>/data/<session>

Exit code 0 when the record was written; 1 when the renders or the train fractions disagree with
the rebuilt split; 2 when the run did not complete (failed runs are never documented).
"""

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from viking_splatad.dataset.run_split import (
    SPLIT_SENSOR_KINDS,
    GUARD_SPLIT_NAME,
    TEST_SPLIT_NAME,
    TRAIN_SPLIT_NAME,
    WHOLE_BLOCK_CORE_FRACTION,
    build_run_split_with_cores,
    compare_with_rendered_stems,
    file_sha256,
    resolve_run_train_fraction,
    split_digest,
)
# read_config_scalar / YAML_NULL_LITERALS moved to the package; the names stay available on this module.
from viking_splatad.io.nerfstudio_config import YAML_NULL_LITERALS, read_config_scalar  # noqa: F401
from viking_splatad.splatad.viking_conversions import open_dataset_view

# Repository root: this file lives in <root>/scripts/ (default dataset location is relative to it).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Local dataset location used by the converter (README "Outputs": data/intermediate/<session>/).
DEFAULT_DATASET_PARENT = PROJECT_ROOT / "data" / "intermediate"
# Version of the split.json layout; bump when keys change (documented in runs/RUN_FOLDER_GLOSSARY.md).
SPLIT_RECORD_SCHEMA = "viking-splatad/run-split/2"
# Output file names inside the run folder.
SPLIT_JSON_NAME = "split.json"
SPLIT_CSV_NAME = "split_frames.csv"
# job_info.txt keys this script needs (written by scripts/slurm/splatad_run.sbatch).
REQUIRED_JOB_INFO_KEYS = ("RUN_ID", "SESSION", "DATASET_START_FRACTION", "DATASET_END_FRACTION", "RENDER_POSE_SOURCE")
# job_info.txt key appended only after the last render step (lidar) succeeded: proof of a complete run.
RENDERS_COMPLETE_JOB_INFO_KEY = "lidar_render_elapsed_s"
# Separator of the splits in RENDER_POSE_SOURCE (e.g. "train+test").
POSE_SOURCE_SEPARATOR = "+"
# Exit codes: written, inconsistent (renders or fractions disagree), run not complete.
EXIT_WRITTEN, EXIT_INCONSISTENT, EXIT_INCOMPLETE = 0, 1, 2
# Render folder per sensor kind and the file extension each renderer writes there.
RENDER_FOLDER_BY_KIND = {"camera": ("rgb", ".jpg"), "lidar": ("lidar", ".npz")}
# Why the seed is recorded even though it does not choose frames (copied into split.json).
SEED_NOTE = (
    "TRAIN_SEED (--machine.seed) seeds training-time sampling only; train/test membership is "
    "deterministic (time window, then per-sensor LINSPACE or shared test blocks; no RNG) and does "
    "not depend on it."
)


def parse_job_info(job_info_path: Path) -> Dict[str, str]:
    """Read the top-level ``KEY: value`` lines of a run's ``job_info.txt``.

    Indented lines (the environment-build block) are skipped; the first occurrence of a key wins.

    Args:
        job_info_path: ``<run dir>/job_info.txt``.

    Returns:
        key -> value strings.
    """
    job_info: Dict[str, str] = {}
    for info_line in job_info_path.read_text().splitlines():
        if not info_line or info_line[0].isspace() or ": " not in info_line:
            continue
        info_key, info_value = info_line.split(": ", 1)
        job_info.setdefault(info_key.strip(), info_value.strip())
    return job_info


def list_rendered_stems(run_dir: Path) -> Dict[Tuple[str, str], List[str]]:
    """Stems of the rendered files per (sensor kind, split), for the render folders that exist.

    Args:
        run_dir: Run folder.

    Returns:
        (kind, split) -> sorted stems; folders that do not exist are left out.
    """
    rendered_stems = {}
    for sensor_kind in SPLIT_SENSOR_KINDS:
        folder_name, file_suffix = RENDER_FOLDER_BY_KIND[sensor_kind]
        for split_name in (TRAIN_SPLIT_NAME, TEST_SPLIT_NAME):
            render_folder = run_dir / "renders" / split_name / folder_name
            if render_folder.is_dir():
                rendered_stems[(sensor_kind, split_name)] = sorted(
                    rendered_path.stem for rendered_path in render_folder.glob(f"*{file_suffix}")
                )
    return rendered_stems


class RunNotDocumented(Exception):
    """The run is not documented; ``exit_code`` says why (incomplete or inconsistent)."""

    def __init__(self, message: str, exit_code: int):
        """Store the reason.

        Args:
            message: Human-readable reason, printed to stderr.
            exit_code: ``EXIT_INCOMPLETE`` or ``EXIT_INCONSISTENT``.
        """
        super().__init__(message)
        self.exit_code = exit_code


def resolve_checked_train_fraction(job_info: Dict[str, str], config_path: Path, manifest: dict) -> Tuple[float, str]:
    """The run's train fraction, after checking that every place recording it agrees.

    Args:
        job_info: Parsed job_info.txt.
        config_path: The run's config.yml.
        manifest: Parsed dataset manifest.

    Returns:
        (fraction, source description) for split.json.

    Raises:
        RunNotDocumented: (inconsistent) if job_info, config.yml and the dataparser's resolution disagree.
    """
    # Step 1: what the dataparser used (override, else the manifest; T27 resolve_train_fraction).
    override_text = read_config_scalar(config_path, "train_split_fraction_override")
    train_fraction, resolution_source = resolve_run_train_fraction(
        manifest, None if override_text is None else float(override_text)
    )
    # Step 2: legacy runs did not export TRAIN_SPLIT_FRACTION; their config.yml holds the 0.5 default.
    if "TRAIN_SPLIT_FRACTION" not in job_info:
        return train_fraction, f"{resolution_source} (legacy run: config.yml train_split_fraction is the class default)"
    # Step 3: newer runs record it twice more; all three values must be the same float.
    recorded_fractions = {
        "job_info.txt TRAIN_SPLIT_FRACTION": float(job_info["TRAIN_SPLIT_FRACTION"]),
        "config.yml train_split_fraction": float(read_config_scalar(config_path, "train_split_fraction") or "nan"),
        f"dataparser ({resolution_source})": train_fraction,
    }
    if len(set(recorded_fractions.values())) != 1:
        raise RunNotDocumented(f"train fractions disagree: {recorded_fractions}", EXIT_INCONSISTENT)
    return train_fraction, f"{resolution_source} (= job_info.txt TRAIN_SPLIT_FRACTION = config.yml)"


def resolve_checked_test_block_count(job_info: Dict[str, str], config_path: Path) -> Optional[int]:
    """The run's test block count (T60), after checking job_info.txt and config.yml agree.

    config.yml holds what the dataparser used (``test_block_count``, saved with the CLI value).
    job_info.txt records TEST_BLOCK_COUNT for runs submitted since T60; older runs have neither
    and used the LINSPACE split.

    Args:
        job_info: Parsed job_info.txt.
        config_path: The run's config.yml.

    Returns:
        K, or None for the LINSPACE split.

    Raises:
        RunNotDocumented: (inconsistent) if job_info.txt records a different value than config.yml.
    """
    config_text = read_config_scalar(config_path, "test_block_count")
    config_block_count = None if config_text is None else int(config_text)
    if "TEST_BLOCK_COUNT" in job_info and int(job_info["TEST_BLOCK_COUNT"]) != config_block_count:
        raise RunNotDocumented(
            f"test block counts disagree: job_info.txt TEST_BLOCK_COUNT {job_info['TEST_BLOCK_COUNT']}, "
            f"config.yml test_block_count {config_block_count}", EXIT_INCONSISTENT)
    return config_block_count


def resolve_checked_test_block_core_fraction(job_info: Dict[str, str], config_path: Path) -> float:
    """The run's test block core fraction (T62), after checking job_info.txt and config.yml agree.

    config.yml holds what the dataparser used (``test_block_core_fraction``). Runs from before T62
    have neither entry and used whole blocks (1.0).

    Args:
        job_info: Parsed job_info.txt.
        config_path: The run's config.yml.

    Returns:
        c in (0, 1].

    Raises:
        RunNotDocumented: (inconsistent) if job_info.txt records a different value than config.yml.
    """
    config_text = read_config_scalar(config_path, "test_block_core_fraction")
    config_core_fraction = WHOLE_BLOCK_CORE_FRACTION if config_text is None else float(config_text)
    if "TEST_BLOCK_CORE_FRACTION" in job_info and float(job_info["TEST_BLOCK_CORE_FRACTION"]) != config_core_fraction:
        raise RunNotDocumented(
            f"test block core fractions disagree: job_info.txt TEST_BLOCK_CORE_FRACTION "
            f"{job_info['TEST_BLOCK_CORE_FRACTION']}, config.yml test_block_core_fraction {config_core_fraction}",
            EXIT_INCONSISTENT)
    return config_core_fraction


def remove_stale_record(run_dir: Path) -> None:
    """Delete split.json / split_frames.csv left from an earlier extraction, so no run keeps a wrong record.

    Args:
        run_dir: Run folder.
    """
    for record_name in (SPLIT_JSON_NAME, SPLIT_CSV_NAME):
        (run_dir / record_name).unlink(missing_ok=True)


def build_split_record(run_dir: Path, dataset_dir_argument: Optional[Path]) -> Tuple[dict, list]:
    """Rebuild and verify the split of a completed run.

    Args:
        run_dir: Run folder.
        dataset_dir_argument: ``--dataset-dir``, or None for data/intermediate/<SESSION>.

    Returns:
        (split.json content, frames).

    Raises:
        RunNotDocumented: When the run is incomplete or its renders/fractions disagree.
    """
    # Step 1: the run must have finished every render step.
    job_info_path = run_dir / "job_info.txt"
    if not job_info_path.is_file():
        raise RunNotDocumented(f"{job_info_path} is missing", EXIT_INCOMPLETE)
    job_info = parse_job_info(job_info_path)
    missing_keys = [info_key for info_key in REQUIRED_JOB_INFO_KEYS if info_key not in job_info]
    if missing_keys:
        raise RunNotDocumented(f"{job_info_path} lacks {missing_keys}", EXIT_INCOMPLETE)
    if RENDERS_COMPLETE_JOB_INFO_KEY not in job_info:
        raise RunNotDocumented(f"run did not finish rendering (no {RENDERS_COMPLETE_JOB_INFO_KEY})", EXIT_INCOMPLETE)
    session = job_info["SESSION"]
    config_path = run_dir / "train" / session / "splatad" / job_info["RUN_ID"] / "config.yml"
    if not config_path.is_file():
        raise RunNotDocumented(f"{config_path} is missing", EXIT_INCOMPLETE)
    # Step 2: every rendered split must have camera and lidar outputs.
    rendered_splits = job_info["RENDER_POSE_SOURCE"].split(POSE_SOURCE_SEPARATOR)
    rendered_stems = list_rendered_stems(run_dir)
    empty_outputs = [f"renders/{split_name}/{RENDER_FOLDER_BY_KIND[sensor_kind][0]}"
                     for split_name in rendered_splits for sensor_kind in SPLIT_SENSOR_KINDS
                     if not rendered_stems.get((sensor_kind, split_name))]
    if empty_outputs:
        raise RunNotDocumented(f"no rendered output in {empty_outputs}", EXIT_INCOMPLETE)

    # Step 3: the dataset, read exactly as the dataparser reads it, and the checked fraction.
    dataset_dir = dataset_dir_argument or (DEFAULT_DATASET_PARENT / session)
    view = open_dataset_view(dataset_dir)
    train_fraction, train_fraction_source = resolve_checked_train_fraction(job_info, config_path, view.manifest)
    test_block_count = resolve_checked_test_block_count(job_info, config_path)
    test_block_core_fraction = resolve_checked_test_block_core_fraction(job_info, config_path)
    max_eval_text = read_config_scalar(config_path, "max_eval_frames")
    # Runs from before the overfitting monitor lack the field: their training eval covered the cores only.
    eval_whole_test_blocks = read_config_scalar(config_path, "eval_whole_test_blocks") == "true"
    start_fraction = float(job_info["DATASET_START_FRACTION"])
    end_fraction = float(job_info["DATASET_END_FRACTION"])

    # Step 4: rebuild the split; the renders must name exactly its frames.
    frames, window_bounds, test_blocks, test_cores = build_run_split_with_cores(
        view, start_fraction, end_fraction, train_fraction,
        max_eval_frames=None if max_eval_text is None else int(max_eval_text),
        test_block_count=test_block_count,
        test_block_core_fraction=test_block_core_fraction,
    )
    render_check = compare_with_rendered_stems(frames, rendered_stems)
    mismatched = {check_name: check for check_name, check in render_check.items() if not check["match"]}
    if mismatched:
        raise RunNotDocumented(f"renders disagree with the rebuilt split: {mismatched}", EXIT_INCONSISTENT)

    # Step 5: per-sensor counts (frames in the recording, in the window, per split). "val" is what
    # training evaluated: the whole test blocks (test + guard frames) or, for old runs, the cores (test).
    sensor_counts = {}
    for sensor in view.manifest["sensors"]:
        if sensor["kind"] not in SPLIT_SENSOR_KINDS:
            continue
        sensor_frames = [frame for frame in frames if frame.sensor == sensor["name"]]
        sensor_counts[sensor["name"]] = {
            "kind": sensor["kind"],
            "frames_in_recording": len(view.sensor_times[sensor["name"]]),
            # At fraction 1.0 a frame is in both splits, so count distinct files.
            "frames_in_window": len({frame.file for frame in sensor_frames}),
            TRAIN_SPLIT_NAME: sum(frame.split == TRAIN_SPLIT_NAME for frame in sensor_frames),
            TEST_SPLIT_NAME: sum(frame.split == TEST_SPLIT_NAME for frame in sensor_frames),
            GUARD_SPLIT_NAME: sum(frame.split == GUARD_SPLIT_NAME for frame in sensor_frames),
        }
        sensor_counts[sensor["name"]]["val"] = (
            sensor_counts[sensor["name"]][TEST_SPLIT_NAME] + sensor_counts[sensor["name"]][GUARD_SPLIT_NAME]
            if eval_whole_test_blocks else sensor_counts[sensor["name"]][TEST_SPLIT_NAME])

    # Step 6: the record (sorted keys and no wall-clock values keep it byte-reproducible).
    split_record = {
        "schema": SPLIT_RECORD_SCHEMA,
        "run": {
            "run_id": job_info["RUN_ID"],
            "session": session,
            "git_sha": job_info.get("GIT_SHA"),
            "train_seed": job_info.get("TRAIN_SEED"),
            "add_missing_points": job_info.get("ADD_MISSING_POINTS"),
        },
        "seed_note": SEED_NOTE,
        "selection": {
            "algorithm": (
                "neurad-studio ADDataParser: time window (T58), then per-sensor LINSPACE (T59)"
                if test_block_count is None else
                "time window (T58), then middle test blocks shared by every sensor (T60, VikingHillDataParser)"
            ),
            "test_block_count": test_block_count,
            "test_blocks_s": [list(block) for block in test_blocks],
            "test_block_core_fraction": test_block_core_fraction,
            "test_cores_s": [list(core) for core in test_cores],
            "eval_whole_test_blocks": eval_whole_test_blocks,
            # Time stretches (s) the training eval (val) covered.
            "eval_during_training_s": [list(stretch) for stretch in (test_blocks if eval_whole_test_blocks else test_cores)],
            "dataset_start_fraction": start_fraction,
            "dataset_end_fraction": end_fraction,
            "window_bounds_s": None if window_bounds is None else list(window_bounds),
            "train_fraction": train_fraction,
            "train_fraction_source": train_fraction_source,
            "max_eval_frames": None,
        },
        "dataset": {
            "name": dataset_dir.name,
            "manifest_sha256": file_sha256(dataset_dir / "manifest.json"),
            "poses_sha256": file_sha256(dataset_dir / "poses.npz"),
        },
        "counts": sensor_counts,
        "membership_sha256": split_digest(frames),
        "render_check": render_check,
        "frames": [asdict(frame) for frame in frames],
    }
    return split_record, frames


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point.

    Args:
        argv: Argument list (default sys.argv[1:]).

    Returns:
        EXIT_WRITTEN, EXIT_INCONSISTENT or EXIT_INCOMPLETE (see the module docstring).
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True, help="run folder holding job_info.txt")
    parser.add_argument("--dataset-dir", type=Path, default=None,
                        help="dataset the run trained on (default: data/intermediate/<SESSION>)")
    arguments = parser.parse_args(argv)
    run_dir: Path = arguments.run_dir

    # Step 1: build and verify; an undocumentable run leaves no record behind.
    try:
        split_record, frames = build_split_record(run_dir, arguments.dataset_dir)
    except RunNotDocumented as refusal:
        remove_stale_record(run_dir)
        print(f"not documented: {refusal}", file=sys.stderr)
        return refusal.exit_code

    # Step 2: write split.json and split_frames.csv.
    (run_dir / SPLIT_JSON_NAME).write_text(json.dumps(split_record, indent=1, sort_keys=True) + "\n")
    with open(run_dir / SPLIT_CSV_NAME, "w", newline="") as csv_handle:
        csv_writer = csv.DictWriter(csv_handle, fieldnames=list(asdict(frames[0]).keys()), lineterminator="\n")
        csv_writer.writeheader()
        csv_writer.writerows(asdict(frame) for frame in frames)

    # Step 3: human summary.
    for sensor_name, counts in split_record["counts"].items():
        print(f"{sensor_name}: {counts['frames_in_window']}/{counts['frames_in_recording']} frames in window, "
              f"train {counts[TRAIN_SPLIT_NAME]}, test {counts[TEST_SPLIT_NAME]}")
    selection = split_record["selection"]
    print(f"train fraction {selection['train_fraction']} ({selection['train_fraction_source']})")
    if selection["test_block_count"] is not None:
        block_text = ", ".join(f"[{block_start:.3f}, {block_end:.3f})" for block_start, block_end in selection["test_blocks_s"])
        print(f"{selection['test_block_count']} test blocks (s): {block_text}")
    print(f"membership sha256 {split_record['membership_sha256']}; renders match for "
          f"{', '.join(split_record['render_check'])}")
    print(f"wrote {run_dir / SPLIT_JSON_NAME} and {run_dir / SPLIT_CSV_NAME}")
    return EXIT_WRITTEN


if __name__ == "__main__":
    sys.exit(main())
