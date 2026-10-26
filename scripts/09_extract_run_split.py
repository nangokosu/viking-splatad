"""Write the exact train/test frames of one SplatAD run into its run folder (CPU, no torch).

SplatAD's split is deterministic (time window + per-sensor LINSPACE, no RNG; see
``viking_splatad.dataset.run_split``), so it is rebuilt from the run's recorded settings
(``job_info.txt``, ``config.yml``) and its dataset (``manifest.json``, ``poses.npz``, frame files).
When the run has renders, their file names are checked against the rebuilt split.

Outputs, in ``<run dir>``:
    split.json        provenance (run, git SHA, seed, window, train fraction and its source,
                      dataset hashes), per-sensor counts, every frame with its split, the
                      membership digest and the render check. Deterministic: re-running on the
                      same inputs writes the same bytes.
    split_frames.csv  one row per frame: sensor, kind, split, frame_index, timestamp_ns, time_s, file.

Usage:
    # Locally, after `submit_run.py fetch <run_id>` (dataset default: data/intermediate/<SESSION>):
    uv run python scripts/09_extract_run_split.py --run-dir runs/<run_id>
    # On the cluster (splatad_run.sbatch runs this before training and after rendering):
    python scripts/09_extract_run_split.py --run-dir <run dir> --dataset-dir <stage>/data/<session>

Exit code 0 on success; 1 if the renders disagree with the rebuilt split.
"""

import argparse
import csv
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from viking_splatad.dataset.run_split import (
    SPLIT_SENSOR_KINDS,
    TEST_SPLIT_NAME,
    TRAIN_SPLIT_NAME,
    build_run_split,
    compare_with_rendered_stems,
    file_sha256,
    resolve_run_train_fraction,
    split_digest,
)
from viking_splatad.splatad.viking_conversions import open_dataset_view

# Repository root: this file lives in <root>/scripts/ (default dataset location is relative to it).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Local dataset location used by the converter (README "Outputs": data/intermediate/<session>/).
DEFAULT_DATASET_PARENT = PROJECT_ROOT / "data" / "intermediate"
# Version of the split.json layout; bump when keys change (documented in runs/RUN_FOLDER_GLOSSARY.md).
SPLIT_RECORD_SCHEMA = "viking-splatad/run-split/1"
# Output file names inside the run folder.
SPLIT_JSON_NAME = "split.json"
SPLIT_CSV_NAME = "split_frames.csv"
# job_info.txt keys this script needs (written by scripts/slurm/splatad_run.sbatch).
REQUIRED_JOB_INFO_KEYS = ("RUN_ID", "SESSION", "DATASET_START_FRACTION", "DATASET_END_FRACTION")
# Render folder per sensor kind and the file extension each renderer writes there.
RENDER_FOLDER_BY_KIND = {"camera": ("rgb", ".jpg"), "lidar": ("lidar", ".npz")}
# YAML literal nerfstudio writes for None.
YAML_NULL_LITERALS = ("null", "~", "None")
# Why the seed is recorded even though it does not choose frames (copied into split.json).
SEED_NOTE = (
    "TRAIN_SEED (--machine.seed) seeds training-time sampling only; train/test membership is "
    "deterministic (time window + per-sensor LINSPACE, no RNG) and does not depend on it."
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


def read_config_scalar(config_path: Path, field_name: str) -> Optional[str]:
    """Read one scalar field of nerfstudio's ``config.yml`` without constructing its Python objects.

    config.yml holds ``!!python/object`` tags that need nerfstudio to load, so the field is read
    with a line regex instead (field names here are unique within the dataparser config).

    Args:
        config_path: The run's ``config.yml``.
        field_name: Field such as ``train_split_fraction_override``.

    Returns:
        The raw value string, None for a YAML null or an absent field/file.
    """
    if not config_path.is_file():
        return None
    field_match = re.search(rf"^\s*{re.escape(field_name)}:\s*(\S+)\s*$", config_path.read_text(), flags=re.MULTILINE)
    if field_match is None or field_match.group(1) in YAML_NULL_LITERALS:
        return None
    return field_match.group(1)


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


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point.

    Args:
        argv: Argument list (default sys.argv[1:]).

    Returns:
        0 on success, 1 when renders disagree with the rebuilt split.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True, help="run folder holding job_info.txt")
    parser.add_argument("--dataset-dir", type=Path, default=None,
                        help="dataset the run trained on (default: data/intermediate/<SESSION>)")
    arguments = parser.parse_args(argv)
    run_dir: Path = arguments.run_dir

    # Step 1: the run's recorded settings.
    job_info = parse_job_info(run_dir / "job_info.txt")
    missing_keys = [info_key for info_key in REQUIRED_JOB_INFO_KEYS if info_key not in job_info]
    if missing_keys:
        raise SystemExit(f"{run_dir / 'job_info.txt'} lacks {missing_keys}")
    session = job_info["SESSION"]
    start_fraction = float(job_info["DATASET_START_FRACTION"])
    end_fraction = float(job_info["DATASET_END_FRACTION"])
    # config.yml exists once ns-train started; before that, the dataparser defaults apply (None).
    config_path = run_dir / "train" / session / "splatad" / job_info["RUN_ID"] / "config.yml"
    override_text = read_config_scalar(config_path, "train_split_fraction_override")
    max_eval_text = read_config_scalar(config_path, "max_eval_frames")

    # Step 2: the dataset, read exactly as the dataparser reads it.
    dataset_dir: Path = arguments.dataset_dir or (DEFAULT_DATASET_PARENT / session)
    view = open_dataset_view(dataset_dir)
    train_fraction, train_fraction_source = resolve_run_train_fraction(
        view.manifest, None if override_text is None else float(override_text)
    )

    # Step 3: rebuild the split and fingerprint it.
    frames, window_bounds = build_run_split(
        view, start_fraction, end_fraction, train_fraction,
        max_eval_frames=None if max_eval_text is None else int(max_eval_text),
    )
    membership_digest = split_digest(frames)

    # Step 4: per-sensor counts (frames in the recording, in the window, per split).
    sensor_counts = {}
    for sensor in view.manifest["sensors"]:
        if sensor["kind"] not in SPLIT_SENSOR_KINDS:
            continue
        sensor_frames = [frame for frame in frames if frame.sensor == sensor["name"]]
        train_count = sum(frame.split == TRAIN_SPLIT_NAME for frame in sensor_frames)
        test_count = sum(frame.split == TEST_SPLIT_NAME for frame in sensor_frames)
        sensor_counts[sensor["name"]] = {
            "kind": sensor["kind"],
            "frames_in_recording": len(view.sensor_times[sensor["name"]]),
            # At fraction 1.0 every windowed frame is in both splits, so train alone is the window.
            "frames_in_window": len({frame.file for frame in sensor_frames}),
            TRAIN_SPLIT_NAME: train_count,
            TEST_SPLIT_NAME: test_count,
        }

    # Step 5: compare with the renders, when the run has any.
    render_check = compare_with_rendered_stems(frames, list_rendered_stems(run_dir))
    renders_agree = all(check["match"] for check in render_check.values())

    # Step 6: write the deterministic record (sorted keys, no wall-clock values).
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
            "algorithm": "neurad-studio ADDataParser: time window (T58), then per-sensor LINSPACE (T59)",
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
        "membership_sha256": membership_digest,
        "render_check": render_check,
        "frames": [asdict(frame) for frame in frames],
    }
    (run_dir / SPLIT_JSON_NAME).write_text(json.dumps(split_record, indent=1, sort_keys=True) + "\n")
    with open(run_dir / SPLIT_CSV_NAME, "w", newline="") as csv_handle:
        csv_writer = csv.DictWriter(csv_handle, fieldnames=list(asdict(frames[0]).keys()) if frames else [],
                                    lineterminator="\n")
        csv_writer.writeheader()
        csv_writer.writerows(asdict(frame) for frame in frames)

    # Step 7: human summary.
    for sensor_name, counts in sensor_counts.items():
        print(f"{sensor_name}: {counts['frames_in_window']}/{counts['frames_in_recording']} frames in window, "
              f"train {counts[TRAIN_SPLIT_NAME]}, test {counts[TEST_SPLIT_NAME]}")
    print(f"train fraction {train_fraction} ({train_fraction_source}); membership sha256 {membership_digest}")
    for check_name, check in render_check.items():
        print(f"renders {check_name}: {'match' if check['match'] else 'MISMATCH'}"
              f" (missing {len(check['missing_from_renders'])}, unexpected {len(check['unexpected_in_renders'])})")
    print(f"wrote {run_dir / SPLIT_JSON_NAME} and {run_dir / SPLIT_CSV_NAME}")
    return 0 if renders_agree else 1


if __name__ == "__main__":
    sys.exit(main())
