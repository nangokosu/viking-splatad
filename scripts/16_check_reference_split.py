"""Check that a dataset yields exactly the committed reference train/test/guard split of its session.

The split is deterministic, so anyone who re-creates a session's bag data (``scripts/03_extract.py``
with the session YAML's pinned ``window.chunk_files``) must get the split of the reference run. This
rebuilds the split from the dataset with the SESSION CONFIG values (the same values
``scripts/slurm/submit_run.py`` passes to the dataparser) and compares it, frame by ``timestamp_ns``,
with ``configs/reference_splits/<session>.json``.

Usage:
    # Check (default dataset: data/intermediate/<session>):
    uv run python scripts/16_check_reference_split.py --session june_tall_grass [--dataset-dir D]
    # Write the record from a fetched run folder (its split.json), then commit it:
    uv run python scripts/16_check_reference_split.py --session june_tall_grass \
        --write-from-run runs/<run_id>

Exit code 0 when identical; 1 when different (the differences are printed); 2 when the session has no
reference record (not an error: e.g. the smoke datasets have none).
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from viking_splatad.config import load_session_config
from viking_splatad.dataset.reference_split import (
    REFERENCE_SETTINGS_KEYS,
    build_reference_record,
    compare_with_reference,
)
from viking_splatad.dataset.run_split import SplitFrame, build_run_split_with_cores, split_digest
from viking_splatad.splatad.viking_conversions import open_dataset_view

# Repository root: this file lives in <root>/scripts/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SESSION_CONFIG_DIRECTORY = PROJECT_ROOT / "configs" / "sessions"
REFERENCE_SPLIT_DIRECTORY = PROJECT_ROOT / "configs" / "reference_splits"
DEFAULT_DATASET_PARENT = PROJECT_ROOT / "data" / "intermediate"
# Run folder file holding every frame with its split (written by scripts/09_extract_run_split.py).
RUN_SPLIT_JSON_NAME = "split.json"
# The reference runs use the whole extracted recording: the profile sets no window (the session config
# already cuts it to the scene), so the dataparser's window fractions are 0 and 1.
WHOLE_RECORDING_START_FRACTION, WHOLE_RECORDING_END_FRACTION = 0.0, 1.0
# Exit codes: identical, different, no reference record.
EXIT_IDENTICAL, EXIT_DIFFERENT, EXIT_NO_REFERENCE = 0, 1, 2


def reference_record_path(session: str, reference_directory: Path = REFERENCE_SPLIT_DIRECTORY) -> Path:
    """Where the committed reference record of a session lives.

    Args:
        session: Session name (``configs/sessions/<session>.yaml``).
        reference_directory: Directory of the records.

    Returns:
        ``<reference_directory>/<session>.json`` (may not exist).
    """
    return reference_directory / f"{session}.json"


def session_split_settings(session: str, session_config_directory: Path = SESSION_CONFIG_DIRECTORY) -> dict:
    """The split-determining settings of a session, from its YAML.

    Args:
        session: Session name.
        session_config_directory: Directory of the session YAMLs.

    Returns:
        A dict with every key of ``REFERENCE_SETTINGS_KEYS``.
    """
    session_config = load_session_config(session_config_directory / f"{session}.yaml")
    split_section = session_config["split"]
    return {
        "session": session_config["session"]["name"],
        "dataset_start_fraction": WHOLE_RECORDING_START_FRACTION,
        "dataset_end_fraction": WHOLE_RECORDING_END_FRACTION,
        "train_fraction": split_section["train_fraction"],
        "test_split_method": split_section["method"],
        "test_render_fraction": split_section["test_render_fraction"],
        "chunk_files": list(session_config["window"]["chunk_files"]),
    }


def rebuild_frames(dataset_dir: Path, settings: dict) -> List[SplitFrame]:
    """Rebuild the split of a dataset exactly as the dataparser does.

    Args:
        dataset_dir: Dataset directory (manifest.json, poses.npz, frame files).
        settings: Session split settings (``session_split_settings``).

    Returns:
        All frames with their split.
    """
    view = open_dataset_view(dataset_dir)
    frames, _, _, _ = build_run_split_with_cores(
        view, settings["dataset_start_fraction"], settings["dataset_end_fraction"], settings["train_fraction"],
        test_split_method=settings["test_split_method"], test_render_fraction=settings["test_render_fraction"])
    return frames


def settings_differences(settings: dict, record: dict) -> List[str]:
    """Differences between the session YAML's settings and the record's (chunk list included).

    Args:
        settings: ``session_split_settings`` of the working tree.
        record: The reference record.

    Returns:
        One line per differing key; empty when the same.
    """
    differences = []
    for settings_key in REFERENCE_SETTINGS_KEYS:
        if settings[settings_key] != record["settings"][settings_key]:
            if settings_key == "chunk_files":
                differences.append(
                    f"the session YAML window.chunk_files {settings[settings_key]} differ from the chunks of the "
                    f"reference run {record['settings'][settings_key]}: the dataset was not extracted from "
                    f"the reference bag chunks")
            else:
                differences.append(f"session YAML {settings_key} {settings[settings_key]!r}, "
                                   f"record {record['settings'][settings_key]!r}")
    return differences


def check_session(session: str, dataset_dir: Path, reference_directory: Path = REFERENCE_SPLIT_DIRECTORY,
                  session_config_directory: Path = SESSION_CONFIG_DIRECTORY) -> int:
    """Rebuild the split of a session's dataset and compare it with the reference record.

    Args:
        session: Session name.
        dataset_dir: Dataset to check.
        reference_directory: Directory of the records.
        session_config_directory: Directory of the session YAMLs.

    Returns:
        ``EXIT_IDENTICAL``, ``EXIT_DIFFERENT`` or ``EXIT_NO_REFERENCE``.
    """
    # Step 1: no record is not an error (sessions without a committed reference run).
    record_path = reference_record_path(session, reference_directory)
    if not record_path.is_file():
        print(f"{session}: no reference split record ({record_path}); nothing to check")
        return EXIT_NO_REFERENCE
    record = json.loads(record_path.read_text())
    # Step 2: the session YAML (what the dataparser is given) against the record's settings.
    settings = session_split_settings(session, session_config_directory)
    differences = settings_differences(settings, record)
    # Step 3: the dataset's own train fraction (T27: the dataparser reads the manifest's) must be the YAML's.
    view = open_dataset_view(dataset_dir)
    manifest_fraction = float(view.manifest["session"]["train_fraction"])
    if manifest_fraction != settings["train_fraction"]:
        differences.append(f"dataset manifest train_fraction {manifest_fraction}, session YAML {settings['train_fraction']}")
    # Step 4: rebuild and compare frame by frame.
    differences += compare_with_reference(rebuild_frames(dataset_dir, settings), record)
    if differences:
        print(f"{session}: the split of {dataset_dir} DIFFERS from the reference run {record['source']['run_id']}:")
        for difference in differences:
            print(f"  - {difference}")
        return EXIT_DIFFERENT
    counts_text = ", ".join(f"{sensor_name}: {sensor_record['counts']}"
                            for sensor_name, sensor_record in record["sensors"].items())
    print(f"{session}: split of {dataset_dir} is identical to the reference run {record['source']['run_id']} "
          f"({counts_text})")
    return EXIT_IDENTICAL


def write_record_from_run(session: str, run_dir: Path, reference_directory: Path = REFERENCE_SPLIT_DIRECTORY,
                          session_config_directory: Path = SESSION_CONFIG_DIRECTORY) -> Path:
    """Write the reference record from a run folder's split.json.

    Args:
        session: Session name (must be the run's session).
        run_dir: Run folder holding ``split.json`` (scripts/09).
        reference_directory: Output directory.
        session_config_directory: Directory of the session YAMLs.

    Returns:
        The written path.

    Raises:
        ValueError: If the run belongs to another session or its window is not the whole recording.
    """
    run_split = json.loads((run_dir / RUN_SPLIT_JSON_NAME).read_text())
    if run_split["run"]["session"] != session:
        raise ValueError(f"run {run_dir} is of session {run_split['run']['session']!r}, not {session!r}")
    # Step 1: settings are the session YAML's, and must agree with what the run recorded.
    settings = session_split_settings(session, session_config_directory)
    selection = run_split["selection"]
    recorded_settings = {
        "dataset_start_fraction": selection["dataset_start_fraction"],
        "dataset_end_fraction": selection["dataset_end_fraction"],
        "train_fraction": selection["train_fraction"],
        "test_split_method": selection["test_split_method"],
        "test_render_fraction": selection["test_render_fraction"],
    }
    disagreeing = {key: (settings[key], value) for key, value in recorded_settings.items() if settings[key] != value}
    if disagreeing:
        raise ValueError(f"session YAML and run disagree (YAML, run): {disagreeing}")
    # Step 2: frames of the run, rebuilt from its split.json rows.
    frames = [SplitFrame(**frame_row) for frame_row in run_split["frames"]]
    if split_digest(frames) != run_split["membership_sha256"]:
        raise ValueError("split.json frames do not match its membership_sha256")
    source = {"run_id": run_split["run"]["run_id"], "git_sha": run_split["run"]["git_sha"]}
    record = build_reference_record(frames, settings, source)
    # Step 3: sorted keys and no wall-clock values keep the file byte-reproducible.
    output_path = reference_record_path(session, reference_directory)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    return output_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point.

    Args:
        argv: Argument list (default sys.argv[1:]).

    Returns:
        The exit code described in the module docstring.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session", required=True, help="session name (configs/sessions/<session>.yaml)")
    parser.add_argument("--dataset-dir", type=Path, default=None,
                        help="dataset to check (default: data/intermediate/<session>)")
    parser.add_argument("--write-from-run", type=Path, default=None,
                        help="write configs/reference_splits/<session>.json from this run folder")
    arguments = parser.parse_args(argv)
    if arguments.write_from_run is not None:
        print(f"wrote {write_record_from_run(arguments.session, arguments.write_from_run)}")
        return EXIT_IDENTICAL
    return check_session(arguments.session, arguments.dataset_dir or DEFAULT_DATASET_PARENT / arguments.session)


if __name__ == "__main__":
    sys.exit(main())
