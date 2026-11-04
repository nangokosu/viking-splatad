"""Committed reference train/test/guard split of a session, and the check against a rebuilt split.

The split is deterministic (``viking_splatad.dataset.run_split``), but the proof of what the reference
run used lived only in gitignored ``runs/``. A reference record keeps, in git, the settings that
determine the split and the held-out frames (test and guard) of every sensor. A collaborator who
re-creates the bag data rebuilds the split and compares it with the record
(``scripts/16_check_reference_split.py``), so a different split is reported, never silently trained on.

Frames are identified by ``timestamp_ns`` (the stamp in the frame file name), not by ``frame_index`` or
file name, so the comparison survives a re-extraction that renumbers or renames the files. Train is
every other windowed frame, so only test and guard are listed (the lists are ~10% of the frames).

Pure functions on ``SplitFrame`` lists and dicts; no torch, no file access.
"""

from typing import Dict, List, Sequence, Set

from viking_splatad.dataset.run_split import (
    GUARD_SPLIT_NAME,
    TEST_SPLIT_NAME,
    TRAIN_SPLIT_NAME,
    SplitFrame,
    split_digest,
)

# Version of the record layout; bump when keys change.
REFERENCE_RECORD_SCHEMA = "viking-splatad/reference-split/1"
# Settings keys a record must carry (the inputs that determine the split, plus the chunk list).
REFERENCE_SETTINGS_KEYS = (
    "session", "dataset_start_fraction", "dataset_end_fraction", "train_fraction",
    "test_split_method", "test_render_fraction", "chunk_files",
)
# Provenance keys of a record (informational: they never enter the comparison).
REFERENCE_SOURCE_KEYS = ("run_id", "git_sha")
# Listing every missing timestamp of a wholesale mismatch would bury the message; show this many and
# the total (a display limit only, it never affects the comparison).
MAX_LISTED_TIMESTAMPS = 5


def split_counts_of_frames(sensor_frames: Sequence[SplitFrame]) -> Dict[str, int]:
    """Frame counts of one sensor's rebuilt split.

    Args:
        sensor_frames: The frames of a single sensor.

    Returns:
        ``frames_in_window`` (distinct timestamps) and the train, test and guard counts.
    """
    return {
        # At train fraction 1.0 a frame is in two splits, so count distinct stamps for the window.
        "frames_in_window": len({frame.timestamp_ns for frame in sensor_frames}),
        TRAIN_SPLIT_NAME: sum(frame.split == TRAIN_SPLIT_NAME for frame in sensor_frames),
        TEST_SPLIT_NAME: sum(frame.split == TEST_SPLIT_NAME for frame in sensor_frames),
        GUARD_SPLIT_NAME: sum(frame.split == GUARD_SPLIT_NAME for frame in sensor_frames),
    }


def timestamps_of_split(sensor_frames: Sequence[SplitFrame], split_name: str) -> List[int]:
    """Sorted ``timestamp_ns`` of one sensor's frames in one split.

    Args:
        sensor_frames: The frames of a single sensor.
        split_name: ``train``, ``test`` or ``guard``.

    Returns:
        Ascending timestamps.
    """
    return sorted(frame.timestamp_ns for frame in sensor_frames if frame.split == split_name)


def build_reference_record(frames: List[SplitFrame], settings: dict, source: dict) -> dict:
    """The JSON-serialisable reference record of a split.

    Args:
        frames: Output of ``build_run_split_with_cores`` (all splits, all sensors).
        settings: The split-determining settings, with every key of ``REFERENCE_SETTINGS_KEYS``.
        source: Provenance with every key of ``REFERENCE_SOURCE_KEYS`` (``run_id``, ``git_sha``).

    Returns:
        Record with the schema string, settings, source, the informational ``split_digest``
        membership hash, and per sensor its kind, counts and the sorted test and guard timestamps.

    Raises:
        ValueError: If ``settings`` or ``source`` lacks a required key.
    """
    # Step 1: refuse an incomplete header, so a record always says what produced it.
    missing_settings = [key for key in REFERENCE_SETTINGS_KEYS if key not in settings]
    missing_source = [key for key in REFERENCE_SOURCE_KEYS if key not in source]
    if missing_settings or missing_source:
        raise ValueError(f"reference record needs settings {missing_settings} and source {missing_source}")
    # Step 2: per sensor, counts plus the held-out stamps (train is the rest of the window).
    sensors: Dict[str, dict] = {}
    for sensor_name in sorted({frame.sensor for frame in frames}):
        sensor_frames = [frame for frame in frames if frame.sensor == sensor_name]
        sensors[sensor_name] = {
            "kind": sensor_frames[0].kind,
            "counts": split_counts_of_frames(sensor_frames),
            "test_timestamps_ns": timestamps_of_split(sensor_frames, TEST_SPLIT_NAME),
            "guard_timestamps_ns": timestamps_of_split(sensor_frames, GUARD_SPLIT_NAME),
        }
    return {
        "schema": REFERENCE_RECORD_SCHEMA,
        "settings": {key: settings[key] for key in REFERENCE_SETTINGS_KEYS},
        "source": {key: source[key] for key in REFERENCE_SOURCE_KEYS},
        # Informational only: it hashes file names (with frame_index), so a renumbered
        # re-extraction changes it even when the split is the same.
        "split_digest": split_digest(frames),
        "sensors": sensors,
    }


def describe_timestamps(timestamps: Set[int]) -> str:
    """A short text of a timestamp set: the total and the first few in ascending order.

    Args:
        timestamps: Non-empty set of ``timestamp_ns``.

    Returns:
        E.g. ``3 [1, 2, 3]`` or ``9 [1, 2, 3, 4, 5, ...]``.
    """
    ordered_timestamps = sorted(timestamps)
    shown_text = ", ".join(str(stamp) for stamp in ordered_timestamps[:MAX_LISTED_TIMESTAMPS])
    ellipsis_text = ", ..." if len(ordered_timestamps) > MAX_LISTED_TIMESTAMPS else ""
    return f"{len(ordered_timestamps)} [{shown_text}{ellipsis_text}]"


def compare_with_reference(frames: List[SplitFrame], record: dict) -> List[str]:
    """Differences between a rebuilt split and a reference record.

    Per sensor it checks: the counts (window, train, test, guard); the test and guard stamps the
    record lists but the rebuilt split does not hold in that split, and the reverse; record stamps
    the dataset lacks altogether; and the train membership. Train is "windowed and neither test nor
    guard", so a frame the rebuilt split trains on must not be a test or guard frame of the record
    (reported as the test/guard stamps missing from the rebuilt split), and every record stamp must
    exist in the rebuilt window (reported as absent from the dataset).

    Args:
        frames: Output of ``build_run_split_with_cores`` for the collaborator's dataset.
        record: A reference record (``build_reference_record``, possibly loaded from JSON).

    Returns:
        Human-readable difference lines; empty when the split is identical.
    """
    differences: List[str] = []
    # Step 1: the schema decides whether the rest of the record can be read at all.
    if record.get("schema") != REFERENCE_RECORD_SCHEMA:
        return [f"record schema {record.get('schema')!r} is not {REFERENCE_RECORD_SCHEMA!r}"]
    rebuilt_sensor_names = {frame.sensor for frame in frames}
    record_sensors = record["sensors"]
    # Step 2: sensors present on one side only.
    for sensor_name in sorted(set(record_sensors) - rebuilt_sensor_names):
        differences.append(f"{sensor_name}: in the record but not in the rebuilt split")
    for sensor_name in sorted(rebuilt_sensor_names - set(record_sensors)):
        differences.append(f"{sensor_name}: in the rebuilt split but not in the record")
    # Step 3: sensors on both sides, compared stamp by stamp.
    for sensor_name in sorted(rebuilt_sensor_names & set(record_sensors)):
        sensor_frames = [frame for frame in frames if frame.sensor == sensor_name]
        record_sensor = record_sensors[sensor_name]
        # Counts first: the quickest sign that the window or the hold-out rule differs.
        rebuilt_counts = split_counts_of_frames(sensor_frames)
        for count_name, rebuilt_count in rebuilt_counts.items():
            record_count = record_sensor["counts"][count_name]
            if rebuilt_count != record_count:
                differences.append(f"{sensor_name}: {count_name} count {rebuilt_count}, record {record_count}")
        window_stamps = {frame.timestamp_ns for frame in sensor_frames}
        for split_name, record_key in ((TEST_SPLIT_NAME, "test_timestamps_ns"),
                                       (GUARD_SPLIT_NAME, "guard_timestamps_ns")):
            record_stamps = set(record_sensor[record_key])
            rebuilt_stamps = set(timestamps_of_split(sensor_frames, split_name))
            # Record stamps that the dataset does not have at all (different recording or window).
            absent_stamps = record_stamps - window_stamps
            if absent_stamps:
                differences.append(
                    f"{sensor_name}: {split_name} frames of the record absent from the dataset: "
                    f"{describe_timestamps(absent_stamps)}")
            # Present in the dataset but not held out in this split: the rebuilt split trains on them
            # (or puts them in the other held-out split).
            missing_stamps = (record_stamps - rebuilt_stamps) - absent_stamps
            if missing_stamps:
                differences.append(
                    f"{sensor_name}: {split_name} frames of the record not {split_name} in the rebuilt split: "
                    f"{describe_timestamps(missing_stamps)}")
            unexpected_stamps = rebuilt_stamps - record_stamps
            if unexpected_stamps:
                differences.append(
                    f"{sensor_name}: unexpected {split_name} frames (not {split_name} in the record): "
                    f"{describe_timestamps(unexpected_stamps)}")
        # Train needs no list: the record keeps no train stamps. A rebuilt train frame that the record
        # holds out is reported above as missing; an extra train frame shows in the train and window counts.
    return differences
