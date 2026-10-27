"""Tests for T65: the dataparser's held-out split depends on who asks (training eval vs renders)."""

from pathlib import Path

import pytest

from tests.unit.test_block_split import _neurad_ready
from viking_splatad.dataset.run_split import (
    EVAL_WHOLE_BLOCKS_SPLIT_NAME,
    WHOLE_BLOCK_CORE_FRACTION,
    build_run_split_with_cores,
    core_fraction_for_split,
)
from viking_splatad.splatad.viking_conversions import open_dataset_view


@pytest.mark.covers("T65")
def test_core_fraction_for_split():
    """Only "val" with the flag on gets the whole block; every other case keeps the configured core."""
    core_fraction = 0.1
    assert EVAL_WHOLE_BLOCKS_SPLIT_NAME == "val"
    assert core_fraction_for_split("val", core_fraction, True) == WHOLE_BLOCK_CORE_FRACTION == 1.0
    assert core_fraction_for_split("val", core_fraction, False) == core_fraction
    for other_split in ("test", "eval", "train"):
        assert core_fraction_for_split(other_split, core_fraction, True) == core_fraction


# Synthetic settings of the neurad test below. f = 0.5 (> 1/(K+1)) makes each T60 block long enough to
# hold 3 of the 10 frames, so c = 0.5 leaves guard frames; at f = 0.8 each block holds only its centre frame.
SYNTHETIC_TRAIN_FRACTION, SYNTHETIC_BLOCK_COUNT, SYNTHETIC_CORE_FRACTION = 0.5, 2, 0.5


@pytest.mark.covers("T65")
def test_synthetic_settings_have_guard_frames_for_both_sensors(tmp_path):
    """Precondition of the neurad test (Mac-runnable): whole blocks differ from cores, per sensor kind."""
    from tests.unit.test_viking_dataparser import _write_synthetic

    _, root = _write_synthetic(tmp_path)
    dataset_view = open_dataset_view(root)
    split_counts = {}
    for core_fraction in (WHOLE_BLOCK_CORE_FRACTION, SYNTHETIC_CORE_FRACTION):
        frames, _, _, _ = build_run_split_with_cores(
            dataset_view, 0.0, 1.0, SYNTHETIC_TRAIN_FRACTION, test_block_count=SYNTHETIC_BLOCK_COUNT,
            test_block_core_fraction=core_fraction)
        for sensor_kind in ("camera", "lidar"):
            for record_split in ("train", "test", "guard"):
                split_counts[(core_fraction, sensor_kind, record_split)] = sum(
                    frame.kind == sensor_kind and frame.split == record_split for frame in frames)
    for sensor_kind in ("camera", "lidar"):
        whole_test = split_counts[(WHOLE_BLOCK_CORE_FRACTION, sensor_kind, "test")]
        core_test = split_counts[(SYNTHETIC_CORE_FRACTION, sensor_kind, "test")]
        guard = split_counts[(SYNTHETIC_CORE_FRACTION, sensor_kind, "guard")]
        assert guard > 0 and core_test > 0
        # Guard frames are exactly the test frames the cores drop; train is unchanged.
        assert whole_test == core_test + guard
        assert split_counts[(WHOLE_BLOCK_CORE_FRACTION, sensor_kind, "train")] == \
            split_counts[(SYNTHETIC_CORE_FRACTION, sensor_kind, "train")]


@pytest.mark.covers("T65")
@pytest.mark.skipif(not _neurad_ready(), reason="needs torch + patched neurad-studio")
def test_real_dataparser_val_is_whole_blocks(tmp_path):
    """val = whole T60 blocks; test and eval = the c cores; train = train frames, flag on or off."""
    from tests.unit.test_viking_dataparser import SYNTHETIC_CAMERA, SYNTHETIC_LIDAR, _write_synthetic
    from viking_splatad.splatad.viking_dataparser import VikingHillDataParserConfig

    _, root = _write_synthetic(tmp_path)
    train_fraction, block_count, core_fraction = (
        SYNTHETIC_TRAIN_FRACTION, SYNTHETIC_BLOCK_COUNT, SYNTHETIC_CORE_FRACTION)
    config_arguments = dict(data=tmp_path, session_name=root.name, cameras=(SYNTHETIC_CAMERA,),
                            lidars=(SYNTHETIC_LIDAR,), train_split_fraction_override=train_fraction,
                            test_block_count=block_count, test_block_core_fraction=core_fraction)
    flag_on = VikingHillDataParserConfig(eval_whole_test_blocks=True, **config_arguments).setup()
    flag_off = VikingHillDataParserConfig(**config_arguments).setup()
    parsed = {split_name: flag_on.get_dataparser_outputs(split_name) for split_name in ("val", "test", "eval", "train")}
    parsed_train_flag_off = flag_off.get_dataparser_outputs("train")
    dataset_view = open_dataset_view(root)
    whole_frames, _, _, _ = build_run_split_with_cores(
        dataset_view, 0.0, 1.0, train_fraction, test_block_count=block_count, test_block_core_fraction=1.0)
    core_frames, _, _, _ = build_run_split_with_cores(
        dataset_view, 0.0, 1.0, train_fraction, test_block_count=block_count, test_block_core_fraction=core_fraction)

    def camera_names(outputs):
        """Sorted camera file names of dataparser outputs."""
        return sorted(Path(image_path).name for image_path in outputs.image_filenames)

    def expected_names(frames, record_split):
        """Sorted camera file names and lidar count of one split of a rebuilt record."""
        names = sorted(Path(frame.file).name for frame in frames if frame.kind == "camera" and frame.split == record_split)
        lidar_count = sum(frame.kind == "lidar" and frame.split == record_split for frame in frames)
        return names, lidar_count

    for parser_split, frames, record_split in (("val", whole_frames, "test"), ("test", core_frames, "test"),
                                               ("eval", core_frames, "test"), ("train", core_frames, "train")):
        expected_camera, expected_lidar_count = expected_names(frames, record_split)
        assert camera_names(parsed[parser_split]) == expected_camera
        assert len(parsed[parser_split].metadata["lidars"]) == expected_lidar_count
    # Whole blocks are strictly larger than the cores (guard frames exist; checked on the Mac above).
    assert whole_frames != core_frames
    assert len(parsed["val"].image_filenames) > len(parsed["test"].image_filenames)
    assert len(parsed["val"].metadata["lidars"]) > len(parsed["test"].metadata["lidars"])
    # Train is the same with the flag off.
    assert camera_names(parsed["train"]) == camera_names(parsed_train_flag_off)
    assert len(parsed["train"].metadata["lidars"]) == len(parsed_train_flag_off.metadata["lidars"])
