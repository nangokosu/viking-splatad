"""Tests for T66: a render-only job's test split is exactly the requested guard frames of a finished run.

Covered here: the transform (known answers, every refusal), the file-name and selection-file helpers
the dataparser feeds it with, the CPU rebuild's guard frames as valid requests (Mac-runnable), and
(when torch + patched neurad-studio are importable) the real dataparser with
``render_frame_selection_file`` set.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from tests.unit.test_block_split import _neurad_ready
from tests.unit.test_eval_split_whole_blocks import (
    SYNTHETIC_BLOCK_COUNT,
    SYNTHETIC_CORE_FRACTION,
    SYNTHETIC_TRAIN_FRACTION,
)
from viking_splatad.dataset.run_split import (
    RENDER_SELECTION_SPLIT_NAMES,
    build_run_split_with_cores,
    frame_indices_of_paths,
    load_render_frame_selection,
    middle_block_train_eval_indices,
    restrict_test_to_block_cores,
    select_guard_frames_for_render,
)
from viking_splatad.splatad.viking_conversions import open_dataset_view

# Frame index = position + this offset in the synthetic arrays below, so a test that confused
# positions with frame indices would fail (any non-zero offset works; 100 keeps them readable).
FRAME_INDEX_OFFSET = 100


def _half_core_split():
    """T60 + T62 on camera times 0..12, f = 0.5, K = 2, c = 0.5 (test_block_cores' known answer).

    Returns:
        (train positions, core positions, guard positions, frame index per position): train
        0, 1, 2, 6, 10, 11, 12; cores 4, 8; guard 3, 5, 7, 9; frame index = position + offset.
    """
    camera_times = np.arange(13, dtype=np.float64)
    positions, blocks = middle_block_train_eval_indices({"cam": camera_times}, 0.5, 2)
    core_positions, guard_positions, _ = restrict_test_to_block_cores({"cam": camera_times}, positions, blocks, 0.5)
    train_positions, test_positions = core_positions["cam"]
    frame_indices = np.arange(len(camera_times), dtype=np.int64) + FRAME_INDEX_OFFSET
    return train_positions, test_positions, guard_positions["cam"], frame_indices


@pytest.mark.covers("T66")
def test_requested_guard_frames_give_their_sorted_positions():
    """Guard positions 3, 5, 7, 9; requesting frames 109 and 103 (unsorted) gives positions [3, 9]."""
    train_positions, test_positions, guard_positions, frame_indices = _half_core_split()
    assert guard_positions.tolist() == [3, 5, 7, 9]
    selected_positions = select_guard_frames_for_render(
        guard_positions, frame_indices, [9 + FRAME_INDEX_OFFSET, 3 + FRAME_INDEX_OFFSET])
    assert selected_positions.tolist() == [3, 9]
    assert selected_positions.dtype == np.int64
    # Invariants: a subset of the guard, disjoint from train and from the cores.
    assert set(selected_positions.tolist()) <= set(guard_positions.tolist())
    assert not set(selected_positions.tolist()) & set(train_positions.tolist())
    assert not set(selected_positions.tolist()) & set(test_positions.tolist())


@pytest.mark.covers("T66")
def test_every_guard_frame_can_be_requested_at_once():
    """Requesting the whole guard returns exactly the guard positions."""
    _, _, guard_positions, frame_indices = _half_core_split()
    selected_positions = select_guard_frames_for_render(guard_positions, frame_indices, frame_indices[guard_positions])
    np.testing.assert_array_equal(selected_positions, guard_positions)


@pytest.mark.covers("T66")
def test_non_guard_and_unknown_frames_are_refused_and_all_named():
    """A train frame (100), a core frame (104) and a frame outside the window (999) are each named."""
    _, _, guard_positions, frame_indices = _half_core_split()
    train_frame_index = 0 + FRAME_INDEX_OFFSET
    core_frame_index = 4 + FRAME_INDEX_OFFSET
    unknown_frame_index = 999
    with pytest.raises(ValueError) as raised:
        select_guard_frames_for_render(
            guard_positions, frame_indices, [3 + FRAME_INDEX_OFFSET, train_frame_index, core_frame_index,
                                             unknown_frame_index])
    message = str(raised.value)
    assert f"[{unknown_frame_index}]" in message
    assert f"[{train_frame_index}, {core_frame_index}]" in message


@pytest.mark.covers("T66")
def test_repeated_and_empty_requests_are_refused():
    """A repeated index and an empty request are caller errors, not silently deduplicated or ignored."""
    _, _, guard_positions, frame_indices = _half_core_split()
    repeated_frame_index = 5 + FRAME_INDEX_OFFSET
    with pytest.raises(ValueError, match=f"repeats frame indices \\[{repeated_frame_index}\\]"):
        select_guard_frames_for_render(guard_positions, frame_indices, [repeated_frame_index, repeated_frame_index])
    with pytest.raises(ValueError, match="empty"):
        select_guard_frames_for_render(guard_positions, frame_indices, [])


@pytest.mark.covers("T66")
def test_frame_indices_are_parsed_from_dataset_file_names():
    """``<frame_index>_<ns>.jpg|npy`` gives the frame index; any other name is refused."""
    frame_paths = [Path("camera/cam/000795_1700000000000000000.jpg"), Path("lidar/os/000012_5.npy")]
    assert frame_indices_of_paths(frame_paths).tolist() == [795, 12]
    with pytest.raises(ValueError, match="frame file"):
        frame_indices_of_paths([Path("camera/cam/795.jpg")])


@pytest.mark.covers("T66")
def test_selection_file_reads_both_kinds_and_refuses_bad_values(tmp_path):
    """Extra keys are ignored; a missing kind, a non-list or a non-integer (bool included) is refused."""
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps({"camera": [795, 797], "lidar": [801], "files": {"camera": []}}))
    assert load_render_frame_selection(selection_path) == {"camera": [795, 797], "lidar": [801]}
    for bad_selection in ({"camera": [1]}, {"camera": 1, "lidar": [1]}, {"camera": [1.5], "lidar": [1]},
                          {"camera": [True], "lidar": [1]}):
        selection_path.write_text(json.dumps(bad_selection))
        with pytest.raises(ValueError):
            load_render_frame_selection(selection_path)


@pytest.mark.covers("T66")
def test_render_selection_replaces_only_the_render_splits():
    """ns-render and scripts/08 ask for "test"; "eval" is its base-class alias; "val" and "train" stay."""
    assert set(RENDER_SELECTION_SPLIT_NAMES) == {"test", "eval"}


@pytest.mark.covers("T66")
def test_rebuilt_guard_frames_are_valid_requests(tmp_path):
    """On the synthetic dataset, scripts/09's guard frames per kind are exactly what T66 accepts."""
    from tests.unit.test_viking_dataparser import _write_synthetic

    _, root = _write_synthetic(tmp_path)
    frames, _, _, _ = build_run_split_with_cores(
        open_dataset_view(root), 0.0, 1.0, SYNTHETIC_TRAIN_FRACTION, test_block_count=SYNTHETIC_BLOCK_COUNT,
        test_block_core_fraction=SYNTHETIC_CORE_FRACTION)
    for sensor_kind in ("camera", "lidar"):
        kind_frames = sorted((frame for frame in frames if frame.kind == sensor_kind), key=lambda frame: frame.time_s)
        frame_indices = np.array([frame.frame_index for frame in kind_frames], dtype=np.int64)
        guard_positions = np.array([position for position, frame in enumerate(kind_frames) if frame.split == "guard"],
                                   dtype=np.int64)
        assert guard_positions.size > 0
        selected_positions = select_guard_frames_for_render(guard_positions, frame_indices,
                                                            frame_indices[guard_positions].tolist())
        np.testing.assert_array_equal(selected_positions, guard_positions)
        # A test (core) frame of the record is refused.
        core_frame_index = next(frame.frame_index for frame in kind_frames if frame.split == "test")
        with pytest.raises(ValueError, match="not guard"):
            select_guard_frames_for_render(guard_positions, frame_indices, [core_frame_index])


@pytest.mark.covers("T66")
@pytest.mark.skipif(not _neurad_ready(), reason="needs torch + patched neurad-studio")
def test_real_dataparser_test_split_is_the_selected_guard_frames(tmp_path):
    """With the selection file, test and eval = the selected guard frames; train and val unchanged."""
    from tests.unit.test_viking_dataparser import SYNTHETIC_CAMERA, SYNTHETIC_LIDAR, _write_synthetic
    from viking_splatad.splatad.viking_dataparser import VikingHillDataParserConfig

    _, root = _write_synthetic(tmp_path)
    frames, _, _, _ = build_run_split_with_cores(
        open_dataset_view(root), 0.0, 1.0, SYNTHETIC_TRAIN_FRACTION, test_block_count=SYNTHETIC_BLOCK_COUNT,
        test_block_core_fraction=SYNTHETIC_CORE_FRACTION)
    # Step 1: request every guard frame of each kind (the rebuild above is the source of truth).
    guard_frames = {sensor_kind: [frame for frame in frames if frame.kind == sensor_kind and frame.split == "guard"]
                    for sensor_kind in ("camera", "lidar")}
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps({sensor_kind: [frame.frame_index for frame in kind_frames]
                                          for sensor_kind, kind_frames in guard_frames.items()}))
    config_arguments = dict(data=tmp_path, session_name=root.name, cameras=(SYNTHETIC_CAMERA,),
                            lidars=(SYNTHETIC_LIDAR,), train_split_fraction_override=SYNTHETIC_TRAIN_FRACTION,
                            test_block_count=SYNTHETIC_BLOCK_COUNT, test_block_core_fraction=SYNTHETIC_CORE_FRACTION,
                            eval_whole_test_blocks=True)
    selected_parser = VikingHillDataParserConfig(render_frame_selection_file=selection_path, **config_arguments).setup()
    plain_parser = VikingHillDataParserConfig(**config_arguments).setup()

    def camera_names(outputs):
        """Sorted camera file names of dataparser outputs."""
        return sorted(Path(image_path).name for image_path in outputs.image_filenames)

    # Step 2: test and eval are exactly the guard frames, per kind.
    expected_camera_names = sorted(Path(frame.file).name for frame in guard_frames["camera"])
    for split_name in RENDER_SELECTION_SPLIT_NAMES:
        outputs = selected_parser.get_dataparser_outputs(split_name)
        assert camera_names(outputs) == expected_camera_names
        assert len(outputs.metadata["lidars"]) == len(guard_frames["lidar"])
    # Step 3: train and val are what the parser without a selection gives.
    for split_name in ("train", "val"):
        selected_outputs = selected_parser.get_dataparser_outputs(split_name)
        plain_outputs = plain_parser.get_dataparser_outputs(split_name)
        assert camera_names(selected_outputs) == camera_names(plain_outputs)
        assert len(selected_outputs.metadata["lidars"]) == len(plain_outputs.metadata["lidars"])
    # Step 4: a core (test) frame in the selection is refused.
    core_camera_index = next(frame.frame_index for frame in frames if frame.kind == "camera" and frame.split == "test")
    selection_path.write_text(json.dumps({"camera": [core_camera_index],
                                          "lidar": [guard_frames["lidar"][0].frame_index]}))
    with pytest.raises(ValueError, match="guard"):
        VikingHillDataParserConfig(render_frame_selection_file=selection_path,
                                   **config_arguments).setup().get_dataparser_outputs("test")
    # Step 5: without guard frames (c = 1) the selection is a configuration error.
    whole_block_arguments = {**config_arguments, "test_block_core_fraction": 1.0}
    with pytest.raises(ValueError, match="render_frame_selection_file"):
        VikingHillDataParserConfig(render_frame_selection_file=selection_path,
                                   **whole_block_arguments).setup().get_dataparser_outputs("test")


@pytest.mark.covers("T66")
@pytest.mark.skipif(not _neurad_ready(), reason="needs torch + patched neurad-studio")
def test_config_saved_before_the_field_loads_without_a_selection(tmp_path):
    """A config.yml without render_frame_selection_file (older runs) loads with it None: the cores."""
    import yaml

    from viking_splatad.splatad.viking_dataparser import VikingHillDataParserConfig

    # Step 1: dump a config the way nerfstudio's save_config does, then drop the field's line, as in
    # every config.yml written before the field existed.
    dumped_text = yaml.dump(VikingHillDataParserConfig(test_block_count=2, test_block_core_fraction=0.1))
    old_text = "\n".join(line for line in dumped_text.splitlines() if "render_frame_selection_file" not in line)
    assert "render_frame_selection_file" in dumped_text and "render_frame_selection_file" not in old_text
    # Step 2: eval_setup's loader; the missing attribute falls back to the class default.
    loaded_config = yaml.load(old_text, Loader=yaml.Loader)
    assert loaded_config.render_frame_selection_file is None
    assert (loaded_config.test_block_count, loaded_config.test_block_core_fraction) == (2, 0.1)
