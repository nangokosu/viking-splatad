"""Contract validation checks all frames without retaining the decoded recording."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from tests.unit.test_writer import build_synthetic_dataset
from viking_splatad.contract.splatad_contract import CONTRACT_CHECKS, validate_dataset
from viking_splatad.dataset import writer


@pytest.mark.covers("T26")
def test_lazy_reader_decodes_only_requested_images_and_does_not_cache(tmp_path, monkeypatch):
    """Three constant images support positive/negative indexing while retaining no cache."""
    dataset = build_synthetic_dataset(num_frames=3)
    camera = dataset.streams["cam_front"]
    intensity_step = np.iinfo(np.uint8).max // (len(camera.frames) + 1)
    camera.frames = [np.full_like(frame, intensity_step * (frame_index + 1))
                     for frame_index, frame in enumerate(camera.frames)]
    writer.write_dataset(tmp_path, dataset)
    original_imread = cv2.imread
    decoded_paths = []

    def recording_imread(path, *arguments, **kwargs):
        """Record image reads and delegate to the actual JPEG decoder."""
        decoded_paths.append(path)
        return original_imread(path, *arguments, **kwargs)

    monkeypatch.setattr(writer.cv2, "imread", recording_imread)
    lazy_dataset = writer.read_dataset(tmp_path, lazy_frames=True)
    frames = lazy_dataset.streams["cam_front"].frames
    assert len(frames) == len(camera.frames)
    assert decoded_paths == []
    first_read, second_read = frames[0], frames[0]
    assert len(decoded_paths) == 2
    assert first_read is not second_read
    np.testing.assert_array_equal(first_read, camera.frames[0])
    np.testing.assert_array_equal(frames[-1], camera.frames[-1])
    assert len(frames[::2]) == len(range(0, len(camera.frames), 2))


def test_streamed_contract_matches_every_eager_check(tmp_path):
    """All existing check names, outcomes, and details remain unchanged for known valid input."""
    writer.write_dataset(tmp_path, build_synthetic_dataset(num_frames=3))
    eager = writer.read_dataset(tmp_path)
    expected_checks = [check(eager) for check in CONTRACT_CHECKS]
    report = validate_dataset(tmp_path)
    assert report.results == expected_checks


def test_streamed_contract_checks_last_camera_frame(tmp_path):
    """A wrong shape only in the final image cannot escape streaming validation."""
    dataset = build_synthetic_dataset(num_frames=3)
    writer.write_dataset(tmp_path, dataset)
    image_path = sorted((tmp_path / "camera/cam_front").glob("*.jpg"))[-1]
    height, width, channels = dataset.streams["cam_front"].frames[-1].shape
    cv2.imwrite(str(image_path), np.zeros((height // 2, width // 2, channels), dtype=np.uint8))
    report = validate_dataset(tmp_path)
    image_check = next(check for check in report.results if check.name == "image_shape_matches")
    assert not image_check.passed
    assert "cam_front[2]" in image_check.message


def test_deferred_array_load_error_is_reported_as_unloadable_dataset(tmp_path):
    """A truncated final sweep still produces the load-failure report rather than an exception."""
    writer.write_dataset(tmp_path, build_synthetic_dataset(num_frames=3))
    sweep_path = sorted((tmp_path / "lidar/lidar_top").glob("*.npy"))[-1]
    sweep_path.write_bytes(b"invalid numpy data")
    report = validate_dataset(tmp_path)
    assert not report.passed
    assert report.results[0].name == "dataset_loadable"
