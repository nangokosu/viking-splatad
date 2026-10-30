"""Tests of the pure helpers in scripts/14_build_fiftyone_dataset.py (no FiftyOne needed)."""

import importlib.util
from pathlib import Path

import numpy as np

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "14_build_fiftyone_dataset.py"


def load_script_module():
    """Import the numerically named script as a module.

    Returns:
        The loaded module (fiftyone is imported lazily inside it, so this works without it).
    """
    module_spec = importlib.util.spec_from_file_location("build_fiftyone_dataset_script", SCRIPT_PATH)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def test_timestamp_is_the_number_after_the_frame_index():
    """``<index>_<timestamp_ns>`` -> the timestamp."""
    assert load_script_module().timestamp_ns_from_stem("000569_1714571368050209280") == 1714571368050209280


def test_nearest_image_minimises_time_difference():
    """Of images at t = 100, 200, 300 ns a sweep at 260 ns picks 300 (|40| < |60|); at 240 it picks 200."""
    module = load_script_module()
    image_paths = [Path(f"000{frame_index}_{timestamp_ns}.jpg") for frame_index, timestamp_ns in enumerate([100, 200, 300])]
    assert module.nearest_image(260, image_paths).stem == "0002_300"
    assert module.nearest_image(240, image_paths).stem == "0001_200"


def test_cuboid_records_become_detections_with_the_stored_rotation():
    """A yaw-only box keeps its centre, full size and label; rotation = (0, 0, yaw)."""
    module = load_script_module()
    yaw = 0.4

    class FakeDetection:
        """Stand-in for fo.Detection that just records its keyword arguments."""

        def __init__(self, **keyword_arguments):
            self.__dict__.update(keyword_arguments)

    class FakeFiftyOne:
        Detection = FakeDetection

    record = {"class_name": "tree_trunk", "center_xyz": [1.0, 2.0, 3.0], "dimensions_xyz": [0.5, 0.6, 7.0],
              "quaternion_xyzw": [0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)], "box_id": 9, "class_id": 1}
    (detection,) = module.cuboids_to_detections([record], FakeFiftyOne)
    assert detection.label == "tree_trunk" and detection.box_id == 9
    assert detection.location == [1.0, 2.0, 3.0] and detection.dimensions == [0.5, 0.6, 7.0]
    np.testing.assert_allclose(detection.rotation, [0.0, 0.0, yaw], atol=1e-12)
