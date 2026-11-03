"""Project rules 5 and 6: radar is never a SplatAD sensor, ground truth never feeds training.

* Rule 5: ``radars`` stays empty, raw radar stays a sidecar, and ``radar_seeds`` (the only way radar
  may enter) never appears in the held-out split's dataparser output.
* Rule 6: ground-truth cuboids are evaluation-only, so no training-side module may import the label
  or evaluation packages or read ``data/gt``, and no future seed builder may either.

The static checks parse source with ``ast`` and cover every training-side module found on disk, so a
new module that breaks a rule is caught without editing this file.
"""

import ast
import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest

from tests.unit.test_writer import build_synthetic_dataset
from viking_splatad.dataset.writer import write_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src" / "viking_splatad"
# Packages that build or consume SplatAD training inputs; they must never see ground truth.
TRAINING_SIDE_PACKAGES = ("dataset", "splatad", "extract", "pipeline", "io", "contract", "align")
# Packages that hold GT-dependent code (labels) or evaluation-only code (eval, viz).
GROUND_TRUTH_PACKAGES = ("labels", "eval", "viz")
# Where the offline radar seed builder lives (CLAUDE.md rule 5); may not exist yet.
SEED_BUILDER_GLOBS = ("scripts/07_build_radar_seeds.py", "src/viking_splatad/radar_init/**/*.py")
# Text that means "reads ground truth" in a file (paths of the GT directory and its files).
GROUND_TRUTH_TEXT_MARKERS = ("data/gt", "cuboids.json", "occupancy.npz", "gt_dir")


def imported_module_names(source_path: Path) -> set:
    """Dotted names of every module a Python file imports (``import a.b`` and ``from a.b import c``)."""
    tree = ast.parse(source_path.read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            # "from viking_splatad import labels" imports the subpackage labels.
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return names


def ground_truth_violations(source_path: Path) -> list:
    """Reasons a file touches ground truth: GT package imports or GT path text."""
    violations = []
    forbidden_prefixes = tuple(f"viking_splatad.{package}" for package in GROUND_TRUTH_PACKAGES)
    for module_name in sorted(imported_module_names(source_path)):
        if module_name.startswith(forbidden_prefixes):
            violations.append(f"imports {module_name}")
    text = source_path.read_text()
    violations.extend(f"mentions {marker!r}" for marker in GROUND_TRUTH_TEXT_MARKERS if marker in text)
    return violations


def training_side_files() -> list:
    """Every .py file of the training-side packages."""
    return sorted(path for package in TRAINING_SIDE_PACKAGES for path in (SOURCE_ROOT / package).rglob("*.py"))


def test_training_side_discovery_is_not_vacuous():
    """The scan sees the dataparser, the converter and the writer (guards against a broken path)."""
    names = {path.name for path in training_side_files()}
    assert {"viking_dataparser.py", "viking_conversions.py", "writer.py", "bag_to_dataset.py"} <= names


def test_ground_truth_detector_flags_a_leaking_file(tmp_path):
    """Known-answer check of the detector itself, so a silent detector cannot make the real scan pass."""
    leaking_file = tmp_path / "leak.py"
    leaking_file.write_text("from viking_splatad.labels.cuboids import parse_cuboids\nPATH = 'data/gt/x/cuboids.json'\n")
    clean_file = tmp_path / "clean.py"
    clean_file.write_text("import numpy as np\n")
    assert ground_truth_violations(leaking_file) == [
        "imports viking_splatad.labels.cuboids", "imports viking_splatad.labels.cuboids.parse_cuboids",
        "mentions 'data/gt'", "mentions 'cuboids.json'"]
    assert ground_truth_violations(clean_file) == []


def test_no_training_side_module_touches_ground_truth():
    """Rule 6: no dataset / dataparser / extract / pipeline module imports labels, eval or viz or reads data/gt."""
    offenders = {str(path.relative_to(PROJECT_ROOT)): ground_truth_violations(path)
                 for path in training_side_files() if ground_truth_violations(path)}
    assert offenders == {}


def test_radar_seed_builder_never_touches_ground_truth():
    """Rules 5 and 6: the offline seed builder (when present) uses no GT cuboid, occupancy or label module."""
    builder_files = sorted(path for pattern in SEED_BUILDER_GLOBS for path in PROJECT_ROOT.glob(pattern))
    offenders = {str(path.relative_to(PROJECT_ROOT)): ground_truth_violations(path)
                 for path in builder_files if ground_truth_violations(path)}
    assert offenders == {}


def test_no_source_package_outside_ground_truth_ones_imports_them():
    """No module outside labels/eval/viz imports labels (cuboids) so GT cannot flow into src/ training code."""
    offenders = []
    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        package = path.relative_to(SOURCE_ROOT).parts[0]
        if package in GROUND_TRUTH_PACKAGES:
            continue
        if any(name.startswith("viking_splatad.labels") for name in imported_module_names(path)):
            offenders.append(str(path.relative_to(PROJECT_ROOT)))
    assert offenders == []


def test_dataset_writer_output_contains_no_ground_truth_or_seed_files(tmp_path):
    """A written dataset holds sensor data and the radar SIDECAR only: no cuboid, GT, or seed file."""
    write_dataset(tmp_path / "session", build_synthetic_dataset(num_frames=4))
    names = {path.name.lower() for path in (tmp_path / "session").rglob("*")}
    assert not [name for name in names if "cuboid" in name or "occupancy" in name or name.startswith("gt")]
    assert (tmp_path / "session" / "sidecar" / "radar").is_dir()


# ---------------------------------------------------------------- real dataparser (needs torch + neurad)

SYNTHETIC_CAMERA, SYNTHETIC_LIDAR = "cam_front", "lidar_top"
DATASET_FRAME_COUNT = 40
FRAME_PERIOD_SECONDS = 0.1  # build_synthetic_dataset's sweep_period
TEST_BLOCK_TRAIN_FRACTION = 0.75
CORE_FRACTION = 0.5


def neurad_is_ready() -> bool:
    """True when torch and the patched neurad-studio are importable (the reference image)."""
    try:
        from nerfstudio.cameras.lidars import LidarType
    except ImportError:
        return False
    return hasattr(LidarType, "OUSTER_OS0_32")


def write_block_split_dataset(tmp_path):
    """Synthetic dataset typed OUSTER_OS0_32 (what the dataparser registers) under tmp_path/session."""
    dataset = build_synthetic_dataset(num_frames=DATASET_FRAME_COUNT)
    dataset.lidar = dataclasses.replace(dataset.lidar, lidar_type="OUSTER_OS0_32")
    write_dataset(tmp_path / "session", dataset)


def make_block_split_config(tmp_path):
    """Dataparser config: one centre test block (K=1) with a central core, 0.75 train fraction."""
    from viking_splatad.splatad.viking_dataparser import VikingHillDataParserConfig
    return VikingHillDataParserConfig(
        data=tmp_path, session_name="session", cameras=(SYNTHETIC_CAMERA,), lidars=(SYNTHETIC_LIDAR,),
        train_split_fraction_override=TEST_BLOCK_TRAIN_FRACTION, test_block_count=1,
        test_block_core_fraction=CORE_FRACTION)


def frame_index_of(image_path) -> int:
    """Frame index in a ``<index>_<ns>.<ext>`` file name."""
    return int(Path(image_path).name.split("_")[0])


@pytest.mark.skipif(not neurad_is_ready(), reason="needs torch + patched neurad-studio (reference image)")
def test_dataparser_keeps_radar_out_and_seeds_out_of_the_held_out_split(tmp_path):
    """Rule 5: radars config empty, no radar key in any split's metadata, no radar_seeds in eval, sidecar untouched."""
    write_block_split_dataset(tmp_path)
    config = make_block_split_config(tmp_path)
    assert tuple(config.radars) == ()
    for split_name in ("train", "eval"):
        outputs = make_block_split_config(tmp_path).setup().get_dataparser_outputs(split_name)
        radar_keys = [key for key in outputs.metadata if "radar" in key.lower() and key != "radar_seeds"]
        assert radar_keys == [], split_name
        sensor_names = set(outputs.metadata["sensor_idx_to_name"].values())
        assert sensor_names == {SYNTHETIC_CAMERA, SYNTHETIC_LIDAR}
        # The held-out split never carries seeds (they exist only for training).
        if split_name == "eval":
            assert "radar_seeds" not in outputs.metadata


@pytest.mark.skipif(not neurad_is_ready(), reason="needs torch + patched neurad-studio (reference image)")
def test_dataparser_does_not_load_ground_truth_cuboids(tmp_path):
    """Rule 6: cuboid actors are off and no trajectories reach either split."""
    write_block_split_dataset(tmp_path)
    config = make_block_split_config(tmp_path)
    assert config.load_cuboids is False
    for split_name in ("train", "eval"):
        outputs = make_block_split_config(tmp_path).setup().get_dataparser_outputs(split_name)
        assert outputs.metadata["trajectories"] == []


@pytest.mark.skipif(not neurad_is_ready(), reason="needs torch + patched neurad-studio (reference image)")
def test_dataparser_train_eval_disjoint_and_guard_frames_excluded(tmp_path):
    """T60/T62 through the real dataparser: train and eval share no frame, and guard frames are in neither.

    Expected sets come from the block formulas on the known 0.1 s frame times (D = 0.1 (N-1),
    L = D (1 - f), centre D/2 for K = 1): train outside the block, eval inside its core,
    guard = block minus core.
    """
    write_block_split_dataset(tmp_path)
    frame_times = np.arange(DATASET_FRAME_COUNT) * FRAME_PERIOD_SECONDS
    span = frame_times[-1] - frame_times[0]
    block_length = span * (1.0 - TEST_BLOCK_TRAIN_FRACTION)
    distance_to_centre = np.abs(frame_times - span / 2.0)
    expected_train = set(np.flatnonzero(distance_to_centre >= block_length / 2.0))
    expected_eval = set(np.flatnonzero(distance_to_centre < CORE_FRACTION * block_length / 2.0))
    expected_guard = set(range(DATASET_FRAME_COUNT)) - expected_train - expected_eval
    assert expected_guard and expected_eval and expected_train  # the scenario exercises all three

    train = make_block_split_config(tmp_path).setup().get_dataparser_outputs("train")
    evaluation = make_block_split_config(tmp_path).setup().get_dataparser_outputs("eval")
    train_indices = {frame_index_of(path) for path in train.image_filenames}
    eval_indices = {frame_index_of(path) for path in evaluation.image_filenames}
    assert train_indices == expected_train and eval_indices == expected_eval
    assert train_indices.isdisjoint(eval_indices)
    assert (train_indices | eval_indices).isdisjoint(expected_guard)
    # Lidar sweeps follow the same shared blocks: same counts, and disjoint timestamps.
    assert len(train.metadata["lidars"]) == len(expected_train)
    assert len(evaluation.metadata["lidars"]) == len(expected_eval)
    train_times = set(np.round(train.metadata["lidars"].times.squeeze(-1).numpy(), 6))
    eval_times = set(np.round(evaluation.metadata["lidars"].times.squeeze(-1).numpy(), 6))
    assert train_times.isdisjoint(eval_times)
    assert math.isclose(len(train_times | eval_times), len(expected_train) + len(expected_eval))
