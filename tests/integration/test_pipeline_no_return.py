"""Bag -> dataset pipeline writes the no-return sidecar (Phase B, option C) on the synthetic bag.

Reuses the synthetic scene of ``test_pipeline_synthetic.py``: 32 rings x 64 columns, azimuth
2*pi*column/64, elevation -20..+20 deg over the rings, robot driving along +x with identity
orientation, and exactly one no-return cell per sweep (ring 0, column 3).
"""

import json

import numpy as np

from tests.integration import test_pipeline_synthetic as synthetic
from tests.integration.test_pipeline_synthetic import (  # noqa: F401 (synthetic_session is a fixture)
    COLUMN_COUNT,
    LIDAR_PERIOD,
    NANOSECONDS_PER_SECOND,
    RING_COUNT,
    make_config,
    synthetic_session,
    write_chunk,
)
from tests.integration.test_sweep_period_coverage import STRETCHED_LIDAR_PERIOD, longest_point_time_seconds
from viking_splatad.contract.splatad_contract import validate_dataset
from viking_splatad.dataset.writer import NO_RETURN_DIRECTORY, read_dataset
from viking_splatad.pipeline.bag_to_dataset import DEFAULT_BEAM_FIT_MAX_MEDIAN_RESIDUAL_COLUMNS, run_pipeline
from viking_splatad.pipeline.derived import DERIVED_FILE_NAME

# The synthetic scene's single no-return cell (lidar_scene_start_frame).
NO_RETURN_RING, NO_RETURN_COLUMN = 0, 3
# Synthetic elevation of a ring: -20 + 40 * ring / (rings - 1) degrees (lidar_scene_start_frame).
SYNTHETIC_LOWEST_ELEVATION_DEG, SYNTHETIC_ELEVATION_SPAN_DEG = -20.0, 40.0


def analytic_no_return_direction() -> np.ndarray:
    """Unit direction of the no-return cell in the synthetic scene."""
    elevation = np.radians(SYNTHETIC_LOWEST_ELEVATION_DEG + SYNTHETIC_ELEVATION_SPAN_DEG * NO_RETURN_RING / (RING_COUNT - 1))
    azimuth = 2.0 * np.pi * NO_RETURN_COLUMN / COLUMN_COUNT
    return np.array([np.cos(elevation) * np.cos(azimuth), np.cos(elevation) * np.sin(azimuth), np.sin(elevation)])


def test_pipeline_writes_one_no_return_file_per_sweep(synthetic_session):
    """One file per sweep, with the no-return cell's channel, column, time and direction; contract passes."""
    config = make_config(synthetic_session)
    run_pipeline(config)
    output_dir = config["session"]["output_dir"]
    # Beam model: the synthetic columns turn counter-clockwise by 360/64 deg.
    beam_model = json.loads((output_dir / DERIVED_FILE_NAME).read_text())["beam_direction_model"]
    assert beam_model["column_step_deg"] == 360.0 / COLUMN_COUNT
    # Same file names in lidar/ and lidar_no_return/.
    sweep_names = sorted(path.name for path in (output_dir / "lidar" / "ouster").glob("*.npy"))
    no_return_names = sorted(path.name for path in (output_dir / NO_RETURN_DIRECTORY / "ouster").glob("*.npy"))
    assert sweep_names and no_return_names == sweep_names
    # Column 3 time: column * (period_ns // columns) ns after the sweep start (= reference, 'start').
    expected_time = NO_RETURN_COLUMN * (round(LIDAR_PERIOD * NANOSECONDS_PER_SECOND) // COLUMN_COUNT) / NANOSECONDS_PER_SECOND
    # Fit gate in degrees: the direction of a cell is trusted to within half an azimuth column.
    direction_bound_deg = DEFAULT_BEAM_FIT_MAX_MEDIAN_RESIDUAL_COLUMNS * 360.0 / COLUMN_COUNT
    for name in no_return_names:
        rows = np.load(output_dir / NO_RETURN_DIRECTORY / "ouster" / name)
        assert rows.shape == (1, 6) and rows.dtype == np.float32
        assert rows[0, 3] == np.float32(expected_time)
        assert (rows[0, 4], rows[0, 5]) == (NO_RETURN_RING, NO_RETURN_COLUMN)
        # Identity orientation: no rotation since the sweep start, so the beam direction itself.
        angle_deg = np.degrees(np.arccos(np.clip(rows[0, :3].astype(np.float64) @ analytic_no_return_direction(), -1, 1)))
        assert angle_deg < direction_bound_deg
    report = validate_dataset(output_dir)
    assert report.passed, report.to_markdown()


def test_no_return_rows_do_not_depend_on_the_lidar_source(synthetic_session):
    """The sidecar comes from the raw cells and our trajectory: 'deskewed' and 'raw_own_deskew' agree.

    'raw_own_deskew' skips a sweep whose point times run past the trajectory (existing returns
    behaviour), so the comparison covers the sweeps both sources wrote.
    """
    deskewed_config = make_config(synthetic_session, "deskewed", "out_deskewed")
    raw_config = make_config(synthetic_session, "raw_own_deskew", "out_raw")
    run_pipeline(deskewed_config)
    run_pipeline(raw_config)
    deskewed_directory = deskewed_config["session"]["output_dir"] / NO_RETURN_DIRECTORY / "ouster"
    raw_directory = raw_config["session"]["output_dir"] / NO_RETURN_DIRECTORY / "ouster"
    deskewed_names = {path.name for path in deskewed_directory.glob("*.npy")}
    raw_names = {path.name for path in raw_directory.glob("*.npy")}
    assert raw_names and raw_names <= deskewed_names
    for name in sorted(raw_names):
        np.testing.assert_array_equal(np.load(deskewed_directory / name), np.load(raw_directory / name))


def test_pre_phase_b_dataset_gets_a_beam_model_and_the_contract_flags_the_gap(synthetic_session):
    """A dataset extracted before Phase B (no model, no sidecar) that gets a new chunk: the model is
    fitted and saved, only the new sweeps get files, and the contract names the gap."""
    config = make_config(synthetic_session)
    run_pipeline(config)
    output_dir = config["session"]["output_dir"]
    # Turn the output into a pre-Phase-B dataset: drop the model from the cache and the sidecar.
    derived_path = output_dir / DERIVED_FILE_NAME
    derived_payload = json.loads(derived_path.read_text())
    derived_payload.pop("beam_direction_model")
    derived_path.write_text(json.dumps(derived_payload))
    for path in (output_dir / NO_RETURN_DIRECTORY / "ouster").glob("*.npy"):
        path.unlink()
    write_chunk(synthetic_session / "bag" / "short_grass__ros2_1.mcap", 1)
    run_pipeline(config)
    assert json.loads(derived_path.read_text())["beam_direction_model"] is not None
    sweep_count = len(list((output_dir / "lidar" / "ouster").glob("*.npy")))
    no_return_count = len(list((output_dir / NO_RETURN_DIRECTORY / "ouster").glob("*.npy")))
    assert 0 < no_return_count < sweep_count
    assert validate_dataset(output_dir).failed_check_names() == ["lidar_no_return_sidecar"]


def test_sweep_period_covers_no_return_times(synthetic_session, monkeypatch):
    """T49 with no-returns: a stretched sweep whose LAST column returns nothing still fits the period.

    Chunk 1's sweeps last STRETCHED_LIDAR_PERIOD and column 63 is a no-return in every ring, so the
    latest returns sit at column 62 and the latest no-returns at column 63. The manifest period
    must reach the column-63 time, or SplatAD would get no-return rows with |t| > sweep period.
    """
    config = make_config(synthetic_session)
    output_dir = config["session"]["output_dir"]
    run_pipeline(config)
    original_scene = synthetic.lidar_scene_start_frame

    def scene_without_last_column():
        """Synthetic scene with every ring's last column turned into a no-return."""
        points = original_scene()
        points[:, -1] = 0.0
        return points

    with monkeypatch.context() as patch:
        patch.setattr(synthetic, "LIDAR_PERIOD", STRETCHED_LIDAR_PERIOD)
        patch.setattr(synthetic, "lidar_scene_start_frame", scene_without_last_column)
        synthetic.write_chunk(synthetic_session / "bag" / "short_grass__ros2_1.mcap", 1)
    run_pipeline(config)
    # Column-63 time of a stretched sweep, float32 as written (analytic, see that helper).
    assert read_dataset(output_dir).lidar.sweep_period == longest_point_time_seconds(STRETCHED_LIDAR_PERIOD)
    report = validate_dataset(output_dir)
    assert report.passed, report.to_markdown()
