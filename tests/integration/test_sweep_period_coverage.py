"""Regression: the manifest sweep period must cover the per-point time of EVERY extracted sweep.

Real Ouster sweeps jitter in length by a fraction of a millisecond. ``sweep_period_seconds`` is
derived once from a small sample of the first chunk, so a later chunk can hold sweeps whose last
point time exceeds it (June chunks 21-22 failed ``lidar_point_time_range`` this way). These tests
reproduce that with the synthetic bag of ``test_pipeline_synthetic``: chunk 0 has nominal sweeps,
chunk 1 has stretched sweeps whose last point lies beyond the period derived from chunk 0.

Expected values are analytic: a sweep of length P has COLUMN_COUNT columns at t = c * (P_ns //
COLUMN_COUNT) ns, so its longest point time is (COLUMN_COUNT - 1) * (P_ns // COLUMN_COUNT) ns.
"""

import json
import shutil

import numpy as np
import pytest

from tests.integration import test_pipeline_synthetic as synthetic
from viking_splatad.contract.splatad_contract import validate_dataset
from viking_splatad.dataset.writer import read_dataset
from viking_splatad.pipeline.bag_to_dataset import run_pipeline
from viking_splatad.pipeline.derived import DERIVED_FILE_NAME, covering_sweep_period

# Stretched sweep length for chunk 1. It must satisfy (COLUMN_COUNT-1)/COLUMN_COUNT * P > the
# period derived from chunk 0 (= LIDAR_PERIOD, the median stamp spacing), i.e. P > 0.1 * 64/63
# = 0.10159 s. 0.105 s is a 5% stretch: clearly beyond that threshold. Chunk 1's stamps are
# spaced 0.105 s too, which keeps them strictly increasing and inside its window.
STRETCHED_LIDAR_PERIOD = 0.105


def longest_point_time_seconds(period_seconds: float) -> float:
    """Analytic longest per-point time of one synthetic sweep of length ``period_seconds``.

    Args:
        period_seconds: Sweep length used by ``build_pointcloud_message``.

    Returns:
        (COLUMN_COUNT - 1) * floor(P_ns / COLUMN_COUNT) nanoseconds, in seconds, rounded to
        float32 exactly as the written scan arrays store t (SplatAD scan layout is float32).
    """
    period_ns = round(period_seconds * synthetic.NANOSECONDS_PER_SECOND)
    exact_seconds = (synthetic.COLUMN_COUNT - 1) * (period_ns // synthetic.COLUMN_COUNT) / synthetic.NANOSECONDS_PER_SECOND
    return float(np.float32(exact_seconds))


def write_stretched_chunk(session_root, chunk_number, monkeypatch) -> None:
    """Write a synthetic chunk whose lidar sweeps last STRETCHED_LIDAR_PERIOD.

    ``build_pointcloud_message`` and the stamp spacing both read the module-level LIDAR_PERIOD
    at call time, so patching it only while this chunk is written stretches exactly its sweeps.

    Args:
        session_root: Synthetic session directory (holds ``bag``).
        chunk_number: Chunk index to write.
        monkeypatch: pytest monkeypatch fixture.
    """
    with monkeypatch.context() as patch:
        patch.setattr(synthetic, "LIDAR_PERIOD", STRETCHED_LIDAR_PERIOD)
        synthetic.write_chunk(session_root / "bag" / f"short_grass__ros2_{chunk_number}.mcap", chunk_number)


@pytest.mark.covers("T49")
def test_covering_sweep_period_is_the_larger_bound():
    """T49 known answers: the derived period wins unless an observed point time exceeds it."""
    # Observed point times shorter than the derived period: keep the derived period.
    assert covering_sweep_period(0.100194, 0.100119) == 0.100194
    # A longer observed point time (the June case, 0.100395 s) raises the bound to it.
    assert covering_sweep_period(0.100194, 0.100395) == 0.100395
    # Equal values are a fixed point.
    assert covering_sweep_period(0.1, 0.1) == 0.1


def test_later_chunk_with_longer_sweeps_passes_contract(synthetic_session, monkeypatch):
    """Incremental run: chunk 1 sweeps exceed the period derived from chunk 0; the contract passes."""
    config = synthetic.make_config(synthetic_session)
    output_dir = config["session"]["output_dir"]
    # Chunk 0 (nominal) derives and caches the period; chunk 1 (stretched) is added later.
    run_pipeline(config)
    derived_period = json.loads((output_dir / DERIVED_FILE_NAME).read_text())["sweep_period_seconds"]
    write_stretched_chunk(synthetic_session, 1, monkeypatch)
    run_pipeline(config)

    # Precondition of the regression: some stretched sweep really exceeds the derived period.
    stretched_longest = longest_point_time_seconds(STRETCHED_LIDAR_PERIOD)
    assert stretched_longest > derived_period

    report = validate_dataset(output_dir)
    assert report.passed, report.to_markdown()
    # The manifest bound equals the longest per-point time actually written (analytic value).
    manifest_period = json.loads((output_dir / "manifest.json").read_text())["lidar"]["sweep_period"]
    written_longest = max(np.abs(sweep[:, 4]).max() for sweep in read_dataset(output_dir).streams["ouster"].frames)
    assert written_longest == stretched_longest
    assert manifest_period == stretched_longest


def test_covering_period_survives_resume_and_matches_fresh_run(synthetic_session, monkeypatch):
    """The bound is the same for incremental, from-scratch and idempotent re-runs."""
    write_stretched_chunk(synthetic_session, 1, monkeypatch)
    incremental_config = synthetic.make_config(synthetic_session)
    run_pipeline(incremental_config, chunk_names=["short_grass__ros2_0.mcap"])
    run_pipeline(incremental_config)
    fresh_config = synthetic.make_config(synthetic_session, output_name="fresh")
    run_pipeline(fresh_config)

    def manifest_period(config) -> float:
        """Sweep period stored in a dataset's manifest."""
        return json.loads((config["session"]["output_dir"] / "manifest.json").read_text())["lidar"]["sweep_period"]

    expected = longest_point_time_seconds(STRETCHED_LIDAR_PERIOD)
    assert manifest_period(incremental_config) == expected
    assert manifest_period(fresh_config) == expected
    # A third run processes nothing and must not lose the bound.
    run_pipeline(incremental_config)
    assert manifest_period(incremental_config) == expected
    assert validate_dataset(incremental_config["session"]["output_dir"]).passed


@pytest.fixture
def synthetic_session(tmp_path):
    """Same layout as ``test_pipeline_synthetic.synthetic_session``: calibration plus nominal chunk 0.

    Args:
        tmp_path: pytest temporary directory.

    Returns:
        Session root holding ``bag/short_grass__ros2_0.mcap`` and ``calibration/extrinsics``.
    """
    (tmp_path / "bag").mkdir()
    (tmp_path / "calibration" / "extrinsics").mkdir(parents=True)
    shutil.copy(synthetic.FIXTURE_EXTRINSICS, tmp_path / "calibration" / "extrinsics" / "extrinsics.txt")
    synthetic.write_chunk(tmp_path / "bag" / "short_grass__ros2_0.mcap", 0)
    return tmp_path


def test_state_without_longest_point_time_is_recomputed_from_frames(synthetic_session, monkeypatch):
    """Datasets extracted before T49 lack the state key; it is rebuilt from the recorded frames.

    Only frames below the recorded count may contribute: a stale higher-index file (left by an
    interrupted run) holding a huge t must not inflate the bound.
    """
    write_stretched_chunk(synthetic_session, 1, monkeypatch)
    config = synthetic.make_config(synthetic_session)
    output_dir = config["session"]["output_dir"]
    run_pipeline(config)
    state_path = output_dir / "extract_state.json"
    # Simulate a pre-T49 state file: drop the key the fix adds.
    state = json.loads(state_path.read_text())
    del state["longest_point_time_seconds"]
    state_path.write_text(json.dumps(state))
    # A stale frame beyond the recorded count, with |t| = 1 s, would break the bound if counted.
    lidar_directory = output_dir / "lidar" / "ouster"
    recorded_count = len(list(lidar_directory.glob("*.npy")))
    stale_scan = np.zeros((1, 6), dtype=np.float32)
    stale_scan[0, 4] = 1.0
    np.save(lidar_directory / f"{recorded_count:06d}_1.npy", stale_scan)

    # A new chunk forces a run (and a manifest rewrite) on the resumed state.
    synthetic.write_chunk(synthetic_session / "bag" / "short_grass__ros2_2.mcap", 2)
    run_pipeline(config)

    manifest_period = json.loads((output_dir / "manifest.json").read_text())["lidar"]["sweep_period"]
    assert manifest_period == longest_point_time_seconds(STRETCHED_LIDAR_PERIOD)
    assert validate_dataset(output_dir).passed
