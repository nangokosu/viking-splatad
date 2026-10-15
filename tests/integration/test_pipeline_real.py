"""Run the pipeline on the first seconds of the real May chunk 0 (skipped if the bag is absent)."""

import time
from pathlib import Path

import numpy as np
import pytest

from viking_splatad.config import find_project_root, load_session_config
from viking_splatad.contract.splatad_contract import validate_dataset
from viking_splatad.dataset.writer import read_dataset
from viking_splatad.pipeline.bag_to_dataset import run_pipeline

SESSION_CONFIG_RELATIVE_PATH = Path("configs") / "sessions" / "may_short_grass.yaml"
# Seconds of the session to extract: 5 s = ~50 sweeps, enough for rate checks and for the
# 50-scan model derivation, while keeping the run short.
EXTRACT_SECONDS = 5.0
# Nominal sensor rate from the Phase 1 report (10 Hz camera and lidar); the check allows a
# generous +-20 % band because the window edges cut partial periods.
NOMINAL_RATE_HZ = 10.0
RATE_BAND = 0.2


@pytest.mark.real_data
def test_real_may_chunk0_first_seconds(tmp_path, capsys):
    """Contract passes on real data; camera and lidar run at ~10 Hz with strictly increasing stamps."""
    config = load_session_config(find_project_root() / SESSION_CONFIG_RELATIVE_PATH)
    chunk_path = Path(config["session"]["bag_dir"]) / "short_grass__ros2_0.mcap"
    if not chunk_path.is_file():
        pytest.skip(f"real chunk not downloaded: {chunk_path}")
    # Write into tmp_path so the real output directory is untouched.
    config["session"]["output_dir"] = tmp_path / "out"
    start_time = time.perf_counter()
    result = run_pipeline(config, chunk_names=[chunk_path.name], max_seconds=EXTRACT_SECONDS)
    elapsed_seconds = time.perf_counter() - start_time
    report = validate_dataset(config["session"]["output_dir"])
    with capsys.disabled():
        print(f"\nREAL RUN: {result.frames_total} frames in {elapsed_seconds:.1f} s; "
              f"dropped {result.skipped_counts}; warnings {result.warnings}")
        print(report.to_markdown())
    assert report.passed, report.to_markdown()
    dataset = read_dataset(config["session"]["output_dir"])
    for sensor_name in (config["camera"]["name"], config["lidar"]["name"]):
        times = dataset.streams[sensor_name].times
        assert len(times) > 1 and np.all(np.diff(times) > 0)
        measured_rate = 1.0 / np.median(np.diff(times))
        assert abs(measured_rate - NOMINAL_RATE_HZ) <= RATE_BAND * NOMINAL_RATE_HZ
