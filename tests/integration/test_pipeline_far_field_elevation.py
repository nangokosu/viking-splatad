"""Bag -> dataset pipeline writes the T61 far-field elevation mapping into the manifest.

Reuses the synthetic scene of ``test_pipeline_synthetic.py`` (32 rings, beams from the sensor origin).
"""

import json

import numpy as np
import pytest

from tests.integration.test_pipeline_synthetic import (  # noqa: F401 (synthetic_session is a fixture)
    make_config,
    synthetic_session,
)
from viking_splatad.contract.splatad_contract import validate_dataset
from viking_splatad.dataset.writer import read_dataset
from viking_splatad.extract.no_return import BeamDirectionModel, elevation_mapping_from_beam_model
from viking_splatad.pipeline.bag_to_dataset import ELEVATION_MAPPING_SOURCE, MANIFEST_FILE_NAME, run_pipeline
from viking_splatad.pipeline.derived import DERIVED_FILE_NAME


@pytest.mark.covers("T61")
def test_manifest_elevation_mapping_is_the_beam_model_far_field(synthetic_session):
    """manifest lidar.elevation_mapping = radians(T61(derived beam model, T16 ring set)); source recorded."""
    config = make_config(synthetic_session)
    run_pipeline(config)
    output_dir = config["session"]["output_dir"]
    derived = json.loads((output_dir / DERIVED_FILE_NAME).read_text())
    # Expected mapping from the cached T51 model over the cached T16 channel keys, in radians.
    expected_degrees = elevation_mapping_from_beam_model(
        BeamDirectionModel.from_dict(derived["beam_direction_model"]),
        [int(channel) for channel in derived["elevation_mapping_deg"]])
    written = read_dataset(output_dir).lidar.elevation_mapping
    assert written == {channel: float(np.radians(angle)) for channel, angle in expected_degrees.items()}
    # Written in ascending elevation order, as SplatAD's raster rows expect.
    assert list(written.values()) == sorted(written.values())
    manifest = json.loads((output_dir / MANIFEST_FILE_NAME).read_text())
    assert manifest["pipeline"]["elevation_mapping_source"] == ELEVATION_MAPPING_SOURCE
    report = validate_dataset(output_dir)
    assert report.passed, report.to_markdown()
