"""A radar artifact verifies when its dataset folder is named differently from the extraction session.

The dataparser and the contract name a dataset by its folder (``<data root>/<folder>``), while the
artifact and its clock report carry the session the data was extracted with (``manifest.json``
``session.name``). ``may_chunks0_1_smoke`` was extracted with ``may_short_grass``; its artifact must
still verify when named by its folder, and a name of neither kind must still be refused.
"""

import json

import pytest

from tests.unit.test_loss_only_artifact import synthetic_artifact
from viking_splatad.contract.splatad_contract import validate_radar_inputs
from viking_splatad.radar_init.artifact import load_verified_anchors
from viking_splatad.radar_init.build import build_loss_only_artifact


def build_artifact_in_differently_named_folder(tmp_path):
    """Build the synthetic artifact; its dataset folder name differs from its session name.

    Args:
        tmp_path: pytest temporary directory.

    Returns:
        (artifact directory, dataset directory, split settings, extraction session name).
    """
    dataset_directory, settings, session_config, calibration, options, _ = synthetic_artifact(tmp_path)
    extraction_session = session_config["session"]["name"]
    # The premise of every test below: folder name and extraction session differ.
    assert dataset_directory.name != extraction_session
    artifact_directory = tmp_path / "anchors"
    build_loss_only_artifact(dataset_directory, session_config, settings, options, artifact_directory, calibration)
    return artifact_directory, dataset_directory, settings, extraction_session


def test_artifact_verifies_when_named_by_its_dataset_folder(tmp_path):
    """The dataparser passes the folder name; the artifact's session tag is the extraction session."""
    artifact_directory, dataset_directory, settings, extraction_session = (
        build_artifact_in_differently_named_folder(tmp_path))
    _, metadata = load_verified_anchors(artifact_directory, dataset_directory, dataset_directory.name, settings)
    # The tag stays the extraction session; only the caller's naming differs.
    assert metadata["session"] == extraction_session


def test_artifact_still_verifies_when_named_by_its_extraction_session(tmp_path):
    """The builder's own check passes the extraction session; that must keep working."""
    artifact_directory, dataset_directory, settings, extraction_session = (
        build_artifact_in_differently_named_folder(tmp_path))
    load_verified_anchors(artifact_directory, dataset_directory, extraction_session, settings)


def test_artifact_refuses_a_name_of_neither_the_folder_nor_the_session(tmp_path):
    """A caller naming another dataset is refused even though the artifact itself is intact."""
    artifact_directory, dataset_directory, settings, extraction_session = (
        build_artifact_in_differently_named_folder(tmp_path))
    unrelated_name = f"{dataset_directory.name}_{extraction_session}_other"
    with pytest.raises(ValueError, match="requested for session"):
        load_verified_anchors(artifact_directory, dataset_directory, unrelated_name, settings)


def test_artifact_with_a_session_tag_other_than_the_manifest_is_refused(tmp_path):
    """The tag is compared with the dataset manifest; its checksum covers only the npz, so this is the guard."""
    artifact_directory, dataset_directory, settings, extraction_session = (
        build_artifact_in_differently_named_folder(tmp_path))
    metadata_path = artifact_directory / "radar_inputs.json"
    metadata = json.loads(metadata_path.read_text())
    # Retag the artifact with the folder name: a name the caller may use, but not the extraction session.
    metadata["session"] = dataset_directory.name
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="tags do not match"):
        load_verified_anchors(artifact_directory, dataset_directory, dataset_directory.name, settings)


def test_contract_passes_when_named_by_the_dataset_folder(tmp_path):
    """Scripts 06/07 and submit_run's gates pass the dataset folder name to the contract."""
    artifact_directory, dataset_directory, settings, _ = build_artifact_in_differently_named_folder(tmp_path)
    report = validate_radar_inputs(artifact_directory, dataset_directory.name, dataset_directory, settings,
                                   require_seed_colours=False)
    assert report.passed, report.to_markdown()
