"""Tests for the OUSTER_OS0_32 neurad-studio patch (T28).

Text-level tests run anywhere. The behavioural test needs torch + a neurad-studio checkout
patched by ``install_into_neurad.sh``; it skips cleanly otherwise (see the command at the
bottom of ``test_viking_dataparser.py``'s docstring).
"""

import importlib
from pathlib import Path

import pytest

from viking_splatad.splatad.patches.apply_ouster_patch import (
    PATCH_MARKER,
    PatchError,
    patch_lidars_source,
)

# Minimal stand-in reproducing every anchor the patch relies on in upstream lidars.py.
UPSTREAM_STAND_IN = '''
from typing import Dict
from enum import Enum, auto


class LidarType(Enum):
    VELODYNE16 = auto()
    WOD64 = auto()


LIDAR_MODEL_TO_TYPE = {
    "VELODYNE16": LidarType.VELODYNE16,
    "WOD64": LidarType.WOD64,
}


def get_lidar_elevation_mapping(lidar_type: LidarType) -> dict:
    if lidar_type == LidarType.VELODYNE16:
        raise NotImplementedError("none")
    else:
        raise ValueError(f"Invalid lidar type: {lidar_type}")


def get_lidar_azimuth_resolution(lidar_type: LidarType) -> float:
    if lidar_type == LidarType.VELODYNE16:
        return 0.2
    else:
        raise ValueError(f"Invalid lidar type: {lidar_type}")


def get_lidar_relovution_time(lidar_type: LidarType) -> float:
    if lidar_type == LidarType.VELODYNE16:
        return 0.1
    else:
        raise ValueError(f"Invalid lidar type: {lidar_type}")
'''


def _exec_module(source: str) -> dict:
    """Execute source text and return its namespace."""
    namespace: dict = {}
    exec(compile(source, "patched_lidars", "exec"), namespace)
    return namespace


@pytest.mark.covers("T28")
def test_patch_adds_type_and_registry_backed_accessors():
    """Patched stand-in: new enum member, lookup entry, accessors return registered values."""
    namespace = _exec_module(patch_lidars_source(UPSTREAM_STAND_IN))
    ouster_type = namespace["LidarType"].OUSTER_OS0_32
    assert namespace["LIDAR_MODEL_TO_TYPE"]["OUSTER_OS0_32"] is ouster_type
    # Existing members keep their numeric values (enum member appended last).
    assert namespace["LidarType"].WOD64.value == 2
    # Unregistered -> loud ValueError, never a silent default.
    with pytest.raises(ValueError):
        namespace["get_lidar_elevation_mapping"](ouster_type)
    mapping_degrees = {channel: -10.0 + 2.0 * channel for channel in range(8)}
    namespace["register_lidar_model"](ouster_type, mapping_degrees, 360.0 / 512, 0.1)
    assert namespace["get_lidar_elevation_mapping"](ouster_type) == mapping_degrees
    assert namespace["get_lidar_azimuth_resolution"](ouster_type) == 360.0 / 512
    assert namespace["get_lidar_relovution_time"](ouster_type) == 0.1
    # Untouched branches still work.
    assert namespace["get_lidar_azimuth_resolution"](namespace["LidarType"].VELODYNE16) == 0.2


@pytest.mark.covers("T28")
def test_patch_is_idempotent_and_fails_loudly_on_drift():
    """Second application is a no-op; a source missing an anchor raises PatchError."""
    once = patch_lidars_source(UPSTREAM_STAND_IN)
    assert PATCH_MARKER in once
    assert patch_lidars_source(once) == once
    with pytest.raises(PatchError):
        patch_lidars_source(UPSTREAM_STAND_IN.replace("WOD64 = auto()", "OTHER = auto()"))


@pytest.mark.covers("T28")
def test_patch_applies_to_real_checkout_source_if_present():
    """The real upstream file (third_party checkout) patches into valid Python."""
    lidars_path = Path(__file__).parents[2] / "third_party/neurad-studio/nerfstudio/cameras/lidars.py"
    if not lidars_path.is_file():
        pytest.skip("third_party/neurad-studio not cloned")
    patched = patch_lidars_source(lidars_path.read_text())
    compile(patched, str(lidars_path), "exec")
    assert patched.count("OUSTER_OS0_32") >= 5


@pytest.mark.covers("T28")
def test_patched_neurad_module_returns_registered_values():
    """In a neurad env with the patch installed, the real accessors use the registry."""
    pytest.importorskip("torch", reason="neurad env only (see test_viking_dataparser docstring)")
    lidars_module = importlib.import_module("nerfstudio.cameras.lidars")
    if not hasattr(lidars_module.LidarType, "OUSTER_OS0_32"):
        pytest.skip("run src/viking_splatad/splatad/install_into_neurad.sh first")
    ouster_type = lidars_module.LidarType.OUSTER_OS0_32
    registry = lidars_module._RUNTIME_LIDAR_MODELS
    registry.pop(ouster_type, None)
    with pytest.raises(ValueError):
        lidars_module.get_lidar_elevation_mapping(ouster_type)
    mapping_degrees = {channel: 45.0 - 3.0 * channel for channel in range(32)}
    lidars_module.register_lidar_model(ouster_type, mapping_degrees, 360.0 / 1024, 0.1)
    try:
        assert lidars_module.get_lidar_elevation_mapping(ouster_type) == mapping_degrees
        assert lidars_module.get_lidar_azimuth_resolution(ouster_type) == 360.0 / 1024
        assert lidars_module.LIDAR_MODEL_TO_TYPE["OUSTER_OS0_32"] is ouster_type
    finally:
        registry.pop(ouster_type, None)
