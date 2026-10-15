"""Patch neurad-studio's ``nerfstudio/cameras/lidars.py`` with an Ouster OS0-32 lidar (T28).

Design (why a runtime registry instead of a hardcoded elevation table):
    The OS0-32 beam elevations are DERIVED from data (T16) and stored per dataset in the
    manifest, so they must not be baked into neurad-studio from memory. The datamanager
    calls ``get_lidar_elevation_mapping(lidar_type)`` / ``get_lidar_azimuth_resolution``
    with only the ``LidarType`` (no dataset handle), so the patched functions look the
    model up in a module-level registry that ``VikingHillDataParser`` fills with the
    manifest values (``register_lidar_model``) before any datamanager code runs, in the
    same process (the datamanager loads lidars through a thread pool, not subprocesses).
    If nothing was registered the functions raise ``ValueError`` (never a silent default).
    Rejected alternative: pass mappings through ``ADDataParserConfig.lidar_elevation_mapping``;
    that field only feeds ``_get_missing_points`` and is invisible to the datamanager.

Channel ordering: SplatAD sorts elevation VALUES ascending to build raster rows
(``sorted(elevation_mapping.values())``) and bins points by their computed elevation, never
by channel id. Our rings run top-down (ring 0 = highest elevation) while raster rows run
bottom-up; this is harmless because ids matter only in ``_get_missing_points``, which
matches ``point[:, channel] == id`` against ``mapping[id]``; both come from the same
ring numbering, so they stay consistent.

The patch is idempotent (marker comment) and fails loudly if upstream anchors move.
"""

import argparse
from pathlib import Path

from viking_splatad.registry import transform

# Marker written into the patched file; its presence means "already patched".
PATCH_MARKER = "# viking-splatad: OUSTER_OS0_32 patch"
# Path of the patched file relative to a neurad-studio checkout root.
LIDARS_RELATIVE_PATH = Path("nerfstudio/cameras/lidars.py")

# Registry + accessor inserted just before ``get_lidar_elevation_mapping``.
REGISTRY_SOURCE = f'''{PATCH_MARKER}
_RUNTIME_LIDAR_MODELS: Dict[LidarType, dict] = {{}}


def register_lidar_model(
    lidar_type: LidarType,
    elevation_mapping_degrees: Dict[int, float],
    azimuth_resolution_degrees: float,
    revolution_time_seconds: float,
) -> None:
    """Register a data-derived lidar model (called by the dataparser, values from the manifest)."""
    _RUNTIME_LIDAR_MODELS[lidar_type] = {{
        "elevation_mapping": dict(elevation_mapping_degrees),
        "azimuth_resolution": float(azimuth_resolution_degrees),
        "revolution_time": float(revolution_time_seconds),
    }}


def _get_registered_lidar_model(lidar_type: LidarType) -> dict:
    """Return the registered model or fail loudly (no silent defaults)."""
    if lidar_type not in _RUNTIME_LIDAR_MODELS:
        raise ValueError(
            f"{{lidar_type}} has no registered model; the dataparser must call register_lidar_model first"
        )
    return _RUNTIME_LIDAR_MODELS[lidar_type]


'''

# (function name, key into the registered model) pairs that gain an OUSTER branch.
PATCHED_FUNCTIONS = (
    ("get_lidar_elevation_mapping", "elevation_mapping"),
    ("get_lidar_azimuth_resolution", "azimuth_resolution"),
    ("get_lidar_relovution_time", "revolution_time"),
)
# Text every upstream accessor ends with; the new branch goes right before it.
ELSE_ANCHOR = '    else:\n        raise ValueError(f"Invalid lidar type: {lidar_type}")'


class PatchError(RuntimeError):
    """Raised when upstream source no longer matches the anchors this patch relies on."""


def _replace_once(source: str, anchor: str, replacement: str, description: str) -> str:
    """Replace exactly one occurrence of ``anchor``.

    Args:
        source: Text to edit.
        anchor: Text that must occur exactly once.
        replacement: Its replacement.
        description: What the edit does (for the error message).

    Returns:
        Edited text.

    Raises:
        PatchError: If the anchor is absent or ambiguous.
    """
    if source.count(anchor) != 1:
        raise PatchError(f"anchor for '{description}' found {source.count(anchor)}x, expected 1")
    return source.replace(anchor, replacement)


@transform("T28", "Patch neurad lidars.py: OUSTER_OS0_32 LidarType + manifest-fed registry")
def patch_lidars_source(source: str) -> str:
    """Return ``lidars.py`` source with the Ouster OS0-32 support added (idempotent).

    Edits: (1) enum member appended last (existing enum values keep their numbers);
    (2) ``LIDAR_MODEL_TO_TYPE`` entry; (3) registry before the accessors;
    (4) an ``OUSTER_OS0_32`` branch in each of the three accessors.

    Args:
        source: Upstream (or already patched) file text.

    Returns:
        Patched text; identical to the input when the marker is already present.

    Raises:
        PatchError: If an upstream anchor is missing (upstream changed).
    """
    # Idempotence: a patched file is returned untouched.
    if PATCH_MARKER in source:
        return source
    # 1) enum member, last so no existing auto() value shifts.
    source = _replace_once(
        source, "    WOD64 = auto()\n", f"    WOD64 = auto()\n    OUSTER_OS0_32 = auto()  {PATCH_MARKER}\n", "enum member"
    )
    # 2) model-name lookup.
    source = _replace_once(
        source,
        '    "WOD64": LidarType.WOD64,\n}',
        '    "WOD64": LidarType.WOD64,\n    "OUSTER_OS0_32": LidarType.OUSTER_OS0_32,\n}',
        "LIDAR_MODEL_TO_TYPE entry",
    )
    # 3) registry, placed right before the first accessor.
    source = _replace_once(
        source,
        "def get_lidar_elevation_mapping(",
        REGISTRY_SOURCE + "def get_lidar_elevation_mapping(",
        "registry",
    )
    # 4) branch in every accessor: cut each function body out and patch its final else.
    for function_name, registry_key in PATCHED_FUNCTIONS:
        function_start = source.index(f"def {function_name}(")
        # The accessor ends at the next top-level statement (or end of file).
        next_top_level = source.find("\n\n\ndef ", function_start)
        function_end = len(source) if next_top_level == -1 else next_top_level
        body = source[function_start:function_end]
        branch = (
            "    elif lidar_type == LidarType.OUSTER_OS0_32:\n"
            f'        return _get_registered_lidar_model(lidar_type)["{registry_key}"]\n'
        )
        source = source[:function_start] + _replace_once(
            body, ELSE_ANCHOR, branch + ELSE_ANCHOR, f"{function_name} branch"
        ) + source[function_end:]
    return source


def apply_to_checkout(neurad_root: Path) -> bool:
    """Patch ``lidars.py`` of a neurad-studio checkout in place.

    Args:
        neurad_root: Checkout root (contains ``nerfstudio/``).

    Returns:
        True if the file changed, False if it was already patched.
    """
    lidars_path = Path(neurad_root) / LIDARS_RELATIVE_PATH
    original = lidars_path.read_text()
    patched = patch_lidars_source(original)
    if patched != original:
        lidars_path.write_text(patched)
    return patched != original


def main() -> None:
    """CLI: ``python -m viking_splatad.splatad.patches.apply_ouster_patch <neurad_root>``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("neurad_root", type=Path)
    arguments = parser.parse_args()
    changed = apply_to_checkout(arguments.neurad_root)
    print("patched" if changed else "already patched")


if __name__ == "__main__":
    main()
