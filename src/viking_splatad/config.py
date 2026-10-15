"""Session configuration loading and validation."""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

# Top-level sections every session YAML must provide.
REQUIRED_TOP_LEVEL_KEYS = (
    "session", "window", "topics", "frames", "camera", "lidar", "radar", "split", "splatad",
)
# Keys required inside the ``session`` section.
REQUIRED_SESSION_KEYS = (
    "name", "extrinsics_variant", "bag_dir", "calibration_dir", "labels_json",
    "labels_sample_key", "output_dir", "gt_dir",
)
# Subset of session keys that are filesystem paths (resolved against the project root).
PATH_SESSION_KEYS = ("bag_dir", "calibration_dir", "labels_json", "output_dir", "gt_dir")
# File whose presence marks the project root directory.
PROJECT_ROOT_MARKER = "pyproject.toml"


def find_project_root(start_path: Optional[Path] = None) -> Path:
    """Locate the project root: the nearest ancestor containing pyproject.toml.

    Args:
        start_path: Where to start walking upward (default: this file's location).

    Returns:
        Absolute path of the project root.

    Raises:
        FileNotFoundError: If no ancestor contains the marker file.
    """
    # Step 1: begin at the given path (or this module) and walk toward the filesystem root.
    current_path = (start_path or Path(__file__)).resolve()
    for candidate_directory in [current_path, *current_path.parents]:
        # Step 2: the first directory holding the marker is the root.
        if (candidate_directory / PROJECT_ROOT_MARKER).is_file():
            return candidate_directory
    raise FileNotFoundError(f"No {PROJECT_ROOT_MARKER} found above {current_path}")


def load_session_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load and validate a session YAML.

    Args:
        config_path: Path of the session YAML file.

    Returns:
        Plain nested dict; every path under ``session`` listed in PATH_SESSION_KEYS
        is an absolute Path (relative values are resolved against the project root).

    Raises:
        ValueError: If a required top-level or session key is missing (message names it).
    """
    # Step 1: parse the YAML.
    config = yaml.safe_load(Path(config_path).read_text())
    if not isinstance(config, dict):
        raise ValueError(f"{config_path}: top level must be a mapping")
    # Step 2: check required top-level sections.
    for required_key in REQUIRED_TOP_LEVEL_KEYS:
        if required_key not in config:
            raise ValueError(f"Missing required top-level key: {required_key!r}")
    # Step 3: check required session keys.
    session_section = config["session"]
    for required_key in REQUIRED_SESSION_KEYS:
        if required_key not in session_section:
            raise ValueError(f"Missing required session key: {required_key!r}")
    # Step 4: resolve session paths. Path join keeps already-absolute values unchanged.
    project_root = find_project_root()
    for path_key in PATH_SESSION_KEYS:
        session_section[path_key] = project_root / Path(session_section[path_key])
    return config
