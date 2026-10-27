"""Read scalar fields of nerfstudio's ``config.yml`` without constructing its Python objects."""

import re
from pathlib import Path
from typing import Optional

# YAML literal nerfstudio writes for None.
YAML_NULL_LITERALS = ("null", "~", "None")


def read_config_scalar(config_path: Path, field_name: str) -> Optional[str]:
    """Read one scalar field of nerfstudio's ``config.yml`` without constructing its Python objects.

    config.yml holds ``!!python/object`` tags that need nerfstudio to load, so the field is read
    with a line regex instead (field names here are unique within the dataparser config).

    Args:
        config_path: The run's ``config.yml``.
        field_name: Field such as ``train_split_fraction_override``.

    Returns:
        The raw value string, None for a YAML null or an absent field/file.
    """
    # Step 1: an absent file means an absent field.
    if not config_path.is_file():
        return None
    # Step 2: match "<indent>field: value" on one line; the value is a single token.
    field_match = re.search(rf"^\s*{re.escape(field_name)}:\s*(\S+)\s*$", config_path.read_text(), flags=re.MULTILINE)
    if field_match is None or field_match.group(1) in YAML_NULL_LITERALS:
        return None
    return field_match.group(1)
