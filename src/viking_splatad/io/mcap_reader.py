"""Helpers for reading ROS2 MCAP bags with the pure-python `mcap-ros2-support` decoder."""

import re
from typing import BinaryIO

from mcap.reader import make_reader
from mcap_ros2._vendor.rosidl_adapter import parser as rosidl_parser
from mcap_ros2.decoder import DecoderFactory

# The Viking Hill bags embed a sensor_msgs/CameraInfo schema with the legacy uppercase field
# names (D, K, R, P). The vendored rosidl parser only accepts lowercase snake_case, so it
# raises NameError on that schema. This is the permissive pattern the library itself keeps
# commented out next to its strict one (rosidl_adapter/parser.py): letters, digits, underscore.
PERMISSIVE_FIELD_NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


def allow_legacy_field_names() -> None:
    """Make the vendored ROS message parser accept uppercase field names (idempotent).

    The parser's `is_valid_field_name` reads the module-level pattern at call time, so
    swapping that attribute is enough; no parser code is modified.

    Returns:
        None. Effect is process-wide.
    """
    rosidl_parser.VALID_FIELD_NAME_PATTERN = PERMISSIVE_FIELD_NAME_PATTERN


def make_decoded_reader(handle: BinaryIO):
    """Open an MCAP reader that decodes ROS2 CDR messages, tolerating legacy schemas.

    Args:
        handle: Binary file object of an MCAP file.

    Returns:
        An `mcap.reader.McapReader` whose `iter_decoded_messages` yields decoded ROS messages.
    """
    # Patch before the first schema is parsed, otherwise decoding CameraInfo raises NameError.
    allow_legacy_field_names()
    return make_reader(handle, decoder_factories=[DecoderFactory()])
