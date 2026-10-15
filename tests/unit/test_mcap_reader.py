"""Tests for the MCAP reader helper (legacy uppercase field names)."""

import struct

from mcap.writer import Writer

from viking_splatad.io.mcap_reader import make_decoded_reader

# CDR little-endian encapsulation header: representation id 0x0001, options 0x0000.
CDR_LITTLE_ENDIAN_HEADER = bytes([0x00, 0x01, 0x00, 0x00])


def write_single_message_bag(path, schema_text: bytes, payload: bytes) -> None:
    """Write a one-message MCAP with a ros2msg schema and CDR payload.

    Args:
        path: Output file path.
        schema_text: ROS .msg definition text.
        payload: CDR-serialised message body (without the encapsulation header).
    """
    with open(path, "wb") as handle:
        writer = Writer(handle)
        writer.start()
        schema_id = writer.register_schema(name="test_msgs/msg/Legacy", encoding="ros2msg", data=schema_text)
        channel_id = writer.register_channel(topic="/legacy", message_encoding="cdr", schema_id=schema_id)
        writer.add_message(channel_id=channel_id, log_time=1, publish_time=1,
                           data=CDR_LITTLE_ENDIAN_HEADER + payload)
        writer.finish()


def test_uppercase_field_schema_decodes(tmp_path):
    """A schema with an uppercase field name (as in the bag's CameraInfo) decodes to the written value."""
    bag_path = tmp_path / "legacy.mcap"
    expected_value = -1234567  # arbitrary int32 distinguishing a real decode from a default
    write_single_message_bag(bag_path, b"int32 D\n", struct.pack("<i", expected_value))
    with open(bag_path, "rb") as handle:
        reader = make_decoded_reader(handle)
        decoded_messages = [decoded for _, _, _, decoded in reader.iter_decoded_messages()]
    assert len(decoded_messages) == 1
    assert decoded_messages[0].D == expected_value
