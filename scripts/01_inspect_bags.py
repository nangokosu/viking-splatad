"""Inspect real Viking Hill bag contents to resolve the plan's "unverified" items.

Reports, for one MCAP file: PointCloud2 field layouts and frame ids (lidar raw/deskewed,
radar), camera_info (distortion model + coefficients), first odometry pose, tf_static
transforms, header-stamp vs bag-log-time offsets, message rates; plus raw dumps of the
calibration text files and the structure of the cuboid-label JSON.

Output is Markdown printed to stdout (redirect it into reports/).
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from viking_splatad.io.mcap_reader import make_decoded_reader

# Topics whose first message is described in detail (others only get counts).
POINTCLOUD_TOPICS = ["/ouster/points", "/point_cloud_deskewed", "/hugin_raf_1/radar_data"]
# How many messages to sample per topic for stamp/rate statistics; a small sample is enough
# to see the median spacing and offsets, and keeps the scan fast on a 2 GB file.
STAMP_SAMPLE_COUNT = 50
# Max entries printed when summarising a list in the cuboid JSON.
JSON_PREVIEW_COUNT = 3
# JSON outline limits, chosen for a readable report (no data-derived value exists):
# the deepest nesting level shown, dict keys listed per node, dict children recursed into
# per node, and the max characters of a printed leaf line.
JSON_OUTLINE_MAX_DEPTH = 6
JSON_LISTED_KEY_COUNT = 12
JSON_RECURSED_KEY_COUNT = JSON_PREVIEW_COUNT + 3
JSON_LEAF_LINE_MAX_CHARS = 160
# /tf messages scanned for dynamic frame pairs: a bounded prefix (a few seconds of a high-rate
# topic) is enough to see every pair the robot publishes, without reading the whole bag.
TF_DYNAMIC_MESSAGE_LIMIT = 2000
# sensor_msgs/PointField datatype codes (fixed by the ROS message definition) -> numpy dtypes.
POINTFIELD_DTYPES = {1: "i1", 2: "u1", 3: "i2", 4: "u2", 5: "i4", 6: "u4", 7: "f4", 8: "f8"}
# Topics reported as camera and odometry sections.
CAMERA_INFO_TOPIC = "/ids_camera/image_raw/camera_info"
CAMERA_IMAGE_TOPIC = "/ids_camera/image_raw/compressed"
ODOMETRY_TOPICS = ["/icp_odom", "/husky_udp_bridge/odom", "/imu_odom"]


def stamp_to_seconds(stamp) -> float:
    """Convert a ROS time (sec, nanosec) message field to float seconds.

    Args:
        stamp: Decoded builtin_interfaces/Time.

    Returns:
        Seconds as float.
    """
    return stamp.sec + stamp.nanosec * 1e-9


def collect_samples(mcap_path: Path, topics: List[str], per_topic: int) -> Dict[str, list]:
    """Read the first `per_topic` decoded messages for each requested topic.

    Args:
        mcap_path: MCAP file path.
        topics: Topic names to sample.
        per_topic: Messages to keep per topic.

    Returns:
        Mapping topic -> list of (log_time_seconds, decoded_message).
    """
    samples: Dict[str, list] = {topic: [] for topic in topics}
    with open(mcap_path, "rb") as handle:
        reader = make_decoded_reader(handle)
        # Iterate topic-by-topic so a quiet topic is not starved by a busy one.
        for topic in topics:
            for _, _, message, decoded in reader.iter_decoded_messages(topics=[topic]):
                samples[topic].append((message.log_time * 1e-9, decoded))
                if len(samples[topic]) >= per_topic:
                    break
    return samples


def describe_pointcloud(topic: str, samples: list) -> None:
    """Print layout and stamp facts of a PointCloud2 topic.

    Args:
        topic: Topic name (section heading).
        samples: (log_time_seconds, decoded PointCloud2) pairs from collect_samples.

    Returns:
        None; prints Markdown.
    """
    # Layout facts come from the first message only; they are constant per topic.
    _, message = samples[0]
    print(f"\n### {topic}")
    print(f"- frame_id: `{message.header.frame_id}`, width x height: {message.width} x {message.height}, "
          f"point_step: {message.point_step}, is_dense: {message.is_dense}, big_endian: {message.is_bigendian}")
    print("- fields: " + ", ".join(
        f"{field.name}(off={field.offset}, dtype={field.datatype}, n={field.count})" for field in message.fields))
    # Stamp statistics over all samples: median header spacing is the sensor period; the median
    # of (log_time - header_stamp) is the recording latency.
    header_seconds = np.array([stamp_to_seconds(sample_message.header.stamp) for _, sample_message in samples])
    log_seconds = np.array([log_time_seconds for log_time_seconds, _ in samples])
    print(f"- header stamp[0]: {header_seconds[0]:.6f}s; median spacing: {np.median(np.diff(header_seconds)):.4f}s; "
          f"median (log_time - header_stamp): {np.median(log_seconds - header_seconds):.4f}s")


def describe_pointcloud_values(samples: list) -> None:
    """Print value ranges of every field of the first cloud (finite values only).

    Args:
        samples: (log_time_seconds, decoded PointCloud2) pairs; only the first is used.

    Returns:
        None; prints Markdown.
    """
    _, message = samples[0]
    # Structured dtype mirroring the PointCloud2 layout: one named field per PointField at its
    # byte offset, with point_step as the record size (covers any padding).
    structured_dtype = np.dtype({
        "names": [field.name for field in message.fields],
        "formats": [POINTFIELD_DTYPES[field.datatype] for field in message.fields],
        "offsets": [field.offset for field in message.fields],
        "itemsize": message.point_step,
    })
    cloud = np.frombuffer(bytes(message.data), dtype=structured_dtype)
    print(f"- points in first cloud: {len(cloud)}")
    for field in message.fields:
        values = cloud[field.name].astype(np.float64)
        # NaN marks a missing return in lidar clouds; exclude it from the ranges.
        finite_values = values[np.isfinite(values)]
        if len(finite_values):
            print(f"  - {field.name}: min {finite_values.min():.4g}, max {finite_values.max():.4g}, "
                  f"nonzero {np.count_nonzero(finite_values)}/{len(values)}")


def describe_camera(info_samples: list, image_samples: list) -> None:
    """Print camera_info and the first compressed image's header facts.

    Args:
        info_samples: (log_time_seconds, CameraInfo) pairs; only the first is used.
        image_samples: (log_time_seconds, CompressedImage) pairs; all are used for the spacing.

    Returns:
        None; prints Markdown.
    """
    _, info = info_samples[0]
    print(f"\n### {CAMERA_INFO_TOPIC}")
    print(f"- frame_id `{info.header.frame_id}`, size {info.width}x{info.height}, "
          f"distortion_model `{info.distortion_model}`")
    print(f"- D = {list(info.D)}")
    print(f"- K = {list(info.K)}")
    print(f"- P = {list(info.P)}")
    _, image = image_samples[0]
    # The first 4 bytes are the file magic (e.g. ffd8ffe0 for JPEG), confirming the codec.
    print(f"\n### {CAMERA_IMAGE_TOPIC}\n- format `{image.format}`, bytes {len(image.data)}, "
          f"frame_id `{image.header.frame_id}`, magic {bytes(image.data[:4]).hex()}")
    header_seconds = np.array([stamp_to_seconds(image_message.header.stamp) for _, image_message in image_samples])
    print(f"- median spacing {np.median(np.diff(header_seconds)):.4f}s")


def describe_odometry(topic: str, samples: list) -> None:
    """Print first pose, frame ids and short path length of an Odometry topic.

    Args:
        topic: Topic name (section heading).
        samples: (log_time_seconds, Odometry) pairs.

    Returns:
        None; prints Markdown.
    """
    _, message = samples[0]
    position = message.pose.pose.position
    orientation = message.pose.pose.orientation
    print(f"\n### {topic}\n- header.frame_id `{message.header.frame_id}`, child_frame_id `{message.child_frame_id}`")
    print(f"- first position ({position.x:.3f}, {position.y:.3f}, {position.z:.3f}), "
          f"quat xyzw ({orientation.x:.4f}, {orientation.y:.4f}, {orientation.z:.4f}, {orientation.w:.4f})")
    positions = np.array([[odometry.pose.pose.position.x, odometry.pose.pose.position.y,
                           odometry.pose.pose.position.z] for _, odometry in samples])
    # Path length = sum of Euclidean distances between consecutive positions.
    print(f"- path length over first {len(samples)} msgs: {np.linalg.norm(np.diff(positions, axis=0), axis=1).sum():.2f} m")


def describe_tf_static(samples: list) -> None:
    """Print all static transforms.

    Args:
        samples: (log_time_seconds, TFMessage) pairs from /tf_static.

    Returns:
        None; prints Markdown.
    """
    print("\n### /tf_static")
    for _, message in samples:
        for transform in message.transforms:
            translation = transform.transform.translation
            rotation = transform.transform.rotation
            print(f"- `{transform.header.frame_id}` -> `{transform.child_frame_id}`: "
                  f"t=({translation.x:.4f}, {translation.y:.4f}, {translation.z:.4f}) "
                  f"q_xyzw=({rotation.x:.4f}, {rotation.y:.4f}, {rotation.z:.4f}, {rotation.w:.4f})")


def describe_tf_dynamic(mcap_path: Path) -> None:
    """Print the set of dynamic /tf parent->child pairs seen in the first messages.

    Args:
        mcap_path: MCAP file path.

    Returns:
        None; prints Markdown.
    """
    pairs = set()
    with open(mcap_path, "rb") as handle:
        reader = make_decoded_reader(handle)
        for message_index, (_, _, _, decoded) in enumerate(reader.iter_decoded_messages(topics=["/tf"])):
            for transform in decoded.transforms:
                pairs.add((transform.header.frame_id, transform.child_frame_id))
            if message_index > TF_DYNAMIC_MESSAGE_LIMIT:
                break
    print("\n### /tf (dynamic) frame pairs seen\n" + "\n".join(f"- `{parent}` -> `{child}`" for parent, child in sorted(pairs)))


def describe_json_structure(json_path: Path) -> None:
    """Print a compact structural outline of the cuboid-label JSON.

    Args:
        json_path: Cuboid-label JSON file.

    Returns:
        None; prints Markdown.
    """
    document = json.loads(json_path.read_text())
    print(f"\n## Cuboid labels `{json_path.name}`")

    def outline(node, path: str, depth: int) -> None:
        """Recursively print dict keys / list lengths / leaf values down to a fixed depth.

        Args:
            node: Current JSON value.
            path: JSONPath-like location of node (``$`` is the root).
            depth: Nesting level of node (root = 0).

        Returns:
            None; prints Markdown.
        """
        if depth > JSON_OUTLINE_MAX_DEPTH:
            return
        if isinstance(node, dict):
            # Dicts: list some keys, then recurse into the first few children.
            print(f"- `{path}` dict keys: {list(node)[:JSON_LISTED_KEY_COUNT]}")
            for key in list(node)[:JSON_RECURSED_KEY_COUNT]:
                outline(node[key], f"{path}.{key}", depth + 1)
        elif isinstance(node, list):
            # Lists: length, then the first element only (labels lists are homogeneous).
            print(f"- `{path}` list len {len(node)}")
            if node and isinstance(node[0], (dict, list)):
                outline(node[0], f"{path}[0]", depth + 1)
        else:
            # Leaves: the value, with the whole line truncated.
            print(f"- `{path}` = {node!r}"[:JSON_LEAF_LINE_MAX_CHARS])

    outline(document, "$", 0)


def main() -> None:
    """Run the inspection and print a Markdown report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mcap", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--labels-json", type=Path, required=True)
    arguments = parser.parse_args()

    # Step 1: sample every reported topic once (clouds, camera, odometry, /tf_static).
    topics = POINTCLOUD_TOPICS + [CAMERA_INFO_TOPIC, CAMERA_IMAGE_TOPIC] + ODOMETRY_TOPICS + ["/tf_static"]
    samples = collect_samples(arguments.mcap, topics, STAMP_SAMPLE_COUNT)

    # Step 2: per-topic sections.
    print(f"# Bag inspection: `{arguments.mcap.name}`\n\n## Topics")
    for topic in POINTCLOUD_TOPICS:
        describe_pointcloud(topic, samples[topic])
        describe_pointcloud_values(samples[topic])
    describe_camera(samples[CAMERA_INFO_TOPIC], samples[CAMERA_IMAGE_TOPIC])
    for topic in ODOMETRY_TOPICS:
        describe_odometry(topic, samples[topic])
    describe_tf_static(samples["/tf_static"])
    describe_tf_dynamic(arguments.mcap)

    # Step 3: raw calibration text and the label JSON outline.
    print("\n## Calibration files")
    for calibration_file in sorted(arguments.calibration_dir.rglob("*.txt")):
        print(f"\n### {calibration_file.name}\n```\n{calibration_file.read_text().strip()}\n```")
    describe_json_structure(arguments.labels_json)


if __name__ == "__main__":
    main()
