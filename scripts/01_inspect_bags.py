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
    """Print layout and stamp facts of a PointCloud2 topic."""
    log_time, message = samples[0]
    print(f"\n### {topic}")
    print(f"- frame_id: `{message.header.frame_id}`, width x height: {message.width} x {message.height}, "
          f"point_step: {message.point_step}, is_dense: {message.is_dense}, big_endian: {message.is_bigendian}")
    print("- fields: " + ", ".join(
        f"{field.name}(off={field.offset}, dtype={field.datatype}, n={field.count})" for field in message.fields))
    header_seconds = np.array([stamp_to_seconds(m.header.stamp) for _, m in samples])
    log_seconds = np.array([t for t, _ in samples])
    print(f"- header stamp[0]: {header_seconds[0]:.6f}s; median spacing: {np.median(np.diff(header_seconds)):.4f}s; "
          f"median (log_time - header_stamp): {np.median(log_seconds - header_seconds):.4f}s")


def describe_pointcloud_values(samples: list) -> None:
    """Print value ranges of the first cloud's float fields (finite points only)."""
    _, message = samples[0]
    dtype_map = {1: "i1", 2: "u1", 3: "i2", 4: "u2", 5: "i4", 6: "u4", 7: "f4", 8: "f8"}
    structured_dtype = np.dtype({
        "names": [field.name for field in message.fields],
        "formats": [dtype_map[field.datatype] for field in message.fields],
        "offsets": [field.offset for field in message.fields],
        "itemsize": message.point_step,
    })
    cloud = np.frombuffer(bytes(message.data), dtype=structured_dtype)
    print(f"- points in first cloud: {len(cloud)}")
    for field in message.fields:
        values = cloud[field.name].astype(np.float64)
        finite = values[np.isfinite(values)]
        if len(finite):
            print(f"  - {field.name}: min {finite.min():.4g}, max {finite.max():.4g}, "
                  f"nonzero {np.count_nonzero(finite)}/{len(values)}")


def describe_camera(info_samples: list, image_samples: list) -> None:
    """Print camera_info and the first compressed image's header facts."""
    _, info = info_samples[0]
    print("\n### /ids_camera/image_raw/camera_info")
    print(f"- frame_id `{info.header.frame_id}`, size {info.width}x{info.height}, "
          f"distortion_model `{info.distortion_model}`")
    print(f"- D = {list(info.D)}")
    print(f"- K = {list(info.K)}")
    print(f"- P = {list(info.P)}")
    _, image = image_samples[0]
    print(f"\n### /ids_camera/image_raw/compressed\n- format `{image.format}`, bytes {len(image.data)}, "
          f"frame_id `{image.header.frame_id}`, magic {bytes(image.data[:4]).hex()}")
    header_seconds = np.array([stamp_to_seconds(m.header.stamp) for _, m in image_samples])
    print(f"- median spacing {np.median(np.diff(header_seconds)):.4f}s")


def describe_odometry(topic: str, samples: list) -> None:
    """Print first pose and frame ids of an Odometry topic."""
    _, message = samples[0]
    position = message.pose.pose.position
    orientation = message.pose.pose.orientation
    print(f"\n### {topic}\n- header.frame_id `{message.header.frame_id}`, child_frame_id `{message.child_frame_id}`")
    print(f"- first position ({position.x:.3f}, {position.y:.3f}, {position.z:.3f}), "
          f"quat xyzw ({orientation.x:.4f}, {orientation.y:.4f}, {orientation.z:.4f}, {orientation.w:.4f})")
    positions = np.array([[m.pose.pose.position.x, m.pose.pose.position.y, m.pose.pose.position.z]
                          for _, m in samples])
    print(f"- path length over first {len(samples)} msgs: {np.linalg.norm(np.diff(positions, axis=0), axis=1).sum():.2f} m")


def describe_tf_static(samples: list) -> None:
    """Print all static transforms."""
    print("\n### /tf_static")
    for _, message in samples:
        for transform in message.transforms:
            translation = transform.transform.translation
            rotation = transform.transform.rotation
            print(f"- `{transform.header.frame_id}` -> `{transform.child_frame_id}`: "
                  f"t=({translation.x:.4f}, {translation.y:.4f}, {translation.z:.4f}) "
                  f"q_xyzw=({rotation.x:.4f}, {rotation.y:.4f}, {rotation.z:.4f}, {rotation.w:.4f})")


def describe_tf_dynamic(mcap_path: Path) -> None:
    """Print the set of dynamic /tf parent->child pairs seen in the first messages."""
    pairs = set()
    with open(mcap_path, "rb") as handle:
        reader = make_decoded_reader(handle)
        for message_index, (_, _, _, decoded) in enumerate(reader.iter_decoded_messages(topics=["/tf"])):
            for transform in decoded.transforms:
                pairs.add((transform.header.frame_id, transform.child_frame_id))
            if message_index > 2000:
                break
    print("\n### /tf (dynamic) frame pairs seen\n" + "\n".join(f"- `{parent}` -> `{child}`" for parent, child in sorted(pairs)))


def describe_json_structure(json_path: Path) -> None:
    """Print a compact structural outline of the cuboid-label JSON."""
    document = json.loads(json_path.read_text())
    print(f"\n## Cuboid labels `{json_path.name}`")

    def outline(node, path, depth):
        """Recursively print dict keys / list lengths down to a fixed depth."""
        if depth > 6:
            return
        if isinstance(node, dict):
            print(f"- `{path}` dict keys: {list(node)[:12]}")
            for key in list(node)[:JSON_PREVIEW_COUNT + 3]:
                outline(node[key], f"{path}.{key}", depth + 1)
        elif isinstance(node, list):
            print(f"- `{path}` list len {len(node)}")
            if node and isinstance(node[0], (dict, list)):
                outline(node[0], f"{path}[0]", depth + 1)
        else:
            print(f"- `{path}` = {node!r}"[:160])

    outline(document, "$", 0)


def main() -> None:
    """Run the inspection and print a Markdown report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mcap", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--labels-json", type=Path, required=True)
    arguments = parser.parse_args()

    topics = POINTCLOUD_TOPICS + [
        "/ids_camera/image_raw/camera_info", "/ids_camera/image_raw/compressed",
        "/icp_odom", "/husky_udp_bridge/odom", "/imu_odom", "/tf_static",
    ]
    samples = collect_samples(arguments.mcap, topics, STAMP_SAMPLE_COUNT)

    print(f"# Bag inspection: `{arguments.mcap.name}`\n\n## Topics")
    for topic in POINTCLOUD_TOPICS:
        describe_pointcloud(topic, samples[topic])
        describe_pointcloud_values(samples[topic])
    describe_camera(samples["/ids_camera/image_raw/camera_info"], samples["/ids_camera/image_raw/compressed"])
    for topic in ["/icp_odom", "/husky_udp_bridge/odom", "/imu_odom"]:
        describe_odometry(topic, samples[topic])
    describe_tf_static(samples["/tf_static"])
    describe_tf_dynamic(arguments.mcap)

    print("\n## Calibration files")
    for calibration_file in sorted(arguments.calibration_dir.rglob("*.txt")):
        print(f"\n### {calibration_file.name}\n```\n{calibration_file.read_text().strip()}\n```")
    describe_json_structure(arguments.labels_json)


if __name__ == "__main__":
    main()
