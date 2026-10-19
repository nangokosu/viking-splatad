"""Quick look at one bag chunk, used to accept or reject a Scene A guess (PLAN.md §1).

For one chunk it writes, under ``reports/scene_a/``:
  * ``<session>_chunk<N>_contact_sheet.jpg``: evenly spaced camera frames (judge the grass height
    and whether trunks are visible);
  * ``<session>_chunk<N>_track.npz``: the ``/icp_odom`` xy track (map frame) and its stamps;
  * ``<session>_chunk<N>_track_map.png``: that track (plus any ``--overlay`` tracks, e.g. the other
    run's chunk) drawn over the session's trunk / bush / rock cuboids.
It also prints the path length and how many trunk cuboids lie within ``--near-radius`` of the track.

The two label maps share coordinates (the June map is registered to May), so overlaying a May and a
June track only tells whether they cover the same place. It is never used to align data.

Usage:
    uv run python scripts/02_scene_quicklook.py --config configs/sessions/june_tall_grass.yaml --chunk 18 \
        [--overlay reports/scene_a/may_short_grass_chunk18_track.npz ...]
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from viking_splatad.config import find_project_root, load_session_config
from viking_splatad.labels.cuboids import Cuboid, parse_cuboids
from viking_splatad.pipeline.bag_to_dataset import open_chunk_messages, read_odometry

PROJECT_ROOT = find_project_root()

# Output folder for all Scene A quick-look artefacts.
OUTPUT_DIRECTORY = PROJECT_ROOT / "reports" / "scene_a"

# Contact sheet: 4 x 3 = 12 frames covers an ~83 s chunk every ~7 s and still fits one screen.
DEFAULT_SHEET_COLUMNS = 4
DEFAULT_SHEET_ROWS = 3
# Sheet width in pixels: the width of a typical laptop screen, so thumbnails stay readable unzoomed.
DEFAULT_SHEET_WIDTH_PX = 2048

# Track-map image width; the height follows from the label extent's aspect ratio.
DEFAULT_MAP_WIDTH_PX = 1600

# "Near the track" radius for counting trunks. Not derivable from the bag alone: 20 m is where the
# OS0-32's ~2.8 deg ring spacing (Phase-1 finding) still puts a ring every ~1 m on a trunk, i.e. a
# trunk at that distance gets several lidar rings and is clearly visible in the camera.
DEFAULT_NEAR_RADIUS_M = 20.0

# Cuboid classes drawn on the map (ids from the labelling protocol) and their BGR colours.
DRAWN_CLASSES = {1: ("tree_trunk", (40, 160, 40)), 3: ("rock", (30, 180, 230)), 4: ("bush_small_tree", (150, 50, 140))}
TRUNK_CLASS_ID = 1
# Track colours, cycled: this chunk first, then each overlay.
TRACK_COLOURS_BGR = [(0, 0, 230), (230, 120, 0), (0, 160, 230), (120, 120, 120)]


def parse_arguments(argument_list: Optional[Sequence[str]]) -> argparse.Namespace:
    """Parse the command line.

    Args:
        argument_list: Arguments (default: sys.argv[1:]).

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path, help="session YAML")
    parser.add_argument("--chunk", required=True, type=int, help="bag chunk index N")
    parser.add_argument("--overlay", type=Path, nargs="*", default=[], help="other *_track.npz to draw")
    parser.add_argument("--columns", type=int, default=DEFAULT_SHEET_COLUMNS)
    parser.add_argument("--rows", type=int, default=DEFAULT_SHEET_ROWS)
    parser.add_argument("--sheet-width", type=int, default=DEFAULT_SHEET_WIDTH_PX)
    parser.add_argument("--map-width", type=int, default=DEFAULT_MAP_WIDTH_PX)
    parser.add_argument("--near-radius", type=float, default=DEFAULT_NEAR_RADIUS_M)
    return parser.parse_args(argument_list)


def build_contact_sheet(
    chunk_path: Path, image_topic: str, columns: int, rows: int, sheet_width_px: int
) -> np.ndarray:
    """Tile evenly spaced camera frames of one chunk into a single image.

    Args:
        chunk_path: MCAP file.
        image_topic: sensor_msgs/CompressedImage topic.
        columns, rows: Grid size; columns * rows frames are shown.
        sheet_width_px: Output width; thumbnail width = sheet_width_px // columns.

    Returns:
        BGR uint8 sheet with each frame's offset (seconds from the chunk's first frame) printed on it.
    """
    # Pass 1: collect only the stamps (cheap), so frames can be chosen evenly over the chunk.
    stamps_ns = [
        message.header.stamp.sec * 10**9 + message.header.stamp.nanosec
        for _, message in open_chunk_messages(chunk_path, [image_topic])
    ]
    frame_count = columns * rows
    # Evenly spaced indices over the whole chunk, endpoints included.
    wanted_indices = set(np.linspace(0, len(stamps_ns) - 1, frame_count).round().astype(int).tolist())

    # Pass 2: decode only the chosen frames.
    thumbnail_width = sheet_width_px // columns
    thumbnails: List[np.ndarray] = []
    for message_index, (_, message) in enumerate(open_chunk_messages(chunk_path, [image_topic])):
        if message_index not in wanted_indices:
            continue
        image = cv2.imdecode(np.frombuffer(bytes(message.data), dtype=np.uint8), cv2.IMREAD_COLOR)
        # Keep the aspect ratio: height scales by the same factor as the width.
        thumbnail_height = round(image.shape[0] * thumbnail_width / image.shape[1])
        thumbnail = cv2.resize(image, (thumbnail_width, thumbnail_height), interpolation=cv2.INTER_AREA)
        offset_seconds = (stamps_ns[message_index] - stamps_ns[0]) / 1e9
        # Label size scales with the thumbnail so it stays legible at any sheet width.
        font_scale = thumbnail_width / 400.0
        cv2.putText(thumbnail, f"+{offset_seconds:.0f}s", (10, round(40 * font_scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), max(1, round(2 * font_scale)))
        thumbnails.append(thumbnail)

    # Tile row by row; pad with black if the chunk has fewer frames than requested.
    while len(thumbnails) < frame_count:
        thumbnails.append(np.zeros_like(thumbnails[0]))
    grid_rows = [np.hstack(thumbnails[row_index * columns:(row_index + 1) * columns]) for row_index in range(rows)]
    return np.vstack(grid_rows)


def trunks_near_track(cuboids: Sequence[Cuboid], track_xy: np.ndarray, radius_m: float) -> int:
    """Count trunk cuboids whose centre lies within ``radius_m`` (xy) of any track pose.

    Brute force: O(n_trunks * n_poses) distances, a few hundred x ~1000, so cheap.

    Args:
        cuboids: Parsed cuboids of the session.
        track_xy: [n,2] track positions in the map frame.
        radius_m: Distance threshold in metres.

    Returns:
        Number of trunk cuboids near the track.
    """
    trunk_centres = np.array([cuboid.center_xyz[:2] for cuboid in cuboids if cuboid.class_id == TRUNK_CLASS_ID])
    if trunk_centres.size == 0 or track_xy.size == 0:
        return 0
    # Pairwise xy distances [n_trunks, n_poses]; a trunk is near if its minimum distance is small.
    distances = np.linalg.norm(trunk_centres[:, None, :] - track_xy[None, :, :], axis=2)
    return int((distances.min(axis=1) <= radius_m).sum())


def draw_track_map(
    cuboids: Sequence[Cuboid], tracks: Sequence[Tuple[str, np.ndarray]], map_width_px: int
) -> np.ndarray:
    """Draw cuboid centres and xy tracks on a top-down image of the label map.

    Args:
        cuboids: Parsed cuboids (only DRAWN_CLASSES are drawn).
        tracks: (name, [n,2] xy) pairs; the first is this chunk.
        map_width_px: Image width; metres-per-pixel follows from the x extent.

    Returns:
        BGR uint8 image, x to the right and y up (map convention).
    """
    # Extent = all cuboids and tracks, so everything is on the image.
    all_xy = [np.array([cuboid.center_xyz[:2] for cuboid in cuboids])] + [track for _, track in tracks]
    stacked = np.vstack([xy for xy in all_xy if xy.size])
    minimum_xy, maximum_xy = stacked.min(axis=0), stacked.max(axis=0)
    # Uniform scale (pixels per metre) so distances are not distorted; margin = 2% of the width each side.
    margin_px = map_width_px // 50
    pixels_per_metre = (map_width_px - 2 * margin_px) / max(maximum_xy[0] - minimum_xy[0], 1e-6)
    map_height_px = round((maximum_xy[1] - minimum_xy[1]) * pixels_per_metre) + 2 * margin_px
    canvas = np.full((map_height_px, map_width_px, 3), 255, dtype=np.uint8)

    def to_pixels(xy: np.ndarray) -> np.ndarray:
        """Map-frame xy [n,2] -> integer pixel coordinates (image y grows downwards, so y is flipped)."""
        column = margin_px + (xy[:, 0] - minimum_xy[0]) * pixels_per_metre
        row = map_height_px - margin_px - (xy[:, 1] - minimum_xy[1]) * pixels_per_metre
        return np.stack([column, row], axis=1).round().astype(np.int32)

    # Cuboids as filled dots whose radius is their half-width in pixels (at least 2 px to stay visible).
    for cuboid in cuboids:
        if cuboid.class_id not in DRAWN_CLASSES:
            continue
        centre = to_pixels(np.array([cuboid.center_xyz[:2]]))[0]
        radius_px = max(2, round(0.5 * min(cuboid.dimensions_xyz[:2]) * pixels_per_metre))
        cv2.circle(canvas, tuple(int(value) for value in centre), radius_px, DRAWN_CLASSES[cuboid.class_id][1], -1)

    # Tracks as polylines with a start marker, plus a legend line per track.
    for track_index, (track_name, track_xy) in enumerate(tracks):
        colour = TRACK_COLOURS_BGR[track_index % len(TRACK_COLOURS_BGR)]
        pixels = to_pixels(track_xy)
        cv2.polylines(canvas, [pixels.reshape(-1, 1, 2)], False, colour, 2)
        cv2.circle(canvas, tuple(int(value) for value in pixels[0]), 6, colour, -1)
        cv2.putText(canvas, f"{track_name} (dot = start)", (margin_px, margin_px + 25 * (track_index + 1)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
    legend_row = margin_px + 25 * (len(tracks) + 1)
    for legend_index, (class_name, colour) in enumerate(DRAWN_CLASSES.values()):
        cv2.putText(canvas, class_name, (margin_px, legend_row + 25 * legend_index),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
    return canvas


def main(argument_list: Optional[Sequence[str]] = None) -> int:
    """Write the contact sheet, track and track map for one chunk; print summary numbers.

    Args:
        argument_list: Command-line arguments (default: sys.argv[1:]).

    Returns:
        Process exit code (0 on success).
    """
    arguments = parse_arguments(argument_list)
    config = load_session_config(arguments.config)
    session_name = config["session"]["name"]

    # Locate the chunk file by its index suffix ("..._<N>.mcap") directly in bag_dir, not through
    # find_chunk_files: that honours window.chunk_files, which would hide every other guess.
    bag_directory = PROJECT_ROOT / config["session"]["bag_dir"]
    chunk_paths = sorted(bag_directory.glob(f"*_{arguments.chunk}.mcap"))
    if not chunk_paths:
        print(f"chunk {arguments.chunk} not found in {config['session']['bag_dir']}", file=sys.stderr)
        return 1
    chunk_path = chunk_paths[0]
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    output_stem = OUTPUT_DIRECTORY / f"{session_name}_chunk{arguments.chunk}"

    # 1. Contact sheet.
    sheet = build_contact_sheet(chunk_path, config["topics"]["camera_image"],
                                arguments.columns, arguments.rows, arguments.sheet_width)
    cv2.imwrite(f"{output_stem}_contact_sheet.jpg", sheet)

    # 2. Track (map-frame xy) and its stamps.
    odometry = read_odometry(chunk_path, config["topics"]["pose"], None)
    track_xy = odometry["positions"][:, :2]
    stamps_seconds = odometry["seconds"] + odometry["nanoseconds"] * 1e-9
    np.savez(f"{output_stem}_track.npz", xy=track_xy, stamps_seconds=stamps_seconds,
             session=session_name, chunk=arguments.chunk)

    # 3. Track map over this session's cuboids, with optional overlay tracks.
    labels_document = json.loads((PROJECT_ROOT / config["session"]["labels_json"]).read_text())
    cuboids = parse_cuboids(labels_document, sample_key=config["session"]["labels_sample_key"])
    tracks = [(f"{session_name} chunk {arguments.chunk}", track_xy)]
    for overlay_path in arguments.overlay:
        overlay = np.load(overlay_path)
        tracks.append((f"{overlay['session']} chunk {int(overlay['chunk'])}", overlay["xy"]))
    cv2.imwrite(f"{output_stem}_track_map.png", draw_track_map(cuboids, tracks, arguments.map_width))

    # 4. Summary: path length = sum of consecutive xy step lengths.
    path_length_m = float(np.linalg.norm(np.diff(track_xy, axis=0), axis=1).sum())
    print(f"{session_name} chunk {arguments.chunk}: {len(track_xy)} poses, "
          f"{stamps_seconds[-1] - stamps_seconds[0]:.1f} s, path {path_length_m:.1f} m, "
          f"xy from {track_xy.min(axis=0).round(1)} to {track_xy.max(axis=0).round(1)}, "
          f"{trunks_near_track(cuboids, track_xy, arguments.near_radius)} trunks within {arguments.near_radius} m")
    print(f"wrote {output_stem}_contact_sheet.jpg, _track.npz, _track_map.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
