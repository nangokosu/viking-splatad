#!/usr/bin/env python3
"""Check that colours baked into a SuperSplat export match a real camera image.

Pipeline: read the PLY (converted back to the map frame through ``supersplat_frame.json`` when that sidecar
records the SuperSplat viewer frame; a map-frame sidecar leaves the means as stored; a PLY without any
sidecar is a legacy map-frame, box-cropped export), decode colour = 0.5 + C0 * f_dc and alpha = sigmoid(opacity),
keep Gaussians above an opacity threshold, project their centres into one camera frame with
the repo's ``project_map_points``, z-buffer per pixel (nearest centre wins), then compare the
winning colours with the image pixels under them (per-channel absolute error, RGB in [0, 1]).

The dataset defaults to ``data/intermediate/<session>``, the session being the one named by the
run's ``job_info.txt`` (as ``scripts/22_export_supersplat_ply.py`` derives it). Its manifest and
poses are verified against the hashes in the run's ``split.json``, and the frame's file name is
checked against the dataset's, in both modes.

Limits, stated because they bound what the numbers mean: occlusion uses Gaussian centres only
(not footprints) and the baked f_dc is the DC term of SH fitted to the decoder, so a small error
means "consistent with the camera", not "pixel-exact". The raw ``features_dc`` baseline needs the
torch checkpoint (``gaussians.npz`` does not store colours), so it is skipped here.

Usage: uv run python scripts/23_check_supersplat_colours.py --run-dir runs/<run_id>

Diagnostic mode (``--diagnose-test-frame``, needs torch, so it runs in the reference image):
splits the colour error into its causes on one TEST frame that SplatAD itself rendered
(``renders/test/rgb`` = its render, ``renders/test/gt-rgb`` = the photo). At the same z-buffer
winning splat pixels it compares against the photo: (a) the PLY colour (SH degree 0 = mean over the
sphere), (b) each Gaussian baked at this camera's own view direction, (c) SplatAD's own rendered
pixel (the model's ceiling), and (d) the render vs the photo over the whole image. Reading (the rule
is also written into the JSON): (b) ~ (a) means averaging over the sphere (H1) is not the cause;
(b) ~ (c) means per-Gaussian decoding is as good as it can be for this direction; (b) far from (c)
means decoding alpha-blended features differs from decoding single Gaussians (H2), which no SH
degree can fix.

    VIKING_WRITABLE_DIR=reports/supersplat_colour_check scripts/test_in_image.sh \\
        python scripts/23_check_supersplat_colours.py --run-dir runs/<run_id> --diagnose-test-frame
"""

import argparse
import copy
import csv
import functools
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from viking_splatad.export.crop import check_dataset_identity  # noqa: E402
from viking_splatad.export.spherical_harmonics import SH_C0, SH_COLOUR_OFFSET  # noqa: E402
from viking_splatad.export.splatad_colours import RGB_MAX, RGB_MIN, camera_sensor_index  # noqa: E402
from viking_splatad.export.supersplat_ply import read_supersplat_ply  # noqa: E402
from viking_splatad.export.viewer_frame import (  # noqa: E402
    VIEWER_FRAME_FILE_NAME, VIEWER_FRAME_NAME, check_record_matches_ply, read_viewer_frame_record,
    viewer_to_map_points)
from viking_splatad.pipeline.derived import DERIVED_FILE_NAME, load_derived  # noqa: E402
from viking_splatad.radar_init.seeds import project_map_points  # noqa: E402
from viking_splatad.splatad.viking_conversions import open_dataset_view  # noqa: E402

# A Gaussian counts as "there" when alpha > 0.5: more than half opaque, so its colour dominates
# what a viewer sees at its centre. A judgement threshold; the JSON records it.
OPACITY_THRESHOLD = 0.5
# Percentile reported next to the median (the task asks for the 90th).
TAIL_PERCENTILE = 90.0
# Camera whose frames are checked (the ``sensor`` column of split_frames.csv and the manifest key).
CAMERA_SENSOR = "ids_camera"
# Side-by-side panel width in pixels (modest, keeps the PNG small); height keeps the aspect ratio.
PANEL_WIDTH_PIXELS = 900
# Radius of the dot drawn per surviving Gaussian in the splat panel, in panel pixels.
SPLAT_DOT_RADIUS_PIXELS = 1
# Names of the splits in split_frames.csv.
TRAIN_SPLIT = "train"
TEST_SPLIT = "test"
# Folder names inside <run>/renders/test (SplatAD's render and the photo it is compared with).
RENDER_RGB_FOLDER = "renders/test/rgb"
PHOTO_RGB_FOLDER = "renders/test/gt-rgb"
# The exported PLY inside a run.
PLY_RELATIVE_PATH = "renders/supersplat.ply"
# Repository root: this file lives in <root>/scripts/; the default output folder is anchored here, not on the cwd.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_RELATIVE_DIR = Path("reports") / "supersplat_colour_check"
# Script 22 (export), reused for checkpoint loading, run layout and the decoder batch size.
EXPORT_SCRIPT_PATH = Path(__file__).resolve().parent / "22_export_supersplat_ply.py"
# Two errors x and y count as "approximately equal" when |x - y| <= CLOSENESS_FRACTION * max(x, y).
# A judgement threshold (10 %), recorded in the JSON; the raw numbers are always reported as well.
CLOSENESS_FRACTION = 0.10
# Panel width of the diagnostic PNG (3 panels side by side), modest to keep the file small.
DIAGNOSTIC_PANEL_WIDTH_PIXELS = 640
# Colour channel count (RGB) and the channel names in order.
CHANNEL_COUNT = 3
CHANNEL_NAMES = ("r", "g", "b")
# PLY position field names, in x, y, z order.
POSITION_FIELD_NAMES = ("x", "y", "z")
# Largest value of an 8-bit image channel: the scale between [0, 1] floats and uint8.
UINT8_MAX = np.iinfo(np.uint8).max
# Safety factor on float32 rounding when checking the PLY means against the checkpoint means: a float32
# stores a value with relative error <= eps / 2, and 2 leaves room for one extra rounding.
FLOAT32_ROUNDING_SAFETY_FACTOR = 2.0
# A pixel index u names the interval [u, u + 1); its centre is u + 0.5 (definition of the pixel grid).
PIXEL_CENTRE_OFFSET = 0.5
# Smallest number of Gaussians per decoder batch.
MIN_BATCH_SIZE = 1


def zbuffer_nearest(pixel_indices: np.ndarray, depths: np.ndarray) -> np.ndarray:
    """Index of the nearest point per distinct pixel.

    Sorting by (pixel, depth) puts each pixel's nearest point first in its run: O(N log N).

    Args:
        pixel_indices: int64 [N] flattened pixel id of each point.
        depths: float [N] depth of each point.

    Returns:
        int64 [P] indices into the inputs, one per distinct pixel, the nearest point of each
        (empty when there are no points).
    """
    # Step 0: no points, no pixels (the run-start mask below would otherwise be longer than the order).
    if len(pixel_indices) == 0:
        return np.empty(0, dtype=np.int64)
    # Step 1: lexsort uses the LAST key as primary, so pixel is primary and depth secondary.
    order = np.lexsort((depths, pixel_indices))
    # Step 2: the first element of every pixel run is its nearest point.
    sorted_pixels = pixel_indices[order]
    is_run_start = np.concatenate([[True], sorted_pixels[1:] != sorted_pixels[:-1]])
    return order[is_run_start]


def error_summary(absolute_errors: np.ndarray) -> dict:
    """Median and tail percentile of per-channel absolute error.

    Args:
        absolute_errors: float [N,3] absolute errors for R, G, B, N >= 1.

    Returns:
        Dict with per-channel and all-channel median and tail percentile.

    Raises:
        ValueError: If there is no sample (a median of nothing is NaN, which must not be reported).
    """
    if absolute_errors.shape[0] == 0:
        raise ValueError("error_summary needs at least one sample")
    tail_name = f"p{int(TAIL_PERCENTILE)}"
    summary = {}
    # Step 1: per channel, the median and the tail percentile.
    for channel_index, channel_name in enumerate(CHANNEL_NAMES):
        channel_errors = absolute_errors[:, channel_index]
        summary[f"{channel_name}_median"] = float(np.median(channel_errors))
        summary[f"{channel_name}_{tail_name}"] = float(np.percentile(channel_errors, TAIL_PERCENTILE))
    # Step 2: the same over all channels pooled.
    summary["all_channels_median"] = float(np.median(absolute_errors))
    summary[f"all_channels_{tail_name}"] = float(np.percentile(absolute_errors, TAIL_PERCENTILE))
    return summary


def pick_median_frame(split_csv: Path, split_name: str) -> dict:
    """Pick the median camera frame of one split by frame index (deterministic).

    Args:
        split_csv: ``split_frames.csv`` of the run.
        split_name: ``train`` or ``test``.

    Returns:
        The CSV row (dict) of the element at position ``len // 2`` of the frames sorted by frame index.

    Raises:
        SystemExit: If the CSV has no camera row of this split.
    """
    with open(split_csv, newline="") as split_file:
        split_rows = [row for row in csv.DictReader(split_file)
                      if row["sensor"] == CAMERA_SENSOR and row["split"] == split_name]
    if not split_rows:
        raise SystemExit(f"{split_csv} has no {CAMERA_SENSOR} row in the {split_name} split")
    split_rows.sort(key=lambda row: int(row["frame_index"]))
    return split_rows[len(split_rows) // 2]


def view_directions_map(means_map: np.ndarray, camera_centre_map: np.ndarray) -> np.ndarray:
    """Unit viewing directions camera -> Gaussian, one per Gaussian, in the map frame.

    The sign is the 3DGS convention ``dir = mean - campos`` (camera towards scene), as the export's
    Fibonacci directions use.

    Args:
        means_map: float [N,3] Gaussian centres.
        camera_centre_map: float [3] camera centre in the map frame.

    Returns:
        float [N,3] unit vectors.
    """
    offsets = means_map - camera_centre_map[None, :]
    return offsets / np.linalg.norm(offsets, axis=1, keepdims=True)


def rotate_directions_to_dataparser(directions_map: np.ndarray, dataparser_transform: np.ndarray) -> np.ndarray:
    """Rotate map-frame directions into the dataparser frame: d_dp = A d_map.

    The decoder was trained in the dataparser frame x_dp = A x_map + t; a direction is a difference of
    two points, so t cancels. A is rotation times a uniform scale s (s = 1 for this run), so the result is
    renormalised: unit length is the decoder's input contract whatever s is.

    Args:
        directions_map: float [N,3] unit directions in the map frame.
        dataparser_transform: float [3,4] or [4,4] ``transform`` of dataparser_transforms.json.

    Returns:
        float [N,3] unit directions in the dataparser frame.
    """
    # One shared convention: script 22's helper (rotate, then renormalise).
    return load_export_script_module().rotate_directions_to_dataparser(directions_map, dataparser_transform)


def scale_pixel_coordinates(columns: np.ndarray, rows: np.ndarray, source_size_hw: tuple,
                            target_size_hw: tuple) -> tuple:
    """Map integer pixel coordinates of one image size onto another by the size ratio.

    Pixel (u, v) covers [u, u + 1), so its centre is u + 0.5. Scaling centres by W' / W and going back to
    an index gives u' = (u + 0.5) * W' / W - 0.5 (likewise for rows); the result is rounded and clamped
    into the target. With equal sizes the coordinates are unchanged.

    Args:
        columns: int [N] column (u) indices in the source image.
        rows: int [N] row (v) indices in the source image.
        source_size_hw: (height, width) of the source image.
        target_size_hw: (height, width) of the target image.

    Returns:
        (columns, rows) int64 arrays in the target image.
    """
    scaled_columns = np.rint((columns + PIXEL_CENTRE_OFFSET) * (target_size_hw[1] / source_size_hw[1])
                             - PIXEL_CENTRE_OFFSET).astype(np.int64)
    scaled_rows = np.rint((rows + PIXEL_CENTRE_OFFSET) * (target_size_hw[0] / source_size_hw[0])
                          - PIXEL_CENTRE_OFFSET).astype(np.int64)
    return (np.clip(scaled_columns, 0, target_size_hw[1] - 1), np.clip(scaled_rows, 0, target_size_hw[0] - 1))


def is_close(first_value: float, second_value: float) -> bool:
    """True when |x - y| <= CLOSENESS_FRACTION * max(x, y) (relative closeness of two errors).

    Args:
        first_value: non-negative error.
        second_value: non-negative error.

    Returns:
        Whether the two are approximately equal under the recorded threshold.
    """
    return abs(first_value - second_value) <= CLOSENESS_FRACTION * max(first_value, second_value)


def fraction_of_gap_closed(ply_error: float, bake_error: float, render_error: float) -> float:
    """Share of the PLY-to-render error gap that the camera-direction bake closes.

    (a - b) / (a - c): 1 = the bake reaches the model's own render, 0 = no better than the PLY,
    negative = worse. NaN when a == c (no gap to close).

    Args:
        ply_error: error (a) of the PLY colour.
        bake_error: error (b) of the camera-direction bake.
        render_error: error (c) of SplatAD's own pixels.

    Returns:
        The fraction (float).
    """
    gap = ply_error - render_error
    return float("nan") if gap == 0.0 else (ply_error - bake_error) / gap


def interpret_errors(ply_error: float, bake_error: float, render_error: float) -> dict:
    """Apply the interpretation rule to one statistic (a, b, c).

    Rule: (b) ~ (a) -> H1 (sphere averaging) is not the cause; (b) ~ (c) -> per-Gaussian decoding is as
    good as possible for this direction; (b) far from (c) -> H2 (blended-feature decoding) dominates.

    Args:
        ply_error: error (a).
        bake_error: error (b).
        render_error: error (c).

    Returns:
        Dict with the two closeness flags, the gap-closed fraction and the verdict strings.
    """
    bake_close_to_ply = is_close(bake_error, ply_error)
    bake_close_to_render = is_close(bake_error, render_error)
    return {
        "b_close_to_a": bake_close_to_ply, "b_close_to_c": bake_close_to_render,
        "fraction_of_a_to_c_gap_closed_by_b": fraction_of_gap_closed(ply_error, bake_error, render_error),
        "h1_sphere_averaging_is_not_the_cause": bake_close_to_ply,
        "per_gaussian_decoding_is_as_good_as_possible": bake_close_to_render,
        "h2_blended_decoding_dominates": not bake_close_to_render,
    }


def visible_zbuffer_winners(means_map: np.ndarray, pose_map_camera: np.ndarray, camera_matrix: np.ndarray,
                            image_size_hw: tuple) -> dict:
    """Project centres, keep those in front and in the image, z-buffer one winner per pixel.

    Args:
        means_map: float [N,3] centres in the map frame.
        pose_map_camera: float [4,4] OpenCV ``T_map_camera``.
        camera_matrix: float [3,3] intrinsics.
        image_size_hw: (height, width) of the image the intrinsics refer to.

    Returns:
        Dict with ``column``/``row`` (int64 [N], -1 where not visible), ``visible_indices`` (int64) and
        ``winners`` (int64 indices into ``means_map``, nearest centre per pixel).
    """
    image_height, image_width = image_size_hw
    # Step 1: repo projection; keep positive depth and finite pixels.
    pixels, depths = project_map_points(means_map, pose_map_camera, camera_matrix)
    in_view = (depths > 0.0) & np.isfinite(pixels).all(axis=1)
    column = np.full(len(pixels), -1, dtype=np.int64)
    row = np.full(len(pixels), -1, dtype=np.int64)
    # Step 2: nearest pixel, as the repo's painter does (pixel centres on integers).
    column[in_view] = np.rint(pixels[in_view, 0]).astype(np.int64)
    row[in_view] = np.rint(pixels[in_view, 1]).astype(np.int64)
    in_view &= (column >= 0) & (column < image_width) & (row >= 0) & (row < image_height)
    visible_indices = np.flatnonzero(in_view)
    # Step 3: z-buffer, one Gaussian per pixel.
    flat_pixels = row[visible_indices] * image_width + column[visible_indices]
    winners = visible_indices[zbuffer_nearest(flat_pixels, depths[visible_indices])]
    return {"column": column, "row": row, "visible_indices": visible_indices, "winners": winners}


def require_nonempty_winners(visibility: dict) -> None:
    """Stop with a clear message when no Gaussian is visible (every statistic would be of nothing).

    Args:
        visibility: Output of ``visible_zbuffer_winners``.

    Raises:
        SystemExit: If there is no z-buffer winner.
    """
    if len(visibility["winners"]) == 0:
        raise SystemExit("no Gaussian above the opacity threshold is visible in this camera frame "
                         "(wrong dataset or pose, or an empty export)")


def decode_ply(ply_path: Path) -> dict:
    """Read an export and decode its centres (map frame), degree-0 colours and opacities.

    colour = clip(0.5 + C0 * f_dc, 0, 1) (the degree-0 SH basis is the constant C0) and
    alpha = sigmoid(opacity logit). If ``supersplat_frame.json`` sits next to the PLY it is first checked
    against the file (``check_record_matches_ply``: an interrupted or mixed export is refused), then its
    ``frame`` decides: the viewer frame, p_viewer = M (p_map - origin), is mapped back with
    p_map = M^T p_viewer + origin (``viewer_to_map_points``); the map frame keeps the centres as stored.
    Without the sidecar the centres are used as stored (map frame). This function stays lenient so it can
    be unit-tested on a bare PLY; the refusal of a missing sidecar lives at the CLI boundary
    (``require_frame_record``, called by both entry points).

    Args:
        ply_path: ``supersplat.ply``.

    Returns:
        Dict with ``means`` float64 [N,3] in the MAP frame, ``colours`` float64 [N,3] in [0,1],
        ``alpha`` float64 [N], ``opacity_logits`` float32 [N] as stored, ``crop`` (the sidecar's crop
        record, or None for a legacy export without sidecar) and ``frame`` (the sidecar's frame name, or None).

    Raises:
        SystemExit: If the sidecar does not describe this PLY (size mismatch): re-export.
    """
    fields = read_supersplat_ply(ply_path)
    means = np.stack([fields[axis_name] for axis_name in POSITION_FIELD_NAMES], axis=1).astype(np.float64)
    # Step 1: viewer-frame export -> back to the map frame, where the cameras and poses live.
    frame_record_path = Path(ply_path).parent / VIEWER_FRAME_FILE_NAME
    crop_record = None
    frame_name = None
    if frame_record_path.is_file():
        frame_record = read_viewer_frame_record(frame_record_path)
        # The sidecar and the PLY are written by two renames; refuse a pair that does not belong together.
        try:
            check_record_matches_ply(frame_record, ply_path)
        except ValueError as error:
            raise SystemExit(f"{error}; re-export with scripts/22_export_supersplat_ply.py --overwrite") from error
        frame_name = frame_record["frame"]
        if frame_name == VIEWER_FRAME_NAME:
            means = viewer_to_map_points(means, frame_record["origin_map"], frame_record["rotation"])
        crop_record = frame_record["crop"]
    albedo_coefficients = np.stack([fields[f"f_dc_{channel_index}"] for channel_index in range(CHANNEL_COUNT)],
                                   axis=1).astype(np.float64)
    colours = np.clip(SH_COLOUR_OFFSET + SH_C0 * albedo_coefficients, RGB_MIN, RGB_MAX)
    # expit(x) = 1 / (1 + exp(-x)) is stable for large |x|.
    alpha = expit(fields["opacity"].astype(np.float64))
    return {"means": means, "colours": colours, "alpha": alpha, "opacity_logits": fields["opacity"],
            "crop": crop_record, "frame": frame_name}


def select_opaque_indices(alpha: np.ndarray) -> np.ndarray:
    """Indices of the Gaussians with alpha above the threshold.

    Args:
        alpha: float [N] opacities.

    Returns:
        int64 indices.

    Raises:
        SystemExit: If none passes.
    """
    opaque_indices = np.flatnonzero(alpha > OPACITY_THRESHOLD)
    if len(opaque_indices) == 0:
        raise SystemExit(f"no Gaussian has alpha > {OPACITY_THRESHOLD} (of {len(alpha)})")
    return opaque_indices


def camera_matrix_from_intrinsics(intrinsics: dict) -> np.ndarray:
    """Pinhole matrix [[fx, 0, cx], [0, fy, cy], [0, 0, 1]] from a manifest camera entry.

    Args:
        intrinsics: Dict with ``fx, fy, cx, cy``.

    Returns:
        float [3,3].
    """
    return np.array([[intrinsics["fx"], 0.0, intrinsics["cx"]],
                     [0.0, intrinsics["fy"], intrinsics["cy"]], [0.0, 0.0, 1.0]])


def read_rgb_unit(image_path: Path) -> np.ndarray:
    """Read an image as float RGB in [0,1].

    Args:
        image_path: JPEG/PNG path.

    Returns:
        float64 [H,W,3].

    Raises:
        SystemExit: If OpenCV cannot read the file.
    """
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise SystemExit(f"cannot read {image_path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float64) / UINT8_MAX


def rgb_unit_to_bgr_uint8(rgb_unit: np.ndarray) -> np.ndarray:
    """Float RGB in [0,1] to uint8 BGR for OpenCV, rounded to the nearest level (not truncated).

    Truncation would bias every level down by half a step on average; rounding is exact for values that
    came from uint8 (v / 255 * 255 rounds back to v).

    Args:
        rgb_unit: float [...,3] in [0,1].

    Returns:
        uint8 [...,3] in BGR order.
    """
    return np.rint(np.clip(rgb_unit, RGB_MIN, RGB_MAX)[..., ::-1] * UINT8_MAX).astype(np.uint8)


def build_photo_panel(image_rgb: np.ndarray, reference_size_hw: tuple, panel_width: int) -> tuple:
    """Resize an image to the panel size, which keeps the aspect ratio of the reference camera size.

    Args:
        image_rgb: float [H,W,3] in [0,1].
        reference_size_hw: (height, width) of the pixel space the dot coordinates live in.
        panel_width: Panel width in pixels.

    Returns:
        (panel uint8 BGR, scale = panel_width / reference width).
    """
    scale = panel_width / reference_size_hw[1]
    panel_size = (panel_width, int(round(reference_size_hw[0] * scale)))
    return cv2.resize(rgb_unit_to_bgr_uint8(image_rgb), panel_size, interpolation=cv2.INTER_AREA), scale


def draw_dot_panel(panel_shape_like: np.ndarray, columns: np.ndarray, rows: np.ndarray,
                   colours_rgb: np.ndarray, scale: float) -> np.ndarray:
    """Black panel with one filled dot per Gaussian at its winning pixel, in its colour.

    Args:
        panel_shape_like: uint8 BGR panel whose shape and dtype the result copies.
        columns: int [P] winning pixel columns in the reference pixel space.
        rows: int [P] winning pixel rows in the reference pixel space.
        colours_rgb: float [P,3] colours in [0,1].
        scale: panel pixels per reference pixel.

    Returns:
        uint8 BGR panel.
    """
    panel = np.zeros_like(panel_shape_like)
    dot_colours = rgb_unit_to_bgr_uint8(colours_rgb)
    for dot_index in range(len(columns)):
        centre = (int(round(columns[dot_index] * scale)), int(round(rows[dot_index] * scale)))
        cv2.circle(panel, centre, SPLAT_DOT_RADIUS_PIXELS,
                   tuple(int(channel_value) for channel_value in dot_colours[dot_index]), cv2.FILLED)
    return panel


@functools.lru_cache(maxsize=None)
def load_export_script_module():
    """Import scripts/22 (its name starts with a digit, so by file path) to reuse its helpers.

    Returns:
        The loaded module.
    """
    module_spec = importlib.util.spec_from_file_location("export_supersplat_ply_script", EXPORT_SCRIPT_PATH)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def resolve_dataset_dir(arguments: argparse.Namespace) -> Path:
    """The dataset of the run, verified to be the one the run trained on.

    Default: ``data/intermediate/<session>`` with the session from the run's ``job_info.txt`` via
    script 22's ``find_train_dir`` / ``session_of_train_dir`` (as the export derives it).

    Args:
        arguments: Parsed CLI (``run_dir``, ``dataset`` or None).

    Returns:
        The dataset directory.

    Raises:
        SystemExit: If the dataset is not the run's (hash mismatch in ``split.json``).
        FileNotFoundError, ValueError: From ``find_train_dir`` when no ``--dataset`` is given and the
            run's ``job_info.txt`` / train dir cannot be resolved.
    """
    if arguments.dataset is not None:
        dataset_dir = arguments.dataset
    else:
        export_module = load_export_script_module()
        session = export_module.session_of_train_dir(export_module.find_train_dir(arguments.run_dir))
        dataset_dir = export_module.DEFAULT_DATASET_PARENT / session
    try:
        check_dataset_identity(arguments.run_dir, dataset_dir)
    except ValueError as error:
        raise SystemExit(f"wrong dataset: {error}") from error
    return dataset_dir


def resize_factor_of_dataset(dataset_dir: Path, run_dir: Path) -> float:
    """``resize_factor`` that extraction recorded in the dataset's ``derived.json``.

    Args:
        dataset_dir: Dataset directory.
        run_dir: Run folder; its ``job_info.txt`` names the session the cache must belong to.

    Returns:
        The factor.

    Raises:
        SystemExit: If ``derived.json`` is absent.
    """
    export_module = load_export_script_module()
    session_name = export_module.session_of_train_dir(export_module.find_train_dir(run_dir))
    derived = load_derived(dataset_dir / DERIVED_FILE_NAME, session_name)
    if derived is None:
        raise SystemExit(f"{dataset_dir / DERIVED_FILE_NAME} not found")
    return float(derived.resize_factor)


def locate_dataset_frame(view, frame_row: dict) -> tuple:
    """Dataset frame index and path of a split row, verified by file name.

    Args:
        view: Dataset view (``open_dataset_view``).
        frame_row: Row of ``split_frames.csv``.

    Returns:
        (frame_index, frame_path).

    Raises:
        SystemExit: If the dataset's file at that index has another name than the row (wrong dataset).
    """
    frame_index = int(frame_row["frame_index"])
    frame_paths = view.sensor_frame_paths[CAMERA_SENSOR]
    if not 0 <= frame_index < len(frame_paths):
        raise SystemExit(f"frame_index {frame_index} outside the dataset's {len(frame_paths)} frames")
    frame_path = frame_paths[frame_index]
    if frame_path.name != Path(frame_row["file"]).name:
        raise SystemExit(f"dataset frame {frame_path.name} != split row {Path(frame_row['file']).name}")
    return frame_index, frame_path


def project_winners_onto_frame(means_map: np.ndarray, view, frame_index: int) -> dict:
    """Z-buffer winners of ``means_map`` in one dataset camera frame.

    Args:
        means_map: float [N,3] centres (map frame).
        view: Dataset view.
        frame_index: Camera frame index.

    Returns:
        ``visible_zbuffer_winners`` dict plus ``pose_map_camera`` and ``manifest_size_hw``.

    Raises:
        SystemExit: If there are no winners.
    """
    manifest_camera = view.manifest["cameras"][CAMERA_SENSOR]
    manifest_size_hw = (int(manifest_camera["height"]), int(manifest_camera["width"]))
    pose_map_camera = view.sensor_poses[CAMERA_SENSOR][frame_index]
    visibility = visible_zbuffer_winners(means_map, pose_map_camera, camera_matrix_from_intrinsics(manifest_camera),
                                         manifest_size_hw)
    require_nonempty_winners(visibility)
    visibility.update(pose_map_camera=pose_map_camera, manifest_size_hw=manifest_size_hw)
    return visibility


def sample_image_at_winners(image_rgb: np.ndarray, visibility: dict) -> np.ndarray:
    """Colours of an image under the winning pixels (coordinates scaled from the manifest size).

    Args:
        image_rgb: float [H,W,3] in [0,1], possibly another resolution than the manifest camera.
        visibility: Output of ``project_winners_onto_frame``.

    Returns:
        float [P,3].
    """
    winners = visibility["winners"]
    columns, rows = scale_pixel_coordinates(visibility["column"][winners], visibility["row"][winners],
                                            visibility["manifest_size_hw"], image_rgb.shape[:2])
    return image_rgb[rows, columns]


def appearance_row_of_checked_camera(camera_names: list) -> int:
    """Appearance-embedding row of the checked camera, which must be the row script 22 baked with.

    Script 22 uses the first camera of the config; the photo comes from ``CAMERA_SENSOR``. If they
    differ the baked colours would be compared against another camera's photo.

    Args:
        camera_names: ``settings["cameras"]`` (dataparser order).

    Returns:
        The row.

    Raises:
        SystemExit: If the checked camera is absent or its row differs from script 22's.
    """
    if CAMERA_SENSOR not in camera_names:
        raise SystemExit(f"{CAMERA_SENSOR} is not among the run's cameras {camera_names}")
    checked_row = camera_sensor_index(camera_names, CAMERA_SENSOR)
    export_row = camera_sensor_index(camera_names, camera_names[0])
    if checked_row != export_row:
        raise SystemExit(f"appearance row of {CAMERA_SENSOR} is {checked_row} but script 22 baked with row "
                         f"{export_row} ({camera_names[0]}): the comparison would mix cameras")
    return checked_row


def bake_at_per_gaussian_directions(decoder, features_dc: np.ndarray, features_rest: np.ndarray,
                                    appearance_vector: np.ndarray, directions_dataparser: np.ndarray,
                                    batch_size: int) -> np.ndarray:
    """Bake each Gaussian with its OWN direction, batched.

    Duplicates the 3-line combine of ``splatad_colours.bake_gaussian_colours`` (replicate-padded decoder
    copy, spec input = [features_rest, appearance, direction], colour = clamp(albedo * (1 + gain) +
    offset)) because that function shares ONE direction set across all Gaussians and so would need one
    call per Gaussian (a decoder copy and a Python loop each). Suggested change to the module: accept
    per-Gaussian directions ((N, 3) next to the (M, 3) set) so this copy can go. A test checks equality
    with ``bake_gaussian_colours`` called per Gaussian.

    Cost: N single-pixel decoder evaluations in ceil(N / batch_size) forwards.

    Args:
        decoder: upstream RGB decoder (not modified; a deep copy is used).
        features_dc: float [N,3] albedo features.
        features_rest: float [N,F] remaining features.
        appearance_vector: float [A] embedding row of the camera.
        directions_dataparser: float [N,3] unit directions, dataparser frame, row n for Gaussian n.
        batch_size: Gaussians per forward, >= 1.

    Returns:
        float32 [N,3] RGB in [0,1].

    Raises:
        ValueError: If ``batch_size < 1``.
    """
    import torch  # pylint: disable=import-outside-toplevel

    if batch_size < MIN_BATCH_SIZE:
        raise ValueError(f"batch_size must be >= {MIN_BATCH_SIZE}, got {batch_size}")
    # Step 1: ONE replicate-padded CPU float32 copy, as the module builds it: with replicate padding a 1x1
    # image behaves like a uniform patch of that Gaussian's feature.
    replicate_decoder = copy.deepcopy(decoder).cpu().float().eval()
    for module in replicate_decoder.modules():
        if isinstance(module, torch.nn.Conv2d):
            module.padding_mode = "replicate"
    albedo_channel_count = replicate_decoder.skip_dim
    albedo_all = torch.as_tensor(features_dc, dtype=torch.float32, device="cpu")
    rest_all = torch.as_tensor(features_rest, dtype=torch.float32, device="cpu")
    appearance = torch.as_tensor(appearance_vector, dtype=torch.float32, device="cpu")
    directions_all = torch.as_tensor(directions_dataparser, dtype=torch.float32, device="cpu")
    gaussian_count = albedo_all.shape[0]
    baked = np.empty((gaussian_count, albedo_channel_count), dtype=np.float32)
    with torch.no_grad():
        for batch_start in range(0, gaussian_count, batch_size):
            batch_stop = min(batch_start + batch_size, gaussian_count)
            # Step 2: spec input channels in the upstream order: features_rest, appearance, direction.
            spec_input = torch.cat([rest_all[batch_start:batch_stop],
                                    appearance.expand(batch_stop - batch_start, -1),
                                    directions_all[batch_start:batch_stop]], dim=-1)
            network_output = replicate_decoder.net(spec_input[:, :, None, None])[:, :, 0, 0]
            # Step 3: albedo * (1 + gain) + offset, clamped as the model does.
            gain = network_output[:, :albedo_channel_count]
            offset = network_output[:, albedo_channel_count:]
            baked[batch_start:batch_stop] = torch.clamp(
                albedo_all[batch_start:batch_stop] * (1 + gain) + offset, RGB_MIN, RGB_MAX).numpy()
    return baked


def max_difference_within_float32(stored: np.ndarray, exported: np.ndarray) -> tuple:
    """Largest absolute difference of two arrays and the float32 rounding tolerance for their magnitude.

    A float32 stores a value with relative error <= eps / 2, so the tolerance
    ``FLOAT32_ROUNDING_SAFETY_FACTOR * eps * max(|stored|max, 1)`` holds for an exact match and fails for
    any reordering or different Gaussian.

    Args:
        stored: Checkpoint values.
        exported: PLY values (same shape).

    Returns:
        (max_abs_difference, tolerance).
    """
    tolerance = (FLOAT32_ROUNDING_SAFETY_FACTOR * np.finfo(np.float32).eps
                 * max(float(np.abs(stored).max()), 1.0))
    return float(np.abs(stored - exported).max()), float(tolerance)


def load_checkpoint_rows_matching_ply(arguments: argparse.Namespace, dataset_dir: Path, decoded_ply: dict) -> dict:
    """Load the run's checkpoint and prove that its kept rows are the PLY rows.

    Rebuilds the export's crop with script 22's ``select_crop`` so kept checkpoint row k is PLY row k,
    then compares every mean and opacity logit. With a sidecar the crop is REPLAYED from its record (the
    recorded corridor width / z range, the recorded box, or none), never re-derived. Without a sidecar the
    export is a legacy one: every row when the PLY holds every Gaussian (--no-crop), else the box (T86).

    Args:
        arguments: Parsed CLI (``run_dir``).
        dataset_dir: Verified dataset directory.
        decoded_ply: Output of ``decode_ply``.

    Returns:
        Dict with ``tensors``, ``settings``, ``dataparser_transform``, ``keep_mask`` and ``row_match`` (the
        JSON block).

    Raises:
        SystemExit: If the counts or values disagree.
    """
    from viking_splatad.eval.gaussian_occupancy import (  # pylint: disable=import-outside-toplevel
        checkpoint_gaussians_to_map_frame, latest_checkpoint_path)
    from viking_splatad.export.splatad_colours import model_settings_from_config_yml  # pylint: disable=import-outside-toplevel

    export_module = load_export_script_module()
    train_dir = export_module.find_train_dir(arguments.run_dir)
    tensors = export_module.load_checkpoint_tensors(
        latest_checkpoint_path(train_dir / export_module.CHECKPOINT_DIR_NAME))
    settings = model_settings_from_config_yml(train_dir / export_module.CONFIG_FILE_NAME)
    transforms = json.loads((train_dir / export_module.DATAPARSER_TRANSFORMS_FILE_NAME).read_text())
    dataparser_transform = np.asarray(transforms["transform"], dtype=np.float64)
    gauss_params = tensors["gauss_params"]
    gaussians = checkpoint_gaussians_to_map_frame(
        means=gauss_params["means"], log_scales=gauss_params["scales"], quaternions_wxyz=gauss_params["quats"],
        opacity_logits=gauss_params["opacities"], actor_ids=gauss_params["id"],
        actor_count=gauss_params["actor_count"], dataparser_transform=dataparser_transform,
        dataparser_scale=float(transforms["scale"]))
    total_count = len(gaussians.means)
    recorded_crop = decoded_ply.get("crop")
    # Legacy export (no sidecar): a PLY with as many rows as the checkpoint has Gaussians was exported with
    # --no-crop, so every row is kept; any other legacy export is box-cropped.
    legacy_without_crop = recorded_crop is None and total_count == len(decoded_ply["means"])
    if legacy_without_crop:
        keep_mask = np.ones(total_count, dtype=bool)
    else:
        try:
            keep_mask, _crop_record = export_module.select_crop(
                arguments.run_dir, dataset_dir, gaussians.means, gaussians.opacities,
                export_module.CROP_MODE_BOX, float(settings["mcmc_min_opacity"]), recorded_crop=recorded_crop)
        except ValueError as error:
            raise SystemExit(f"cannot rebuild the export's crop: {error}") from error
    kept_means = gaussians.means[keep_mask]
    kept_logits = np.asarray(gauss_params["opacities"], dtype=np.float64).reshape(-1)[keep_mask]
    if len(kept_means) != len(decoded_ply["means"]):
        raise SystemExit(f"crop keeps {len(kept_means)} rows but the PLY has {len(decoded_ply['means'])}")
    mean_difference, mean_tolerance = max_difference_within_float32(kept_means, decoded_ply["means"])
    logit_difference, logit_tolerance = max_difference_within_float32(kept_logits, decoded_ply["opacity_logits"])
    if mean_difference > mean_tolerance or logit_difference > logit_tolerance:
        raise SystemExit(f"checkpoint rows do not match the PLY: means {mean_difference} (tol "
                         f"{mean_tolerance}), logits {logit_difference} (tol {logit_tolerance})")
    row_match = {
        "method": ("no crop (PLY row count equals the checkpoint's Gaussian count: exported with --no-crop); "
                   if legacy_without_crop else
                   "rebuilt the export's crop with select_crop (replayed from the sidecar's recorded "
                   "parameters, or train_lidar_bounds_map + crop_mask for a legacy box export); ")
                  + "kept checkpoint row k is PLY row k; verified by comparing every mean and opacity logit",
        "rows_kept": len(kept_means), "max_abs_mean_difference": mean_difference,
        "mean_tolerance_float32": mean_tolerance, "max_abs_logit_difference": logit_difference,
        "logit_tolerance_float32": logit_tolerance}
    return {"tensors": tensors, "settings": settings, "dataparser_transform": dataparser_transform,
            "keep_mask": keep_mask, "row_match": row_match}


def bake_winners_at_camera_direction(checkpoint: dict, winner_rows_ply: np.ndarray, winner_means_map: np.ndarray,
                                     camera_centre_map: np.ndarray) -> np.ndarray:
    """Bake the winning Gaussians at this camera's own view direction with its appearance row.

    Args:
        checkpoint: Output of ``load_checkpoint_rows_matching_ply``.
        winner_rows_ply: int [P] PLY rows (= kept checkpoint rows) of the winners.
        winner_means_map: float [P,3] their centres.
        camera_centre_map: float [3] camera centre in the map frame.

    Returns:
        float32 [P,3] RGB.
    """
    from viking_splatad.export.splatad_colours import load_upstream_rgb_decoder  # pylint: disable=import-outside-toplevel

    export_module = load_export_script_module()
    tensors, settings, keep_mask = checkpoint["tensors"], checkpoint["settings"], checkpoint["keep_mask"]
    directions_dataparser = rotate_directions_to_dataparser(
        view_directions_map(winner_means_map, camera_centre_map), checkpoint["dataparser_transform"])
    decoder, appearance_weight = load_upstream_rgb_decoder(tensors["pipeline_state"], settings)
    appearance_vector = appearance_weight[appearance_row_of_checked_camera(settings["cameras"])].numpy()
    return bake_at_per_gaussian_directions(
        decoder, tensors["features_dc"][keep_mask][winner_rows_ply],
        tensors["features_rest"][keep_mask][winner_rows_ply], appearance_vector, directions_dataparser,
        export_module.decoder_batch_size_for_budget(decoder))


def mean_absolute_difference(first: np.ndarray, second: np.ndarray) -> float:
    """Mean of |first - second| over all elements.

    Args:
        first: float array.
        second: float array of the same shape.

    Returns:
        The mean absolute difference.
    """
    return float(np.abs(first - second).mean())


def build_diagnostic_report(run_id: str, frame_row: dict, frame_index: int, images: dict, visibility: dict,
                            counts: dict, checkpoint: dict, camera_centre_map: np.ndarray,
                            summaries: dict, colours: dict, resize_factor: float) -> dict:
    """Assemble the diagnostic JSON.

    Args:
        run_id: Run folder name.
        frame_row: ``split_frames.csv`` row.
        frame_index: Dataset frame index.
        images: ``dataset``, ``render`` and ``photo`` RGB arrays.
        visibility: Output of ``project_winners_onto_frame``.
        counts: The ``counts`` block.
        checkpoint: Output of ``load_checkpoint_rows_matching_ply``.
        camera_centre_map: float [3].
        summaries: The four ``error_summary`` dicts, keyed a/b/c/d.
        colours: ``ply``, ``baked``, ``render`` colours at the winners.
        resize_factor: ``derived.json`` resize factor.

    Returns:
        The report dict.
    """
    manifest_size_hw = visibility["manifest_size_hw"]
    render_hw, photo_hw = images["render"].shape[:2], images["photo"].shape[:2]
    tail_key = f"all_channels_p{int(TAIL_PERCENTILE)}"
    interpretation = {
        statistic: interpret_errors(summaries["a_ply_colour_degree0"][statistic],
                                    summaries["b_baked_at_camera_direction"][statistic],
                                    summaries["c_splatad_render_at_same_pixels"][statistic])
        for statistic in ("all_channels_median", tail_key)}
    return {
        "run_id": run_id,
        "frame": {"sensor": CAMERA_SENSOR, "split": TEST_SPLIT, "frame_index": frame_index,
                  "timestamp_ns": int(frame_row["timestamp_ns"]), "file": frame_row["file"],
                  "selection_rule": "median element (len // 2) of test camera frames sorted by frame_index"},
        "resolution": {
            "dataset_image_hw": list(images["dataset"].shape[:2]), "manifest_camera_hw": list(manifest_size_hw),
            "render_hw": list(render_hw), "photo_hw": list(photo_hw),
            "render_over_manifest_scale_hw": [render_hw[0] / manifest_size_hw[0], render_hw[1] / manifest_size_hw[1]],
            "dataset_image_minus_photo_mean_abs": mean_absolute_difference(images["dataset"], images["photo"])
            if images["dataset"].shape == images["photo"].shape else None,
            "derived_resize_factor": resize_factor,
            "resize_factor_note": f"derived.json resize_factor is {resize_factor}; coordinates are scaled by the "
                                  f"render/manifest size ratio ({render_hw[1] / manifest_size_hw[1]})"},
        "opacity_threshold": OPACITY_THRESHOLD,
        "counts": counts,
        "checkpoint_row_match": checkpoint["row_match"],
        "bake": {"direction": "d_map = normalize(mean - camera_centre_map) per Gaussian, rotated by the "
                              "dataparser rotation (renormalised), camera->scene sign",
                 "camera_centre_map": camera_centre_map.tolist(), "appearance": CAMERA_SENSOR,
                 "appearance_row": appearance_row_of_checked_camera(checkpoint["settings"]["cameras"])},
        "absolute_error_vs_photo": summaries,
        "colour_difference_between_variants_mean_abs": {
            "a_vs_b": mean_absolute_difference(colours["ply"], colours["baked"]),
            "b_vs_c": mean_absolute_difference(colours["baked"], colours["render"]),
            "a_vs_c": mean_absolute_difference(colours["ply"], colours["render"])},
        "interpretation_rule": (
            f"closeness: |x - y| <= {CLOSENESS_FRACTION} * max(x, y). If (b) ~ (a), H1 (averaging over all "
            "sphere directions) is not the cause. If (b) ~ (c), the export's per-Gaussian decoding is as good "
            "as possible for this direction. If (b) is far from (c), H2 (decoding alpha-blended features vs "
            "single Gaussians) dominates."),
        "interpretation": interpretation,
    }


def require_frame_record(ply_path: Path) -> None:
    """Refuse a PLY that has no ``supersplat_frame.json`` beside it.

    Every export since 2026-11-22 writes the sidecar and no legacy map-frame exports remain, so a missing
    sidecar means an incomplete export (e.g. a first export killed between the PLY and sidecar renames) and
    the frame of the centres would be a guess.

    Args:
        ply_path: ``supersplat.ply`` to check.

    Returns:
        None.

    Raises:
        SystemExit: If the sidecar is absent, telling the user to re-export with scripts/22.
    """
    frame_record_path = Path(ply_path).parent / VIEWER_FRAME_FILE_NAME
    if not frame_record_path.is_file():
        raise SystemExit(f"{ply_path} has no {VIEWER_FRAME_FILE_NAME} beside it, so its frame is unknown "
                         "(an incomplete export, or one older than the sidecar); re-export it with "
                         "scripts/22_export_supersplat_ply.py")


def run_test_frame_diagnostic(arguments: argparse.Namespace) -> None:
    """Split the colour error into PLY colour, camera-direction bake and SplatAD's own render.

    Writes ``<run_id>_test_frame_diagnostic.{json,png}`` into ``output_dir`` and prints the JSON.

    Args:
        arguments: parsed CLI (run_dir, dataset, output_dir).

    Returns:
        None.
    """
    run_dir, run_id = arguments.run_dir, arguments.run_dir.name
    # Step 1: dataset (verified), PLY decode, opacity filter.
    dataset_dir = resolve_dataset_dir(arguments)
    require_frame_record(run_dir / PLY_RELATIVE_PATH)
    decoded_ply = decode_ply(run_dir / PLY_RELATIVE_PATH)
    opaque_indices = select_opaque_indices(decoded_ply["alpha"])
    # Step 2: median TEST frame, looked up in the dataset by frame index and verified by file name.
    frame_row = pick_median_frame(run_dir / "split_frames.csv", TEST_SPLIT)
    view = open_dataset_view(dataset_dir)
    frame_index, dataset_frame_path = locate_dataset_frame(view, frame_row)
    frame_file_name = Path(frame_row["file"]).name
    images = {"render": read_rgb_unit(run_dir / RENDER_RGB_FOLDER / frame_file_name),
              "photo": read_rgb_unit(run_dir / PHOTO_RGB_FOLDER / frame_file_name),
              "dataset": read_rgb_unit(dataset_frame_path)}
    # Step 3: z-buffer winners among the opaque Gaussians, sampled in render and photo.
    visibility = project_winners_onto_frame(decoded_ply["means"][opaque_indices], view, frame_index)
    winners = visibility["winners"]
    winner_rows_ply = opaque_indices[winners]
    photo_samples = sample_image_at_winners(images["photo"], visibility)
    render_samples = sample_image_at_winners(images["render"], visibility)
    # Step 4: checkpoint rows proven equal to the PLY rows, then the bake at the camera's direction.
    checkpoint = load_checkpoint_rows_matching_ply(arguments, dataset_dir, decoded_ply)
    camera_centre_map = visibility["pose_map_camera"][:3, 3]
    baked_colours = bake_winners_at_camera_direction(
        checkpoint, winner_rows_ply, decoded_ply["means"][winner_rows_ply], camera_centre_map
    ).astype(np.float64)
    # Step 5: errors vs the photo.
    ply_colours = decoded_ply["colours"][winner_rows_ply]
    summaries = {
        "a_ply_colour_degree0": error_summary(np.abs(ply_colours - photo_samples)),
        "b_baked_at_camera_direction": error_summary(np.abs(baked_colours - photo_samples)),
        "c_splatad_render_at_same_pixels": error_summary(np.abs(render_samples - photo_samples)),
        "d_splatad_render_whole_image": error_summary(
            np.abs(images["render"] - images["photo"]).reshape(-1, CHANNEL_COUNT))}
    counts = {"gaussians_in_ply": len(decoded_ply["means"]), "above_opacity_threshold": len(opaque_indices),
              "in_front_and_in_image": len(visibility["visible_indices"]), "pixels_after_zbuffer": len(winners)}
    resize_factor = resize_factor_of_dataset(dataset_dir, run_dir)
    report = build_diagnostic_report(
        run_id, frame_row, frame_index, images, visibility, counts, checkpoint, camera_centre_map, summaries,
        {"ply": ply_colours, "baked": baked_colours, "render": render_samples}, resize_factor)
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    (arguments.output_dir / f"{run_id}_test_frame_diagnostic.json").write_text(json.dumps(report, indent=2))
    # Step 6: 3-panel PNG: photo | PLY colours | camera-direction bake (dots at the winning pixels on black).
    photo_panel, scale = build_photo_panel(images["photo"], visibility["manifest_size_hw"],
                                           DIAGNOSTIC_PANEL_WIDTH_PIXELS)
    winner_columns, winner_rows = visibility["column"][winners], visibility["row"][winners]
    splat_panels = [draw_dot_panel(photo_panel, winner_columns, winner_rows, panel_colours, scale)
                    for panel_colours in (ply_colours, baked_colours)]
    cv2.imwrite(str(arguments.output_dir / f"{run_id}_test_frame_diagnostic.png"),
                np.hstack([photo_panel, *splat_panels]))
    print(json.dumps(report, indent=2))


def run_train_frame_check(arguments: argparse.Namespace) -> None:
    """Compare the PLY colours with the median TRAIN camera frame.

    Writes ``<run_id>.json`` and ``<run_id>.png`` into ``output_dir`` and prints the JSON.

    Args:
        arguments: parsed CLI (run_dir, dataset, output_dir).

    Returns:
        None.
    """
    run_id = arguments.run_dir.name
    # Step 1: dataset (verified), PLY decode, opacity filter.
    dataset_dir = resolve_dataset_dir(arguments)
    require_frame_record(arguments.run_dir / PLY_RELATIVE_PATH)
    decoded_ply = decode_ply(arguments.run_dir / PLY_RELATIVE_PATH)
    opaque_indices = select_opaque_indices(decoded_ply["alpha"])
    # Step 2: frame, image, z-buffer winners.
    frame_row = pick_median_frame(arguments.run_dir / "split_frames.csv", TRAIN_SPLIT)
    view = open_dataset_view(dataset_dir)
    frame_index, frame_path = locate_dataset_frame(view, frame_row)
    image_rgb = read_rgb_unit(frame_path)
    visibility = project_winners_onto_frame(decoded_ply["means"][opaque_indices], view, frame_index)
    winners = visibility["winners"]
    # Step 3: compare winner colours with the image under them.
    winner_colours = decoded_ply["colours"][opaque_indices[winners]]
    absolute_errors = np.abs(winner_colours - sample_image_at_winners(image_rgb, visibility))
    image_height, image_width = image_rgb.shape[:2]
    report = {
        "run_id": run_id,
        "ply": str(arguments.run_dir / PLY_RELATIVE_PATH),
        "dataset": str(dataset_dir),
        "frame": {"sensor": CAMERA_SENSOR, "frame_index": frame_index,
                  "timestamp_ns": int(frame_row["timestamp_ns"]), "file": frame_row["file"],
                  "selection_rule": "median element (len // 2) of train camera frames sorted by frame_index",
                  "image_size_hw": [image_height, image_width]},
        "opacity_threshold": OPACITY_THRESHOLD,
        "colour_decode": "0.5 + C0 * f_dc, C0 = 1/(2 sqrt(pi)), clipped to [0,1]; alpha = sigmoid(opacity)",
        "counts": {"gaussians_total": len(decoded_ply["means"]), "above_opacity_threshold": len(opaque_indices),
                   "in_front_and_in_image": int(len(visibility["visible_indices"])),
                   "pixels_after_zbuffer": int(len(winners))},
        "absolute_error_baked_colour_vs_image": error_summary(absolute_errors),
        "baseline_raw_features_dc": "skipped: gaussians.npz has no colours; needs the torch checkpoint",
    }
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    (arguments.output_dir / f"{run_id}.json").write_text(json.dumps(report, indent=2))
    # Step 4: side-by-side PNG (camera | splat colours at the winning pixels on black).
    camera_panel, scale = build_photo_panel(image_rgb, visibility["manifest_size_hw"], PANEL_WIDTH_PIXELS)
    splat_panel = draw_dot_panel(camera_panel, visibility["column"][winners], visibility["row"][winners],
                                 winner_colours, scale)
    cv2.imwrite(str(arguments.output_dir / f"{run_id}.png"), np.hstack([camera_panel, splat_panel]))
    print(json.dumps(report, indent=2))


def main() -> None:
    """Run the train-frame colour check (or, with ``--diagnose-test-frame``, the test-frame diagnostic).

    Returns:
        None.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=None,
                        help="Dataset the run trained on (default data/intermediate/<session of the run>).")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / DEFAULT_OUTPUT_RELATIVE_DIR)
    parser.add_argument("--diagnose-test-frame", action="store_true",
                        help="Split the error on the median TEST frame (needs torch: run in the image).")
    arguments = parser.parse_args()
    if arguments.diagnose_test_frame:
        run_test_frame_diagnostic(arguments)
    else:
        run_train_frame_check(arguments)


if __name__ == "__main__":
    main()
