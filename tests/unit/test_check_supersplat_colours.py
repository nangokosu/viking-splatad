"""Known-answer tests of the pure helpers of scripts/23_check_supersplat_colours.py.

Expected values are computed in each test from first principles on tiny synthetic inputs.
"""

import csv
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "23_check_supersplat_colours.py"


def load_script_module():
    """Import the script by file path (its file name starts with a digit).

    Returns:
        The loaded script module.
    """
    module_spec = importlib.util.spec_from_file_location("check_supersplat_colours_script", SCRIPT_PATH)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


SCRIPT = load_script_module()


def test_zbuffer_nearest_keeps_nearest_point_per_pixel():
    """Two points on pixel 5 (depths 3, 1) and one on pixel 2: winners are the depth-1 point and the lone one."""
    pixel_indices = np.array([5, 2, 5])
    depths = np.array([3.0, 7.0, 1.0])
    winners = SCRIPT.zbuffer_nearest(pixel_indices, depths)
    assert sorted(winners.tolist()) == [1, 2]


def test_error_summary_median_and_tail_per_channel():
    """Channel values 0..4 give median 2 and the 90th percentile 3.6 (linear interpolation, 0.9 * 4)."""
    channel_values = np.arange(5, dtype=np.float64)
    errors = np.stack([channel_values, channel_values, channel_values], axis=1)
    summary = SCRIPT.error_summary(errors)
    assert summary["r_median"] == 2.0
    assert math.isclose(summary["g_p90"], 0.9 * 4.0)
    assert summary["all_channels_median"] == 2.0


def test_view_directions_are_unit_and_point_from_camera_to_gaussian():
    """A Gaussian at (3, 0, 4) seen from (0, 0, 0) has direction (0.6, 0, 0.8); sign camera -> scene."""
    means = np.array([[3.0, 0.0, 4.0], [0.0, -2.0, 0.0]])
    directions = SCRIPT.view_directions_map(means, np.zeros(3))
    assert np.allclose(directions[0], [3.0 / 5.0, 0.0, 4.0 / 5.0])
    assert np.allclose(directions[1], [0.0, -1.0, 0.0])
    assert np.allclose(np.linalg.norm(directions, axis=1), 1.0)
    shifted = SCRIPT.view_directions_map(means, np.array([3.0, 0.0, 0.0]))
    assert np.allclose(shifted[0], [0.0, 0.0, 1.0])


def test_rotate_directions_applies_rotation_and_ignores_translation_and_scale():
    """A = 2 * Rz(90 deg) maps x to 2y; translation column is ignored and the result renormalised."""
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    transform = np.hstack([2.0 * rotation, [[10.0], [20.0], [30.0]]])
    rotated = SCRIPT.rotate_directions_to_dataparser(np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]), transform)
    assert np.allclose(rotated, [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


def test_scale_pixel_coordinates_identity_and_halving():
    """Equal sizes keep pixels; a half-size target halves coordinates; results are clamped inside."""
    columns, rows = np.array([10, 99]), np.array([4, 49])
    same = SCRIPT.scale_pixel_coordinates(columns, rows, (50, 100), (50, 100))
    assert same[0].tolist() == [10, 99] and same[1].tolist() == [4, 49]
    half = SCRIPT.scale_pixel_coordinates(columns, rows, (50, 100), (25, 50))
    assert half[0].tolist() == [5, 49] and half[1].tolist() == [2, 24]


def test_is_close_and_gap_fraction():
    """10 % relative closeness; gap fraction (a-b)/(a-c) = (0.4-0.3)/(0.4-0.2) = 0.5."""
    assert SCRIPT.is_close(1.0, 0.95)
    assert not SCRIPT.is_close(1.0, 0.5)
    assert math.isclose(SCRIPT.fraction_of_gap_closed(0.4, 0.3, 0.2), 0.5)
    assert math.isnan(SCRIPT.fraction_of_gap_closed(0.3, 0.1, 0.3))


def test_interpret_errors_follows_the_rule():
    """b = a -> H1 not the cause and H2 dominates (b far from c); b = c -> decoding is as good as possible."""
    same_as_ply = SCRIPT.interpret_errors(0.40, 0.40, 0.10)
    assert same_as_ply["h1_sphere_averaging_is_not_the_cause"] and same_as_ply["h2_blended_decoding_dominates"]
    same_as_render = SCRIPT.interpret_errors(0.40, 0.10, 0.10)
    assert same_as_render["per_gaussian_decoding_is_as_good_as_possible"]
    assert not same_as_render["h1_sphere_averaging_is_not_the_cause"]


def test_visible_zbuffer_winners_projects_and_resolves_occlusion():
    """Identity pose, fx = fy = 10, principal point (5, 5): (0,0,2) hits pixel (5,5) and hides (0,0,4);
    (0,0,-1) is behind; (100,0,1) projects to column 1005, outside the 10x10 image."""
    camera_matrix = np.array([[10.0, 0.0, 5.0], [0.0, 10.0, 5.0], [0.0, 0.0, 1.0]])
    means = np.array([[0.0, 0.0, 4.0], [0.0, 0.0, 2.0], [0.0, 0.0, -1.0], [100.0, 0.0, 1.0]])
    result = SCRIPT.visible_zbuffer_winners(means, np.eye(4), camera_matrix, (10, 10))
    assert result["visible_indices"].tolist() == [0, 1]
    assert result["winners"].tolist() == [1]
    assert (result["column"][1], result["row"][1]) == (5, 5)


def test_pick_median_frame_uses_len_floor_half_of_sorted_indices(tmp_path):
    """Five test frames (indices 40, 10, 30, 20, 50 in file order) -> sorted median 30; train rows ignored."""
    split_csv = tmp_path / "split_frames.csv"
    with open(split_csv, "w", newline="") as split_file:
        writer = csv.writer(split_file)
        writer.writerow(["sensor", "kind", "split", "frame_index", "timestamp_ns", "time_s", "file"])
        for frame_index in (40, 10, 30, 20, 50):
            writer.writerow([SCRIPT.CAMERA_SENSOR, "camera", "test", frame_index, 0, 0.0, f"f{frame_index}.jpg"])
        writer.writerow([SCRIPT.CAMERA_SENSOR, "camera", "train", 99, 0, 0.0, "train.jpg"])
    assert SCRIPT.pick_median_frame(split_csv, "test")["frame_index"] == "30"


def write_split_csv(split_csv, rows):
    """Write a ``split_frames.csv`` with the given (sensor, split, frame_index) rows.

    Args:
        split_csv: Target path.
        rows: Iterable of (sensor, split, frame_index) tuples.
    """
    with open(split_csv, "w", newline="") as split_file:
        writer = csv.writer(split_file)
        writer.writerow(["sensor", "kind", "split", "frame_index", "timestamp_ns", "time_s", "file"])
        for sensor, split, frame_index in rows:
            writer.writerow([sensor, "camera", split, frame_index, 0, 0.0, f"f{frame_index}.jpg"])


def test_pick_median_frame_without_rows_of_the_split_exits(tmp_path):
    """Only train rows exist, so asking for the test split must stop with a message, not an IndexError."""
    split_csv = tmp_path / "split_frames.csv"
    write_split_csv(split_csv, [(SCRIPT.CAMERA_SENSOR, "train", 1)])
    with pytest.raises(SystemExit, match="no ids_camera row"):
        SCRIPT.pick_median_frame(split_csv, "test")


def test_pick_median_frame_ignores_other_sensors(tmp_path):
    """A row of another sensor in the requested split does not count as a camera row."""
    split_csv = tmp_path / "split_frames.csv"
    write_split_csv(split_csv, [("other_camera", "test", 1)])
    with pytest.raises(SystemExit):
        SCRIPT.pick_median_frame(split_csv, "test")


def test_no_visible_winners_exits_with_a_message():
    """Every centre is behind the camera: zero winners, and the require step stops instead of going on."""
    camera_matrix = np.array([[10.0, 0.0, 5.0], [0.0, 10.0, 5.0], [0.0, 0.0, 1.0]])
    means = np.array([[0.0, 0.0, -1.0], [1.0, 0.0, -2.0]])
    visibility = SCRIPT.visible_zbuffer_winners(means, np.eye(4), camera_matrix, (10, 10))
    assert len(visibility["winners"]) == 0
    with pytest.raises(SystemExit, match="no Gaussian"):
        SCRIPT.require_nonempty_winners(visibility)


def test_zbuffer_nearest_of_no_points_is_empty():
    """Empty input must give an empty index array (it used to fail with an IndexError)."""
    winners = SCRIPT.zbuffer_nearest(np.empty(0, dtype=np.int64), np.empty(0))
    assert winners.shape == (0,)


def test_select_opaque_indices_threshold_and_empty():
    """alpha 0.9 and 0.6 pass the 0.5 threshold, 0.5 itself (strict >) and 0.1 do not; none passing exits."""
    assert SCRIPT.select_opaque_indices(np.array([0.9, 0.1, 0.6, 0.5])).tolist() == [0, 2]
    with pytest.raises(SystemExit):
        SCRIPT.select_opaque_indices(np.array([0.1, 0.5]))


def test_error_summary_on_empty_input_raises():
    """A median of nothing is NaN; the summary refuses it instead of reporting NaN."""
    with pytest.raises(ValueError, match="at least one"):
        SCRIPT.error_summary(np.empty((0, SCRIPT.CHANNEL_COUNT)))


def test_decode_ply_sigmoid_is_stable_for_large_logits(tmp_path, monkeypatch):
    """Logits +-1000 decode to alpha 1 and 0 without overflow warnings; f_dc = 0 decodes to colour 0.5."""
    fields = {name: np.zeros(2, dtype=np.float32) for name in
              ("x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2")}
    fields["opacity"] = np.array([1000.0, -1000.0], dtype=np.float32)
    monkeypatch.setattr(SCRIPT, "read_supersplat_ply", lambda path: fields)
    with np.errstate(all="raise"):
        decoded = SCRIPT.decode_ply(tmp_path / "unused.ply")
    assert decoded["alpha"].tolist() == [1.0, 0.0]
    assert np.allclose(decoded["colours"], 0.5)


def test_decode_ply_colour_follows_the_sh_dc_formula(monkeypatch, tmp_path):
    """colour = 0.5 + C0 * f_dc with C0 = 1 / (2 sqrt(pi)): f_dc = 1 gives 0.5 + 0.2820948, and f_dc = 10 clips to 1."""
    fields = {name: np.zeros(2, dtype=np.float32) for name in ("x", "y", "z", "opacity")}
    fields["f_dc_0"] = np.array([1.0, 10.0], dtype=np.float32)
    fields["f_dc_1"] = np.zeros(2, dtype=np.float32)
    fields["f_dc_2"] = np.zeros(2, dtype=np.float32)
    monkeypatch.setattr(SCRIPT, "read_supersplat_ply", lambda path: fields)
    colours = SCRIPT.decode_ply(tmp_path / "unused.ply")["colours"]
    assert math.isclose(colours[0, 0], 0.5 + 1.0 / (2.0 * math.sqrt(math.pi)), rel_tol=1e-6)
    assert colours[1, 0] == 1.0


def test_fraction_of_gap_closed_is_negative_when_the_bake_is_worse():
    """a = 0.4, b = 0.5, c = 0.2: (0.4 - 0.5) / (0.4 - 0.2) = -0.5."""
    assert math.isclose(SCRIPT.fraction_of_gap_closed(0.4, 0.5, 0.2), -0.5)


def test_rgb_unit_to_bgr_uint8_rounds_and_swaps_channels():
    """0.5 * 255 = 127.5 rounds to 128 (rint: half to even gives 128), 0.4 * 255 = 102 exactly; RGB becomes BGR."""
    rgb = np.array([[0.5, 0.4, 1.0]])
    assert SCRIPT.rgb_unit_to_bgr_uint8(rgb).tolist() == [[255, 102, 128]]


def test_draw_dot_panel_paints_the_scaled_pixel_in_bgr():
    """A red (1, 0, 0) dot at reference pixel (column 4, row 2) with scale 0.5 lands at (2, 1) as BGR (0, 0, 255)."""
    panel = SCRIPT.draw_dot_panel(np.zeros((6, 6, 3), dtype=np.uint8), np.array([4]), np.array([2]),
                                  np.array([[1.0, 0.0, 0.0]]), 0.5)
    assert panel[1, 2].tolist() == [0, 0, 255]
    assert panel[5, 5].tolist() == [0, 0, 0]


def test_appearance_row_must_match_the_export_row():
    """With cameras [ids_camera, other] the checked camera is row 0, the export's row; [other, ids_camera]
    makes them differ (row 1 vs the export's 0), and an absent camera stops too."""
    assert SCRIPT.appearance_row_of_checked_camera([SCRIPT.CAMERA_SENSOR, "other"]) == 0
    with pytest.raises(SystemExit, match="mix cameras"):
        SCRIPT.appearance_row_of_checked_camera(["other", SCRIPT.CAMERA_SENSOR])
    with pytest.raises(SystemExit, match="not among"):
        SCRIPT.appearance_row_of_checked_camera(["other"])


def test_locate_dataset_frame_checks_the_file_name():
    """A matching name returns (index, path); a different name (wrong dataset) exits."""
    class FakeView:
        sensor_frame_paths = {SCRIPT.CAMERA_SENSOR: [Path("a/000000_1.jpg"), Path("a/000001_2.jpg")]}

    row = {"frame_index": "1", "file": "elsewhere/000001_2.jpg"}
    assert SCRIPT.locate_dataset_frame(FakeView, row) == (1, Path("a/000001_2.jpg"))
    with pytest.raises(SystemExit, match="split row"):
        SCRIPT.locate_dataset_frame(FakeView, {"frame_index": "1", "file": "000001_999.jpg"})
    with pytest.raises(SystemExit, match="outside"):
        SCRIPT.locate_dataset_frame(FakeView, {"frame_index": "7", "file": "000001_2.jpg"})


def test_max_difference_within_float32_scales_with_magnitude():
    """Tolerance is 2 * eps(float32) * max(|stored|max, 1): 1000 gives 2000 eps, and values below 1 use 1."""
    eps = float(np.finfo(np.float32).eps)
    difference, tolerance = SCRIPT.max_difference_within_float32(np.array([1000.0]), np.array([999.0]))
    assert difference == 1.0 and math.isclose(tolerance, 2.0 * eps * 1000.0)
    _, small_tolerance = SCRIPT.max_difference_within_float32(np.array([0.1]), np.array([0.1]))
    assert math.isclose(small_tolerance, 2.0 * eps)


def build_fake_decoder(torch_module, feature_dim, appearance_dim):
    """Small seeded stand-in with the upstream decoder's interface (``skip_dim`` and ``net``).

    The net maps (feature_dim + appearance_dim + 3) channels to 2 * skip_dim (gain, offset) through two
    3x3 convolutions with zero padding, as the upstream CNN does.

    Args:
        torch_module: The imported torch module.
        feature_dim: Width of ``features_rest``.
        appearance_dim: Width of the appearance vector.

    Returns:
        The module.
    """
    torch_module.manual_seed(0)
    skip_dim = SCRIPT.CHANNEL_COUNT
    decoder = torch_module.nn.Module()
    decoder.skip_dim = skip_dim
    decoder.net = torch_module.nn.Sequential(
        torch_module.nn.Conv2d(feature_dim + appearance_dim + 3, 8, kernel_size=3, padding=1),
        torch_module.nn.ReLU(),
        torch_module.nn.Conv2d(8, 2 * skip_dim, kernel_size=3, padding=1))
    return decoder.eval()


def test_batched_bake_equals_bake_gaussian_colours_per_gaussian():
    """The batched per-Gaussian-direction bake equals ``bake_gaussian_colours`` called once per Gaussian
    with that Gaussian's single direction, for a batch size that does not divide N (5 Gaussians, batch 2)."""
    torch_module = pytest.importorskip("torch")
    from viking_splatad.export.splatad_colours import bake_gaussian_colours

    feature_dim, appearance_dim, gaussian_count = 4, 2, 5
    decoder = build_fake_decoder(torch_module, feature_dim, appearance_dim)
    generator = np.random.default_rng(0)
    features_dc = generator.uniform(size=(gaussian_count, SCRIPT.CHANNEL_COUNT)).astype(np.float32)
    features_rest = generator.normal(size=(gaussian_count, feature_dim)).astype(np.float32)
    appearance_vector = generator.normal(size=appearance_dim).astype(np.float32)
    directions = generator.normal(size=(gaussian_count, 3))
    directions = (directions / np.linalg.norm(directions, axis=1, keepdims=True)).astype(np.float32)
    batched = SCRIPT.bake_at_per_gaussian_directions(decoder, features_dc, features_rest, appearance_vector,
                                                     directions, 2)
    for gaussian_index in range(gaussian_count):
        single = bake_gaussian_colours(
            decoder, features_dc[gaussian_index:gaussian_index + 1], features_rest[gaussian_index:gaussian_index + 1],
            appearance_vector, directions[gaussian_index:gaussian_index + 1], 1)[0, 0]
        assert np.allclose(batched[gaussian_index], single, atol=1e-5)
    # The decoder passed in keeps its zero padding (the bake works on a copy).
    assert all(module.padding_mode == "zeros" for module in decoder.modules()
               if isinstance(module, torch_module.nn.Conv2d))


def test_batched_bake_rejects_a_batch_size_below_one():
    """batch_size 0 would loop forever or return empty colours; it must raise."""
    pytest.importorskip("torch")
    with pytest.raises(ValueError, match="batch_size"):
        SCRIPT.bake_at_per_gaussian_directions(None, np.zeros((1, 3)), np.zeros((1, 1)), np.zeros(1),
                                               np.zeros((1, 3)), 0)


def test_scale_pixel_coordinates_uses_pixel_centres():
    """u' = (u + 0.5) * W'/W - 0.5: doubling maps pixel 3 (centre 3.5) to 6.5 -> 6 (rint, ties to even);
    pixel 0 of a 4 -> 2 downscale (centre 0.5 -> 0.25 - 0.5 = -0.25) rounds to 0."""
    doubled = SCRIPT.scale_pixel_coordinates(np.array([3]), np.array([3]), (10, 10), (20, 20))
    assert doubled[0].tolist() == [6] and doubled[1].tolist() == [6]
    halved = SCRIPT.scale_pixel_coordinates(np.array([0, 3]), np.array([0, 3]), (4, 4), (2, 2))
    assert halved[0].tolist() == [0, 1] and halved[1].tolist() == [0, 1]


def test_export_script_loader_is_cached_and_default_output_dir_is_anchored_to_the_repo():
    """The script-22 module is loaded once; the default output folder does not depend on the cwd."""
    assert SCRIPT.load_export_script_module() is SCRIPT.load_export_script_module()
    assert SCRIPT.PROJECT_ROOT == Path(__file__).resolve().parents[2]
    assert (SCRIPT.PROJECT_ROOT / SCRIPT.DEFAULT_OUTPUT_RELATIVE_DIR).is_absolute()


def test_require_frame_record_refuses_a_missing_sidecar_and_accepts_a_present_one(tmp_path):
    """The CLI entry check exits telling to re-export when no sidecar sits beside the PLY."""
    ply_path = tmp_path / "supersplat.ply"
    with pytest.raises(SystemExit, match="re-export"):
        SCRIPT.require_frame_record(ply_path)
    (tmp_path / SCRIPT.VIEWER_FRAME_FILE_NAME).write_text("{}")
    SCRIPT.require_frame_record(ply_path)
