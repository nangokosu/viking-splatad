"""Matched P1/P2 pilot on June trunks, recorded occlusion evidence and metric pictures.

This is evaluation-only. It never changes a dataset, run, anchor or checkpoint.
Undefined observations remain unknown, and a nearer return is not a grass label.
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import cv2
import numpy as np
import yaml

from viking_splatad.eval.detection import (
    cuboid_point_counts, median_half_footprint, scored_region_cuboids,
    visible_neighbourhood_mask, voxel_centers,
)
from viking_splatad.eval.hidden_geometry import gaussian_occupancy_at_centres, recorded_ray_evidence, occupied_cell_metrics
from viking_splatad.eval.rendered_pointcloud import load_test_lidar_sweeps, ray_directions, points_to_map_voxels
from viking_splatad.labels.cuboids import compute_cuboid_geometry, points_in_cuboid
from viking_splatad.viz.hidden_trunks import write_box_comparison
from viking_splatad.viz.radar_bev import bev_pixels


def load_script(root, filename):
    """Load an existing repo CLI helper without reimplementing its scientific definitions."""
    specification = importlib.util.spec_from_file_location(Path(filename).stem, root / "scripts" / filename)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def digest(path):
    """Return a streaming source checksum without recording machine/account paths."""
    checksum = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def completed_run(run_dir):
    """Return verified completion/split records; fail rather than score partial outputs."""
    information = dict(line.split(": ", 1) for line in (run_dir / "job_info.txt").read_text().splitlines() if ": " in line)
    if (not information.get("end_utc") or information.get("split_render_check") != "match"
            or information.get("overfitting_plot") != "written"):
        raise ValueError("comparison requires a successful complete run, including renders and split")
    split = json.loads((run_dir / "split.json").read_text())
    return information, split


def matching_runs(baseline, radar):
    """Require identical scene/data/split/seed/step budget and explicit loss-only L configuration."""
    baseline_info, baseline_split = completed_run(baseline)
    radar_info, radar_split = completed_run(radar)
    for key in ("SESSION", "TRAIN_SEED", "MAX_NUM_ITERATIONS", "TRAIN_LIDAR_ONLY", "ADD_MISSING_POINTS"):
        if baseline_info.get(key) is None or baseline_info[key] != radar_info.get(key):
            raise ValueError(f"comparison arms disagree on {key}")
    if baseline_split["dataset"] != radar_split["dataset"] or baseline_split["membership_sha256"] != radar_split["membership_sha256"]:
        raise ValueError("comparison requires exactly the same dataset and timestamp split")
    if (baseline_info.get("TRAIN_METHOD", "splatad") != "splatad"
            or radar_info.get("TRAIN_METHOD") != "splatad-radar-lidar" or radar_info.get("RADAR_LIDAR_AUTO_WEIGHT") != "True"):
        raise ValueError("comparison requires stock B and the configured loss-only L")
    initialization = [json.loads(line) for line in (radar / "radar_diagnostics.jsonl").read_text().splitlines()]
    initialized = next(record for record in initialization if record["event"] == "stock_initialization")
    if initialized["radar_gaussians_added"] != 0:
        raise ValueError("this comparison is restricted to no radar initialization")
    return baseline_info, baseline_split


def extract_truth(root, labels_path, gt_dir, session):
    """Build June-only GT using existing script 05 definitions, kept outside model inputs.

    Args:
        root: Source snapshot; labels_path: original labels JSON; gt_dir: eval-only output;
        session: committed session settings selecting its own label sample.
    Returns:
        Cuboids and grid. Saved metadata hashes source labels to prevent stale GT reuse.
    """
    extractor = load_script(root, "05_extract_cuboids.py")
    document = json.loads(labels_path.read_text())
    cuboids = extractor.parse_cuboids(document, class_names=extractor.build_class_names(document),
                                     labelset_name="ground-truth", sample_key=session["session"]["labels_sample_key"])
    voxel_size = extractor.derive_default_voxel_size(cuboids)
    extractor.plan_grid(cuboids, voxel_size, extractor.DEFAULT_MAX_GRID_MB)
    # A content-addressed subdirectory keeps previous GT untouched.
    destination = gt_dir / digest(labels_path)
    destination.mkdir(parents=True, exist_ok=True)
    grid_path = destination / "occupancy.npz"
    if not grid_path.exists():
        grid = extractor.voxelize_cuboids(cuboids, voxel_size)
        np.savez_compressed(grid_path, class_ids=grid.class_ids.astype(np.uint8), origin=grid.origin, voxel_size=voxel_size)
        (destination / "cuboids.json").write_text(json.dumps([extractor.cuboid_to_json(box) for box in cuboids], indent=2) + "\n")
    with np.load(grid_path, allow_pickle=False) as archive:
        grid = {key: archive[key] for key in archive.files}
    return cuboids, grid, destination


def p1_cells(sweeps, point_attribute, indices, origin, voxel_size, shape):
    """Voxelize this arm's own points onto the common selected cells; return bool [V]."""
    wanted = np.ravel_multi_index(indices.T, shape)
    occupied = np.zeros(len(indices), dtype=bool)
    for sweep in sweeps:
        _, point_indices, inside = points_to_map_voxels(getattr(sweep, point_attribute), sweep.lidar_to_map, origin, voxel_size, shape)
        keys = np.ravel_multi_index(point_indices[inside].T, shape)
        occupied |= np.isin(wanted, keys)
    return occupied


def write_alignment(path, real_points, cuboids):
    """Draw real June held-out returns with scored box outlines before interpreting metrics."""
    corners = [compute_cuboid_geometry(box).corners for box in cuboids]
    combined = np.concatenate([real_points, *corners])[:, :2]
    bounds = np.stack([combined.min(axis=0), combined.max(axis=0)])
    pixels = 1024  # Same diagnostic display resolution as the existing BEV viewer.
    canvas = np.full((pixels, pixels, 3), 255, dtype=np.uint8)
    locations, _ = bev_pixels(real_points[:, :2], bounds, (pixels, pixels))
    canvas[tuple(locations.T)] = (160, 160, 160)
    for box in cuboids:
        geometry = compute_cuboid_geometry(box)
        locations, _ = bev_pixels(geometry.corners[:, :2], bounds, (pixels, pixels))
        hull = cv2.convexHull(locations[:, ::-1].astype(np.int32))
        cv2.polylines(canvas, [hull], True, (20, 110, 20), 1)
    if not cv2.imwrite(str(path), canvas):
        raise OSError(f"could not write {path}")


def main(argument_list=None):
    """Score a matched pilot and write cell arrays, numerical reports and all trunk views."""
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--radar", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--alignment-only", action="store_true")
    parser.add_argument("--alignment-receipt", type=Path)
    arguments = parser.parse_args(argument_list)
    if not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("run numerical evaluation in an allocated Slurm CPU job")
    arguments.out.mkdir(parents=True, exist_ok=True)
    baseline_info, split = completed_run(arguments.baseline)
    if baseline_info["SESSION"] != "june_tall_grass":
        raise ValueError("this pilot evaluates June only; no cross-session mapping")
    if not arguments.alignment_only:
        if arguments.radar is None:
            raise ValueError("a completed radar arm is required")
        matching_runs(arguments.baseline, arguments.radar)
    dataset = arguments.stage_root / "data" / baseline_info["SESSION"]
    if digest(dataset / "manifest.json") != split["dataset"]["manifest_sha256"] or digest(dataset / "poses.npz") != split["dataset"]["poses_sha256"]:
        raise ValueError("published data differs from the runs")
    manifest = json.loads((dataset / "manifest.json").read_text())
    session = yaml.safe_load((root / "configs/sessions/june_tall_grass.yaml").read_text())
    labels_path = arguments.stage_root / "source-data" / session["session"]["labels_json"]
    boxes, grid, gt_dir = extract_truth(root, labels_path, arguments.stage_root / "evaluation-inputs/june_tall_grass", session)
    settings = yaml.safe_load((root / "configs/eval.yaml").read_text())["eval"]
    threshold = float(settings["render_accumulation_threshold"])
    expected = sorted(Path(frame["file"]).stem for frame in split["frames"] if frame["kind"] == "lidar" and frame["split"] == "test")
    baseline_directory = arguments.baseline / "renders/test/lidar"
    baseline_sweeps = load_test_lidar_sweeps(baseline_directory, expected, threshold)
    poses = [sweep.lidar_to_map for sweep in baseline_sweeps]
    trunk_id = settings["scored_class_ids"]["tree_trunk"]
    scored, max_range, half_fov = scored_region_cuboids(boxes, [trunk_id], poses,
        [sweep.real_points for sweep in baseline_sweeps], manifest["lidar"]["elevation_mapping"])
    trunks = scored[trunk_id]
    if not trunks:
        raise ValueError("no June trunk boxes are in the evaluated range/FOV")
    margin = median_half_footprint([box for box in boxes if box.class_id == trunk_id])
    voxel_size = float(grid["voxel_size"])
    neighbourhood = visible_neighbourhood_mask(trunks, margin, grid["origin"], voxel_size,
                                                grid["class_ids"].shape, poses, max_range, half_fov)
    indices = np.argwhere(neighbourhood)
    centres = voxel_centers(indices, grid["origin"], voxel_size)
    truth = grid["class_ids"][tuple(indices.T)] == trunk_id
    real_points = np.concatenate([sweep.real_points @ sweep.lidar_to_map[:3, :3].T + sweep.lidar_to_map[:3, 3] for sweep in baseline_sweeps])
    write_alignment(arguments.out / "label_alignment.png", real_points, trunks)
    alignment = {"session": baseline_info["SESSION"], "manifest_sha256": digest(dataset / "manifest.json"),
                 "cuboids_sha256": digest(gt_dir / "cuboids.json"), "labels_sha256": digest(labels_path),
                 "scored_trunks": len(trunks), "real_points_per_trunk": cuboid_point_counts(real_points, trunks).tolist(),
                 "reviewed": False, "note": "Review label_alignment.png and per-trunk views; no frame transform is estimated."}
    (arguments.out / "alignment.json").write_text(json.dumps(alignment, indent=2) + "\n")
    if arguments.alignment_only:
        print(json.dumps(alignment), flush=True)
        return 0
    alignment_reviewed = False
    if arguments.alignment_receipt:
        receipt = json.loads(arguments.alignment_receipt.read_text())
        if any(receipt.get(key) != alignment[key] for key in ("session", "manifest_sha256", "cuboids_sha256", "labels_sha256")) or receipt.get("reviewed") is not True:
            raise ValueError("alignment receipt must match the exact dataset/labels and record a completed review")
        alignment_reviewed = True
    # Enforce identical recorded observations; model predictions must never set
    # visibility strata or change the common scoring domain.
    radar_directory = arguments.radar / "renders/test/lidar"
    radar_sweeps = load_test_lidar_sweeps(radar_directory, expected, threshold)
    evidence = np.zeros((len(centres), 4), dtype=bool)
    for sweep_index, stem in enumerate(expected):
        with np.load(baseline_directory / (stem + ".npz"), allow_pickle=False) as baseline_arrays, np.load(radar_directory / (stem + ".npz"), allow_pickle=False) as radar_arrays:
            for key in ("raster_pts", "raster_did_return", "lidar_to_map", "gt_points"):
                if not np.array_equal(baseline_arrays[key], radar_arrays[key]):
                    raise ValueError(f"recorded observation or pose differs between arms: {stem}/{key}")
            raster = baseline_arrays["raster_pts"].reshape(-1, 5)
            return_mask = baseline_arrays["raster_did_return"].reshape(-1)
            valid = np.isfinite(raster).all(axis=1) & (raster[:, 2] > 0)
            # Rotate rays into map-grid axes; translate cells relative to the
            # same sweep origin. Rotating centres alone would rotate voxel boxes.
            pose = baseline_sweeps[sweep_index].lidar_to_map
            rays = ray_directions(raster[valid, 0], raster[valid, 1]) @ pose[:3, :3].T
            rays /= np.linalg.norm(rays, axis=1, keepdims=True)
            evidence |= recorded_ray_evidence(centres - pose[:3, 3], voxel_size, rays, raster[valid, 2], return_mask[valid])
        print(f"recorded evidence {sweep_index + 1}/{len(expected)}", flush=True)
    masks, p1_masks, probabilities, reports = {}, {}, {}, {}
    for arm_name, run, sweeps in (("B", arguments.baseline, baseline_sweeps), ("L", arguments.radar, radar_sweeps)):
        if any(sweep.ungated_points is None for sweep in sweeps):
            raise ValueError("binding ungated P1 cannot be reconstructed from these outputs")
        p1 = p1_cells(sweeps, "ungated_points", indices, grid["origin"], voxel_size, grid["class_ids"].shape)
        p1_masks[arm_name] = p1
        with np.load(run / "renders/gaussians.npz", allow_pickle=False) as archive:
            if int(archive["checkpoint_step"]) != int(baseline_info["MAX_NUM_ITERATIONS"]) - 1:
                raise ValueError("comparison requires final equal-budget Gaussian checkpoints")
            probability = gaussian_occupancy_at_centres(centres, archive["means"], archive["covariances"], archive["opacities"],
                                                        session["splatad"]["radar_lidar_gate_chunk_size"])
        masks[arm_name], probabilities[arm_name] = probability >= threshold, probability
        reports[arm_name] = {"P1": occupied_cell_metrics(p1, truth, evidence),
                            "P2": occupied_cell_metrics(masks[arm_name], truth, evidence),
                            "present_but_hidden_gt_cells": int(np.count_nonzero(truth & masks[arm_name] & ~p1)),
                            "run_id": run.name, "gaussians_sha256": digest(run / "renders/gaussians.npz")}
        print(f"scored {arm_name}: {json.dumps(reports[arm_name])}", flush=True)
    box_reports = []
    for box_index, box in enumerate(trunks):
        inside = points_in_cuboid(compute_cuboid_geometry(box), centres) & truth
        count = int(inside.sum())
        fractions = {arm: float((inside & masks[arm]).sum() / count) if count else None for arm in masks}
        filename = f"trunk-{box_index:04d}.png"
        write_box_comparison(arguments.out / filename, box, centres, truth, masks["B"], masks["L"], evidence, real_points, margin, p1_masks)
        box_reports.append({"box_id": box.box_id, "gt_cells": count, "P2_occupied_fraction": fractions, "picture": filename})
    report = {"schema": "viking-splatad/hidden-trunk-pilot/1", "session": baseline_info["SESSION"],
              "source_revision": root.name, "slurm_job_id": os.environ["SLURM_JOB_ID"], "alignment_reviewed": alignment_reviewed,
              "membership_sha256": split["membership_sha256"], "labels_sha256": digest(labels_path),
              "threshold": threshold, "voxel_size_m": voxel_size, "margin_m": margin, "scored_trunks": len(trunks),
              "arms": reports, "trunks": box_reports,
              "limitations": ["One matched pilot; no statistical repeat evidence.", "Blocked is geometric occlusion, not a grass semantic label.",
                              "Off-trunk neighbourhood occupancy is not an unmatched-tree false-positive count.",
                              "No camera visibility classification; camera loss is disabled in both arms.",
                              "Label/map alignment must be reviewed before interpreting the metrics."]}
    np.savez_compressed(arguments.out / "cells.npz", indices=indices, centres=centres, truth=truth, evidence=evidence,
                        B=probabilities["B"], L=probabilities["L"], B_P1=p1_masks["B"], L_P1=p1_masks["L"], threshold=threshold)
    (arguments.out / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    import html
    images = "\n".join(f'<p>GT box {html.escape(str(row["box_id"]))}</p><a href="{row["picture"]}"><img width="100%" src="{row["picture"]}"></a>' for row in box_reports)
    (arguments.out / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>June hidden trunk pilot</title><h1>Matched June B/L pilot</h1><p>All scored trunks, annotation order. Green: on-GT occupancy; red: off-GT occupancy. Review alignment before interpretation. Single pilot; no repeat evidence.</p>' + images)
    print(json.dumps(report), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
