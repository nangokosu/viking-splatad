"""Build a FiftyOne grouped dataset from one fetched SplatAD run, for read-only viewing.

Each test lidar sweep becomes one group with three slices:
    ``lidar``      a 3D scene (.fo3d) holding the rendered and the real point cloud in the MAP frame,
                   the GT cuboids as 3D labels and the sweep's per-scan metrics as fields;
    ``render_rgb`` the rendered camera image nearest in time;
    ``gt_rgb``     the real camera image of the same pose.
Points and cuboids share the map frame (T70 applies the sweep's ``lidar_to_map``; the GT boxes are
already map-frame). The dataparser frame is never used. GT cuboids are display labels only: they never
enter SplatAD inputs (CLAUDE.md rule 6). Nothing is written into the run's renders; scenes go to
``<output dir>`` (default ``<run dir>/fiftyone``).

Needs the isolated viewer environment (FiftyOne pins opencv<5, so it is not in the root lock):

    uv sync --project viz --locked
    uv run --project viz python scripts/14_build_fiftyone_dataset.py runs/<run_id> --session may_short_grass
    uv run --project viz python scripts/14_build_fiftyone_dataset.py runs/<run_id> --session may_short_grass --launch
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from viking_splatad.config import find_project_root, load_session_config
from viking_splatad.viz.viewer_export import (
    intensity_colors,
    quaternion_to_fiftyone_rotation,
    sweep_points_to_map,
    write_binary_pcd,
)

RENDERS_SPLIT_DIR = Path("renders") / "test"
OUTPUT_DIR_NAME = "fiftyone"
# Display-only choices (no effect on any metric): orange = rendered, cyan = real, sized to stay legible
# for a ~16k-point sweep in the 3D panel.
RENDERED_CLOUD_COLOR = "#ff7f0e"
REAL_CLOUD_COLOR = "#17becf"
POINT_SIZE_PIXELS = 2


def build_argument_parser() -> argparse.ArgumentParser:
    """Command-line interface.

    Returns:
        Parser with the run folder, session, output folder, dataset name and launch options.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, help="fetched run folder (runs/<run_id>)")
    parser.add_argument("--session", required=True, help="session config name (configs/sessions/<name>.yaml) whose GT cuboids to show")
    parser.add_argument("--output-dir", type=Path, default=None, help="where point clouds and scenes go (default <run dir>/fiftyone)")
    parser.add_argument("--dataset-name", default=None, help="FiftyOne dataset name (default: the run folder name)")
    parser.add_argument("--launch", action="store_true", help="open the FiftyOne App after building")
    return parser


def timestamp_ns_from_stem(file_stem: str) -> int:
    """Timestamp encoded in a render file name ``<index>_<timestamp_ns>``.

    Args:
        file_stem: Name without extension, e.g. ``000569_1714571368050209280``.

    Returns:
        Nanosecond timestamp.
    """
    return int(file_stem.split("_", 1)[1])


def nearest_image(lidar_timestamp_ns: int, image_paths: List[Path]) -> Path:
    """Camera image closest in time to a lidar sweep.

    Args:
        lidar_timestamp_ns: Sweep timestamp.
        image_paths: Candidate images named ``<index>_<timestamp_ns>.jpg``.

    Returns:
        The path minimising |image time - sweep time|.
    """
    return min(image_paths, key=lambda path: abs(timestamp_ns_from_stem(path.stem) - lidar_timestamp_ns))


def cuboids_to_detections(cuboid_records: List[Dict[str, Any]], fiftyone_module) -> list:
    """GT cuboid JSON records -> FiftyOne 3D detections in the map frame.

    Args:
        cuboid_records: Parsed ``cuboids.json`` (center_xyz, dimensions_xyz full sizes, quaternion_xyzw).
        fiftyone_module: The imported ``fiftyone`` module (kept lazy so tests need not install it).

    Returns:
        One ``fo.Detection`` per cuboid; rotation from T69.
    """
    return [
        fiftyone_module.Detection(
            label=record["class_name"],
            location=list(record["center_xyz"]),
            dimensions=list(record["dimensions_xyz"]),
            rotation=quaternion_to_fiftyone_rotation(record["quaternion_xyzw"]).tolist(),
            box_id=record["box_id"],
            class_id=record["class_id"],
        )
        for record in cuboid_records
    ]


def write_sweep_scene(sweep_npz_path: Path, output_dir: Path, fiftyone_module) -> Path:
    """Write one sweep's rendered and real clouds (map frame) and the .fo3d scene that holds them.

    Args:
        sweep_npz_path: ``renders/test/lidar/<index>_<timestamp_ns>.npz``.
        output_dir: Folder for the .pcd and .fo3d files.
        fiftyone_module: The imported ``fiftyone`` module.

    Returns:
        Path of the .fo3d scene file.
    """
    sweep = np.load(sweep_npz_path)
    lidar_to_map = sweep["lidar_to_map"]
    scene = fiftyone_module.Scene()
    # Columns of pred_points: x, y, z, intensity, t; gt_points adds a channel column.
    for cloud_name, points, color in (
        ("rendered", sweep["pred_points"], RENDERED_CLOUD_COLOR),
        ("real", sweep["gt_points"], REAL_CLOUD_COLOR),
    ):
        pcd_path = output_dir / f"{sweep_npz_path.stem}_{cloud_name}.pcd"
        write_binary_pcd(pcd_path, sweep_points_to_map(points[:, :3], lidar_to_map), intensity_colors(points[:, 3]))
        point_cloud = fiftyone_module.PointCloud(cloud_name, str(pcd_path))
        point_cloud.default_material.shading_mode = "custom"
        point_cloud.default_material.custom_color = color
        point_cloud.default_material.point_size = POINT_SIZE_PIXELS
        scene.add(point_cloud)
    scene_path = output_dir / f"{sweep_npz_path.stem}.fo3d"
    scene.write(str(scene_path))
    return scene_path


def build_dataset(run_dir: Path, session_name: str, output_dir: Path, dataset_name: str):
    """Create (or replace) the FiftyOne grouped dataset of one run.

    Args:
        run_dir: Fetched run folder.
        session_name: Session whose GT cuboids are attached.
        output_dir: Folder for generated point clouds and scenes.
        dataset_name: Name in the FiftyOne database; an existing dataset of that name is replaced.

    Returns:
        The persistent ``fiftyone.Dataset``.

    Raises:
        FileNotFoundError: If the run has no lidar renders or the session has no cuboids.json.
    """
    import fiftyone as fo  # lazy: only the isolated viewer environment has it

    # Step 1: inputs of the run and of the session's ground truth.
    render_dir = run_dir / RENDERS_SPLIT_DIR
    sweep_paths = sorted((render_dir / "lidar").glob("*.npz"))
    if not sweep_paths:
        raise FileNotFoundError(f"{render_dir / 'lidar'}: no lidar renders (fetch the run first)")
    session_config = load_session_config(find_project_root() / "configs" / "sessions" / f"{session_name}.yaml")
    cuboids_path = Path(session_config["session"]["gt_dir"]) / "cuboids.json"
    if not cuboids_path.exists():
        raise FileNotFoundError(f"{cuboids_path}: run scripts/05_extract_cuboids.py for this session")
    cuboid_detections = cuboids_to_detections(json.loads(cuboids_path.read_text()), fo)
    per_scan_metrics = json.loads((render_dir / "lidar_metrics.json").read_text()).get("per_scan", {})
    render_images = sorted((render_dir / "rgb").glob("*.jpg"))
    real_images = sorted((render_dir / "gt-rgb").glob("*.jpg"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 2: one group per sweep with lidar, rendered-camera and real-camera slices.
    samples = []
    for sweep_path in sweep_paths:
        sweep_timestamp_ns = timestamp_ns_from_stem(sweep_path.stem)
        group = fo.Group()
        scene_path = write_sweep_scene(sweep_path, output_dir, fo)
        lidar_sample = fo.Sample(
            filepath=str(scene_path),
            group=group.element("lidar"),
            sweep_index=int(np.load(sweep_path)["sweep_index"]),
            timestamp_ns=sweep_timestamp_ns,
            gt_cuboids=fo.Detections(detections=cuboid_detections),
        )
        # Per-scan metrics become fields so the App can sort and filter by them.
        for metric_name, metric_value in per_scan_metrics.get(sweep_path.stem, {}).items():
            lidar_sample[metric_name] = metric_value
        samples.append(lidar_sample)
        if render_images:
            samples.append(fo.Sample(filepath=str(nearest_image(sweep_timestamp_ns, render_images).resolve()), group=group.element("render_rgb"), timestamp_ns=sweep_timestamp_ns))
        if real_images:
            samples.append(fo.Sample(filepath=str(nearest_image(sweep_timestamp_ns, real_images).resolve()), group=group.element("gt_rgb"), timestamp_ns=sweep_timestamp_ns))

    # Step 3: replace any earlier dataset of this name, then store.
    if fo.dataset_exists(dataset_name):
        fo.delete_dataset(dataset_name)
    dataset = fo.Dataset(dataset_name, persistent=True)
    dataset.add_samples(samples)
    dataset.group_slice = "render_rgb" if render_images else "lidar"
    dataset.save()
    return dataset


def main(argument_list: Optional[List[str]] = None) -> int:
    """Entry point.

    Args:
        argument_list: Arguments (default ``sys.argv[1:]``).

    Returns:
        Process exit code.
    """
    arguments = build_argument_parser().parse_args(argument_list)
    run_dir = arguments.run_dir.resolve()
    dataset = build_dataset(
        run_dir,
        arguments.session,
        (arguments.output_dir or run_dir / OUTPUT_DIR_NAME).resolve(),
        arguments.dataset_name or run_dir.name,
    )
    print(f"dataset {dataset.name!r}: {len(dataset.distinct('group.id'))} groups, slices {list(dataset.group_slices)}")
    if arguments.launch:
        import fiftyone as fo

        fo.launch_app(dataset).wait()
    return 0


if __name__ == "__main__":
    sys.exit(main())
