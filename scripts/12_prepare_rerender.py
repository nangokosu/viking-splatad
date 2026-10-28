"""Prepare render configs for a run's checkpoint, and check render-only jobs (render_only.sbatch).

A render-only job renders guard frames (T62: inside a T60 test block, outside its core; never trained
on, never rendered by the run) from the run's checkpoint. Neither subcommand that writes a config
ever edits the run's own config.yml:

    write-config      copies the run's config.yml into the rerender folder with the dataparser's
                      ``render_frame_selection_file`` set to the job's selection.json, so the ``test``
                      split that ns-render and scripts/08 ask for is exactly the selected guard frames
                      (T66), and with point painting off (below).
    write-run-config  copies the run's config.yml with point painting off only, for the run's own
                      ns-render step (splatad_run.sbatch).
    check             compares the rendered file names with selection.json: renders/test/rgb/*.jpg and
                      renders/test/gt-rgb/*.jpg against the camera files, renders/test/lidar/*.npz
                      against the lidar files.

In both copies ``output_dir``, ``experiment_name``, ``method_name`` and ``timestamp`` still point at
the run's train folder, so the run's latest checkpoint is the one loaded.

Point painting off: SplatAD's FullImageLidarDatamanager calls ``paint_points()`` in its constructor
when ``paint_points`` is set. That loads every training image and sweep into RAM (``cached_train``,
``cached_lidar_train``) to colour the seed points. ``ns-render dataset`` builds the data manager
twice (once in ``eval_setup``, once per rendered split; render.py:935, 954-962), and its
``_disable_datamanager_setup`` stubs only setup_train/setup_eval, not the constructor, so the
training set was held twice and job 222994 (May Scene A) was OOM-killed. Rendering does not need
the colours: they only seed the Gaussians, and the checkpoint replaces every Gaussian parameter
(SplatADModel.load_state_dict resizes to the checkpoint's count, then loads), so the model rendered
is the same.

Usage (inside the job, in the cluster environment, which can unpickle nerfstudio configs):
    python scripts/12_prepare_rerender.py write-config --train-config <run>/train/<session>/splatad/<run_id>/config.yml \
        --selection <rerender dir>/selection.json --output <rerender dir>/config.yml
    python scripts/12_prepare_rerender.py write-run-config --train-config <run>/train/<session>/splatad/<run_id>/config.yml \
        --output <run>/render_config/config.yml
    python scripts/12_prepare_rerender.py check --rerender-dir <rerender dir>

Exit codes of ``check``: 0 the renders equal the selection, 1 they differ, 2 outputs (or
selection.json) missing. ``write-config`` exits 0, or 1 when the run's config has no guard frames.
``write-run-config`` exits 0.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

# Name of the selection file in a rerender folder (submit_run.py SELECTION_FILE_NAME).
SELECTION_FILE_NAME = "selection.json"
# The split ns-render and scripts/08 write for --pose-source test (renders/<split>/...).
RENDERED_SPLIT_NAME = "test"
# Render folders (under renders/test/) and file suffix per sensor kind: ns-render writes rgb and
# gt-rgb jpg (--rendered-output-names rgb gt-rgb --image-format jpeg), scripts/08 writes lidar npz.
RENDER_FOLDERS_BY_KIND = {"camera": (("rgb", ".jpg"), ("gt-rgb", ".jpg")), "lidar": (("lidar", ".npz"),)}
# T62 core fraction meaning "the whole block is test" (dataset/run_split.py WHOLE_BLOCK_CORE_FRACTION;
# repeated so this script needs no package import): such a run has no guard frames to render.
WHOLE_BLOCK_CORE_FRACTION = 1.0
# check exit codes (module docstring).
EXIT_MATCH = 0
EXIT_MISMATCH = 1
EXIT_MISSING = 2
# write-config exit code for a run config without guard frames.
EXIT_NO_GUARD = 1


def set_render_frame_selection(trainer_config: Any, selection_path: Path) -> Any:
    """Point the run config's dataparser at a selection file (T66 in VikingHillDataParser).

    Args:
        trainer_config: The run's loaded config.yml (nerfstudio TrainerConfig or any object with
            ``pipeline.datamanager.dataparser``).
        selection_path: Absolute path of the job's selection.json.

    Returns:
        The same config object, with ``render_frame_selection_file`` set.

    Raises:
        ValueError: If the dataparser has no test blocks or a core fraction of 1 (no guard frames).
    """
    dataparser_config = trainer_config.pipeline.datamanager.dataparser
    # Step 1: refuse a run without guard frames here, before the job loads a model (getattr: configs
    # saved before T60/T62 lack the fields; their class defaults are None and 1.0).
    test_block_count = getattr(dataparser_config, "test_block_count", None)
    core_fraction = getattr(dataparser_config, "test_block_core_fraction", WHOLE_BLOCK_CORE_FRACTION)
    if test_block_count is None or core_fraction == WHOLE_BLOCK_CORE_FRACTION:
        raise ValueError(f"run config has no guard frames (test_block_count {test_block_count}, "
                         f"test_block_core_fraction {core_fraction}); nothing to re-render")
    # Step 2: the one edit; every other field (output_dir, timestamp, ...) still names the run.
    dataparser_config.render_frame_selection_file = Path(selection_path)
    return trainer_config


def disable_point_painting(trainer_config: Any) -> Any:
    """Turn off the data manager's point painting for a render (module docstring).

    Args:
        trainer_config: A loaded config.yml (nerfstudio TrainerConfig or any object with
            ``pipeline.datamanager``).

    Returns:
        The same config object, with ``paint_points`` False when the data manager has the field
        (data managers without it never paint, so nothing changes for them).
    """
    datamanager_config = trainer_config.pipeline.datamanager
    if hasattr(datamanager_config, "paint_points"):
        datamanager_config.paint_points = False
    return trainer_config


def write_render_config(train_config_path: Path, selection_path: Path, output_path: Path) -> None:
    """Copy a run's config.yml with the render selection set and point painting off.

    Loads and dumps the way nerfstudio does (eval_setup: ``yaml.load(..., Loader=yaml.Loader)``;
    save_config: ``yaml.dump(config)``), so ns-render and scripts/08 read the copy like the original.

    Args:
        train_config_path: The run's config.yml.
        selection_path: The job's selection.json (stored as an absolute path).
        output_path: Where to write the copy (the rerender folder's config.yml).

    Raises:
        ValueError: As ``set_render_frame_selection``.
    """
    trainer_config = yaml.load(train_config_path.read_text(), Loader=yaml.Loader)
    trainer_config = set_render_frame_selection(trainer_config, selection_path.resolve())
    trainer_config = disable_point_painting(trainer_config)
    output_path.write_text(yaml.dump(trainer_config), "utf8")


def write_run_render_config(train_config_path: Path, output_path: Path) -> None:
    """Copy a run's config.yml with point painting off, for the run's own ns-render step.

    Same load/dump as ``write_render_config``; the split is untouched, so ns-render renders the
    run's own test frames.

    Args:
        train_config_path: The run's config.yml.
        output_path: Where to write the copy (its folder is created if missing).
    """
    trainer_config = disable_point_painting(yaml.load(train_config_path.read_text(), Loader=yaml.Loader))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.dump(trainer_config), "utf8")


def compare_rendered_with_selection(
    selected_files_by_kind: Dict[str, List[str]], rendered_stems_by_folder: Dict[Tuple[str, str], Optional[List[str]]]
) -> Tuple[int, List[str]]:
    """Check the rendered file names against the selection (pure: no I/O).

    Args:
        selected_files_by_kind: selection.json ``files``: kind -> dataset-relative frame paths.
        rendered_stems_by_folder: (kind, folder) -> stems found in renders/test/<folder>/, or None
            when that folder does not exist.

    Returns:
        (exit code, reasons): EXIT_MATCH and [] when every folder holds exactly the selected stems;
        EXIT_MISSING when a folder is absent (the step did not run); else EXIT_MISMATCH. Each reason
        names the folder and the missing / unexpected stems.
    """
    reasons, exit_code = [], EXIT_MATCH
    for sensor_kind, render_folders in RENDER_FOLDERS_BY_KIND.items():
        # Step 1: expected stems = the selected frames' file stems (ns-render and 08 keep them).
        expected_stems = {Path(frame_file).stem for frame_file in selected_files_by_kind.get(sensor_kind, [])}
        for folder_name, _ in render_folders:
            found_stems = rendered_stems_by_folder.get((sensor_kind, folder_name))
            # Step 2: an absent folder means the render step never wrote it.
            if found_stems is None:
                reasons.append(f"renders/{RENDERED_SPLIT_NAME}/{folder_name}/ is missing")
                exit_code = EXIT_MISSING
                continue
            # Step 3: set equality, both directions reported.
            missing_stems = sorted(expected_stems - set(found_stems))
            unexpected_stems = sorted(set(found_stems) - expected_stems)
            if missing_stems or unexpected_stems:
                reasons.append(f"renders/{RENDERED_SPLIT_NAME}/{folder_name}/: missing {missing_stems}, "
                               f"unexpected {unexpected_stems}")
                if exit_code == EXIT_MATCH:
                    exit_code = EXIT_MISMATCH
    return exit_code, reasons


def list_rendered_stems(rerender_dir: Path) -> Dict[Tuple[str, str], Optional[List[str]]]:
    """Stems of the files in each render folder of a rerender folder.

    Args:
        rerender_dir: ``<run dir>/rerenders/<label>/``.

    Returns:
        (kind, folder) -> sorted stems, or None when the folder does not exist.
    """
    rendered_stems: Dict[Tuple[str, str], Optional[List[str]]] = {}
    for sensor_kind, render_folders in RENDER_FOLDERS_BY_KIND.items():
        for folder_name, file_suffix in render_folders:
            render_folder = rerender_dir / "renders" / RENDERED_SPLIT_NAME / folder_name
            rendered_stems[(sensor_kind, folder_name)] = (
                sorted(rendered_path.stem for rendered_path in render_folder.glob(f"*{file_suffix}"))
                if render_folder.is_dir() else None)
    return rendered_stems


def check_rerender(rerender_dir: Path) -> int:
    """Compare a rerender folder's outputs with its selection.json and print the verdict.

    Args:
        rerender_dir: ``<run dir>/rerenders/<label>/``.

    Returns:
        EXIT_MATCH, EXIT_MISMATCH or EXIT_MISSING.
    """
    selection_path = rerender_dir / SELECTION_FILE_NAME
    if not selection_path.is_file():
        print(f"render check: {selection_path} is missing", file=sys.stderr)
        return EXIT_MISSING
    selected_files_by_kind = json.loads(selection_path.read_text())["files"]
    exit_code, reasons = compare_rendered_with_selection(selected_files_by_kind, list_rendered_stems(rerender_dir))
    for reason in reasons:
        print(f"render check: {reason}", file=sys.stderr)
    if exit_code == EXIT_MATCH:
        frame_counts = ", ".join(f"{len(frame_files)} {sensor_kind}"
                                 for sensor_kind, frame_files in selected_files_by_kind.items())
        print(f"render check: match ({frame_counts})")
    return exit_code


def main(argument_list: Optional[Sequence[str]] = None) -> int:
    """CLI entry point.

    Args:
        argument_list: Argument list (default sys.argv[1:]).

    Returns:
        The subcommand's exit code (module docstring).
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)
    write_parser = subparsers.add_parser("write-config", help="copy the run's config.yml with the selection set")
    write_parser.add_argument("--train-config", type=Path, required=True, help="the run's config.yml")
    write_parser.add_argument("--selection", type=Path, required=True, help="the job's selection.json")
    write_parser.add_argument("--output", type=Path, required=True, help="config.yml to write in the rerender dir")
    run_config_parser = subparsers.add_parser("write-run-config",
                                              help="copy the run's config.yml with point painting off")
    run_config_parser.add_argument("--train-config", type=Path, required=True, help="the run's config.yml")
    run_config_parser.add_argument("--output", type=Path, required=True, help="config.yml to write")
    check_parser = subparsers.add_parser("check", help="rendered file names vs selection.json")
    check_parser.add_argument("--rerender-dir", type=Path, required=True, help="<run dir>/rerenders/<label>/")
    arguments = parser.parse_args(argument_list)
    if arguments.command == "write-config":
        try:
            write_render_config(arguments.train_config, arguments.selection, arguments.output)
        except ValueError as error:
            print(f"ERROR: {error}", file=sys.stderr)
            return EXIT_NO_GUARD
        print(f"wrote {arguments.output} (render_frame_selection_file = {arguments.selection.resolve()})")
        return 0
    if arguments.command == "write-run-config":
        write_run_render_config(arguments.train_config, arguments.output)
        print(f"wrote {arguments.output} (paint_points = False)")
        return 0
    return check_rerender(arguments.rerender_dir)


if __name__ == "__main__":
    sys.exit(main())
