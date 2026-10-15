"""Write a lidar/camera alignment report for an extracted session.

Usage:
    uv run python scripts/04_align_report.py --config configs/sessions/<session>.yaml \
        [--frames N] [--out DIR]

Reads only poses.npz and the individual frames it needs (never the whole dataset). Default
output: reports/<session>/align/. Exit code 0 even for a WARN verdict; 1 if the dataset is
missing or invalid.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from viking_splatad.align.report import DEFAULT_FRAME_COUNT, generate_align_report
from viking_splatad.config import find_project_root, load_session_config


def main(argument_list: Optional[List[str]] = None) -> int:
    """Generate the report and print the Markdown.

    Args:
        argument_list: Command-line arguments (default: sys.argv[1:]).

    Returns:
        0 on success (PASS or WARN), 1 when the dataset is missing or invalid.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="session YAML")
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAME_COUNT,
                        help="number of evenly spaced lidar sweeps to examine")
    parser.add_argument("--out", type=Path, default=None, help="output directory (default reports/<session>/align)")
    arguments = parser.parse_args(argument_list)
    # Step 1: resolve the dataset and output directories from the session config.
    config = load_session_config(arguments.config)
    dataset_dir = config["session"]["output_dir"]
    output_dir = arguments.out or (find_project_root() / "reports" / config["session"]["name"] / "align")
    # Step 2: run; a missing/invalid dataset is the only failure mode.
    try:
        report = generate_align_report(
            dataset_dir, output_dir, arguments.frames,
            camera_name=config["camera"].get("name"), lidar_name=config["lidar"].get("name"))
    except (FileNotFoundError, ValueError, KeyError, OSError) as error:
        print(f"ERROR: dataset {dataset_dir} missing or invalid: {error!r}", file=sys.stderr)
        return 1
    print((output_dir / "report.md").read_text())
    print(f"verdict: {report['verdict']}; report written to {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
