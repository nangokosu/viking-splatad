"""Extract ROS2 MCAP bag chunks into the intermediate dataset and validate the SplatAD contract.

Usage:
    uv run python scripts/03_extract.py --config configs/sessions/may_short_grass.yaml \
        [--max-seconds N] [--chunks file1.mcap ...] [--force]

Re-running after downloading another chunk adds only the new chunk's frames.
Exit status is non-zero when any contract check fails.
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

from viking_splatad.config import load_session_config
from viking_splatad.contract.splatad_contract import validate_dataset
from viking_splatad.pipeline.bag_to_dataset import run_pipeline


def main(argument_list: Optional[List[str]] = None) -> int:
    """Run extraction, validation and print the report.

    Args:
        argument_list: Command-line arguments (default: sys.argv[1:]).

    Returns:
        0 when every contract check passes, 1 otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="session YAML")
    parser.add_argument("--max-seconds", type=float, default=None,
                        help="only extract the first N seconds of the session (debug runs)")
    parser.add_argument("--chunks", nargs="+", default=None, help="chunk file names overriding the config")
    parser.add_argument("--force", action="store_true", help="discard previous outputs and re-extract")
    arguments = parser.parse_args(argument_list)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    # Step 1: load config and run the incremental pipeline.
    config = load_session_config(arguments.config)
    start_time = time.perf_counter()
    result = run_pipeline(config, arguments.chunks, arguments.max_seconds, arguments.force)
    elapsed_seconds = time.perf_counter() - start_time
    print(f"processed chunks: {result.processed_chunks}; skipped (already extracted): {result.skipped_chunks}")
    print(f"frames written this run: {result.frames_written}; total: {result.frames_total}")
    print(f"dropped frames: {result.skipped_counts}; extraction time {elapsed_seconds:.1f} s")
    for warning in result.warnings:
        print(f"WARNING: {warning}")
    # Step 2: validate the dataset on disk and print/save the markdown report.
    report = validate_dataset(result.output_dir)
    markdown = report.to_markdown()
    print(markdown)
    (result.output_dir / "contract_report.md").write_text(markdown)
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
