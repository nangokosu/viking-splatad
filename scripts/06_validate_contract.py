"""Validate an intermediate dataset against the SplatAD input contract.

Usage:
    uv run python scripts/06_validate_contract.py --dataset <dir> [--report out.md]

Exit code 0 if every check passes, 1 otherwise.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from viking_splatad.contract.splatad_contract import (
    ContractReport,
    validate_dataset,
    validate_radar_inputs,
)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Validate a dataset, print the contract report and optionally save it.

    Args:
        argv: Command-line arguments (default: sys.argv[1:]).

    Returns:
        0 when the contract report passes, 1 otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="intermediate dataset directory")
    parser.add_argument("--report", type=Path, default=None, help="optional Markdown output path")
    parser.add_argument("--radar-inputs-dir", type=Path, default=None,
                        help="optional versioned radar seed artifact directory")
    parser.add_argument("--session", default=None,
                        help="session tag required when --radar-inputs-dir is set")
    arguments = parser.parse_args(argv)

    if (arguments.radar_inputs_dir is None) != (arguments.session is None):
        parser.error("--radar-inputs-dir and --session must be provided together")

    # Step 1: run all checks and print the Markdown report.
    report = validate_dataset(arguments.dataset)
    if arguments.radar_inputs_dir is not None:
        radar_report = validate_radar_inputs(
            arguments.radar_inputs_dir, arguments.session, arguments.dataset
        )
        report = ContractReport(report.results + radar_report.results)
    markdown = report.to_markdown()
    print(markdown)
    # Step 2: save it only when --report is given.
    if arguments.report is not None:
        arguments.report.parent.mkdir(parents=True, exist_ok=True)
        arguments.report.write_text(markdown)
    # Non-zero exit gates the hand-off to the GPU box.
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
