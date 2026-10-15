"""Validate an intermediate dataset against the SplatAD input contract.

Usage:
    uv run python scripts/06_validate_contract.py --dataset <dir> [--report out.md]

Exit code 0 if every check passes, 1 otherwise.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from viking_splatad.contract.splatad_contract import validate_dataset


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point.

    Args:
        argv: Argument list (default sys.argv[1:]).

    Returns:
        0 when the contract report passes, 1 otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="intermediate dataset directory")
    parser.add_argument("--report", type=Path, default=None, help="optional Markdown output path")
    arguments = parser.parse_args(argv)

    # Run all checks and render the report.
    report = validate_dataset(arguments.dataset)
    markdown = report.to_markdown()
    print(markdown)
    if arguments.report is not None:
        arguments.report.parent.mkdir(parents=True, exist_ok=True)
        arguments.report.write_text(markdown)
    # Non-zero exit gates the hand-off to the GPU box.
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
