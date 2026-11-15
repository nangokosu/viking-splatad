"""Validate an intermediate dataset against the SplatAD input contract.

Usage:
    uv run python scripts/06_validate_contract.py --dataset <dir> [--report out.md]
        [--radar-inputs-dir <dir> --session <s> --profile <p> [--require-seed-colours]]

Exit code 0 if every check passes, 1 otherwise.
"""

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Optional, Sequence

from viking_splatad.contract.splatad_contract import (
    ContractReport,
    validate_dataset,
    validate_radar_inputs,
)


def load_split_settings(code_root: Path, dataset: Path, session: str, profile: str) -> dict:
    """Resolve the Slurm profile's exact split settings via the submission helper.

    Args:
        code_root: Repository root holding scripts/slurm/submit_run.py.
        dataset: Intermediate dataset directory.
        session: Session tag; profile: Slurm profile name.

    Returns:
        Split-settings kwargs identical to what the training job will use.

    Raises:
        RuntimeError: If submit_run.py lacks `profile_radar_split_settings`.
    """
    # Step 1: import submit_run.py by file path (scripts/ is not a package).
    specification = importlib.util.spec_from_file_location(
        "submit_helpers", code_root / "scripts/slurm/submit_run.py")
    helpers = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(helpers)
    if not hasattr(helpers, "profile_radar_split_settings"):
        raise RuntimeError("scripts/slurm/submit_run.py has no profile_radar_split_settings; "
                           "it is required to check radar inputs against the profile split")
    # Step 2: the function returns (session_config, split_settings); only the latter is needed.
    _, split_settings = helpers.profile_radar_split_settings(code_root, dataset, session, profile)
    return split_settings


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
    parser.add_argument("--profile", default=None,
                        help="Slurm profile whose split the artifact must match (with --radar-inputs-dir)")
    parser.add_argument("--require-seed-colours", action="store_true",
                        help="fail unless the radar artifact carries painted seed colours")
    arguments = parser.parse_args(argv)

    # The three radar selectors are only meaningful together.
    radar_selectors = (arguments.radar_inputs_dir, arguments.session, arguments.profile)
    if any(value is None for value in radar_selectors) and any(value is not None for value in radar_selectors):
        parser.error("--radar-inputs-dir, --session and --profile must be provided together")
    if arguments.require_seed_colours and arguments.radar_inputs_dir is None:
        parser.error("--require-seed-colours needs --radar-inputs-dir")

    # Step 1: run all checks and print the Markdown report.
    report = validate_dataset(arguments.dataset)
    if arguments.radar_inputs_dir is not None:
        code_root = Path(__file__).resolve().parent.parent
        split_settings = load_split_settings(code_root, arguments.dataset, arguments.session, arguments.profile)
        radar_report = validate_radar_inputs(
            arguments.radar_inputs_dir, arguments.session, arguments.dataset, split_settings,
            arguments.require_seed_colours,
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
