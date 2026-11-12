"""Summarize at least two independent B/L runs against their within-arm variation."""

import argparse
import json
from pathlib import Path

import numpy as np

from viking_splatad.eval.repeated_hidden import summarize_comparisons, validate_alignment_review
from viking_splatad.viz.repeated_hidden import write_consistency_map


def main(argument_list=None):
    """Load script 17 products, validate shared cells and write repeat report plus maps.

    Args:
        argument_list: CLI options; report directories each contain comparison.json/cells.npz.
    Returns:
        Zero on a valid descriptive summary; refuses a single pair or reused runs.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparisons", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--alignment-review", type=Path, help="Separate completed review bound to the exact label and dataset hashes.")
    args = parser.parse_args(argument_list)
    reports = [json.loads((directory / "comparison.json").read_text()) for directory in args.comparisons]
    arrays = []
    for directory in args.comparisons:
        with np.load(directory / "cells.npz", allow_pickle=False) as archive:
            arrays.append({key: archive[key] for key in archive.files})
    summary, views = summarize_comparisons(reports, arrays)
    if args.alignment_review:
        review = json.loads(args.alignment_review.read_text())
        alignments = [json.loads((directory / "alignment.json").read_text()) for directory in args.comparisons]
        review_hash = validate_alignment_review(review, alignments)
        summary["all_alignment_reviews_complete"] = True
        summary["alignment_review"] = {"receipt_sha256": review_hash, "scope": review["review_scope"],
                                        "policy": "separate post-evaluation review; original numerical products unchanged"}
    args.out.mkdir(parents=True, exist_ok=True)
    for representation, view in views.items():
        write_consistency_map(args.out / f"{representation}-consistency.png", arrays[0]["centres"], arrays[0]["truth"],
                              arrays[0]["evidence"], view["B"], view["L"])
    (args.out / "repeats.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    lines = ["# Matched hidden-trunk repeats", "", f"Runs per arm: {summary['repeats_per_arm']}. Descriptive statistics only.",
             f"All label/frame reviews complete: {summary['all_alignment_reviews_complete']}.", "", summary["caution"], "",
             "| View / metric | B mean [min, max] | L mean [min, max] | Mean L minus B |", "|---|---|---|---|"]
    for representation, metrics in summary["metrics"].items():
        for name, metric in metrics.items():
            baseline, radar, delta = metric["B"], metric["L"], metric["paired_L_minus_B"]
            lines.append(f"| {representation} / {name} | {baseline['mean']} [{baseline['minimum']}, {baseline['maximum']}] | "
                         f"{radar['mean']} [{radar['minimum']}, {radar['maximum']}] | {delta['mean']} |")
    lines += ["", "![P1 consistency](P1-consistency.png)", "", "![P2 consistency](P2-consistency.png)", ""]
    (args.out / "repeats.md").write_text("\n".join(lines))
    print(json.dumps({"out": str(args.out), "repeats_per_arm": summary["repeats_per_arm"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
