"""Read-back sanity report of a supersplat.ply: row count, finiteness, colour/opacity/scale percentiles.

Run with the project venv (JSON is written next to this script):
  uv run python docs/research/supersplat_export/ply_sanity.py
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import expit

from viking_splatad.export.spherical_harmonics import COLOUR_CHANNEL_COUNT, SH_C0, SH_COLOUR_OFFSET
from viking_splatad.export.supersplat_ply import read_supersplat_ply

DEFAULT_PLY_PATH = Path("runs/may_short_grass-may_scene_a-20261116-135432-17bb7a1/renders/supersplat.ply")
DEFAULT_OUTPUT_PATH = Path(__file__).with_name("ply_sanity_output.json")
# Percentiles reported (min, 1, quartiles, 99, max).
PERCENTILES = (0, 1, 25, 50, 75, 99, 100)
# Decimal places kept in the reported percentiles.
REPORT_DECIMALS = 4


def percentile_table(values: np.ndarray) -> dict:
    """Percentiles of a 1-D array as {percentile: rounded value}.

    Args:
        values: 1-D float array.

    Returns:
        Dict mapping each of PERCENTILES (as str) to its value rounded to REPORT_DECIMALS.
    """
    return {str(percentile): float(round(value, REPORT_DECIMALS))
            for percentile, value in zip(PERCENTILES, np.percentile(values, PERCENTILES))}


def main() -> None:
    """Read the PLY, compute the sanity statistics, print and write JSON."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ply", type=Path, default=DEFAULT_PLY_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    arguments = parser.parse_args()
    fields = read_supersplat_ply(arguments.ply)
    # Degree-0 view-independent colour = 0.5 + C0 * f_dc per channel.
    albedo_coefficients = np.stack([fields[f"f_dc_{channel}"] for channel in range(COLOUR_CHANNEL_COUNT)], axis=1)
    colours = SH_COLOUR_OFFSET + SH_C0 * albedo_coefficients
    # PLY stores the opacity logit and the log scale; invert with sigmoid (expit, stable for large |logit|)
    # and exp.
    opacity = expit(fields["opacity"].astype(np.float64))
    result = {
        "inputs": {"ply": arguments.ply.as_posix()},
        "row_count": int(len(fields["x"])),
        "field_count": len(fields),
        "all_finite": bool(all(np.isfinite(column).all() for column in fields.values())),
        "colour_degree0_percentiles": {name: percentile_table(colours[:, channel_index])
                                       for channel_index, name in enumerate("RGB")},
        "opacity_percentiles": percentile_table(opacity),
        "scale_0_metres_percentiles": percentile_table(np.exp(fields["scale_0"])),
        "position_range": {axis: [float(fields[axis].min()), float(fields[axis].max())] for axis in "xyz"},
    }
    print(f"rows {result['row_count']} fields {result['field_count']} all_finite {result['all_finite']}")
    arguments.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
