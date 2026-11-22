"""Degree-3 SH fit residual vs direction oversampling factor (justifies DIRECTION_OVERSAMPLING_FACTOR).

Colour functions are random SH expansions up to degree 4 (one band above the fit, so the fit
cannot represent all of it), with coefficient std decaying as DECAY_PER_DEGREE**degree (smooth).
Each is sampled at M = factor * MAX_SH_BASIS_SIZE Fibonacci directions, fitted with the repo's
`sh_fit_operator`, and scored as RMS error over a dense held-out Fibonacci direction set.

Run in the reference image (writes the JSON next to this script):
  VIKING_WRITABLE_DIR=docs/research/supersplat_export scripts/test_in_image.sh \
      python docs/research/supersplat_export/sh_oversampling_residual.py
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from gsplat.cuda._torch_impl import _eval_sh_bases_fast

from viking_splatad.export.spherical_harmonics import (
    MAX_SH_BASIS_SIZE, MAX_SH_DEGREE, fibonacci_sphere_directions, sh_fit_operator,
)

# Fixed seed so the output is deterministic (any value works; 20260121 is arbitrary, it is not a date
# of writing).
RANDOM_SEED = 20260121
# Random colour functions averaged per oversampling factor (more = smaller Monte-Carlo noise).
FUNCTION_COUNT = 200
# Source functions have one band more than the fit, so the residual floor is the degree-4 energy.
SOURCE_DEGREE = MAX_SH_DEGREE + 1
# Smoothness: coefficient std is DECAY_PER_DEGREE**degree (halves per degree).
DECAY_PER_DEGREE = 0.5
# Dense held-out direction set on which the residual is measured.
EVALUATION_DIRECTION_COUNT = 20000
# Oversampling factors tested (directions per basis function).
OVERSAMPLING_FACTORS = [1, 2, 4, 8, 16, 32, 64, 256]
DEFAULT_OUTPUT_PATH = Path(__file__).with_name("sh_oversampling_residual_output.json")


def source_basis(directions: np.ndarray) -> np.ndarray:
    """Evaluate the degree-SOURCE_DEGREE SH basis through gsplat (float64).

    Args:
        directions: (M, 3) unit vectors.
    Returns:
        (M, (SOURCE_DEGREE + 1)**2) basis matrix.
    """
    unit_vectors = torch.from_numpy(directions)
    return _eval_sh_bases_fast((SOURCE_DEGREE + 1) ** 2, unit_vectors).numpy()


def run_experiment() -> dict:
    """Measure the held-out RMS residual of the degree-MAX_SH_DEGREE fit per oversampling factor.

    Residual = RMS(truth - fit) over held-out directions and functions; "relative" divides by the
    RMS of the truth signal. Cost of the fit is O(N * M * K) with K = MAX_SH_BASIS_SIZE.

    Returns:
        JSON-serialisable dict with the settings and one row per factor.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    source_size = (SOURCE_DEGREE + 1) ** 2
    # SH index k belongs to degree floor(sqrt(k)) (degree d owns indices d^2 .. (d+1)^2 - 1).
    coefficient_degrees = np.floor(np.sqrt(np.arange(source_size))).astype(int)
    coefficient_std = DECAY_PER_DEGREE ** coefficient_degrees
    true_coefficients = rng.normal(size=(FUNCTION_COUNT, source_size)) * coefficient_std
    # Held-out truth, and the fitted-degree columns of the same basis for evaluating fits.
    evaluation_directions = fibonacci_sphere_directions(EVALUATION_DIRECTION_COUNT)
    evaluation_full_basis = source_basis(evaluation_directions)
    evaluation_truth = true_coefficients @ evaluation_full_basis.T
    evaluation_fit_basis = evaluation_full_basis[:, :MAX_SH_BASIS_SIZE]
    signal_rms = float(np.sqrt((evaluation_truth ** 2).mean()))
    rows = []
    for oversampling_factor in OVERSAMPLING_FACTORS:
        direction_count = oversampling_factor * MAX_SH_BASIS_SIZE
        directions = fibonacci_sphere_directions(direction_count)
        # Noise-free samples of every function at the M fit directions, then the least-squares fit.
        samples = true_coefficients @ source_basis(directions).T
        fitted_coefficients = samples @ sh_fit_operator(directions, MAX_SH_DEGREE).T
        residual = evaluation_truth - fitted_coefficients @ evaluation_fit_basis.T
        rms_residual = float(np.sqrt((residual ** 2).mean()))
        rows.append({
            "oversampling_factor": oversampling_factor,
            "direction_count": direction_count,
            "rms_residual": rms_residual,
            "relative_residual": rms_residual / signal_rms,
        })
    return {
        "settings": {
            "random_seed": RANDOM_SEED, "function_count": FUNCTION_COUNT,
            "source_degree": SOURCE_DEGREE, "fit_degree": MAX_SH_DEGREE,
            "decay_per_degree": DECAY_PER_DEGREE,
            "evaluation_direction_count": EVALUATION_DIRECTION_COUNT,
        },
        "signal_rms": signal_rms,
        "rows": rows,
    }


def main() -> None:
    """Run the experiment, print the table and write the JSON."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="output JSON path")
    arguments = parser.parse_args()
    result = run_experiment()
    for row in result["rows"]:
        print(f"factor {row['oversampling_factor']:4d}  M {row['direction_count']:5d}  "
              f"relative residual {row['relative_residual']:.6f}")
    arguments.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
