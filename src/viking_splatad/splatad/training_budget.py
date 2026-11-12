"""Measured Slurm walltime budgets; these never change iterations, inputs or model settings."""

import math


def measured_training_budgets(baseline_wall_seconds: float, gate_seconds: float,
                               index_seconds: float, profile_iterations: dict,
                               headroom_factor: float, maximum_seconds: int) -> dict:
    """Return bounded, whole-minute walltimes from measured B overhead and radar gate cost.

    Args:
        baseline_wall_seconds: Successful full B walltime, including loading/eval/render.
        gate_seconds: Worst measured full-scene GPU forward gate pass.
        index_seconds: Measured one-time anchor-index construction.
        profile_iterations: Profile -> positive iteration budget.
        headroom_factor: Configured allocation headroom, at least one.
        maximum_seconds: Cluster's actual maximum schedulable job time.

    Returns:
        Profile -> requested Slurm time string, seconds and measured prediction.
        Reusing the full B walltime even for smoke is deliberately conservative.
    """
    values = (baseline_wall_seconds, gate_seconds, index_seconds, headroom_factor)
    if (not all(math.isfinite(value) for value in values) or baseline_wall_seconds <= 0
            or gate_seconds <= 0 or index_seconds < 0 or headroom_factor < 1 or maximum_seconds <= 0):
        raise ValueError("finite positive measured times and valid allocation headroom required")
    result = {}
    for profile, iterations in profile_iterations.items():
        if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations <= 0:
            raise ValueError("iteration budgets must be positive integers")
        predicted = baseline_wall_seconds + index_seconds + gate_seconds * iterations
        # Slurm time limits have whole-minute resolution for these job requests.
        seconds = math.ceil(predicted * headroom_factor / 60) * 60
        if seconds > maximum_seconds:
            raise ValueError("measured radar training budget exceeds the cluster maximum; keep jobs held")
        days, remaining = divmod(seconds, 24 * 60 * 60)
        hours, remaining = divmod(remaining, 60 * 60)
        minutes = remaining // 60
        result[profile] = {"time_limit": f"{days}-{hours:02d}:{minutes:02d}:00", "seconds": seconds,
                           "predicted_seconds": predicted, "iterations": iterations}
    return result
