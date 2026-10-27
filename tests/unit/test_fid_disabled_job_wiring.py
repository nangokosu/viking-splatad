"""The job turns off SplatAD's novel-view FIDs, which crash on scenes without dynamic actors.

SplatAD computes the FIDs at a step in `pipeline.calc_fid_steps` that is also an all-images eval
step. Its method config uses (30000,), the last step of a 30001-iteration run, and the actor-edit
FID raises IndexError when the scene has no actors (jobs 222951, 222952).
"""

import re

import yaml

from tests.unit.test_block_split import SBATCH_PATH
from tests.unit.test_slurm_submit import PROJECT_ROOT

FID_FLAG = "--pipeline.calc-fid-steps"


def _ns_train_command(sbatch_text: str) -> str:
    """Return the ns-train invocation, joined across its backslash continuations.

    :param sbatch_text: full text of splatad_run.sbatch.
    :return: the command from `ns-train splatad` up to its first line without a trailing backslash.
    """
    command_lines = []
    # Step 1: start at the ns-train line and collect lines while they continue with a backslash.
    for line in sbatch_text[sbatch_text.index("\nns-train splatad") + 1 :].splitlines():
        command_lines.append(line.rstrip("\\").strip())
        if not line.rstrip().endswith("\\"):
            break
    return " ".join(command_lines)


def _fid_steps(command: str) -> list[int]:
    """Parse the integer values passed to the FID flag (tyro takes a space-separated tuple).

    :param command: the joined ns-train command.
    :return: the FID steps, in order.
    """
    match = re.search(re.escape(FID_FLAG) + r"((?:\s+\d+)+)", command)
    assert match, f"{FID_FLAG} missing from ns-train"
    return [int(value) for value in match.group(1).split()]


def test_fid_flag_is_a_trainer_level_argument():
    """The flag sits before the viking-data subcommand, where tyro reads pipeline arguments."""
    command = _ns_train_command(SBATCH_PATH.read_text())
    assert command.index(FID_FLAG) < command.index(" viking-data ")


def test_no_fid_step_is_reached_by_any_profile():
    """No FID step is a training step, so none is an eval step either, for every profile.

    The trainer runs steps 0..N-1 (range(max_num_iterations)), and the all-images eval runs at the
    multiples of k = (N - 1) // minimum_eval_points (T63) in 1..N-1. An FID step >= N is never reached.
    """
    fid_steps = _fid_steps(_ns_train_command(SBATCH_PATH.read_text()))
    profiles = yaml.safe_load((PROJECT_ROOT / "configs" / "slurm.yaml").read_text())["profiles"]
    for profile_name, profile in profiles.items():
        iteration_count = profile["max_num_iterations"]
        eval_interval = (iteration_count - 1) // profile["minimum_eval_points"]
        # Step 1: every eval step of this profile, from first principles.
        eval_steps = set(range(eval_interval, iteration_count, eval_interval))
        # Step 2: no FID step is a training step, hence none is an eval step.
        for fid_step in fid_steps:
            assert fid_step >= iteration_count, (profile_name, fid_step)
            assert fid_step not in eval_steps, (profile_name, fid_step)
