"""Static checks of job step (e4) in scripts/slurm/splatad_run.sbatch: the SuperSplat PLY export.

Script 22 crops to the train-lidar box, which needs split.json and split_frames.csv from script 09. So
(e4) lives inside the "split check matched" branch of step (f), after 09 and before the end_utc line. It
writes its timing and status keys to job_info.txt, runs under `timeout`, and never aborts the job (the
PLY is a viewer product, not an evaluation input).
"""

import subprocess
import textwrap
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RUN_SCRIPT_PATH = REPOSITORY_ROOT / "scripts" / "slurm" / "splatad_run.sbatch"
GAUSSIAN_EXPORT_SCRIPT = "scripts/10_export_gaussians.py"
SUPERSPLAT_EXPORT_SCRIPT = "scripts/22_export_supersplat_ply.py"
SPLIT_RECORD_SCRIPT = "scripts/09_extract_run_split.py"
END_UTC_LINE = 'echo "end_utc:'
TIMING_KEYS = ("supersplat_export_start_epoch_s", "supersplat_export_end_epoch_s", "supersplat_export_elapsed_s")
STATUS_KEY = "supersplat_export:"
SPLIT_IF_PREFIX = "if python"  # the split check opens with this, on the line that calls script 09
ELSE_LINE = "\nelse\n"  # the split check's else, at column 0 (inner if/else are indented)


def read_run_script() -> str:
    """Read the job script.

    Returns:
        The full text of splatad_run.sbatch.
    """
    return RUN_SCRIPT_PATH.read_text()


def split_check_branches(script_text: str) -> tuple:
    """Return the (matched, failed) branch texts of step (f)'s split check.

    Args:
        script_text: Full text of splatad_run.sbatch.

    Returns:
        ``(matched_text, failed_text)``: the text between the split check's ``then`` and its
        column-0 ``else``, and between that ``else`` and the column-0 ``fi``.
    """
    if_start = script_text.index(f'{SPLIT_IF_PREFIX} "$CODE_DIR/{SPLIT_RECORD_SCRIPT}"')
    then_end = script_text.index("; then\n", if_start) + len("; then\n")
    else_start = script_text.index(ELSE_LINE, then_end)
    fi_start = script_text.index("\nfi\n", else_start)
    return script_text[then_end:else_start], script_text[else_start + len(ELSE_LINE):fi_start]


def supersplat_step_text(script_text: str) -> str:
    """Return the (e4) block: the split check's matched branch, from the (e4) header onward.

    Args:
        script_text: Full text of splatad_run.sbatch.

    Returns:
        The slice of the matched branch starting at "# ---- (e4)".
    """
    matched_text, _ = split_check_branches(script_text)
    return matched_text[matched_text.index("# ---- (e4)"):]


def command_lines(step_text: str) -> str:
    """Drop the comment lines of a script slice, so assertions see only commands.

    Args:
        step_text: Slice of splatad_run.sbatch.

    Returns:
        ``step_text`` without the lines whose first non-blank character is ``#``.
    """
    return "\n".join(line for line in step_text.splitlines() if not line.lstrip().startswith("#"))


def test_supersplat_step_runs_after_split_record_and_before_end_utc():
    """Script 22 is called once: after script 10 and script 09, before the end_utc line."""
    script_text = read_run_script()
    assert script_text.count(SUPERSPLAT_EXPORT_SCRIPT) == 1
    gaussian_position = script_text.index(GAUSSIAN_EXPORT_SCRIPT)
    split_position = script_text.index(SPLIT_RECORD_SCRIPT)
    supersplat_position = script_text.index(SUPERSPLAT_EXPORT_SCRIPT)
    end_utc_position = script_text.index(END_UTC_LINE)
    assert gaussian_position < split_position < supersplat_position < end_utc_position


def test_supersplat_step_is_inside_split_matched_branch_only():
    """Script 22 sits in the matched branch; the failed branch only records a skip and exits."""
    script_text = read_run_script()
    matched_text, failed_text = split_check_branches(script_text)
    assert SUPERSPLAT_EXPORT_SCRIPT in matched_text
    assert '"split_render_check: match"' in matched_text
    assert SUPERSPLAT_EXPORT_SCRIPT not in failed_text
    assert f'"{STATUS_KEY} skipped (no split record)"' in failed_text
    assert "exit 1" in failed_text


def test_supersplat_step_writes_timing_and_status_keys():
    """The three timing keys and the written / timeout / FAILED statuses go to job_info.txt."""
    step_text = supersplat_step_text(read_run_script())
    for timing_key in TIMING_KEYS:
        assert f'echo "{timing_key}:' in step_text
    assert f'"{STATUS_KEY} written"' in step_text
    assert f'"{STATUS_KEY} FAILED (exit $SUPERSPLAT_EXIT_CODE; see Slurm log)"' in step_text
    assert f'"{STATUS_KEY} FAILED (exit 124 timeout' in step_text
    assert "$JOB_INFO" in step_text


def test_supersplat_export_runs_under_timeout():
    """The export command is wrapped in `timeout` with a derived limit, and the Slurm end time is read."""
    step_commands = command_lines(supersplat_step_text(read_run_script()))
    # `-k <grace>` (kill-after) sits between `timeout` and the duration; the derived limit is unchanged.
    assert (f'timeout -k "$SUPERSPLAT_KILL_GRACE_SECONDS" "$SUPERSPLAT_TIMEOUT_SECONDS" '
            f'python "$CODE_DIR/{SUPERSPLAT_EXPORT_SCRIPT}"') in step_commands
    assert "squeue" in step_commands


def test_supersplat_failure_path_does_not_exit():
    """The step's own if/elif/else has no exit and no `set -e` change, so a failure reaches end_utc."""
    step_text = supersplat_step_text(read_run_script())
    step_commands = command_lines(step_text)
    # The step's export if/elif/else: the elif runs the export, an else follows it.
    assert "elif timeout" in step_commands
    export_start = step_commands.index("elif timeout")
    assert "\n  else\n" in step_commands[export_start:]
    # The "(exit N" inside the status messages is text, not a command.
    assert "exit " not in step_commands.replace("(exit $SUPERSPLAT_EXIT_CODE", "").replace("(exit 124", "")
    assert "set -e" not in step_text


def test_supersplat_failure_branch_removes_the_exports_temp_files():
    """After a failed export, only script 22's two mkstemp patterns are removed from renders/."""
    step_commands = command_lines(supersplat_step_text(read_run_script()))
    # Step 1: the patterns are the ones the writers use (mkstemp prefix "<final name>.", suffix ".tmp").
    assert "SUPERSPLAT_PLY_TEMP_PATTERN='supersplat.ply.*.tmp'" in step_commands
    assert "SUPERSPLAT_SIDECAR_TEMP_PATTERN='supersplat_frame.json.*.tmp'" in step_commands
    assert "TIMEOUT_EXIT_CODE=124" in step_commands
    # Step 2: the rm sits in the failure (else) branch, after the exit code is captured.
    export_start = step_commands.index("elif timeout")
    failure_branch = step_commands[export_start:].split("\n  else\n", 1)[1]
    cleanup_line = ('rm -f "$RUN_DIR/renders"/$SUPERSPLAT_PLY_TEMP_PATTERN '
                    '"$RUN_DIR/renders"/$SUPERSPLAT_SIDECAR_TEMP_PATTERN')
    assert cleanup_line in failure_branch
    assert failure_branch.index("SUPERSPLAT_EXIT_CODE=$?") < failure_branch.index(cleanup_line)
    # Step 3: no recursive or broader removal.
    assert "rm -rf" not in step_commands


def test_run_script_is_valid_bash():
    """`bash -n` parses splatad_run.sbatch without a syntax error."""
    completed_process = subprocess.run(["bash", "-n", str(RUN_SCRIPT_PATH)], capture_output=True, text=True)
    assert completed_process.returncode == 0, completed_process.stderr


BUDGET_BEGIN_MARKER = "# BEGIN supersplat_time_budget"
BUDGET_END_MARKER = "# END supersplat_time_budget"
# Synthetic clock for the stub `date`: any epoch works, the budget only uses differences.
STUB_NOW_EPOCH = 1_000_000
STUB_RESERVE_SECONDS = 300
STUB_MEASURED_SECONDS = 192
STUB_SAFETY_FACTOR = 4
FIXED_CAP_SECONDS = STUB_MEASURED_SECONDS * STUB_SAFETY_FACTOR


def run_time_budget(tmp_path: Path, squeue_text: str, seconds_left: int) -> tuple:
    """Run the sbatch's budget function in bash with stub ``squeue`` and ``date`` on PATH.

    Args:
        tmp_path: Folder for the stubs and the call log.
        squeue_text: What the stub squeue prints (the job end-time text).
        seconds_left: Seconds between the stub clock and the stub end time.

    Returns:
        ``(budget_seconds, date_dash_d_call_count)``.
    """
    script_text = read_run_script()
    function_text = script_text[script_text.index(BUDGET_BEGIN_MARKER):script_text.index(BUDGET_END_MARKER)]
    constants_text = "\n".join(
        line.strip() for line in script_text.splitlines() if line.strip().startswith("SUPERSPLAT_") and "=" in line
        and "$(" not in line)
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir()
    call_log = tmp_path / "date_calls.log"
    (stub_dir / "squeue").write_text(f"#!/bin/bash\nprintf '%s' '{squeue_text}'\n")
    (stub_dir / "date").write_text(textwrap.dedent(f"""\
        #!/bin/bash
        if [ "$1" = "-d" ]; then echo "$2" >> {call_log}; echo {STUB_NOW_EPOCH + seconds_left}; else echo {STUB_NOW_EPOCH}; fi
        """))
    for stub_name in ("squeue", "date"):
        (stub_dir / stub_name).chmod(0o755)
    driver_text = f"{constants_text}\n{function_text}\nsupersplat_time_budget_seconds\n"
    completed_process = subprocess.run(["bash", "-c", driver_text], capture_output=True, text=True,
                                       env={"PATH": f"{stub_dir}:/usr/bin:/bin", "SLURM_JOB_ID": "1"})
    assert completed_process.returncode == 0, completed_process.stderr
    call_count = len(call_log.read_text().splitlines()) if call_log.exists() else 0
    return int(completed_process.stdout.strip()), call_count


@pytest.mark.parametrize("squeue_text", ["", "UNLIMITED", "N/A", "Unknown"])
def test_budget_without_a_usable_end_time_is_the_fixed_cap_and_never_calls_date_d(tmp_path, squeue_text):
    """Empty or non-date squeue text gives the fixed cap, and `date -d` is not called (date -d "" is midnight)."""
    budget_seconds, date_call_count = run_time_budget(tmp_path, squeue_text, seconds_left=0)
    assert budget_seconds == FIXED_CAP_SECONDS
    assert date_call_count == 0


def test_budget_with_plenty_of_time_left_is_the_fixed_cap(tmp_path):
    """Remaining - reserve above the fixed cap leaves the fixed cap."""
    budget_seconds, date_call_count = run_time_budget(
        tmp_path, "2026-11-21T12:00:00", seconds_left=FIXED_CAP_SECONDS + STUB_RESERVE_SECONDS + 1000)
    assert budget_seconds == FIXED_CAP_SECONDS
    assert date_call_count == 1


def test_budget_near_the_job_end_is_remaining_minus_reserve(tmp_path):
    """min(fixed cap, remaining - reserve): 500 s left with a 300 s reserve gives 200 s."""
    seconds_left = STUB_RESERVE_SECONDS + 200
    budget_seconds, _ = run_time_budget(tmp_path, "2026-11-21T12:00:00", seconds_left=seconds_left)
    assert budget_seconds == seconds_left - STUB_RESERVE_SECONDS


def test_budget_inside_the_reserve_is_not_positive(tmp_path):
    """Less time left than the reserve gives a value below 1, which the step treats as 'skip'."""
    budget_seconds, _ = run_time_budget(tmp_path, "2026-11-21T12:00:00", seconds_left=STUB_RESERVE_SECONDS - 10)
    assert budget_seconds < 1
