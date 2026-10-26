"""Stage, submit, monitor and fetch SplatAD runs on the CS Lab Slurm cluster (Mac side).

Usage:
    uv run python scripts/slurm/submit_run.py gates  --session S
    uv run python scripts/slurm/submit_run.py build-env
    uv run python scripts/slurm/submit_run.py stage  --session S
    uv run python scripts/slurm/submit_run.py submit --session S --profile smoke [--skip-gates]
    uv run python scripts/slurm/submit_run.py status <job_id>
    uv run python scripts/slurm/submit_run.py fetch  <run_id>

The GPU nodes have no container runtime, so jobs run in a native environment that
``build-env`` builds once from the Dockerfile's recipe (scripts/slurm/build_env.sbatch). The
environment directory is named by a hash of the committed recipe files, so a run always uses
the environment its recipe describes.

Settings come from configs/slurm.yaml. The cluster username is never stored in the repo: ssh
picks it from ~/.ssh/config, or it is taken from the VIKING_SLURM_USER environment variable.
Every external command goes through ``run_command`` so tests can replace it.
"""

import argparse
import datetime
import hashlib
import math
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

# Project root = two levels above this file (scripts/slurm/submit_run.py).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Default Slurm settings file, relative to the project root.
DEFAULT_SLURM_CONFIG_PATH = PROJECT_ROOT / "configs" / "slurm.yaml"
# Batch scripts, as paths inside the staged code snapshot (code/<sha>/...).
RUN_SCRIPT_RELATIVE_PATH = "scripts/slurm/splatad_run.sbatch"
BUILD_SCRIPT_RELATIVE_PATH = "scripts/slurm/build_env.sbatch"
# Repo paths staged as code/<sha>/: what the environment build installs and tests (the same set
# the Dockerfile COPYs, plus the Dockerfile itself, which build_env.sbatch reads pins from).
STAGED_REPO_PATHS = ("pyproject.toml", "src", "tests", "configs", "scripts",
                     "docker/constraints.txt", "docker/Dockerfile.splatad",
                     ".claude/hooks/readonly_bash_guard.py")
# Hex digits of the recipe hash used in environment names: 12 hex = 48 bits, so two different
# recipes collide with probability ~2**-48, negligible for the handful of envs ever built.
RECIPE_HASH_LENGTH = 12
# build_env.sbatch writes <stage_root>/envs/<env name>.marker as its last step; it names the
# env's path, so a marker means a complete build.
ENV_MARKER_SUFFIX = ".marker"
# Local ledger of submitted jobs (runs/ is git-ignored).
SUBMITTED_LEDGER_PATH = PROJECT_ROOT / "runs" / "submitted.tsv"
# Environment variable that, when set, supplies the cluster username for user@host.
SLURM_USER_ENVIRONMENT_VARIABLE = "VIKING_SLURM_USER"

# Unit sizes for the Slurm time format D-HH:MM:SS.
SECONDS_PER_MINUTE = 60
SECONDS_PER_HOUR = 60 * SECONDS_PER_MINUTE
SECONDS_PER_DAY = 24 * SECONDS_PER_HOUR
# Cluster rule: a time limit over 5 days never schedules (support.cs.toronto.edu slurmresource).
MAXIMUM_TIME_LIMIT_SECONDS = 5 * SECONDS_PER_DAY
# Slurm reports MEMORY in MB (MiB, 1024**2 bytes); the --mem "G" suffix means GiB = 1024 MB.
MEGABYTES_PER_GIGABYTE = 1024
# Length of the abbreviated git SHA embedded in run ids (git's conventional short-SHA length).
SHORT_SHA_LENGTH = 7
# UTC timestamp format embedded in run ids (sortable, filesystem-safe).
RUN_TIMESTAMP_FORMAT = "%Y%m%d-%H%M%S"
# Header of the local submission ledger.
LEDGER_COLUMNS = ("run_id", "job_id", "git_sha", "session", "profile", "submitted_utc")


class GateFailure(RuntimeError):
    """Raised when a pre-submission gate (tests, contract, clean tree) fails."""


def run_command(command: Sequence[str], capture_output: bool = True,
                input_text: Optional[str] = None) -> subprocess.CompletedProcess:
    """Run one external command; the single choke point that tests mock.

    Args:
        command: Argument list (never a shell string).
        capture_output: True to capture stdout/stderr as text; False to stream to the terminal.
        input_text: Optional text fed to the command's stdin.

    Returns:
        The CompletedProcess (returncode, and stdout/stderr when captured). Never raises on a
        nonzero exit; callers check ``returncode``.
    """
    return subprocess.run(list(command), capture_output=capture_output, text=True,
                          input=input_text, check=False, cwd=PROJECT_ROOT)


def load_slurm_config(config_path: Path = DEFAULT_SLURM_CONFIG_PATH) -> Dict[str, Any]:
    """Load configs/slurm.yaml.

    Args:
        config_path: Path of the Slurm YAML.

    Returns:
        The parsed mapping.

    Raises:
        ValueError: If the top level is not a mapping or a required key is missing.
    """
    # Step 1: parse the YAML.
    config = yaml.safe_load(Path(config_path).read_text())
    if not isinstance(config, dict):
        raise ValueError(f"{config_path}: top level must be a mapping")
    # Step 2: check the keys every subcommand relies on (stage_root may be null until probed).
    for required_key in ("ssh_host", "partition", "gres", "mem_fraction", "workdisk_root",
                         "stage_root", "micromamba_version", "micromamba_sha256",
                         "environment_recipe_files", "env_build", "profiles"):
        if required_key not in config:
            raise ValueError(f"{config_path}: missing required key {required_key!r}")
    return config


def require_stage_root(config: Dict[str, Any]) -> str:
    """Return the configured stage root (relative to the user's workdisk dir), or fail when unset.

    Args:
        config: Parsed Slurm config.

    Returns:
        The configured stage root, without a trailing slash.

    Raises:
        ValueError: If ``stage_root`` is null/empty.
    """
    stage_root = config.get("stage_root")
    if not stage_root:
        raise ValueError("configs/slurm.yaml: stage_root is not set")
    return str(stage_root).rstrip("/")


def resolve_remote_stage_root(config: Dict[str, Any], ssh_target: str) -> str:
    """Create the stage root on the cluster workdisk and return its absolute path.

    The stage root is ``<workdisk_root>/<cluster username>/<stage_root>``. The username is read
    on the cluster (``id -un``), so neither it nor a home path is stored in the repo; the
    absolute path is only needed in exported job variables.

    Args:
        config: Parsed Slurm config (``workdisk_root``, ``stage_root``).
        ssh_target: ``host`` or ``user@host``.

    Returns:
        Absolute stage root path on the cluster.

    Raises:
        ValueError: If ``stage_root`` is unset or ``workdisk_root`` is not absolute.
        RuntimeError: If the remote command fails (e.g. the per-user workdisk dir does not
            exist) or prints a non-absolute path.
    """
    stage_root = require_stage_root(config)
    workdisk_root = str(config["workdisk_root"]).rstrip("/")
    if not workdisk_root.startswith("/"):
        raise ValueError(f"configs/slurm.yaml: workdisk_root {workdisk_root!r} must be absolute")
    # Paths go in as positional parameters ($1, $2), so the script text needs no quoting of them.
    # `cd` into the per-user dir first (created by the admins; never created here), so a wrong
    # workdisk_root fails instead of silently making a new tree.
    stage_script = 'cd "$1/$(id -un)" && mkdir -p "$2" && cd "$2" && pwd -P'
    completed = run_checked(ssh_command(ssh_target, [
        "sh", "-c", stage_script, "sh", workdisk_root, stage_root]), "resolve stage root")
    absolute_root = completed.stdout.strip()
    if not absolute_root.startswith("/"):
        raise RuntimeError(f"remote stage root resolved to a non-absolute path: {absolute_root!r}")
    return absolute_root


def compute_recipe_hash(recipe_contents: Dict[str, bytes]) -> str:
    """Hash the environment recipe files into a short, order-independent identifier.

    Each file contributes ``path NUL length NUL content`` in sorted path order. The length
    prefix makes the encoding unambiguous (no two different file sets give the same byte
    stream), so the hash changes iff some file's path or content changes.

    Args:
        recipe_contents: repo-relative path -> committed file bytes.

    Returns:
        First RECIPE_HASH_LENGTH hex digits of the SHA-256 of that stream.
    """
    digest = hashlib.sha256()
    for relative_path in sorted(recipe_contents):
        file_bytes = recipe_contents[relative_path]
        digest.update(relative_path.encode() + b"\0" + str(len(file_bytes)).encode() + b"\0")
        digest.update(file_bytes)
    return digest.hexdigest()[:RECIPE_HASH_LENGTH]


def read_committed_file(git_sha: str, relative_path: str) -> bytes:
    """Read a file as committed at ``git_sha`` (ignores uncommitted edits).

    Args:
        git_sha: Commit SHA.
        relative_path: Repo-relative path.

    Returns:
        The file's bytes at that commit.

    Raises:
        RuntimeError: If git cannot show the file.
    """
    completed = subprocess.run(["git", "show", f"{git_sha}:{relative_path}"], capture_output=True,
                               check=False, cwd=PROJECT_ROOT)
    if completed.returncode != 0:
        raise RuntimeError(f"git show {git_sha}:{relative_path} failed: {completed.stderr!r}")
    return completed.stdout


def environment_name(config: Dict[str, Any], git_sha: str) -> str:
    """Name of the environment that the recipe at ``git_sha`` defines.

    Args:
        config: Parsed Slurm config (``environment_recipe_files``).
        git_sha: Commit whose recipe files are hashed.

    Returns:
        ``splatad-<recipe hash>``.
    """
    recipe_contents = {relative_path: read_committed_file(git_sha, relative_path)
                       for relative_path in config["environment_recipe_files"]}
    return f"splatad-{compute_recipe_hash(recipe_contents)}"


def parse_environment_marker(marker_text: str) -> Dict[str, str]:
    """Parse the ``key=value`` lines build_env.sbatch writes.

    Args:
        marker_text: Marker file contents.

    Returns:
        Mapping key -> value.

    Raises:
        ValueError: If ``env_prefix`` is missing (a marker is only written after a complete
            build, so this means corruption).
    """
    marker = {}
    for line_text in marker_text.splitlines():
        key_text, separator, value_text = line_text.partition("=")
        if separator:
            marker[key_text.strip()] = value_text.strip()
    if "env_prefix" not in marker:
        raise ValueError(f"environment marker lacks env_prefix: {marker_text!r}")
    return marker


def resolve_ssh_target(ssh_host: str) -> str:
    """Build the ssh destination without storing any username in the repo.

    Args:
        ssh_host: Host name from the config.

    Returns:
        ``user@host`` when VIKING_SLURM_USER is set, else ``host`` (ssh then uses the User
        from ~/.ssh/config or the local login name).
    """
    user_name = os.environ.get(SLURM_USER_ENVIRONMENT_VARIABLE, "").strip()
    return f"{user_name}@{ssh_host}" if user_name else ssh_host


def gres_type_from_spec(gres_spec: str) -> str:
    """Extract the GPU type from a gres spec like ``gpu:rtx_a6000:1``.

    Args:
        gres_spec: Slurm generic-resource string ``name:type:count``.

    Returns:
        The type field (``rtx_a6000``).

    Raises:
        ValueError: If the spec has no type field (``gpu:1`` cannot filter nodes by GPU model).
    """
    gres_fields = gres_spec.split(":")
    # Format is name:type:count; only the 3-field form names a GPU type.
    if len(gres_fields) != 3 or not gres_fields[1]:
        raise ValueError(f"gres {gres_spec!r} must have the form gpu:<type>:<count>")
    return gres_fields[1]


def parse_slurm_time_limit_seconds(time_limit: str) -> int:
    """Convert a Slurm ``D-HH:MM:SS`` time limit to seconds.

    Args:
        time_limit: Time string in the form days-hours:minutes:seconds, e.g. ``0-06:00:00``.

    Returns:
        Total seconds = days*86400 + hours*3600 + minutes*60 + seconds.

    Raises:
        ValueError: If the string is malformed or a field is out of range.
    """
    # Step 1: split off the day count.
    day_text, separator, clock_text = time_limit.partition("-")
    clock_fields = clock_text.split(":")
    if not separator or len(clock_fields) != 3:
        raise ValueError(f"time limit {time_limit!r} must have the form D-HH:MM:SS")
    try:
        day_count = int(day_text)
        hour_count, minute_count, second_count = (int(field_text) for field_text in clock_fields)
    except ValueError as error:
        raise ValueError(f"time limit {time_limit!r} has a non-integer field") from error
    # Step 2: range-check each field (hours < 24, minutes/seconds < 60).
    if (day_count < 0 or not 0 <= hour_count < 24 or not 0 <= minute_count < 60
            or not 0 <= second_count < 60):
        raise ValueError(f"time limit {time_limit!r} has a field out of range")
    # Step 3: weighted sum of the fields.
    return (day_count * SECONDS_PER_DAY + hour_count * SECONDS_PER_HOUR
            + minute_count * SECONDS_PER_MINUTE + second_count)


def derive_node_resources(sinfo_text: str, gres_type: str, mem_fraction: float) -> Tuple[int, int]:
    """Derive -c and --mem from ``sinfo -O NodeHost,Memory,CPUs,Gres`` output.

    Only nodes whose GRES names ``gres_type`` are considered. Taking the minimum over them means
    the request fits on every matching node, so Slurm may schedule on any of them.

    Args:
        sinfo_text: Whitespace-padded table with a header row (HOSTNAMES MEMORY CPUS GRES).
        gres_type: GPU type to match, e.g. ``rtx_a6000``.
        mem_fraction: Fraction of node RAM to request, in (0, 1].

    Returns:
        (cpus, mem_gb): min CPUS over matching nodes, and
        floor(mem_fraction * min MEMORY_MB / 1024). MEMORY is in MB and Slurm's ``--mem=<n>G``
        means GiB = 1024 MB, so dividing by 1024 converts; flooring keeps the request at or
        under the fraction.

    Raises:
        ValueError: If mem_fraction is outside (0, 1], the header lacks a needed column, or no
            node matches the GPU type.
    """
    if not 0 < mem_fraction <= 1:
        raise ValueError(f"mem_fraction {mem_fraction} must be in (0, 1]")
    # Step 1: locate columns from the header so column order does not matter.
    table_lines = [line_text for line_text in sinfo_text.splitlines() if line_text.strip()]
    if not table_lines:
        raise ValueError("empty sinfo output")
    header_fields = [field_text.upper() for field_text in table_lines[0].split()]
    try:
        memory_column = header_fields.index("MEMORY")
        cpus_column = header_fields.index("CPUS")
        gres_column = header_fields.index("GRES")
    except ValueError as error:
        raise ValueError(f"sinfo header {table_lines[0]!r} lacks MEMORY/CPUS/GRES") from error
    # Step 2: collect (cpus, memory MB) of rows whose GRES names the wanted type.
    matching_nodes: List[Tuple[int, int]] = []
    for row_text in table_lines[1:]:
        row_fields = row_text.split()
        if len(row_fields) < len(header_fields):
            continue
        # GRES may list several entries ("gpu:rtx_a6000:1,tmp:..."); match the type field exactly.
        gres_entries = row_fields[gres_column].split(",")
        if any(gres_entry.split(":")[1:2] == [gres_type] for gres_entry in gres_entries):
            matching_nodes.append((int(row_fields[cpus_column]), int(row_fields[memory_column])))
    if not matching_nodes:
        raise ValueError(f"no sinfo node has GPU type {gres_type!r}")
    # Step 3: smallest node bounds the request; convert MB -> GiB and take the fraction.
    minimum_cpus = min(node_cpus for node_cpus, _ in matching_nodes)
    minimum_memory_mb = min(node_memory_mb for _, node_memory_mb in matching_nodes)
    memory_gb = math.floor(mem_fraction * minimum_memory_mb / MEGABYTES_PER_GIGABYTE)
    return minimum_cpus, memory_gb


def build_run_id(session: str, profile: str, git_sha: str, now_utc: datetime.datetime) -> str:
    """Build the run id ``<session>-<profile>-<YYYYmmdd-HHMMSS>-<sha7>``.

    Args:
        session: Session (dataset) name.
        profile: Profile name from the config.
        git_sha: Full commit SHA of the code being run.
        now_utc: Submission time (UTC).

    Returns:
        The run id string.
    """
    return f"{session}-{profile}-{now_utc.strftime(RUN_TIMESTAMP_FORMAT)}-{git_sha[:SHORT_SHA_LENGTH]}"


def dataset_time_span(session_dir: Path) -> Tuple[float, float]:
    """Earliest and latest timestamp over every sensor stream of an intermediate dataset.

    These are the bounds ADDataParser._filter_based_on_time measures its fractions against
    (``min``/``max`` of all camera and lidar times).

    Args:
        session_dir: ``data/intermediate/<session>`` (holds ``poses.npz``).

    Returns:
        (earliest, latest) in seconds.

    Raises:
        ValueError: If ``poses.npz`` has no ``*__times`` arrays.
    """
    with np.load(session_dir / "poses.npz") as poses:
        time_arrays = [poses[key_name] for key_name in poses.files if key_name.endswith("__times")]
    if not time_arrays:
        raise ValueError(f"{session_dir}/poses.npz has no '<sensor>__times' arrays")
    return (float(min(times.min() for times in time_arrays)),
            float(max(times.max() for times in time_arrays)))


def window_fractions(earliest_time: float, latest_time: float, window_start_seconds: float,
                     window_seconds: float) -> Tuple[float, float]:
    """Convert a time window into the dataparser's dataset start/end fractions.

    ADDataParser keeps data with times in [t0 + f_start*D, t0 + f_end*D], where t0 is the
    earliest and D = latest - earliest the full span. Solving for the window
    [t0 + s, t0 + s + w] gives f_start = s / D and f_end = (s + w) / D.

    Args:
        earliest_time: Earliest sensor time [s].
        latest_time: Latest sensor time [s].
        window_start_seconds: Window start s, measured from the earliest time.
        window_seconds: Window length w.

    Returns:
        (start_fraction, end_fraction), both in [0, 1], start < end.

    Raises:
        ValueError: If the window is empty or not inside the recording.
    """
    duration = latest_time - earliest_time
    if duration <= 0 or window_seconds <= 0 or window_start_seconds < 0:
        raise ValueError("window and recording must have positive length and a non-negative start")
    if window_start_seconds + window_seconds > duration:
        raise ValueError(f"window [{window_start_seconds}, {window_start_seconds + window_seconds}] s "
                         f"exceeds the {duration:.2f} s recording")
    return window_start_seconds / duration, (window_start_seconds + window_seconds) / duration


def profile_window_fractions(profile_settings: Dict[str, Any], session: str) -> Tuple[float, float]:
    """Dataset fractions for a profile: its time window if set, else the whole recording.

    Args:
        profile_settings: ``profiles.<name>`` mapping (optional ``window_start_seconds`` and
            ``window_seconds``, both or neither).
        session: Dataset name under data/intermediate/ (its timestamps define the span).

    Returns:
        (start_fraction, end_fraction).
    """
    window_keys_present = ["window_start_seconds" in profile_settings, "window_seconds" in profile_settings]
    if not any(window_keys_present):
        return 0.0, 1.0
    # Half a window is a config mistake; say so instead of failing later with a KeyError.
    if not all(window_keys_present):
        raise ValueError("profile sets only one of window_start_seconds / window_seconds; set both or neither")
    earliest_time, latest_time = dataset_time_span(PROJECT_ROOT / "data" / "intermediate" / session)
    return window_fractions(earliest_time, latest_time, float(profile_settings["window_start_seconds"]),
                            float(profile_settings["window_seconds"]))


def build_job_environment(profile_settings: Dict[str, Any], session: str, run_id: str,
                          git_sha: str, absolute_stage_root: str,
                          env_marker: Dict[str, str]) -> Dict[str, str]:
    """Build the variables exported to splatad_run.sbatch (insertion order is kept).

    Args:
        profile_settings: The chosen ``profiles.<name>`` mapping.
        session: Session name.
        run_id: Run id.
        git_sha: Full commit SHA.
        absolute_stage_root: Resolved stage root on the cluster.
        env_marker: Parsed marker of the environment the run activates (its path).

    Returns:
        Mapping variable name -> string value.

    Raises:
        ValueError: If the profile has no integer ``seed`` (every run must record its seed).
    """
    # A run without an explicit seed would silently take nerfstudio's default; require it instead.
    train_seed = profile_settings.get("seed")
    if isinstance(train_seed, bool) or not isinstance(train_seed, int):
        raise ValueError(f"profile needs an integer 'seed' (got {train_seed!r})")
    start_fraction, end_fraction = profile_window_fractions(profile_settings, session)
    return {
        "RUN_ID": run_id,
        "SESSION": session,
        "STAGE_ROOT": absolute_stage_root,
        "GIT_SHA": git_sha,
        "CODE_DIR": f"{absolute_stage_root}/code/{git_sha}",
        "ENV_PREFIX": env_marker["env_prefix"],
        "MAX_NUM_ITERATIONS": str(profile_settings["max_num_iterations"]),
        "STEPS_PER_SAVE": str(profile_settings["steps_per_save"]),
        # ns-train --machine.seed: seeds python/numpy/torch RNGs (not bit-exact, see the sbatch).
        "TRAIN_SEED": str(train_seed),
        "RENDER_POSE_SOURCE": str(profile_settings["render_pose_source"]),
        "RENDER_DOWNSCALE_FACTOR": str(profile_settings["render_downscale_factor"]),
        # repr keeps every digit of the float, so the job sees exactly the computed window.
        "DATASET_START_FRACTION": repr(start_fraction),
        "DATASET_END_FRACTION": repr(end_fraction),
    }


def build_job_script(job_environment: Dict[str, str], script_path: str) -> str:
    """Build the per-job batch script: ``export`` every job variable, then run the committed script.

    CS Lab support (2026-10-26): shells on the compute nodes are locked to Slurm jobs, so the
    submitting environment is not a supported way to pass variables, and ``--export=ALL,K=V`` /
    ``--get-user-env`` make Slurm fetch the login environment, which fails and holds the job
    (``user_env_retrieval_failed_requeued_held``; jobs 222519, 222526). Their recommendation is to
    ``export`` the variables inside the batch script, which this generated script does. It is
    written into the job's own directory, so it also records exactly what the job was given.

    Args:
        job_environment: Variables the job needs (name -> value), in export order.
        script_path: Remote path of the committed batch script to run with those variables.

    Returns:
        Script text: shebang, one ``export NAME=<shell-quoted value>`` per variable, then
        ``exec bash <script>`` (exec: the committed script replaces this shell, so its exit
        code is the job's).

    Raises:
        ValueError: If a variable name is not a valid shell identifier (``export`` would
            misparse it).
    """
    # Step 1: names must be identifiers; values may contain anything, because shlex.quote makes
    # each one a single literal word for bash (no expansion of $, spaces or quotes).
    for variable_name in job_environment:
        if not variable_name.isidentifier():
            raise ValueError(f"job variable name {variable_name!r} is not a valid identifier")
    # Step 2: assemble the lines.
    script_lines = ["#!/bin/bash",
                    "# Generated by scripts/slurm/submit_run.py: the job's variables, exported in the",
                    "# batch script itself (CS Lab support's recommended way), then the committed script."]
    script_lines += [f"export {variable_name}={shlex.quote(variable_value)}"
                     for variable_name, variable_value in job_environment.items()]
    script_lines.append(f"exec bash {shlex.quote(script_path)}")
    return "\n".join(script_lines) + "\n"


def write_remote_file(ssh_target: str, remote_path: str, file_text: str) -> None:
    """Write a text file on the cluster over ssh (contents sent on stdin, path as ``$1``).

    Args:
        ssh_target: ``host`` or ``user@host``.
        remote_path: Absolute destination path; its directory must exist.
        file_text: File contents.

    Raises:
        RuntimeError: If the remote write fails.
    """
    run_checked(ssh_command(ssh_target, ["sh", "-c", 'cat > "$1"', "sh", remote_path]),
                f"write {remote_path}", input_text=file_text)


def build_sbatch_command(partition: str, gres: str, cpus: int, memory_gb: int, time_limit: str,
                         job_name: str, output_path: str, job_script_path: str) -> List[str]:
    """Build the submission command: ``sbatch <resources> <generated job script>``.

    Job variables are not on this command: they are exported inside the job script
    (``build_job_script``). Never uses ``--export`` or ``--get-user-env`` (both hold the job on
    this cluster).

    Args:
        partition: Slurm partition.
        gres: Generic resource spec, e.g. ``gpu:rtx_a6000:1``.
        cpus: CPUs per task (-c).
        memory_gb: Memory in GiB (--mem=<n>G).
        time_limit: ``D-HH:MM:SS``; must be at most 5 days.
        job_name: --job-name value.
        output_path: --output path (may contain %j).
        job_script_path: Remote path of the generated job script (``build_job_script``).

    Returns:
        Argument list ``["sbatch", <resources>..., job_script_path]``.

    Raises:
        ValueError: If the time limit exceeds 5 days or resources are non-positive.
    """
    # Step 1: validate the cluster rules.
    if parse_slurm_time_limit_seconds(time_limit) > MAXIMUM_TIME_LIMIT_SECONDS:
        raise ValueError(f"time limit {time_limit} exceeds the 5-day cluster maximum")
    if cpus <= 0 or memory_gb <= 0:
        raise ValueError(f"cpus ({cpus}) and memory_gb ({memory_gb}) must be positive")
    # Step 2: assemble the arguments.
    return [
        "sbatch",
        f"--partition={partition}",
        f"--gres={gres}",
        "-c", str(cpus),
        f"--mem={memory_gb}G",
        "-t", time_limit,
        f"--job-name={job_name}",
        f"--output={output_path}",
        job_script_path,
    ]


def ssh_command(ssh_target: str, remote_arguments: Sequence[str]) -> List[str]:
    """Wrap a remote command for ssh, quoting each argument for the remote shell.

    Args:
        ssh_target: ``host`` or ``user@host``.
        remote_arguments: Remote argument list.

    Returns:
        Local argument list ``["ssh", target, "<quoted remote command>"]``.
    """
    return ["ssh", ssh_target, shlex.join(list(remote_arguments))]


def get_head_sha() -> str:
    """Return the full SHA of HEAD.

    Returns:
        40-character commit SHA.

    Raises:
        RuntimeError: If git fails.
    """
    completed = run_command(["git", "rev-parse", "HEAD"])
    if completed.returncode != 0:
        raise RuntimeError(f"git rev-parse HEAD failed: {completed.stderr}")
    return completed.stdout.strip()


def check_clean_tree() -> List[str]:
    """Refuse a working tree with tracked modifications.

    Returns:
        Untracked paths (allowed, but reported to the user).

    Raises:
        GateFailure: If any tracked file is modified, staged, deleted or renamed.
    """
    # Step 1: porcelain lines are "XY path"; "??" marks untracked, anything else is tracked.
    completed = run_command(["git", "status", "--porcelain"])
    if completed.returncode != 0:
        raise GateFailure(f"git status failed: {completed.stderr}")
    status_lines = [line_text for line_text in completed.stdout.splitlines() if line_text.strip()]
    untracked_paths = [line_text[3:] for line_text in status_lines if line_text.startswith("??")]
    tracked_changes = [line_text for line_text in status_lines if not line_text.startswith("??")]
    # Step 2: the run must correspond exactly to a commit.
    if tracked_changes:
        raise GateFailure("working tree has tracked modifications; commit first:\n  "
                          + "\n  ".join(tracked_changes))
    return untracked_paths


def run_gates(session: str) -> None:
    """Run the Mac-side gates: clean tree, full test suite, contract validation.

    Args:
        session: Dataset name under data/intermediate/.

    Raises:
        GateFailure: On the first failing gate.
    """
    # Step 1: clean tree first (cheapest, and tests/contract must judge the committed code).
    untracked_paths = check_clean_tree()
    if untracked_paths:
        print("untracked (allowed, not staged):\n  " + "\n  ".join(untracked_paths))
    # Step 2: unit tests.
    if run_command(["uv", "run", "pytest", "-q"], capture_output=False).returncode != 0:
        raise GateFailure("uv run pytest -q failed")
    # Step 3: SplatAD input contract on the dataset to be staged.
    dataset_dir = PROJECT_ROOT / "data" / "intermediate" / session
    contract_command = ["uv", "run", "python", "scripts/06_validate_contract.py",
                        "--dataset", str(dataset_dir)]
    if run_command(contract_command, capture_output=False).returncode != 0:
        raise GateFailure(f"contract validation failed for {dataset_dir}")
    print("gates: PASS")


def run_checked(command: Sequence[str], description: str, capture_output: bool = True,
                input_text: Optional[str] = None) -> subprocess.CompletedProcess:
    """Run a command and raise when it fails.

    Args:
        command: Argument list.
        description: Short label for the error message.
        capture_output: Passed to run_command.
        input_text: Optional stdin text, passed to run_command.

    Returns:
        The CompletedProcess.

    Raises:
        RuntimeError: On a nonzero exit.
    """
    completed = run_command(command, capture_output=capture_output, input_text=input_text)
    if completed.returncode != 0:
        raise RuntimeError(f"{description} failed (exit {completed.returncode}): {completed.stderr}")
    return completed


def stage_code(ssh_target: str, absolute_stage_root: str, git_sha: str) -> str:
    """Copy the committed repo snapshot (STAGED_REPO_PATHS at ``git_sha``) to code/<sha>/.

    ``git archive`` exports exactly the committed tree, so uncommitted edits never reach the
    cluster and the snapshot directory name identifies its content.

    Args:
        ssh_target: ``host`` or ``user@host``.
        absolute_stage_root: Resolved stage root on the cluster.
        git_sha: Commit to export.

    Returns:
        Absolute remote code directory.

    Raises:
        RuntimeError: If git, tar or rsync fails.
    """
    remote_code_dir = f"{absolute_stage_root}/code/{git_sha}"
    run_checked(ssh_command(ssh_target, ["mkdir", "-p", remote_code_dir]), "remote code mkdir")
    with tempfile.TemporaryDirectory() as export_dir:
        # Step 1: export the committed paths and unpack them locally.
        archive_path = Path(export_dir) / "code.tar"
        run_checked(["git", "archive", "--format=tar", "-o", str(archive_path), git_sha,
                     *STAGED_REPO_PATHS], "git archive")
        snapshot_dir = Path(export_dir) / "snapshot"
        snapshot_dir.mkdir()
        run_checked(["tar", "-xf", str(archive_path), "-C", str(snapshot_dir)], "tar extract")
        # Step 2: mirror it (--delete keeps the directory identical to the commit).
        run_checked(["rsync", "-a", "--delete", f"{snapshot_dir}/", f"{ssh_target}:{remote_code_dir}/"],
                    "code rsync", capture_output=False)
    return remote_code_dir


def stage_session(config: Dict[str, Any], session: str) -> str:
    """Copy the dataset and the committed code snapshot to the cluster.

    Args:
        config: Parsed Slurm config.
        session: Dataset name under data/intermediate/.

    Returns:
        The git SHA whose code was staged.

    Raises:
        FileNotFoundError: If the dataset is missing locally.
        RuntimeError: If a remote command fails.
    """
    # Step 1: resolve paths.
    ssh_target = resolve_ssh_target(config["ssh_host"])
    absolute_stage_root = resolve_remote_stage_root(config, ssh_target)
    dataset_dir = PROJECT_ROOT / "data" / "intermediate" / session
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"dataset {dataset_dir} not found")
    git_sha = get_head_sha()
    # Step 2: create remote directories (rsync creates only the last path component).
    run_checked(ssh_command(ssh_target, ["mkdir", "-p", f"{absolute_stage_root}/data/{session}",
                                          f"{absolute_stage_root}/runs"]), "remote mkdir")
    # Step 3: dataset; trailing slashes copy the directory contents; --delete mirrors exactly.
    run_checked(["rsync", "-a", "--partial", "--delete", f"{dataset_dir}/",
                 f"{ssh_target}:{absolute_stage_root}/data/{session}/"], "dataset rsync",
                capture_output=False)
    # Step 4: committed code snapshot.
    stage_code(ssh_target, absolute_stage_root, git_sha)
    print(f"staged session {session} and code {git_sha} under {absolute_stage_root}")
    return git_sha


def read_environment_marker(ssh_target: str, absolute_stage_root: str,
                            env_name: str) -> Optional[Dict[str, str]]:
    """Read and parse the marker of an environment, or None if it was never completed.

    Args:
        ssh_target: ``host`` or ``user@host``.
        absolute_stage_root: Resolved stage root on the workdisk.
        env_name: ``splatad-<recipe hash>``.

    Returns:
        The parsed marker, or None when the marker file does not exist.
    """
    marker_path = f"{absolute_stage_root}/envs/{env_name}{ENV_MARKER_SUFFIX}"
    completed = run_command(ssh_command(ssh_target, ["cat", marker_path]))
    if completed.returncode != 0:
        return None
    return parse_environment_marker(completed.stdout)


def query_node_resources(config: Dict[str, Any], ssh_target: str) -> Tuple[int, int]:
    """Size a job from live ``sinfo`` output (never guessed).

    Args:
        config: Parsed Slurm config.
        ssh_target: ``host`` or ``user@host``.

    Returns:
        (cpus, memory_gb) from ``derive_node_resources``.
    """
    sinfo_result = run_checked(ssh_command(ssh_target, [
        "sinfo", "-p", config["partition"], "-O", "NodeHost,Memory,CPUs,Gres"]), "sinfo")
    return derive_node_resources(
        sinfo_result.stdout, gres_type_from_spec(config["gres"]), float(config["mem_fraction"]))


def submit_sbatch(ssh_target: str, sbatch_arguments: List[str]) -> str:
    """Submit over ssh and return the job id.

    Args:
        ssh_target: ``host`` or ``user@host``.
        sbatch_arguments: Output of ``build_sbatch_command``.

    Returns:
        The Slurm job id.
    """
    # --parsable prints "jobid[;cluster]"; it goes right after the sbatch token.
    sbatch_position = sbatch_arguments.index("sbatch") + 1
    parsable_arguments = (sbatch_arguments[:sbatch_position] + ["--parsable"]
                          + sbatch_arguments[sbatch_position:])
    sbatch_result = run_checked(ssh_command(ssh_target, parsable_arguments), "sbatch")
    return sbatch_result.stdout.strip().split(";")[0]


def build_environment(config: Dict[str, Any]) -> Optional[str]:
    """Stage the committed recipe and submit build_env.sbatch, unless the env already exists.

    Args:
        config: Parsed Slurm config.

    Returns:
        The build job id, or None when the environment is already complete.

    Raises:
        GateFailure: If the tree has tracked modifications (the recipe must be committed).
    """
    # Step 1: the environment is named by the committed recipe, so require a clean tree.
    check_clean_tree()
    ssh_target = resolve_ssh_target(config["ssh_host"])
    absolute_stage_root = resolve_remote_stage_root(config, ssh_target)
    git_sha = get_head_sha()
    env_name = environment_name(config, git_sha)
    # Reuse a built env: it lives on the non-expiring workdisk, so a marker stays valid.
    env_marker = read_environment_marker(ssh_target, absolute_stage_root, env_name)
    if env_marker:
        print(f"environment already built: {env_marker['env_prefix']}")
        return None
    # Step 2: stage the code the build installs and tests.
    remote_code_dir = stage_code(ssh_target, absolute_stage_root, git_sha)
    log_dir = f"{absolute_stage_root}/envs/logs"
    run_checked(ssh_command(ssh_target, ["mkdir", "-p", log_dir]), "remote log mkdir")
    # Step 3: full-node resources from sinfo (the compile is parallel, MAX_JOBS = CPUs).
    cpus, memory_gb = query_node_resources(config, ssh_target)
    # Step 4: job script exporting the build's variables, next to the build logs.
    job_script_path = f"{log_dir}/{env_name}-job.sbatch"
    write_remote_file(ssh_target, job_script_path, build_job_script(
        {
            "STAGE_ROOT": absolute_stage_root,
            "CODE_DIR": remote_code_dir,
            "ENV_NAME": env_name,
            "MICROMAMBA_VERSION": str(config["micromamba_version"]),
            "MICROMAMBA_SHA256": str(config["micromamba_sha256"]),
        },
        f"{remote_code_dir}/{BUILD_SCRIPT_RELATIVE_PATH}"))
    # Step 5: submit.
    sbatch_arguments = build_sbatch_command(
        partition=config["partition"], gres=config["gres"], cpus=cpus, memory_gb=memory_gb,
        time_limit=str(config["env_build"]["time_limit"]), job_name="splatad-build-env",
        output_path=f"{log_dir}/{env_name}-%j.out", job_script_path=job_script_path)
    job_id = submit_sbatch(ssh_target, sbatch_arguments)
    print(f"submitted environment build job {job_id} -> {env_name} ({cpus} CPUs, {memory_gb}G)\n"
          f"log: {log_dir}/{env_name}-{job_id}.out")
    return job_id


def submit_run(config: Dict[str, Any], session: str, profile: str, skip_gates: bool,
               now_utc: Optional[datetime.datetime] = None) -> Tuple[str, str]:
    """Run gates, check the environment, size the job from sinfo, and submit it over ssh.

    Assumes ``stage`` was run for this session and the current HEAD.

    Args:
        config: Parsed Slurm config.
        session: Dataset name.
        profile: Profile name under ``profiles``.
        skip_gates: True to skip the Mac-side gates.
        now_utc: Submission time (default: now); injectable for tests.

    Returns:
        (job_id, run_id).

    Raises:
        GateFailure: If a gate fails.
        KeyError: If the profile is unknown.
        RuntimeError/ValueError: On remote failure, a missing environment, or invalid resources.
    """
    # Step 1: gates and config lookups.
    if not skip_gates:
        run_gates(session)
    if profile not in config["profiles"]:
        raise KeyError(f"unknown profile {profile!r}; known: {sorted(config['profiles'])}")
    profile_settings = config["profiles"][profile]
    ssh_target = resolve_ssh_target(config["ssh_host"])
    absolute_stage_root = resolve_remote_stage_root(config, ssh_target)
    git_sha = get_head_sha()
    now_utc = now_utc or datetime.datetime.now(datetime.timezone.utc)
    run_id = build_run_id(session, profile, git_sha, now_utc)
    # Step 2: the run needs the environment its committed recipe defines.
    env_name = environment_name(config, git_sha)
    env_marker = read_environment_marker(ssh_target, absolute_stage_root, env_name)
    if env_marker is None:
        raise RuntimeError(f"environment {env_name} is not built; run `build-env` first")
    # Step 3: node sizes from sinfo.
    cpus, memory_gb = query_node_resources(config, ssh_target)
    # Step 4: run dir first (it holds the job script, and Slurm cannot write the --output log
    # into a missing directory), then the job script exporting the run's variables.
    remote_run_dir = f"{absolute_stage_root}/runs/{run_id}"
    run_checked(ssh_command(ssh_target, ["mkdir", "-p", remote_run_dir]), "remote run dir mkdir")
    job_script_path = f"{remote_run_dir}/job.sbatch"
    write_remote_file(ssh_target, job_script_path, build_job_script(
        build_job_environment(profile_settings, session, run_id, git_sha, absolute_stage_root, env_marker),
        f"{absolute_stage_root}/code/{git_sha}/{RUN_SCRIPT_RELATIVE_PATH}"))
    sbatch_arguments = build_sbatch_command(
        partition=config["partition"], gres=config["gres"], cpus=cpus, memory_gb=memory_gb,
        time_limit=str(profile_settings["time_limit"]), job_name=f"splatad-{session}",
        output_path=f"{remote_run_dir}/slurm-%j.out", job_script_path=job_script_path)
    # Step 5: submit.
    job_id = submit_sbatch(ssh_target, sbatch_arguments)
    print(f"submitted job {job_id}  run {run_id}  ({cpus} CPUs, {memory_gb}G)")
    # Step 6: append to the local ledger (header written once).
    SUBMITTED_LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not SUBMITTED_LEDGER_PATH.exists()
    with SUBMITTED_LEDGER_PATH.open("a") as ledger_file:
        if write_header:
            ledger_file.write("\t".join(LEDGER_COLUMNS) + "\n")
        ledger_file.write("\t".join([run_id, job_id, git_sha, session, profile,
                                     now_utc.strftime(RUN_TIMESTAMP_FORMAT)]) + "\n")
    return job_id, run_id


def show_status(config: Dict[str, Any], job_id: str) -> int:
    """Print squeue and sacct for a job.

    Args:
        config: Parsed Slurm config.
        job_id: Slurm job id.

    Returns:
        0 if sacct succeeded, else its exit code.
    """
    ssh_target = resolve_ssh_target(config["ssh_host"])
    # squeue lists the job only while pending/running; sacct covers finished jobs.
    run_command(ssh_command(ssh_target, ["squeue", "-j", job_id]), capture_output=False)
    return run_command(ssh_command(ssh_target, [
        "sacct", "-j", job_id, "--format=JobID,State,Elapsed,ExitCode,MaxRSS"]),
        capture_output=False).returncode


def fetch_run(config: Dict[str, Any], run_id: str) -> None:
    """Copy a run directory back, keeping only the latest checkpoint.

    Args:
        config: Parsed Slurm config.
        run_id: Run id (directory name under <stage_root>/runs/).

    Raises:
        RuntimeError: If an rsync fails.
    """
    # Step 1: everything except the checkpoint directories.
    ssh_target = resolve_ssh_target(config["ssh_host"])
    remote_run_dir = f"{resolve_remote_stage_root(config, ssh_target)}/runs/{run_id}"
    local_run_dir = PROJECT_ROOT / "runs" / run_id
    local_run_dir.mkdir(parents=True, exist_ok=True)
    run_checked(["rsync", "-a", "--partial", "--exclude", "nerfstudio_models",
                 f"{ssh_target}:{remote_run_dir}/", f"{local_run_dir}/"], "run rsync",
                capture_output=False)
    # Step 2: list checkpoints; step-<zero-padded>.ckpt names sort lexicographically by step.
    listing = run_command(ssh_command(ssh_target, [
        "find", remote_run_dir, "-name", "step-*.ckpt", "-path", "*/nerfstudio_models/*"]))
    checkpoint_paths = sorted(path_text for path_text in listing.stdout.split() if path_text)
    if not checkpoint_paths:
        print("no checkpoints found")
        return
    # Compare by file name (paths share the prefix within one run).
    latest_checkpoint = max(checkpoint_paths, key=lambda path_text: Path(path_text).name)
    relative_path = Path(latest_checkpoint).relative_to(remote_run_dir)
    (local_run_dir / relative_path.parent).mkdir(parents=True, exist_ok=True)
    run_checked(["rsync", "-a", "--partial", f"{ssh_target}:{latest_checkpoint}",
                 f"{local_run_dir / relative_path}"], "checkpoint rsync", capture_output=False)
    print(f"fetched {run_id} to {local_run_dir} (latest checkpoint {relative_path})")


def main(argument_list: Optional[List[str]] = None) -> int:
    """CLI entry point.

    Args:
        argument_list: Command-line arguments (default: sys.argv[1:]).

    Returns:
        0 on success, 1 on any gate/remote/config failure.
    """
    # Step 1: subcommands.
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_SLURM_CONFIG_PATH, help="Slurm YAML")
    subparsers = parser.add_subparsers(dest="command", required=True)
    gates_parser = subparsers.add_parser("gates", help="tests + contract + clean tree")
    gates_parser.add_argument("--session", required=True)
    subparsers.add_parser("build-env", help="build the native environment (once per recipe)")
    stage_parser = subparsers.add_parser("stage", help="copy data and code to the cluster")
    stage_parser.add_argument("--session", required=True)
    submit_parser = subparsers.add_parser("submit", help="sbatch a run")
    submit_parser.add_argument("--session", required=True)
    submit_parser.add_argument("--profile", required=True)
    submit_parser.add_argument("--skip-gates", action="store_true")
    status_parser = subparsers.add_parser("status", help="squeue + sacct for a job")
    status_parser.add_argument("job_id")
    fetch_parser = subparsers.add_parser("fetch", help="copy a run back (latest checkpoint only)")
    fetch_parser.add_argument("run_id")
    arguments = parser.parse_args(argument_list)
    # Step 2: dispatch; every expected failure becomes a message and exit code 1.
    try:
        if arguments.command == "gates":
            run_gates(arguments.session)
            return 0
        config = load_slurm_config(arguments.config)
        if arguments.command == "build-env":
            build_environment(config)
        elif arguments.command == "stage":
            stage_session(config, arguments.session)
        elif arguments.command == "submit":
            submit_run(config, arguments.session, arguments.profile, arguments.skip_gates)
        elif arguments.command == "status":
            return 0 if show_status(config, arguments.job_id) == 0 else 1
        elif arguments.command == "fetch":
            fetch_run(config, arguments.run_id)
        return 0
    except (GateFailure, ValueError, KeyError, RuntimeError, FileNotFoundError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
