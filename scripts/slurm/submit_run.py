"""Stage, submit, monitor and fetch SplatAD runs on the CS Lab Slurm cluster (Mac side).

Usage:
    uv run python scripts/slurm/submit_run.py gates  --session S
    uv run python scripts/slurm/submit_run.py build-env
    uv run python scripts/slurm/submit_run.py stage  --session S
    uv run python scripts/slurm/submit_run.py submit --session S --profile smoke [--skip-gates]
    uv run python scripts/slurm/submit_run.py status <job_id>
    uv run python scripts/slurm/submit_run.py fetch  <run_id>
    uv run python scripts/slurm/submit_run.py clean  [<run_id> ...] [--yes]
    uv run python scripts/slurm/submit_run.py render <run_id> --camera-frames 795,797-800 --lidar-sweeps 801 [--skip-gates]

``render`` re-renders guard frames of a finished run from its checkpoint, with the code at HEAD
(scripts/slurm/render_only.sbatch), into <run dir>/rerenders/<label>/; no training.

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
import json
import math
import os
import re
import shlex
import shutil
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
RENDER_SCRIPT_RELATIVE_PATH = "scripts/slurm/render_only.sbatch"
BUILD_SCRIPT_RELATIVE_PATH = "scripts/slurm/build_env.sbatch"
# Repo paths staged as code/<sha>/: what the environment build installs and tests (the same set
# the Dockerfile COPYs, plus the Dockerfile itself, which build_env.sbatch reads pins from).
STAGED_REPO_PATHS = ("pyproject.toml", "src", "tests", "configs", "scripts",
                     "docker/constraints.txt", "docker/Dockerfile.splatad",
                     ".claude/hooks/readonly_bash_guard.py",
                     # Read by tests/unit/test_render_lidar_script.py (the build's pytest step).
                     "runs/RUN_FOLDER_GLOSSARY.md")
# Hex digits of the recipe hash used in environment names: 12 hex = 48 bits, so two different
# recipes collide with probability ~2**-48, negligible for the handful of envs ever built.
RECIPE_HASH_LENGTH = 12
# build_env.sbatch writes <stage_root>/envs/<env name>.marker as its last step; it names the
# env's path, so a marker means a complete build.
ENV_MARKER_SUFFIX = ".marker"
# Local datasets (one directory per dataset name) and the session configs they were extracted with.
INTERMEDIATE_DATA_ROOT = PROJECT_ROOT / "data" / "intermediate"
SESSION_CONFIG_DIRECTORY = PROJECT_ROOT / "configs" / "sessions"
# Directory of the per-sweep no-return files inside a dataset (dataset/writer.py NO_RETURN_DIRECTORY;
# repeated here because this script runs without importing the package).
NO_RETURN_DIRECTORY_NAME = "lidar_no_return"
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
# Local folder holding fetched runs (one sub-directory per run id) and the ledger.
LOCAL_RUNS_ROOT = PROJECT_ROOT / "runs"
# A run id is one path component (build_run_id: session-profile-timestamp-sha). Anything else is
# refused before a path is built from it, so `clean` can never delete outside a runs/ folder.
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
# job_info.txt lines that only a successful run has (splatad_run.sbatch writes end_utc after
# `split_render_check: match`); a run without both was cancelled, failed or killed.
SUCCESS_SPLIT_CHECK_VALUE = "match"
# File name of the per-job batch script inside the run dir; squeue's %o names it, which maps an
# active job back to its run id.
JOB_SCRIPT_NAME = "job.sbatch"
# Folder inside a run dir that holds one sub-folder per render-only job (<run>/rerenders/<label>/).
# A job script there also marks its run as active, so `clean` never deletes a checkpoint in use.
RERENDERS_DIRECTORY_NAME = "rerenders"
# Render-only job's frame selection, inside its rerender folder (read by the dataparser, T66, and by
# scripts/12_prepare_rerender.py check).
SELECTION_FILE_NAME = "selection.json"
# Prefix of rerender labels: render-<UTC timestamp>-<short sha of the render code>.
RERENDER_LABEL_PREFIX = "render"
# Split name of guard frames in a run's split.json (dataset/run_split.py GUARD_SPLIT_NAME; repeated
# here because this script runs without importing the package). Only these may be re-rendered.
GUARD_SPLIT_NAME = "guard"
# Fewest camera frames a render-only job accepts. Not a tunable: `ns-render dataset` names each output
# relative to os.path.commonpath of the split's image files (nerfstudio render.py ~998-1013); for a
# single file that path is the file itself, so `relative_to` raises (smoke job 222946 crashed so).
MINIMUM_CAMERA_RENDER_FRAMES = 2


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


def load_dataset_session_config(session: str, intermediate_root: Path = INTERMEDIATE_DATA_ROOT,
                                session_config_directory: Path = SESSION_CONFIG_DIRECTORY) -> Tuple[str, Dict[str, Any]]:
    """The session config a dataset was extracted with, and that session's name.

    The session name comes from the dataset's ``manifest.json`` (a dataset directory may be named
    differently, e.g. ``may_chunks0_1_smoke`` was extracted with ``may_short_grass``). Without a
    local manifest the dataset name itself is taken as the session name.

    Args:
        session: Dataset name under ``intermediate_root``.
        intermediate_root: Directory holding the local datasets.
        session_config_directory: Directory holding ``<session name>.yaml``.

    Returns:
        (session name, parsed session config).

    Raises:
        FileNotFoundError: If the session config does not exist.
    """
    manifest_path = Path(intermediate_root) / session / "manifest.json"
    session_name = json.loads(manifest_path.read_text())["session"]["name"] if manifest_path.is_file() else session
    session_config = yaml.safe_load((Path(session_config_directory) / f"{session_name}.yaml").read_text())
    return session_name, session_config


def session_test_block_count(session: str, intermediate_root: Path = INTERMEDIATE_DATA_ROOT,
                             session_config_directory: Path = SESSION_CONFIG_DIRECTORY) -> int:
    """``split.test_block_count`` of the session config a dataset was extracted with.

    Passed to viking-data as ``--test-block-count``: the test set is that many contiguous time
    blocks shared by every sensor (T60), and config.yml records the value for ns-render, scripts/08
    and scripts/09.

    Args:
        session: Dataset name under ``intermediate_root``.
        intermediate_root: Directory holding the local datasets.
        session_config_directory: Directory holding ``<session name>.yaml``.

    Returns:
        The block count K >= 1.

    Raises:
        FileNotFoundError: If the session config does not exist.
        ValueError: If the value is missing or not a positive YAML integer.
    """
    session_name, session_config = load_dataset_session_config(session, intermediate_root, session_config_directory)
    block_count = session_config.get("split", {}).get("test_block_count")
    # bool is an int subclass; refuse it (and floats/strings) instead of coercing.
    if isinstance(block_count, bool) or not isinstance(block_count, int) or block_count < 1:
        raise ValueError(f"configs/sessions/{session_name}.yaml: split.test_block_count must be a positive "
                         f"integer, got {block_count!r}")
    return block_count


def session_test_block_core_fraction(session: str, intermediate_root: Path = INTERMEDIATE_DATA_ROOT,
                                     session_config_directory: Path = SESSION_CONFIG_DIRECTORY) -> float:
    """``split.test_block_core_fraction`` of the session config a dataset was extracted with.

    Passed to viking-data as ``--test-block-core-fraction``: only the central fraction c of each
    T60 test block is the test split (T62), so only those frames are rendered and fetched; the
    rest of each block is guard (neither trained on nor rendered). config.yml records c for
    ns-render, scripts/08 and scripts/09.

    Args:
        session: Dataset name under ``intermediate_root``.
        intermediate_root: Directory holding the local datasets.
        session_config_directory: Directory holding ``<session name>.yaml``.

    Returns:
        c in (0, 1].

    Raises:
        FileNotFoundError: If the session config does not exist.
        ValueError: If the value is missing, not a YAML number, or outside (0, 1].
    """
    session_name, session_config = load_dataset_session_config(session, intermediate_root, session_config_directory)
    core_fraction = session_config.get("split", {}).get("test_block_core_fraction")
    # bool is an int subclass; refuse it (and strings) instead of coercing.
    if isinstance(core_fraction, bool) or not isinstance(core_fraction, (int, float)) or not 0 < core_fraction <= 1:
        raise ValueError(f"configs/sessions/{session_name}.yaml: split.test_block_core_fraction must be a number "
                         f"in (0, 1], got {core_fraction!r}")
    return float(core_fraction)


def session_add_missing_points(session: str, intermediate_root: Path = INTERMEDIATE_DATA_ROOT,
                               session_config_directory: Path = SESSION_CONFIG_DIRECTORY) -> bool:
    """``splatad.add_missing_points`` of the session config a dataset was extracted with.

    The session name comes from the dataset's ``manifest.json`` (a dataset directory may be named
    differently, e.g. ``may_chunks0_1_smoke`` was extracted with ``may_short_grass``). Without a
    local manifest the dataset name itself is taken as the session name.

    Args:
        session: Dataset name under ``intermediate_root``.
        intermediate_root: Directory holding the local datasets.
        session_config_directory: Directory holding ``<session name>.yaml``.

    Returns:
        The flag (True = SplatAD gets the recorded no-returns as 2000 m rays).

    Raises:
        FileNotFoundError: If the session config does not exist.
        ValueError: If the flag is missing or not a YAML boolean.
    """
    # Step 1: the session config the dataset was extracted with.
    session_name, session_config = load_dataset_session_config(session, intermediate_root, session_config_directory)
    # Step 2: the flag from that config; only a real boolean is accepted (no "yes"/1).
    flag = session_config.get("splatad", {}).get("add_missing_points")
    if not isinstance(flag, bool):
        raise ValueError(f"configs/sessions/{session_name}.yaml: splatad.add_missing_points must be true/false, got {flag!r}")
    return flag


def dataset_train_fraction(session: str, intermediate_root: Path = INTERMEDIATE_DATA_ROOT) -> float:
    """``session.train_fraction`` of the local dataset's manifest, the fraction the dataparser uses.

    Passed to ns-train as ``--train-split-fraction`` so ``config.yml`` records the real value:
    nerfstudio saves config.yml before the dataparser runs, so without the flag it holds the class
    default (0.5) even though the dataparser then splits with the manifest's value.

    Args:
        session: Dataset name under ``intermediate_root``.
        intermediate_root: Directory holding the local datasets.

    Returns:
        The fraction in (0, 1].

    Raises:
        FileNotFoundError: If the dataset's ``manifest.json`` is missing.
        ValueError: If the fraction is not a number in (0, 1].
    """
    manifest_path = Path(intermediate_root) / session / "manifest.json"
    train_fraction = json.loads(manifest_path.read_text())["session"]["train_fraction"]
    # bool is an int subclass; refuse it like any other non-number.
    if isinstance(train_fraction, bool) or not isinstance(train_fraction, (int, float)) or not 0 < train_fraction <= 1:
        raise ValueError(f"{manifest_path}: session.train_fraction must be a number in (0, 1], got {train_fraction!r}")
    return float(train_fraction)


def require_no_return_sidecar(dataset_dir: Path, add_missing_points: bool) -> None:
    """Refuse a dataset that lacks the no-return sidecar when the run needs it.

    Without this check the job would queue for a GPU and then fail in the dataparser.

    Args:
        dataset_dir: Local dataset directory.
        add_missing_points: The session's flag.

    Raises:
        FileNotFoundError: If the flag is set and ``lidar_no_return/`` is missing.
    """
    if add_missing_points and not (Path(dataset_dir) / NO_RETURN_DIRECTORY_NAME).is_dir():
        raise FileNotFoundError(
            f"{dataset_dir} has no {NO_RETURN_DIRECTORY_NAME}/ but add_missing_points is true: "
            "re-extract it with scripts/03_extract.py --force"
        )


def profile_eval_cadence(profile_settings: Dict[str, Any]) -> Tuple[int, int]:
    """Minimum eval count of a profile and the eval interval that guarantees it (T63).

    Args:
        profile_settings: The chosen ``profiles.<name>`` mapping.

    Returns:
        ``(minimum_eval_points, steps_per_eval_all_images)``, where the interval is
        k = (max_num_iterations - 1) // minimum_eval_points.

    Raises:
        ValueError: If ``minimum_eval_points`` is missing or not an int (bool refused), or if
            T63 refuses because the run is too short for that many evals.
    """
    # Step 1: the minimum must be explicit, as the seed is; bool is an int subclass, so refuse it.
    minimum_eval_points = profile_settings.get("minimum_eval_points")
    if isinstance(minimum_eval_points, bool) or not isinstance(minimum_eval_points, int):
        raise ValueError(f"profile needs an integer 'minimum_eval_points' (got {minimum_eval_points!r})")
    # Step 2: T63 derives the interval (and raises ValueError when N - 1 < P).
    # Imported lazily so the other subcommands stay usable without the viking_splatad package.
    from viking_splatad.eval.training_curves import eval_interval_for_minimum_points
    steps_per_eval_all_images = eval_interval_for_minimum_points(
        int(profile_settings["max_num_iterations"]), minimum_eval_points)
    return minimum_eval_points, steps_per_eval_all_images


def build_job_environment(profile_settings: Dict[str, Any], session: str, run_id: str,
                          git_sha: str, absolute_stage_root: str,
                          env_marker: Dict[str, str],
                          add_missing_points: Optional[bool] = None,
                          train_split_fraction: Optional[float] = None,
                          test_block_count: Optional[int] = None,
                          test_block_core_fraction: Optional[float] = None,
                          minimum_eval_points: Optional[int] = None,
                          steps_per_eval_all_images: Optional[int] = None) -> Dict[str, str]:
    """Build the variables exported to splatad_run.sbatch (insertion order is kept).

    Args:
        profile_settings: The chosen ``profiles.<name>`` mapping.
        session: Session name.
        run_id: Run id.
        git_sha: Full commit SHA.
        absolute_stage_root: Resolved stage root on the cluster.
        env_marker: Parsed marker of the environment the run activates (its path).
        add_missing_points: The session's flag (``session_add_missing_points``), exported as
            ADD_MISSING_POINTS. ``submit_run`` always passes it; None leaves it out, and then
            splatad_run.sbatch refuses to start.
        train_split_fraction: The dataset's train fraction (``dataset_train_fraction``), exported
            as TRAIN_SPLIT_FRACTION. Same rule as ``add_missing_points``: None leaves it out.
        test_block_count: The session's test block count (``session_test_block_count``), exported
            as TEST_BLOCK_COUNT. Same rule as ``add_missing_points``: None leaves it out.
        test_block_core_fraction: The session's core fraction (``session_test_block_core_fraction``),
            exported as TEST_BLOCK_CORE_FRACTION. Same rule as ``add_missing_points``: None leaves it out.
        minimum_eval_points: The profile's minimum eval count (``profile_eval_cadence``), exported as
            MINIMUM_EVAL_POINTS. Same rule as ``add_missing_points``: None leaves it out.
        steps_per_eval_all_images: The derived eval interval (``profile_eval_cadence``), exported as
            STEPS_PER_EVAL_ALL_IMAGES. Same rule: None leaves it out.

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
    job_environment = {
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
    # tyro (ns-train) parses booleans as True/False, which is what str(bool) gives.
    if add_missing_points is not None:
        job_environment["ADD_MISSING_POINTS"] = str(bool(add_missing_points))
    # repr keeps every digit, so config.yml, job_info.txt and the manifest hold the same float.
    if train_split_fraction is not None:
        job_environment["TRAIN_SPLIT_FRACTION"] = repr(float(train_split_fraction))
    if test_block_count is not None:
        job_environment["TEST_BLOCK_COUNT"] = str(int(test_block_count))
    # repr keeps every digit, so config.yml and job_info.txt hold the same float.
    if test_block_core_fraction is not None:
        job_environment["TEST_BLOCK_CORE_FRACTION"] = repr(float(test_block_core_fraction))
    if minimum_eval_points is not None:
        job_environment["MINIMUM_EVAL_POINTS"] = str(int(minimum_eval_points))
    if steps_per_eval_all_images is not None:
        job_environment["STEPS_PER_EVAL_ALL_IMAGES"] = str(int(steps_per_eval_all_images))
    return job_environment


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


def parse_frame_index_list(frame_list_text: str) -> List[int]:
    """Parse a frame index list such as ``795,797-800`` (single indices and inclusive ranges).

    Args:
        frame_list_text: Comma-separated items, each ``<n>`` or ``<first>-<last>`` with first <= last.

    Returns:
        Sorted, distinct frame indices (``795,797-800`` -> [795, 797, 798, 799, 800]).

    Raises:
        ValueError: If the text is empty, an item is empty or not a non-negative integer, or a
            range is reversed.
    """
    # Step 1: one item per comma; blank items (",," or a trailing comma) are a typo, not "nothing".
    item_texts = [item_text.strip() for item_text in frame_list_text.split(",")]
    if not frame_list_text.strip() or any(not item_text for item_text in item_texts):
        raise ValueError(f"frame list {frame_list_text!r} is empty or has an empty item")
    frame_indices = set()
    for item_text in item_texts:
        # Step 2: "a-b" is an inclusive range; digits only, so "-3" (negative) fails here too.
        first_text, separator, last_text = item_text.partition("-")
        if not first_text.isdigit() or (separator and not last_text.isdigit()):
            raise ValueError(f"frame list item {item_text!r} is not <n> or <first>-<last> (non-negative integers)")
        first_index = int(first_text)
        last_index = int(last_text) if separator else first_index
        if last_index < first_index:
            raise ValueError(f"frame range {item_text!r} is reversed")
        frame_indices.update(range(first_index, last_index + 1))
    # Step 3: sorted and distinct (overlapping items are merged).
    return sorted(frame_indices)


def validate_render_selection(split_record: Dict[str, Any], camera_frame_indices: Sequence[int],
                              lidar_frame_indices: Sequence[int]) -> Dict[str, List[str]]:
    """Check a render-only request against a finished run's split.json and name the selected files.

    Only guard frames may be requested: frames inside a T60 test block but outside its T62 core, which
    the run never trained on and never rendered (split ``guard`` in split.json, scripts/09).

    Args:
        split_record: The run's parsed split.json (``frames`` entries: sensor, kind, split,
            frame_index, timestamp_ns, time_s, file).
        camera_frame_indices: Camera frame indices to render (at least MINIMUM_CAMERA_RENDER_FRAMES).
        lidar_frame_indices: Lidar sweep indices to render (at least one).

    Returns:
        Sensor kind -> split.json ``file`` of every selected frame, in frame-index order.

    Raises:
        ValueError: If a list is empty or repeats an index, the camera list is shorter than
            MINIMUM_CAMERA_RENDER_FRAMES, or an index is not a guard frame of that kind (each
            offender is listed with its actual split, or "absent" when the record has no such frame).
    """
    requested_by_kind = {"camera": [int(frame_index) for frame_index in camera_frame_indices],
                         "lidar": [int(frame_index) for frame_index in lidar_frame_indices]}
    # Step 1: list sizes (both kinds are always rendered; ns-render needs >= 2 camera frames).
    for sensor_kind, requested_indices in requested_by_kind.items():
        if not requested_indices:
            raise ValueError(f"no {sensor_kind} frames requested: both --camera-frames and --lidar-sweeps are required")
        if len(set(requested_indices)) != len(requested_indices):
            raise ValueError(f"{sensor_kind} frame list repeats an index: {requested_indices}")
    if len(requested_by_kind["camera"]) < MINIMUM_CAMERA_RENDER_FRAMES:
        raise ValueError(
            f"request at least {MINIMUM_CAMERA_RENDER_FRAMES} camera frames: ns-render dataset names outputs "
            "relative to the common path of the split's images, which for one image is the file itself "
            "(render.py relative_to raises; smoke job 222946)")
    # Step 2: each frame's records, per kind (several sensors of one kind share an index: keep all).
    records_by_kind_and_index: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for frame_record in split_record["frames"]:
        records_by_kind_and_index.setdefault((frame_record["kind"], int(frame_record["frame_index"])),
                                             []).append(frame_record)
    # Step 3: every requested index must be guard for its kind; collect all offenders first.
    offenders, selected_files = [], {}
    for sensor_kind, requested_indices in requested_by_kind.items():
        selected_files[sensor_kind] = []
        for frame_index in sorted(requested_indices):
            frame_records = records_by_kind_and_index.get((sensor_kind, frame_index), [])
            actual_splits = sorted({frame_record["split"] for frame_record in frame_records}) or ["absent"]
            if actual_splits != [GUARD_SPLIT_NAME]:
                offenders.append(f"{sensor_kind} {frame_index} ({'/'.join(actual_splits)})")
                continue
            selected_files[sensor_kind] += [frame_record["file"] for frame_record in frame_records]
    if offenders:
        raise ValueError("only guard frames of the run's split.json can be re-rendered; not guard: "
                         + ", ".join(offenders))
    return selected_files


def build_rerender_label(git_sha: str, now_utc: datetime.datetime) -> str:
    """Name of one render-only job's folder: ``render-<YYYYmmdd-HHMMSS>-<sha7>``.

    Args:
        git_sha: Full SHA of the render code.
        now_utc: Submission time (UTC).

    Returns:
        The label (one path component).
    """
    return f"{RERENDER_LABEL_PREFIX}-{now_utc.strftime(RUN_TIMESTAMP_FORMAT)}-{git_sha[:SHORT_SHA_LENGTH]}"


def build_render_job_environment(run_id: str, rerender_label: str, absolute_stage_root: str,
                                 run_job_info: Dict[str, str], git_sha: str,
                                 env_marker: Dict[str, str]) -> Dict[str, str]:
    """Build the variables exported to render_only.sbatch (insertion order is kept).

    Args:
        run_id: The finished run whose checkpoint is rendered.
        rerender_label: ``build_rerender_label`` output.
        absolute_stage_root: Resolved stage root on the cluster.
        run_job_info: The run's parsed job_info.txt (SESSION, GIT_SHA, RENDER_DOWNSCALE_FACTOR).
        git_sha: Full SHA of the render code (HEAD, staged as code/<sha>).
        env_marker: Parsed marker of the environment the job activates.

    Returns:
        Mapping variable name -> string value.

    Raises:
        KeyError: If the run's job_info.txt lacks SESSION, GIT_SHA or RENDER_DOWNSCALE_FACTOR.
    """
    return {
        "RUN_ID": run_id,
        "RERENDER_LABEL": rerender_label,
        "STAGE_ROOT": absolute_stage_root,
        "SESSION": run_job_info["SESSION"],
        # The code that renders (HEAD) and the code that trained the checkpoint can differ; both are recorded.
        "GIT_SHA": git_sha,
        "TRAIN_GIT_SHA": run_job_info["GIT_SHA"],
        "CODE_DIR": f"{absolute_stage_root}/code/{git_sha}",
        "ENV_PREFIX": env_marker["env_prefix"],
        # Same camera resolution as the run's own renders.
        "RENDER_DOWNSCALE_FACTOR": run_job_info["RENDER_DOWNSCALE_FACTOR"],
    }


def build_render_selection_record(camera_frame_indices: Sequence[int], lidar_frame_indices: Sequence[int],
                                  selected_files: Dict[str, List[str]], split_record: Dict[str, Any],
                                  now_utc: datetime.datetime) -> Dict[str, Any]:
    """Contents of a render-only job's selection.json.

    Args:
        camera_frame_indices, lidar_frame_indices: Validated requests.
        selected_files: ``validate_render_selection`` output.
        split_record: The run's split.json (its ``membership_sha256`` ties the selection to that split).
        now_utc: Submission time (UTC).

    Returns:
        Mapping with keys camera, lidar (sorted indices), files, source_split_membership_sha256,
        requested_utc.
    """
    return {
        "camera": sorted(int(frame_index) for frame_index in camera_frame_indices),
        "lidar": sorted(int(frame_index) for frame_index in lidar_frame_indices),
        "files": selected_files,
        "source_split_membership_sha256": split_record["membership_sha256"],
        "requested_utc": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


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
        FileNotFoundError: If the dataset is missing locally, or lacks the no-return sidecar that
            its session's add_missing_points needs.
        RuntimeError: If a remote command fails.
    """
    # Step 1: resolve paths; check the dataset holds what the run will read.
    ssh_target = resolve_ssh_target(config["ssh_host"])
    absolute_stage_root = resolve_remote_stage_root(config, ssh_target)
    dataset_dir = PROJECT_ROOT / "data" / "intermediate" / session
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"dataset {dataset_dir} not found")
    require_no_return_sidecar(dataset_dir, session_add_missing_points(session))
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
    # The eval cadence is checked before any ssh/sbatch, so a profile too short for its minimum fails early.
    minimum_eval_points, steps_per_eval_all_images = profile_eval_cadence(profile_settings)
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
        build_job_environment(profile_settings, session, run_id, git_sha, absolute_stage_root, env_marker,
                              add_missing_points=session_add_missing_points(session),
                              train_split_fraction=dataset_train_fraction(session),
                              test_block_count=session_test_block_count(session),
                              test_block_core_fraction=session_test_block_core_fraction(session),
                              minimum_eval_points=minimum_eval_points,
                              steps_per_eval_all_images=steps_per_eval_all_images),
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


def read_remote_text(ssh_target: str, remote_path: str, description: str) -> str:
    """Read one text file on the cluster with ``cat`` over ssh.

    Args:
        ssh_target: ``host`` or ``user@host``.
        remote_path: Absolute file path.
        description: Label for the error message.

    Returns:
        The file contents.

    Raises:
        RuntimeError: If the file cannot be read (missing run, or not yet written).
    """
    return run_checked(ssh_command(ssh_target, ["cat", remote_path]), description).stdout


def submit_render(config: Dict[str, Any], run_id: str, camera_frames_text: str, lidar_sweeps_text: str,
                  skip_gates: bool, now_utc: Optional[datetime.datetime] = None) -> Tuple[str, str]:
    """Submit a render-only job for guard frames of a finished run (render_only.sbatch).

    The job renders with the code at HEAD (which must be staged) from the run's own config.yml and
    checkpoint, into ``<run dir>/rerenders/<label>/``. It is not added to runs/submitted.tsv: that
    ledger lists training runs, and ``clean`` deletes ledger rows that have no run folder.

    Args:
        config: Parsed Slurm config (``render_only.time_limit``).
        run_id: Finished run under ``<stage root>/runs/``.
        camera_frames_text: ``--camera-frames`` list (``parse_frame_index_list`` syntax).
        lidar_sweeps_text: ``--lidar-sweeps`` list.
        skip_gates: True to skip the Mac-side gates.
        now_utc: Submission time (default: now); injectable for tests.

    Returns:
        (job_id, rerender label).

    Raises:
        GateFailure: If a gate fails.
        ValueError: On a malformed run id or frame list, a selection that is not guard frames, an
            unsuccessful run, or a config without ``render_only.time_limit``.
        RuntimeError: If the run's files cannot be read, the code is not staged, the environment is
            not built, or a remote command fails.
    """
    # Step 1: local checks first (no ssh for a typo).
    if not RUN_ID_PATTERN.match(run_id):
        raise ValueError(f"malformed run id {run_id!r}")
    camera_frame_indices = parse_frame_index_list(camera_frames_text)
    lidar_frame_indices = parse_frame_index_list(lidar_sweeps_text)
    time_limit = (config.get("render_only") or {}).get("time_limit")
    if not time_limit:
        raise ValueError("configs/slurm.yaml: render_only.time_limit is not set")
    # Step 2: the run's own record decides: only a successful run has a verified split.json.
    ssh_target = resolve_ssh_target(config["ssh_host"])
    absolute_stage_root = resolve_remote_stage_root(config, ssh_target)
    remote_run_dir = f"{absolute_stage_root}/runs/{run_id}"
    run_job_info = parse_job_info_text(read_remote_text(ssh_target, f"{remote_run_dir}/job_info.txt",
                                                        f"read job_info.txt of {run_id}"))
    if not run_succeeded(run_job_info):
        raise ValueError(f"run {run_id} did not succeed (no end_utc + split_render_check: match); "
                         "only finished runs can be re-rendered")
    session = run_job_info["SESSION"]
    # Step 3: gates (clean tree: the render code must be a commit; contract on the run's dataset).
    if not skip_gates:
        run_gates(session)
    split_record = json.loads(read_remote_text(ssh_target, f"{remote_run_dir}/split.json",
                                               f"read split.json of {run_id}"))
    selected_files = validate_render_selection(split_record, camera_frame_indices, lidar_frame_indices)
    # Step 4: the render code (HEAD) must be staged, and its environment built.
    git_sha = get_head_sha()
    remote_code_dir = f"{absolute_stage_root}/code/{git_sha}"
    if run_command(ssh_command(ssh_target, ["test", "-d", remote_code_dir])).returncode != 0:
        raise RuntimeError(f"code {git_sha} is not staged; run `stage --session {session}` first")
    env_name = environment_name(config, git_sha)
    env_marker = read_environment_marker(ssh_target, absolute_stage_root, env_name)
    if env_marker is None:
        raise RuntimeError(f"environment {env_name} is not built; run `build-env` first")
    cpus, memory_gb = query_node_resources(config, ssh_target)
    # Step 5: rerender folder (Slurm cannot write --output into a missing dir), selection, job script.
    now_utc = now_utc or datetime.datetime.now(datetime.timezone.utc)
    rerender_label = build_rerender_label(git_sha, now_utc)
    remote_rerender_dir = f"{remote_run_dir}/{RERENDERS_DIRECTORY_NAME}/{rerender_label}"
    run_checked(ssh_command(ssh_target, ["mkdir", "-p", remote_rerender_dir]), "remote rerender dir mkdir")
    selection_record = build_render_selection_record(camera_frame_indices, lidar_frame_indices, selected_files,
                                                     split_record, now_utc)
    write_remote_file(ssh_target, f"{remote_rerender_dir}/{SELECTION_FILE_NAME}",
                      json.dumps(selection_record, indent=2) + "\n")
    job_script_path = f"{remote_rerender_dir}/{JOB_SCRIPT_NAME}"
    write_remote_file(ssh_target, job_script_path, build_job_script(
        build_render_job_environment(run_id, rerender_label, absolute_stage_root, run_job_info, git_sha,
                                     env_marker),
        f"{remote_code_dir}/{RENDER_SCRIPT_RELATIVE_PATH}"))
    # Step 6: submit.
    sbatch_arguments = build_sbatch_command(
        partition=config["partition"], gres=config["gres"], cpus=cpus, memory_gb=memory_gb,
        time_limit=str(time_limit), job_name=f"splatad-render-{session}",
        output_path=f"{remote_rerender_dir}/slurm-%j.out", job_script_path=job_script_path)
    job_id = submit_sbatch(ssh_target, sbatch_arguments)
    print(f"submitted render job {job_id} run {run_id} label {rerender_label}")
    return job_id, rerender_label


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


def parse_job_info_text(job_info_text: str) -> Dict[str, str]:
    """Parse ``key: value`` lines of a job_info.txt (indented and value-less lines are skipped).

    Args:
        job_info_text: File contents.

    Returns:
        key -> value (the last occurrence wins).
    """
    job_info = {}
    for line in job_info_text.splitlines():
        key, separator, value = line.partition(": ")
        if separator and key and not key.startswith(" "):
            job_info[key] = value.strip()
    return job_info


def run_succeeded(job_info: Optional[Dict[str, str]]) -> bool:
    """True iff job_info.txt proves a successful run (split check matched, then end_utc written).

    Args:
        job_info: Parsed job_info.txt, or None when the run has none.

    Returns:
        Whether the run finished every step.
    """
    return bool(job_info) and "end_utc" in job_info and job_info.get("split_render_check") == SUCCESS_SPLIT_CHECK_VALUE


def parse_remote_run_listing(listing_text: str) -> Tuple[Dict[str, Optional[Dict[str, str]]], set]:
    """Parse the output of the remote listing script in ``list_remote_runs``.

    Format: ``RUN<TAB><run_id>`` opens a run, followed by its job_info.txt lines between
    ``INFO_BEGIN`` and ``INFO_END`` (absent if it has none); ``ACTIVE<TAB><job script path>`` is one
    pending or running job of the user: a run's own job (``<runs>/<run_id>/job.sbatch``) or a
    render-only job on its checkpoint (``<runs>/<run_id>/rerenders/<label>/job.sbatch``).

    Args:
        listing_text: Remote stdout.

    Returns:
        (runs, active run ids): run id -> parsed job_info (None if missing); the run ids with a
        queued or running job: their own, or a render-only job reading their checkpoint.
    """
    remote_runs: Dict[str, Optional[Dict[str, str]]] = {}
    active_run_ids = set()
    current_run_id, info_lines, inside_info = None, [], False
    for line in listing_text.splitlines():
        if line.startswith("RUN\t"):
            current_run_id = line.split("\t", 1)[1]
            remote_runs[current_run_id] = None
        elif line == "INFO_BEGIN":
            inside_info, info_lines = True, []
        elif line == "INFO_END":
            inside_info = False
            remote_runs[current_run_id] = parse_job_info_text("\n".join(info_lines))
        elif inside_info:
            info_lines.append(line)
        elif line.startswith("ACTIVE\t"):
            # The job script is <stage>/runs/<run_id>/job.sbatch: its parent folder is the run id.
            script_path = Path(line.split("\t", 1)[1].strip())
            if script_path.name == JOB_SCRIPT_NAME:
                active_run_ids.add(script_path.parent.name)
                # A render-only job (<run_id>/rerenders/<label>/job.sbatch) uses <run_id>'s
                # checkpoint, so that run is active too.
                if script_path.parent.parent.name == RERENDERS_DIRECTORY_NAME:
                    active_run_ids.add(script_path.parent.parent.parent.name)
    return remote_runs, active_run_ids


def list_remote_runs(ssh_target: str, absolute_stage_root: str) -> Tuple[Dict[str, Optional[Dict[str, str]]], set]:
    """Every run folder on the cluster with its job_info.txt, and the runs with a queued/running job.

    One ssh call. squeue's default state filter lists only pending and running jobs, and its
    ``%o`` field is the batch script path (``<run dir>/job.sbatch``), so a job that has not
    written job_info.txt yet (still pending) is still recognised as active.

    Args:
        ssh_target: ``host`` or ``user@host``.
        absolute_stage_root: Resolved stage root on the cluster.

    Returns:
        As ``parse_remote_run_listing``.

    Raises:
        RuntimeError: If the remote command fails (e.g. squeue unavailable), since deleting
            without knowing which jobs are active is unsafe.
    """
    listing_script = (
        'if [ -d "$1/runs" ]; then for run_dir in "$1"/runs/*/; do '
        '[ -d "$run_dir" ] || continue; run_id=$(basename "$run_dir"); printf "RUN\\t%s\\n" "$run_id"; '
        'if [ -f "$run_dir/job_info.txt" ]; then echo INFO_BEGIN; cat "$run_dir/job_info.txt"; echo; echo INFO_END; fi; '
        'done; fi; '
        'squeue -h -u "$(id -un)" -o "%o" | sed "s/^/ACTIVE\\t/"; exit "${PIPESTATUS[0]:-0}"'
    )
    completed = run_checked(ssh_command(ssh_target, ["bash", "-c", listing_script, "bash", absolute_stage_root]),
                            "list remote runs")
    return parse_remote_run_listing(completed.stdout)


def list_local_runs(local_runs_root: Path) -> Dict[str, Optional[Dict[str, str]]]:
    """Every fetched run folder on the Mac with its job_info.txt.

    Args:
        local_runs_root: The local runs/ folder.

    Returns:
        run id -> parsed job_info (None if missing).
    """
    local_runs: Dict[str, Optional[Dict[str, str]]] = {}
    if not local_runs_root.is_dir():
        return local_runs
    for run_dir in sorted(local_runs_root.iterdir()):
        if not run_dir.is_dir() or not RUN_ID_PATTERN.match(run_dir.name):
            continue
        job_info_path = run_dir / "job_info.txt"
        local_runs[run_dir.name] = parse_job_info_text(job_info_path.read_text()) if job_info_path.is_file() else None
    return local_runs


def read_ledger(ledger_path: Path) -> List[Dict[str, str]]:
    """Rows of the local submission ledger (runs/submitted.tsv).

    Args:
        ledger_path: Ledger file.

    Returns:
        One dict per row keyed by LEDGER_COLUMNS ([] when the file does not exist).
    """
    if not ledger_path.is_file():
        return []
    data_lines = [line for line in ledger_path.read_text().splitlines()[1:] if line.strip()]
    return [dict(zip(LEDGER_COLUMNS, line.split("\t"))) for line in data_lines]


def write_ledger(ledger_path: Path, ledger_rows: List[Dict[str, str]]) -> None:
    """Rewrite the ledger with these rows (header always kept).

    Args:
        ledger_path: Ledger file.
        ledger_rows: Rows keyed by LEDGER_COLUMNS.
    """
    lines = ["\t".join(LEDGER_COLUMNS)] + ["\t".join(row.get(column, "") for column in LEDGER_COLUMNS)
                                          for row in ledger_rows]
    ledger_path.write_text("\n".join(lines) + "\n")


def plan_cleanup(remote_runs: Dict[str, Optional[Dict[str, str]]], active_run_ids: set,
                 local_runs: Dict[str, Optional[Dict[str, str]]], ledger_rows: List[Dict[str, str]],
                 named_run_ids: Sequence[str]) -> List[Dict[str, Any]]:
    """Decide, per run, whether ``clean`` deletes it (pure: no I/O).

    Rules, in order:
        1. a queued or running job's run is never deleted (``scancel`` it first);
        2. a run named on the command line is deleted, successful or not;
        3. a run with no proof of success (no ``end_utc`` + ``split_render_check: match`` in its
           job_info.txt, cluster copy preferred over the fetched one) is deleted: it was
           cancelled (by the user or Slurm: time/memory limit), failed, or never started;
        4. a successful run is kept.
    Every place a deleted run lives is cleaned: the cluster run dir, the fetched copy, the ledger row.

    Args:
        remote_runs: From ``list_remote_runs``.
        active_run_ids: From ``list_remote_runs``.
        local_runs: From ``list_local_runs``.
        ledger_rows: From ``read_ledger``.
        named_run_ids: Runs the user asked to delete.

    Returns:
        One dict per known run, sorted by run id: run_id, delete (bool), reason, on_cluster,
        on_mac, in_ledger.

    Raises:
        ValueError: If a named run id is malformed, unknown everywhere, or active.
    """
    # Step 1: validate the named runs before deciding anything.
    ledger_run_ids = {row["run_id"] for row in ledger_rows}
    known_run_ids = set(remote_runs) | set(local_runs) | ledger_run_ids
    for run_id in named_run_ids:
        if not RUN_ID_PATTERN.match(run_id):
            raise ValueError(f"malformed run id {run_id!r}")
        if run_id not in known_run_ids:
            raise ValueError(f"run {run_id} is not on the cluster, on the Mac or in the ledger")
        if run_id in active_run_ids:
            raise ValueError(f"run {run_id} still has a queued or running job; scancel it first")
    # Step 2: one decision per run, wherever it lives.
    plan = []
    for run_id in sorted(known_run_ids):
        # The cluster copy is the job's own record; the fetched copy is used when the cluster one is gone.
        job_info = remote_runs.get(run_id) if run_id in remote_runs else local_runs.get(run_id)
        if run_id in active_run_ids:
            delete, reason = False, "job queued or running"
        elif run_id in named_run_ids:
            delete, reason = True, "named"
        elif not run_succeeded(job_info):
            delete, reason = True, "cancelled, failed or incomplete (no end_utc + split_render_check: match)"
        else:
            delete, reason = False, "successful"
        plan.append({"run_id": run_id, "delete": delete, "reason": reason, "on_cluster": run_id in remote_runs,
                     "on_mac": run_id in local_runs, "in_ledger": run_id in ledger_run_ids})
    return plan


def clean_runs(config: Dict[str, Any], named_run_ids: Sequence[str], confirmed: bool,
               local_runs_root: Path = LOCAL_RUNS_ROOT, ledger_path: Path = SUBMITTED_LEDGER_PATH) -> List[Dict[str, Any]]:
    """Delete every artifact of cancelled/failed runs and of the named runs (dry run unless confirmed).

    Args:
        config: Parsed Slurm config.
        named_run_ids: Runs to delete even if successful.
        confirmed: False = print the plan only; True = delete.
        local_runs_root: Local runs/ folder.
        ledger_path: Local ledger.

    Returns:
        The plan from ``plan_cleanup``.

    Raises:
        ValueError: As ``plan_cleanup``.
        RuntimeError: If a remote listing or deletion fails.
    """
    # Step 1: everything known about every run, then the plan.
    ssh_target = resolve_ssh_target(config["ssh_host"])
    absolute_stage_root = resolve_remote_stage_root(config, ssh_target)
    remote_runs, active_run_ids = list_remote_runs(ssh_target, absolute_stage_root)
    ledger_rows = read_ledger(ledger_path)
    plan = plan_cleanup(remote_runs, active_run_ids, list_local_runs(local_runs_root), ledger_rows, named_run_ids)
    for decision in plan:
        places = [place for place, present in (("cluster", decision["on_cluster"]), ("mac", decision["on_mac"]),
                                               ("ledger", decision["in_ledger"])) if present]
        action = "DELETE" if decision["delete"] else "keep  "
        print(f"{action} {decision['run_id']}  [{', '.join(places)}]  {decision['reason']}")
    doomed = [decision for decision in plan if decision["delete"]]
    if not confirmed:
        print(f"dry run: {len(doomed)} run(s) would be deleted; re-run with --yes to delete")
        return plan
    # Step 2: delete. Run ids passed RUN_ID_PATTERN (one path component), so each path stays
    # inside a runs/ folder.
    for decision in doomed:
        run_id = decision["run_id"]
        if not RUN_ID_PATTERN.match(run_id):
            raise ValueError(f"refusing to delete malformed run id {run_id!r}")
        if decision["on_cluster"]:
            run_checked(ssh_command(ssh_target, ["rm", "-rf", "--", f"{absolute_stage_root}/runs/{run_id}"]),
                        f"delete cluster run {run_id}")
        if decision["on_mac"]:
            shutil.rmtree(local_runs_root / run_id)
    # Step 3: ledger rows of the deleted runs.
    doomed_run_ids = {decision["run_id"] for decision in doomed}
    if ledger_path.is_file():
        write_ledger(ledger_path, [row for row in ledger_rows if row["run_id"] not in doomed_run_ids])
    print(f"deleted {len(doomed)} run(s)")
    return plan


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
    clean_parser = subparsers.add_parser(
        "clean", help="delete cancelled/failed runs and the named runs (cluster dir, fetched copy, ledger row)")
    clean_parser.add_argument("run_ids", nargs="*", help="runs to delete even if successful")
    clean_parser.add_argument("--yes", action="store_true", help="delete; without it only the plan is printed")
    render_parser = subparsers.add_parser(
        "render", help="render guard frames of a finished run from its checkpoint (no training)")
    render_parser.add_argument("run_id")
    render_parser.add_argument("--camera-frames", required=True,
                               help="guard camera frame indices, e.g. 795,797-800 (at least 2)")
    render_parser.add_argument("--lidar-sweeps", required=True, help="guard lidar sweep indices, e.g. 801-803")
    render_parser.add_argument("--skip-gates", action="store_true")
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
        elif arguments.command == "clean":
            clean_runs(config, arguments.run_ids, arguments.yes)
        elif arguments.command == "render":
            submit_render(config, arguments.run_id, arguments.camera_frames, arguments.lidar_sweeps,
                          arguments.skip_gates)
        return 0
    except (GateFailure, ValueError, KeyError, RuntimeError, FileNotFoundError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
