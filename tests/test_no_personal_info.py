"""Meta-test: no tracked file contains the running user's own identifiers.

The repo is public and shared by several people, each with their own CS Lab account. Logins
stay on each user's machine (``~/.ssh/config`` or ``VIKING_SLURM_USER``). This test reads the
identifiers of whoever runs it at runtime (home path, local login, cluster username, git email),
so the test itself stores no personal information and catches leaks on every contributor's machine.
"""

import getpass
import os
import re
import subprocess
from pathlib import Path

import pytest
import yaml

# Repository root: this file lives in <root>/tests/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Shared Slurm config: its ssh_host is the login node whose ssh-config User is the cluster login.
SLURM_CONFIG_PATH = PROJECT_ROOT / "configs" / "slurm.yaml"

# Same variable submit_run.py reads for the cluster username (SLURM_USER_ENVIRONMENT_VARIABLE).
SLURM_USER_ENVIRONMENT_VARIABLE = "VIKING_SLURM_USER"

# GitHub's privacy-preserving commit address domain: such an email is public by design.
GITHUB_NOREPLY_EMAIL_SUFFIX = "@users.noreply.github.com"

# Unavoidable threshold: a login of 1-3 characters (e.g. "kn", "dev") would match ordinary words
# and code tokens, so such identifiers are not searched; real CS Lab logins are longer.
MINIMUM_SEARCHABLE_IDENTIFIER_LENGTH = 4


def read_git_output(*git_arguments: str) -> str:
    """Run a read-only git command in the repo root.

    Args:
        git_arguments: arguments after ``git``.

    Returns:
        Stripped stdout, or "" when git fails (e.g. the key is unset).
    """
    completed = subprocess.run(
        ["git", *git_arguments], cwd=PROJECT_ROOT, capture_output=True, text=True, check=False
    )
    return completed.stdout.strip() if completed.returncode == 0 else ""


def read_ssh_config_cluster_user() -> str:
    """Resolve the cluster login from the user's own ssh config, without connecting.

    ``ssh -G <host>`` only evaluates ``~/.ssh/config`` and prints the effective options
    (one ``key value`` per line); it opens no network connection.

    Returns:
        The ``user`` ssh would log in as on ``ssh_host``, or "" if ssh or the config is unavailable.
    """
    # Step 1: the host comes from the shared config, never hardcoded here.
    if not SLURM_CONFIG_PATH.is_file():
        return ""
    ssh_host = yaml.safe_load(SLURM_CONFIG_PATH.read_text()).get("ssh_host", "")
    if not ssh_host:
        return ""
    # Step 2: evaluate the ssh config for that host; stdin closed so ssh never prompts.
    try:
        completed = subprocess.run(
            ["ssh", "-G", ssh_host], capture_output=True, text=True, check=False, stdin=subprocess.DEVNULL
        )
    except FileNotFoundError:
        return ""
    # Step 3: pick the "user <name>" line.
    for option_line in completed.stdout.splitlines():
        option_key, _, option_value = option_line.partition(" ")
        if option_key == "user":
            return option_value.strip()
    return ""


def collect_personal_identifiers() -> dict:
    """Gather the running user's identifiers that must never appear in tracked files.

    Returns:
        Mapping label -> identifier string, only for identifiers that are set and long enough.
    """
    candidate_identifiers = {
        "home directory": str(Path.home()),
        "local login name": getpass.getuser(),
        f"${SLURM_USER_ENVIRONMENT_VARIABLE}": os.environ.get(SLURM_USER_ENVIRONMENT_VARIABLE, "").strip(),
        "cluster login (~/.ssh/config User)": read_ssh_config_cluster_user(),
        "git user.email": read_git_output("config", "user.email"),
    }
    # A noreply address is meant to be public, so it is not a leak.
    if candidate_identifiers["git user.email"].lower().endswith(GITHUB_NOREPLY_EMAIL_SUFFIX):
        candidate_identifiers["git user.email"] = ""
    return {
        label: identifier
        for label, identifier in candidate_identifiers.items()
        if len(identifier) >= MINIMUM_SEARCHABLE_IDENTIFIER_LENGTH
    }


def test_tracked_files_contain_none_of_the_running_users_identifiers():
    """Every ``git ls-files`` file is free of the home path, logins and private git email."""
    # Step 1: only meaningful in a git checkout (the cluster snapshot and the image have no .git).
    tracked_file_listing = read_git_output("ls-files", "-z")
    if not tracked_file_listing:
        pytest.skip("needs a git checkout (git ls-files)")
    personal_identifiers = collect_personal_identifiers()

    # Step 2: one case-insensitive pattern per identifier. Word boundaries stop a login from
    # matching inside a longer token; re.escape keeps "/", "." and "@" literal.
    identifier_patterns = {
        label: re.compile(rf"(?<![\w.-]){re.escape(identifier)}(?![\w-])", flags=re.IGNORECASE)
        for label, identifier in personal_identifiers.items()
    }

    # Step 3: scan every tracked file (bytes decoded leniently, so binary files cannot crash it).
    # Cost: O(total tracked bytes x number of identifiers), a few MB here.
    leaks = []
    for relative_path in tracked_file_listing.split("\0"):
        tracked_path = PROJECT_ROOT / relative_path
        if not tracked_path.is_file():
            continue
        file_text = tracked_path.read_bytes().decode("utf-8", errors="ignore")
        for label, identifier_pattern in identifier_patterns.items():
            if identifier_pattern.search(file_text):
                leaks.append(f"{relative_path}: contains your {label}")
    assert not leaks, "personal information in tracked files:\n" + "\n".join(leaks)
