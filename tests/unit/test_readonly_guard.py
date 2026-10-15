"""Tests for the Bash guard that keeps the transform-test-runner agent read-only."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

GUARD_PATH = Path(__file__).resolve().parents[2] / ".claude" / "hooks" / "readonly_bash_guard.py"


def load_guard():
    """Import the guard script as a module (it lives outside the package)."""
    spec = importlib.util.spec_from_file_location("readonly_bash_guard", GUARD_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("command", [
    "uv run python scripts/affected_tests.py --run",
    "uv run pytest -q tests/unit/test_pose.py",
    "uv run pytest -q -p no:cacheprovider tests | tail -20",
    "git diff --stat",
    "git status --short",
    "ls tests/unit",
])
def test_allowed_commands_pass(command):
    """Test and read-only commands are permitted."""
    assert load_guard().verdict(command) == ""


@pytest.mark.parametrize("command", [
    "rm -rf src",
    "uv run pytest -q > out.txt",
    "uv run pytest -q; rm -rf src",
    "uv run pytest -q && git commit -am x",
    "git commit -m x",
    "git checkout -- src/viking_splatad/registry.py",
    "uv pip install requests",
    "sed -i s/a/b/ src/viking_splatad/registry.py",
    "uv run pytest -q | tee log.txt",
    "uv run pytest $(echo -q)",
    "python3 -c 'open(\"x\",\"w\")'",
])
def test_writing_or_unlisted_commands_are_blocked(command):
    """Anything that could write, chain or run unlisted programs is rejected."""
    assert load_guard().verdict(command) != ""


def test_hook_exit_codes_through_stdin():
    """The real hook protocol: JSON on stdin; exit 2 blocks, exit 0 allows."""
    def run(command):
        """Invoke the guard as a subprocess with a hook payload."""
        return subprocess.run(
            [sys.executable, str(GUARD_PATH)],
            input=json.dumps({"tool_input": {"command": command}}),
            capture_output=True, text=True,
        )

    blocked = run("rm -rf src")
    assert blocked.returncode == 2 and "read-only" in blocked.stderr
    assert run("git status --short").returncode == 0
