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
    "uv run pytest -q -p no:cacheprovider tests/unit/test_pose.py",
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


# ---------------------------------------------------------------------------------------------
# Hardened per-command argument checks.
# ---------------------------------------------------------------------------------------------

NO_CACHE = "-p no:cacheprovider"


@pytest.mark.parametrize("command", [
    f"uv run pytest -q {NO_CACHE} tests --co",
    f"uv run python -m pytest {NO_CACHE} -k 'a and b' -m 'not gpu' tests/unit",
    f"uv run pytest -q -pno:cacheprovider tests/unit/test_pose.py::test_a",
    "scripts/test_in_image.sh",
    "scripts/test_in_image.sh python scripts/affected_tests.py",
    "scripts/test_in_image.sh python scripts/affected_tests.py --run",
    "scripts/test_in_image.sh python scripts/affected_tests.py --all --run",
    "scripts/test_in_image.sh python scripts/affected_tests.py --base main --run",
    "scripts/test_in_image.sh python scripts/affected_tests.py --base origin/main~2",
    f"scripts/test_in_image.sh python -m pytest {NO_CACHE} -m covers -q",
    "uv run python scripts/affected_tests.py --run",
    "uv run scripts/affected_tests.py --base HEAD~1",
    "git log --oneline -5",
    "git show HEAD:tests/test_traceability.py",
    "git diff --stat HEAD",
])
def test_legitimate_agent_commands_still_pass(command):
    """The commands the agent definition tells the runner to use stay allowed."""
    assert load_guard().verdict(command) == "", command


@pytest.mark.parametrize("command", [
    # git: output-writing options
    "git diff --output=/tmp/x",
    "git diff --output /tmp/x",
    "git log --output=x",
    "git show --output=x HEAD",
    "git diff --ext-diff",
    "git diff --textconv",
    # pytest: output / config / plugin options
    f"uv run pytest {NO_CACHE} --basetemp=/tmp/x tests",
    f"uv run pytest {NO_CACHE} --junitxml=out.xml tests",
    f"uv run pytest {NO_CACHE} --junit-xml=out.xml tests",
    f"uv run pytest {NO_CACHE} --log-file=out.log tests",
    f"uv run pytest {NO_CACHE} --html=out.html tests",
    f"uv run pytest {NO_CACHE} -c other.ini tests",
    f"uv run pytest {NO_CACHE} -o addopts= tests",
    f"uv run pytest {NO_CACHE} --override-ini=addopts= tests",
    f"uv run pytest {NO_CACHE} -p evil_plugin tests",
    f"uv run pytest {NO_CACHE} -pevil tests",
    f"uv run pytest {NO_CACHE} --rootdir=/ tests",
    f"uv run pytest {NO_CACHE} --pdb tests",
    f"uv run pytest {NO_CACHE} /etc/somewhere/test_x.py",
    f"uv run pytest {NO_CACHE} ../other/tests",
    # the cache provider must be disabled
    "uv run pytest -q tests",
    "uv run python -m pytest -q tests",
    "uv run pytest -p evil -q tests",
    # test_in_image.sh: only no args, affected_tests or a vetted pytest
    "scripts/test_in_image.sh python -c 'open(\"x\",\"w\")'",
    "scripts/test_in_image.sh sh",
    "scripts/test_in_image.sh bash -c id",
    "scripts/test_in_image.sh python scripts/affected_tests.py --allow-suspicious",
    "scripts/test_in_image.sh python scripts/affected_tests.py --run extra",
    "scripts/test_in_image.sh python scripts/affected_tests.py --base",
    "scripts/test_in_image.sh python scripts/affected_tests.py --base --run",
    "scripts/test_in_image.sh python scripts/other.py",
    f"scripts/test_in_image.sh python -m pytest --basetemp=x {NO_CACHE}",
    "scripts/test_in_image.sh python -m pytest -q tests",
    # affected_tests via uv: --base must be a plain ref
    "uv run python scripts/affected_tests.py --base --output=x",
    "uv run scripts/affected_tests.py --base -x",
    "uv run scripts/affected_tests.py --base=-x",
    "uv run scripts/affected_tests.py --allow-suspicious",
    "uv run scripts/affected_tests.py --base 'a b'",
    # shell tricks that would build a forbidden argument after the check
    "git diff {--output=x,}",
    "git diff --output=$HOME/x",
    # unbalanced quoting cannot be tokenised
    "git diff 'unterminated",
])
def test_argument_level_abuse_is_blocked(command):
    """Each hardened argument rule has at least one blocked command."""
    assert load_guard().verdict(command) != "", command


def run_hook(raw_stdin: str):
    """Invoke the guard as the hook process with arbitrary stdin text."""
    return subprocess.run([sys.executable, str(GUARD_PATH)], input=raw_stdin, capture_output=True, text=True)


@pytest.mark.parametrize("raw_stdin", [
    "not json at all",
    "",
    "{}",
    json.dumps({"tool_input": {}}),
    json.dumps({"tool_input": {"command": 42}}),
    json.dumps({"tool_input": None}),
    json.dumps([1, 2, 3]),
])
def test_hook_fails_closed_on_malformed_payloads(raw_stdin):
    """Malformed JSON or missing keys must exit 2 (block), never 0 or a traceback exit 1."""
    completed = run_hook(raw_stdin)
    assert completed.returncode == 2, (raw_stdin, completed.stderr)
