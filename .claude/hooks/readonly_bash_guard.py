"""PreToolUse hook for the transform-test-runner agent: allow only read-only / test commands.

Reads the hook JSON from stdin ({"tool_input": {"command": ...}}). Exit code 0 allows the
call; exit code 2 blocks it and sends the stderr text back to the agent as feedback.

Policy: the agent may run the affected-test selector, pytest, and read-only git/ls
commands. Anything that could write files or chain arbitrary commands is rejected, and the
accepted programs are checked argument by argument (a tokenised check, not just a prefix):
  * git: no ``--output*`` (writes a file), ``--ext-diff`` / ``--textconv`` (run external programs);
  * pytest: ``-p no:cacheprovider`` is mandatory (the cache would write into the repo) and is the
    only plugin; no output files (``--basetemp``, ``--junitxml``, ``--log-file``, ``--html``, ...),
    no other config (``-c``, ``-o``, ``--rootdir``), no debugger, no absolute or ``..`` paths
    (pytest would import and run code from there);
  * scripts/test_in_image.sh: no arguments, ``python scripts/affected_tests.py`` with
    ``--run`` / ``--all`` / ``--base <ref>``, or ``python -m pytest`` under the pytest rules;
  * affected_tests.py: ``--base`` takes a plain git ref (never an option), no ``--allow-suspicious``
    (the main agent alone may opt out of the audit gate).
The hook fails closed: any error while judging (malformed JSON, missing keys, wrong types, a
bug in this file) blocks the call.
"""

import json
import re
import shlex
import sys
from typing import List

# Shell features that can write files, chain commands, substitute output or expand into
# forbidden arguments after this check: redirection (> <), chaining (; & and newlines),
# substitution ($( ) and backticks), variable expansion ($) and brace expansion ({ }).
FORBIDDEN_SHELL_FRAGMENTS = [">", "<", ";", "&", "`", "$", "{", "}", "\n"]
ALLOWED_PIPE_TARGETS = re.compile(r"^(tail|head|grep|wc)\b")
# A git ref: letters, digits and the punctuation of branch names / ancestry (main, origin/main~2,
# HEAD^, v1.0, a1b2c3). It can never start with "-" (checked separately), so git cannot read it as an option.
SAFE_REF_PATTERN = re.compile(r"^[A-Za-z0-9_./~^@][A-Za-z0-9_./~^@-]*$")
GIT_SUBCOMMANDS = {"diff", "status", "log", "show", "ls-files"}
GIT_FORBIDDEN_PREFIXES = ("--output", "--ext-diff", "--textconv")
# pytest options that write files, change configuration or open a debugger (matched as prefixes,
# so ``--junitxml=x`` and ``--junitxml x`` are both caught).
PYTEST_FORBIDDEN_PREFIXES = (
    "--basetemp", "--junitxml", "--junit-xml", "--log-file", "--html", "--report-log", "--resultlog",
    "--result-log", "--alluredir", "--cov", "--debug", "--pastebin", "--rootdir", "--config-file",
    "--override-ini", "--pdb", "--trace",
)
PYTEST_FORBIDDEN_EXACT = {"-c", "-o"}
REQUIRED_PLUGIN_ARGUMENT = "no:cacheprovider"
AFFECTED_TESTS_SCRIPT = "scripts/affected_tests.py"


def check_pytest_arguments(arguments: List[str]) -> str:
    """Vet the arguments after ``pytest`` (or ``python -m pytest``).

    Args:
        arguments: Tokens following the pytest program name.

    Returns:
        Empty string if acceptable, else the reason.
    """
    plugin_ok = False
    position = 0
    while position < len(arguments):
        token = arguments[position]
        if token == "-p":
            # Exactly one plugin spelling is allowed: switching the cache provider OFF.
            if position + 1 >= len(arguments) or arguments[position + 1] != REQUIRED_PLUGIN_ARGUMENT:
                return "only `-p no:cacheprovider` may be used with -p"
            plugin_ok = True
            position += 2
            continue
        if token.startswith("-p") and not token.startswith("--"):
            if token != "-p" + REQUIRED_PLUGIN_ARGUMENT:
                return "only `-p no:cacheprovider` may be used with -p"
            plugin_ok = True
        elif token in PYTEST_FORBIDDEN_EXACT or token.startswith(PYTEST_FORBIDDEN_PREFIXES):
            return f"pytest option {token!r} is not allowed (writes files or changes configuration)"
        elif token.startswith(("/", "~")) or ".." in token.split("/"):
            return f"pytest argument {token!r} points outside the repository"
        position += 1
    if not plugin_ok:
        return "pytest must be run with `-p no:cacheprovider` (the cache would write into the repo)"
    return ""


def check_affected_tests_arguments(arguments: List[str]) -> str:
    """Vet the arguments after ``scripts/affected_tests.py``.

    Args:
        arguments: Tokens following the script path.

    Returns:
        Empty string if acceptable, else the reason.
    """
    position = 0
    while position < len(arguments):
        token = arguments[position]
        if token in ("--run", "--all"):
            position += 1
        elif token == "--base":
            if position + 1 >= len(arguments):
                return "--base needs a git ref"
            reference = arguments[position + 1]
            if reference.startswith("-") or not SAFE_REF_PATTERN.match(reference):
                return f"--base value {reference!r} is not a plain git ref"
            position += 2
        else:
            return f"affected_tests.py argument {token!r} is not allowed (only --run, --all, --base <ref>)"
    return ""


def check_git_arguments(arguments: List[str]) -> str:
    """Vet ``git <subcommand> ...`` for options that write files or run external programs.

    Args:
        arguments: Tokens after ``git``.

    Returns:
        Empty string if acceptable, else the reason.
    """
    if not arguments or arguments[0] not in GIT_SUBCOMMANDS:
        return "git subcommand is not on the read-only list"
    for token in arguments[1:]:
        if token.startswith(GIT_FORBIDDEN_PREFIXES):
            return f"git option {token!r} is not allowed (writes a file or runs an external program)"
    return ""


def check_in_image_arguments(arguments: List[str]) -> str:
    """Vet ``scripts/test_in_image.sh`` arguments: none, affected_tests.py, or a vetted pytest.

    Args:
        arguments: Tokens after the wrapper script.

    Returns:
        Empty string if acceptable, else the reason.
    """
    if not arguments:
        return ""
    if arguments[:2] == ["python", AFFECTED_TESTS_SCRIPT]:
        return check_affected_tests_arguments(arguments[2:])
    if arguments[:3] == ["python", "-m", "pytest"]:
        return check_pytest_arguments(arguments[3:])
    return "test_in_image.sh may only run affected_tests.py or pytest"


def judge_first_stage(tokens: List[str]) -> str:
    """Decide on the tokenised first pipe stage.

    Args:
        tokens: shlex tokens of the command before the first pipe.

    Returns:
        Empty string if allowed, else the reason.
    """
    if not tokens:
        return "empty command"
    program = tokens[0]
    if program == "scripts/test_in_image.sh":
        return check_in_image_arguments(tokens[1:])
    if program == "uv" and tokens[1:2] == ["run"]:
        rest = tokens[2:]
        if rest[:1] == ["python"]:
            rest = rest[1:]
            if rest[:2] == ["-m", "pytest"]:
                return check_pytest_arguments(rest[2:])
        if rest[:1] == ["pytest"]:
            return check_pytest_arguments(rest[1:])
        if rest[:1] == [AFFECTED_TESTS_SCRIPT]:
            return check_affected_tests_arguments(rest[1:])
        return "command is not on the read-only allowlist"
    if program == "git":
        return check_git_arguments(tokens[1:])
    if program == "ls":
        return ""
    if program == "pwd" and len(tokens) == 1:
        return ""
    return "command is not on the read-only allowlist"


def verdict(command: str) -> str:
    """Decide whether a Bash command is permitted.

    Args:
        command: The full command string the agent wants to run.

    Returns:
        Empty string if allowed, otherwise a human-readable reason for blocking.
    """
    # Step 1: reject shell features that could write, chain or rewrite arguments, in any stage.
    for fragment in FORBIDDEN_SHELL_FRAGMENTS:
        if fragment in command:
            return f"shell fragment {fragment!r} is not allowed"

    # Step 2: split off pipe stages; only the first stage is the "real" command.
    stages = [stage.strip() for stage in command.split("|")]
    first_stage, piped_stages = stages[0], stages[1:]

    # Step 3: tokenise like the shell would (quotes, escapes) and judge program and arguments.
    try:
        tokens = shlex.split(first_stage)
    except ValueError as error:
        return f"command cannot be tokenised ({error})"
    reason = judge_first_stage(tokens)
    if reason:
        return reason

    # Step 4: later stages may only trim/filter output.
    for stage in piped_stages:
        if not ALLOWED_PIPE_TARGETS.match(stage):
            return f"pipe target {stage!r} is not allowed (use tail/head/grep/wc)"
    return ""


def main() -> int:
    """Read the hook payload from stdin and block disallowed commands (fail closed).

    Returns:
        0 to allow, 2 to block (also on any error while reading or judging the payload).
    """
    try:
        payload = json.load(sys.stdin)
        command = payload["tool_input"]["command"]
        if not isinstance(command, str):
            raise TypeError("command is not a string")
        reason = verdict(command)
    except Exception as error:  # noqa: BLE001 - deliberate: anything unexpected must block
        print(f"Blocked by transform-test-runner guard: could not judge the call ({error!r}).", file=sys.stderr)
        return 2
    if reason:
        print(
            f"Blocked by transform-test-runner guard: {reason}. This agent is read-only: "
            "run scripts/affected_tests.py or pytest, and report findings instead of editing.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
