"""PreToolUse hook for the transform-test-runner agent: allow only read-only / test commands.

Reads the hook JSON from stdin ({"tool_input": {"command": ...}}). Exit code 0 allows the
call; exit code 2 blocks it and sends the stderr text back to the agent as feedback.

Policy: the agent may run the affected-test selector, pytest, and read-only git/ls
commands. Anything that could write files or chain arbitrary commands is rejected.
"""

import json
import re
import sys

# Command prefixes the agent is allowed to run (matched against the whole command start).
ALLOWED_COMMAND_PATTERNS = [
    r"^uv run (python )?scripts/affected_tests\.py\b",
    r"^uv run pytest\b",
    r"^uv run python -m pytest\b",
    r"^git (diff|status|log|show|ls-files)\b",
    r"^ls\b",
    r"^pwd$",
]
# Shell features that can write files, chain commands or substitute output.
# Output redirection (>), command chaining (; & &&), substitution ($( ) and backticks)
# are rejected; a plain pipe into tail/head/grep/wc is allowed for trimming output.
FORBIDDEN_SHELL_FRAGMENTS = [">", "<", ";", "&", "`", "$(", "\n"]
ALLOWED_PIPE_TARGETS = re.compile(r"^(tail|head|grep|wc)\b")


def verdict(command: str) -> str:
    """Decide whether a Bash command is permitted.

    Args:
        command: The full command string the agent wants to run.

    Returns:
        Empty string if allowed, otherwise a human-readable reason for blocking.
    """
    # Step 1: split off pipe stages; only the first stage is the "real" command.
    stages = [stage.strip() for stage in command.split("|")]
    first_stage, piped_stages = stages[0], stages[1:]

    # Step 2: reject shell features that could write or chain commands in any stage.
    for fragment in FORBIDDEN_SHELL_FRAGMENTS:
        if fragment in command:
            return f"shell fragment {fragment!r} is not allowed"

    # Step 3: the first stage must match an allowed command pattern.
    if not any(re.match(pattern, first_stage) for pattern in ALLOWED_COMMAND_PATTERNS):
        return "command is not on the read-only allowlist"

    # Step 4: later stages may only trim/filter output.
    for stage in piped_stages:
        if not ALLOWED_PIPE_TARGETS.match(stage):
            return f"pipe target {stage!r} is not allowed (use tail/head/grep/wc)"
    return ""


def main() -> int:
    """Read the hook payload from stdin and block disallowed commands.

    Returns:
        0 to allow, 2 to block.
    """
    payload = json.load(sys.stdin)
    command = payload.get("tool_input", {}).get("command", "")
    reason = verdict(command)
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
