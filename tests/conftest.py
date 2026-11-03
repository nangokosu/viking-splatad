"""Pytest strictness for the whole suite (a new file: pyproject.toml is a recipe file and stays as is).

Three guards, all of which only ever make a run STRICTER:
  1. Strict markers: an unregistered marker (a typo such as ``@pytest.mark.coverz``) is a
     collection error instead of a silently ignored decoration.
  2. Strict xfail: an ``xfail`` test that unexpectedly passes fails the run.
  3. Exclusion allowlist: when the whole default suite ran, the set of skipped, xfailed and
     marker-deselected (gpu / real_data) tests must match tests/fixtures/test_exclusions_allowlist.json,
     so a test cannot be silenced (skip, xfail, new gpu marker) without a reviewed edit of that file.

"Whole default suite" is decided in ``is_whole_default_suite``: only the configured testpaths were
given (no file, directory or node id narrower than them), no ``-k``, no ``--deselect``, the ``-m``
expression is the one from the ini ``addopts``, and the run was neither collect-only nor stopped
early. Any narrower run (one file, ``-k``, ``-m real_data``, ``-x``) skips guard 3, so developers
and ``scripts/affected_tests.py --run`` keep working. ``scripts/test_in_image.sh`` with no arguments
runs ``python -m pytest -q -p no:cacheprovider`` = the whole default suite = guard 3 applies.
Guards 1 and 2 are on for every run.
"""

import json
import re
import shlex
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

TESTS_DIRECTORY = Path(__file__).resolve().parent
ALLOWLIST_PATH = TESTS_DIRECTORY / "fixtures" / "test_exclusions_allowlist.json"
# Config-stash key under which the session's skipped / xfailed / deselected tests are recorded.
SESSION_RECORD_KEY = pytest.StashKey[Dict[str, list]]()
# Prefix pytest puts in front of a skip reason in the report's long representation.
SKIP_REASON_PREFIX = "Skipped: "


def force_ini_flag(config: "pytest.Config", option_name: str) -> None:
    """Make a boolean ini option read as True regardless of the project's pyproject.toml.

    pytest caches every ``getini`` answer in ``config._inicache``; seeding that cache is the
    supported-in-practice way to override an ini value from a plugin (verified for the strict
    markers and xfail cases in tests/unit/test_conftest_strictness.py).

    Args:
        config: The pytest config object.
        option_name: ini option such as "strict_markers" or "xfail_strict".
    """
    config._inicache[option_name] = True


def pytest_configure(config: "pytest.Config") -> None:
    """Switch on strict markers and strict xfail, and start recording exclusions.

    Args:
        config: The pytest config object.
    """
    # Step 1: pytest >= 9 reads the ini option, older pytest the command-line attribute.
    force_ini_flag(config, "strict_markers")
    config.option.strict_markers = True
    # Step 2: an unexpected pass of an xfail test fails (pytest >= 9.1 calls the option strict_xfail
    # with alias xfail_strict; older versions only know xfail_strict, so both names are seeded).
    force_ini_flag(config, "strict_xfail")
    force_ini_flag(config, "xfail_strict")
    # Step 3: containers for what the session skips, xfails and deselects.
    config.stash[SESSION_RECORD_KEY] = {"skipped": [], "xfailed": [], "deselected": []}


def pytest_deselected(items: List["pytest.Item"]) -> None:
    """Record items removed by a marker / keyword expression.

    Args:
        items: Items pytest deselected.
    """
    # The config is reachable from any item; an empty list means nothing to record.
    if items:
        items[0].config.stash[SESSION_RECORD_KEY]["deselected"].extend(item.nodeid for item in items)


def pytest_runtest_logreport(report: "pytest.TestReport") -> None:
    """Record skipped and xfailed outcomes (setup or call phase).

    Args:
        report: The phase report of one test.
    """
    if not report.skipped:
        return
    # The record lives in the config stash; reports do not carry the config, so use the module-level
    # handle set in pytest_sessionstart.
    record = CURRENT_RECORD.get("record")
    if record is None:
        return
    if hasattr(report, "wasxfail"):
        record["xfailed"].append(report.nodeid)
    else:
        long_representation = report.longrepr
        reason = long_representation[2] if isinstance(long_representation, tuple) else str(long_representation)
        reason = reason[len(SKIP_REASON_PREFIX):] if reason.startswith(SKIP_REASON_PREFIX) else reason
        record["skipped"].append((report.nodeid, reason))


# Handle to the active session record for pytest_runtest_logreport (which has no config argument).
CURRENT_RECORD: Dict[str, Dict[str, list]] = {}


def pytest_sessionstart(session: "pytest.Session") -> None:
    """Expose the session record to the report hook.

    Args:
        session: The pytest session.
    """
    CURRENT_RECORD["record"] = session.config.stash[SESSION_RECORD_KEY]


def default_marker_expression(config: "pytest.Config") -> str:
    """Read the ``-m`` expression that the ini ``addopts`` applies by default.

    Args:
        config: The pytest config object.

    Returns:
        The expression, or "" if addopts has no ``-m``.
    """
    addopts = config.getini("addopts")
    # pytest returns addopts already split into tokens (a str only in very old versions).
    tokens = list(addopts) if isinstance(addopts, (list, tuple)) else shlex.split(str(addopts))
    for position, token in enumerate(tokens[:-1]):
        if token == "-m":
            return tokens[position + 1]
    return ""


def is_whole_default_suite(session: "pytest.Session") -> bool:
    """Decide whether this session ran the whole default suite (the only case guard 3 checks).

    Args:
        session: The finished pytest session.

    Returns:
        True when the arguments name exactly the configured testpaths and nothing narrowed or
        stopped the run (see the module docstring).
    """
    config = session.config
    option = config.option
    # Step 1: arguments must resolve to exactly the configured testpaths (a node id has "::").
    if any("::" in argument for argument in config.args):
        return False
    invocation_directory = Path(config.invocation_params.dir)

    def resolve_argument(argument: str) -> Path:
        """Resolve a path argument: as typed from the invocation directory, else from the rootdir
        (pytest stores the ini ``testpaths`` relative to the rootdir)."""
        typed = (invocation_directory / argument).resolve()
        return typed if typed.exists() else (config.rootpath / argument).resolve()

    given_roots = {resolve_argument(argument) for argument in config.args}
    testpath_roots = {(config.rootpath / testpath).resolve() for testpath in config.getini("testpaths")}
    if not testpath_roots or given_roots != testpath_roots:
        return False
    # Step 2: no narrowing options, default marker expression, not collect-only, not stopped early.
    narrowed = (
        bool(option.keyword) or bool(option.deselect) or option.markexpr != default_marker_expression(config)
        or option.collectonly or session.shouldstop or session.shouldfail
    )
    return not narrowed


def load_allowlist() -> dict:
    """Load the committed allowlist.

    Returns:
        Parsed JSON with keys "skipped" (list of {"nodeid", "reason_regex"}), "xfailed" (list of
        node ids) and "deselected_by_marker" ({"count", "nodeids"}).

    Raises:
        FileNotFoundError: If the file is missing (deleting it must not disable the check).
    """
    return json.loads(ALLOWLIST_PATH.read_text())


def exclusion_differences(record: Dict[str, list], allowlist: dict) -> List[str]:
    """Compare what the session skipped / xfailed / deselected with the allowlist.

    Skips and xfails are an allowed SUBSET (a listed skip need not occur: e.g. the traceability
    skip disappears under VIKING_REQUIRE_COMPLETE=1); marker deselection must match EXACTLY, count
    and ids, because it silently shrinks the suite.

    Args:
        record: {"skipped": [(nodeid, reason)], "xfailed": [nodeid], "deselected": [nodeid]}.
        allowlist: Parsed allowlist.

    Returns:
        Human-readable differences (empty = compliant).
    """
    differences: List[str] = []
    allowed_skips: Dict[str, str] = {entry["nodeid"]: entry["reason_regex"] for entry in allowlist["skipped"]}
    for nodeid, reason in record["skipped"]:
        if nodeid not in allowed_skips:
            differences.append(f"unlisted skip: {nodeid} ({reason!r})")
        elif not re.search(allowed_skips[nodeid], reason):
            differences.append(f"skip reason differs: {nodeid} ({reason!r} !~ {allowed_skips[nodeid]!r})")
    for nodeid in record["xfailed"]:
        if nodeid not in allowlist["xfailed"]:
            differences.append(f"unlisted xfail: {nodeid}")
    recorded = allowlist["deselected_by_marker"]
    observed_ids = sorted(set(record["deselected"]))
    if len(record["deselected"]) != recorded["count"]:
        differences.append(f"deselected count {len(record['deselected'])} != recorded {recorded['count']}")
    for nodeid in sorted(set(observed_ids) - set(recorded["nodeids"])):
        differences.append(f"newly deselected: {nodeid}")
    for nodeid in sorted(set(recorded["nodeids"]) - set(observed_ids)):
        differences.append(f"recorded deselection no longer happens: {nodeid}")
    return differences


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session: "pytest.Session", exitstatus: int) -> None:
    """Fail the session when the exclusions of a whole default run differ from the allowlist.

    Args:
        session: The pytest session.
        exitstatus: Exit status so far.
    """
    # Only complete, usable runs are judged.
    if exitstatus in (pytest.ExitCode.INTERRUPTED, pytest.ExitCode.INTERNAL_ERROR, pytest.ExitCode.USAGE_ERROR):
        return
    if not is_whole_default_suite(session):
        return
    record = session.config.stash[SESSION_RECORD_KEY]
    try:
        differences = exclusion_differences(record, load_allowlist())
    except (OSError, KeyError, ValueError) as error:
        differences = [f"allowlist unreadable: {error!r}"]
    if differences:
        reporter = session.config.pluginmanager.get_plugin("terminalreporter")
        lines = ["", "EXCLUSION ALLOWLIST MISMATCH (tests/fixtures/test_exclusions_allowlist.json):"]
        lines += [f"  {difference}" for difference in differences]
        for line in lines:
            reporter.write_line(line, red=True) if reporter else print(line)
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
