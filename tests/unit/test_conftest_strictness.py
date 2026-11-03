"""Proof that tests/conftest.py enforces strict markers, strict xfail and the exclusions allowlist.

Each case builds a throwaway project (a copy of the real conftest plus a tiny pyproject) and runs
pytest on it in a subprocess, so the real hooks run exactly as they do for the suite.
"""

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REAL_CONFTEST = Path(__file__).resolve().parents[1] / "conftest.py"
# Same marker registration and default selection as the real pyproject (copied, not imported, so
# the throwaway project is self-contained).
TEMP_PYPROJECT = textwrap.dedent('''
    [tool.pytest.ini_options]
    testpaths = ["tests"]
    markers = ["gpu: needs a gpu", "real_data: needs data"]
    addopts = "-m 'not gpu and not real_data'"
''')
PASSING_TEST = "def test_ok():\n    assert 1 == 1\n"


def build_project(project_root: Path, test_sources: dict, allowlist: dict = None) -> None:
    """Write a throwaway project: the real conftest, pyproject, tests and an optional allowlist.

    Args:
        project_root: Empty temp directory.
        test_sources: Mapping file name under tests/ -> source text.
        allowlist: Content of tests/fixtures/test_exclusions_allowlist.json (None = absent).
    """
    tests_directory = project_root / "tests"
    (tests_directory / "fixtures").mkdir(parents=True)
    (project_root / "pyproject.toml").write_text(TEMP_PYPROJECT)
    (tests_directory / "conftest.py").write_text(REAL_CONFTEST.read_text())
    for file_name, source in test_sources.items():
        (tests_directory / file_name).write_text(source)
    if allowlist is not None:
        (tests_directory / "fixtures" / "test_exclusions_allowlist.json").write_text(json.dumps(allowlist))


def run_pytest(project_root: Path, *pytest_arguments: str) -> subprocess.CompletedProcess:
    """Run pytest inside the throwaway project (cache provider off: nothing is written there)."""
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *pytest_arguments],
        cwd=project_root, capture_output=True, text=True,
    )


EMPTY_ALLOWLIST = {"skipped": [], "xfailed": [], "deselected_by_marker": {"count": 0, "nodeids": []}}


def test_unknown_marker_is_a_collection_error(tmp_path):
    """A misspelled / unregistered marker fails the run instead of being silently ignored."""
    build_project(tmp_path, {"test_a.py": "import pytest\n@pytest.mark.bogus\ndef test_a():\n    assert 1\n"},
                  EMPTY_ALLOWLIST)
    completed = run_pytest(tmp_path)
    assert completed.returncode != 0
    assert "bogus" in completed.stdout + completed.stderr


def test_registered_markers_still_work(tmp_path):
    """Control: registered and built-in markers do not trip strict mode."""
    build_project(tmp_path, {"test_a.py": "import pytest\n@pytest.mark.parametrize('v', [1])\n"
                                          "def test_a(v):\n    assert v\n"}, EMPTY_ALLOWLIST)
    assert run_pytest(tmp_path).returncode == 0


def test_unexpected_pass_of_an_xfail_test_fails_the_run(tmp_path):
    """xfail is strict: an XPASS is a failure (the allowlist would also reject the xfail)."""
    build_project(tmp_path, {"test_a.py": "import pytest\n@pytest.mark.xfail\ndef test_a():\n    assert 1\n"},
                  EMPTY_ALLOWLIST)
    completed = run_pytest(tmp_path, "tests/test_a.py")  # single file: allowlist not enforced
    assert completed.returncode != 0 and "XPASS(strict)" in completed.stdout


SKIPPING_TEST = "import pytest\ndef test_skips():\n    pytest.skip('0 things pending')\n"


def test_unlisted_skip_fails_a_whole_default_run(tmp_path):
    """A skip that is not in the allowlist fails the session even though every test 'passed'."""
    build_project(tmp_path, {"test_a.py": PASSING_TEST, "test_skip.py": SKIPPING_TEST}, EMPTY_ALLOWLIST)
    completed = run_pytest(tmp_path)
    assert completed.returncode != 0
    assert "test_skip.py::test_skips" in completed.stdout


def test_listed_skip_with_matching_reason_is_accepted(tmp_path):
    """The allowlisted skip (node id + reason regex) passes."""
    allowlist = dict(EMPTY_ALLOWLIST, skipped=[
        {"nodeid": "tests/test_skip.py::test_skips", "reason_regex": "^0 things pending"}])
    build_project(tmp_path, {"test_a.py": PASSING_TEST, "test_skip.py": SKIPPING_TEST}, allowlist)
    assert run_pytest(tmp_path).returncode == 0


def test_listed_skip_with_a_different_reason_is_rejected(tmp_path):
    """The reason must match too: '3 things pending' is not the allowed '0 things pending'."""
    allowlist = dict(EMPTY_ALLOWLIST, skipped=[
        {"nodeid": "tests/test_skip.py::test_skips", "reason_regex": "^0 things pending"}])
    other_reason = SKIPPING_TEST.replace("0 things", "3 things")
    build_project(tmp_path, {"test_a.py": PASSING_TEST, "test_skip.py": other_reason}, allowlist)
    assert run_pytest(tmp_path).returncode != 0


def test_xfailed_test_is_rejected_unless_allowlisted(tmp_path):
    """A (failing) xfail test is an exclusion too."""
    source = "import pytest\n@pytest.mark.xfail\ndef test_x():\n    assert 0\n"
    build_project(tmp_path, {"test_x.py": source}, EMPTY_ALLOWLIST)
    assert run_pytest(tmp_path).returncode != 0
    allowlist = dict(EMPTY_ALLOWLIST, xfailed=["tests/test_x.py::test_x"])
    build_project_again = tmp_path / "tests" / "fixtures" / "test_exclusions_allowlist.json"
    build_project_again.write_text(json.dumps(allowlist))
    assert run_pytest(tmp_path).returncode == 0


GPU_TEST = "import pytest\n@pytest.mark.gpu\ndef test_g():\n    assert 1\n"


def test_marker_deselection_must_match_recorded_count_and_ids(tmp_path):
    """gpu/real_data deselection is allowed only as recorded: a new or missing id fails."""
    build_project(tmp_path, {"test_a.py": PASSING_TEST, "test_g.py": GPU_TEST}, EMPTY_ALLOWLIST)
    assert run_pytest(tmp_path).returncode != 0  # one deselected, none recorded
    recorded = dict(EMPTY_ALLOWLIST, deselected_by_marker={"count": 1, "nodeids": ["tests/test_g.py::test_g"]})
    (tmp_path / "tests" / "fixtures" / "test_exclusions_allowlist.json").write_text(json.dumps(recorded))
    assert run_pytest(tmp_path).returncode == 0
    wrong_count = dict(EMPTY_ALLOWLIST, deselected_by_marker={"count": 2, "nodeids": ["tests/test_g.py::test_g"]})
    (tmp_path / "tests" / "fixtures" / "test_exclusions_allowlist.json").write_text(json.dumps(wrong_count))
    assert run_pytest(tmp_path).returncode != 0


@pytest.mark.parametrize("narrowing_arguments", [
    ["tests/test_a.py"],                        # a single file
    ["tests/test_skip.py::test_skips"],         # a single node id
    ["-k", "test_ok"],                          # keyword selection
    ["-m", "real_data"],                        # a different marker expression
    ["--deselect", "tests/test_a.py::test_ok"],  # explicit deselection
])
def test_narrowed_runs_do_not_enforce_the_allowlist(tmp_path, narrowing_arguments):
    """Only the whole default suite is checked: narrowed runs (file, node id, -k, -m, --deselect) are not."""
    build_project(tmp_path, {"test_a.py": PASSING_TEST, "test_skip.py": SKIPPING_TEST}, EMPTY_ALLOWLIST)
    completed = run_pytest(tmp_path, *narrowing_arguments)
    assert "exclusion" not in (completed.stdout + completed.stderr).lower()
    assert completed.returncode in (0, 5)  # 5 = nothing collected for the -m real_data case


def test_missing_allowlist_fails_a_whole_default_run(tmp_path):
    """Deleting the allowlist cannot disable the check."""
    build_project(tmp_path, {"test_a.py": PASSING_TEST}, allowlist=None)
    assert run_pytest(tmp_path).returncode != 0


def test_allowlist_writer_script_builds_the_documented_document():
    """scripts/write_exclusions_allowlist.py turns node-id lines into count + ids and keeps the one allowed skip."""
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "write_exclusions_allowlist.py"
    listing = "tests/b.py::test_b\ntests/a.py::test_a[1]\ntests/a.py::test_a[1]\n\n2 tests collected\n"
    completed = subprocess.run([sys.executable, str(script_path)], input=listing, capture_output=True, text=True)
    document = json.loads(completed.stdout)
    assert document["deselected_by_marker"] == {
        "count": 2, "nodeids": ["tests/a.py::test_a[1]", "tests/b.py::test_b"]}
    assert document["xfailed"] == []
    assert [entry["nodeid"] for entry in document["skipped"]] == [
        "tests/test_traceability.py::test_all_planned_transforms_implemented"]
