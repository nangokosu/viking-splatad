"""Tests for the changed-file -> affected-test selector used by the test-runner agent."""

import importlib.util
import textwrap
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_selector_module():
    """Import scripts/affected_tests.py (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location(
        "affected_tests", PROJECT_ROOT / "scripts" / "affected_tests.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_tiny_project(tmp_path: Path) -> Path:
    """Create a miniature project: pkg.base <- pkg.mid <- (test_mid), pkg.other <- test_other."""
    source_dir = tmp_path / "src" / "pkg"
    source_dir.mkdir(parents=True)
    (source_dir / "__init__.py").write_text("")
    (source_dir / "base.py").write_text("VALUE = 1\n")
    (source_dir / "mid.py").write_text("from pkg import base\n")
    (source_dir / "other.py").write_text("OTHER = 2\n")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_traceability.py").write_text("")
    (tests_dir / "test_mid.py").write_text(textwrap.dedent('''
        import pytest
        from pkg import mid
        @pytest.mark.covers("T1")
        def test_mid(): pass
    '''))
    (tests_dir / "test_other.py").write_text(textwrap.dedent('''
        import pytest
        from pkg import other
        @pytest.mark.covers("T2")
        def test_other(): pass
    '''))
    return tmp_path


def test_change_propagates_to_importers_only(tmp_path):
    """Editing base selects test_mid (via mid) but not test_other."""
    selector = load_selector_module()
    project = build_tiny_project(tmp_path)
    selection = selector.select_tests(
        [project / "src" / "pkg" / "base.py"],
        tests_root=project / "tests", source_root=project / "src",
    )
    assert "tests/test_mid.py" in selection["test_files"]
    assert "tests/test_other.py" not in selection["test_files"]
    assert selection["transform_ids"] == ["T1"]
    # The meta-test is always included.
    assert "tests/test_traceability.py" in selection["test_files"]


def test_changed_test_file_selects_itself_and_pyproject_selects_all(tmp_path):
    """A changed test runs by itself; a pyproject change runs the full suite."""
    selector = load_selector_module()
    project = build_tiny_project(tmp_path)
    only_test = selector.select_tests(
        [project / "tests" / "test_other.py"],
        tests_root=project / "tests", source_root=project / "src",
    )
    assert "tests/test_other.py" in only_test["test_files"]
    assert "tests/test_mid.py" not in only_test["test_files"]

    (project / "pyproject.toml").write_text("")
    everything = selector.select_tests(
        [project / "pyproject.toml"],
        tests_root=project / "tests", source_root=project / "src",
    )
    assert everything["full_suite"] is True
    assert {"tests/test_mid.py", "tests/test_other.py"} <= set(everything["test_files"])


def run_git(repo: Path, *arguments: str) -> None:
    """Run git inside a throwaway repo with a neutral identity (no personal info)."""
    import subprocess
    subprocess.run(
        ["git", "-c", "user.name=tester", "-c", "user.email=tester@example.invalid", *arguments],
        cwd=repo, check=True, capture_output=True,
    )


def test_audit_flags_weakened_existing_tests_but_not_new_ones(tmp_path):
    """Removing an assertion / loosening a tolerance in a committed test is flagged; new tests are not."""
    selector = load_selector_module()
    run_git(tmp_path, "init", "-q")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_a.py").write_text(
        "def test_a():\n    assert 1 == 1\n    assert abs(2 - 2) < 1e-9, 'x'\n"
    )
    (tests_dir / "test_b.py").write_text("def test_b():\n    assert 3 == 3\n")
    run_git(tmp_path, "add", "-A")
    run_git(tmp_path, "commit", "-q", "-m", "base")

    # Weaken test_a (drop one assertion), leave test_b alone, add a brand-new test.
    (tests_dir / "test_a.py").write_text("def test_a():\n    assert 1 == 1\n")
    (tests_dir / "test_new.py").write_text("def test_new():\n    assert True\n")
    run_git(tmp_path, "add", "-A")

    findings = selector.audit_modified_tests("HEAD", project_root=tmp_path)
    by_path = {finding["path"]: finding for finding in findings}
    assert set(by_path) == {"tests/test_a.py"}  # new and untouched files are not flagged
    assert by_path["tests/test_a.py"]["removed_assertions"] == 1
    assert by_path["tests/test_a.py"]["suspicious"] is True


def test_audit_flags_tolerance_edit_even_with_same_assertion_count(tmp_path):
    """Loosening a tolerance keeps the assertion count equal but is still suspicious."""
    selector = load_selector_module()
    run_git(tmp_path, "init", "-q")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_c.py").write_text("def test_c():\n    assert abs(1.0 - 1.0) < 1e-9  # tolerance\n")
    run_git(tmp_path, "add", "-A")
    run_git(tmp_path, "commit", "-q", "-m", "base")
    (tests_dir / "test_c.py").write_text("def test_c():\n    assert abs(1.0 - 1.0) < 1e-3  # tolerance\n")

    (finding,) = selector.audit_modified_tests("HEAD", project_root=tmp_path)
    assert finding["tolerance_lines_changed"] >= 1 and finding["suspicious"] is True


# ---------------------------------------------------------------------------------------------
# Hardened audit: AST comparison, merge-base default, exit-code gate, script/viz mapping.
# ---------------------------------------------------------------------------------------------

import subprocess
import sys

import pytest


def commit_files(repo: Path, files: dict) -> None:
    """Initialise a repo on branch main and commit the given {relative path: text} files.

    Args:
        repo: Empty directory to turn into a repository.
        files: Mapping repo-relative path -> file text.
    """
    run_git(repo, "init", "-q", "-b", "main")
    for relative_path, text in files.items():
        target = repo / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    run_git(repo, "add", "-A")
    run_git(repo, "commit", "-q", "-m", "base")


def reasons_of(base_source: str, new_source: str) -> list:
    """Run the pure AST comparison on two dedented sources and return its reason strings."""
    selector = load_selector_module()
    return selector.compare_test_sources(textwrap.dedent(base_source), textwrap.dedent(new_source))


# (case id, base source, weakened source): each pair must produce at least one reason.
WEAKENING_CASES = [
    ("bare_numeric_loosening",
     "def test_a():\n    assert abs(1.0 - 1.0) < 1e-9\n",
     "def test_a():\n    assert abs(1.0 - 1.0) < 1e-3\n"),
    ("allclose_atol",
     "import numpy as np\ndef test_a():\n    assert np.allclose(1.0, 1.0, atol=1e-9)\n",
     "import numpy as np\ndef test_a():\n    assert np.allclose(1.0, 1.0, atol=1e-3)\n"),
    ("custom_tolerance_keyword",
     "def test_a():\n    check(1.0, 1.0, abs_tol=1e-9)\n    assert True\n",
     "def test_a():\n    check(1.0, 1.0, abs_tol=1e-3)\n    assert True\n"),
    ("uppercase_tolerance_keyword",
     "def test_a():\n    check(1.0, ATOL=1e-9)\n    assert True\n",
     "def test_a():\n    check(1.0, ATOL=1e-3)\n    assert True\n"),
    ("approx_abs",
     "import pytest\ndef test_a():\n    value = 1.0\n    assert value == pytest.approx(1.0, abs=1e-9)\n",
     "import pytest\ndef test_a():\n    value = 1.0\n    assert value == pytest.approx(1.0, abs=1e-3)\n"),
    ("expected_value_edit",
     "def test_a():\n    expected = 5\n    result = 5\n    assert result == expected\n",
     "def test_a():\n    expected = 6\n    result = 6\n    assert result == expected\n"),
    ("parametrize_row_changed",
     "import pytest\n@pytest.mark.parametrize('value', [1, 2, 3])\ndef test_a(value):\n    assert value > 0\n",
     "import pytest\n@pytest.mark.parametrize('value', [1, 2])\ndef test_a(value):\n    assert value > 0\n"),
    ("parametrize_removed",
     "import pytest\n@pytest.mark.parametrize('value', [1, 2, 3])\ndef test_a(value):\n    assert value > 0\n",
     "import pytest\ndef test_a(value=1):\n    assert value > 0\n"),
    ("skip_decorator_added",
     "def test_a():\n    assert 1 == 1\n",
     "import pytest\n@pytest.mark.skip(reason='later')\ndef test_a():\n    assert 1 == 1\n"),
    ("skipif_decorator_added",
     "def test_a():\n    assert 1 == 1\n",
     "import pytest\n@pytest.mark.skipif(True, reason='x')\ndef test_a():\n    assert 1 == 1\n"),
    ("xfail_added",
     "def test_a():\n    assert 1 == 1\n",
     "import pytest\n@pytest.mark.xfail\ndef test_a():\n    assert 1 == 1\n"),
    ("gpu_marker_added",
     "def test_a():\n    assert 1 == 1\n",
     "import pytest\n@pytest.mark.gpu\ndef test_a():\n    assert 1 == 1\n"),
    ("real_data_marker_added",
     "def test_a():\n    assert 1 == 1\n",
     "import pytest\n@pytest.mark.real_data\ndef test_a():\n    assert 1 == 1\n"),
    ("pytest_skip_call_added",
     "import pytest\ndef test_a():\n    assert 1 == 1\n",
     "import pytest\ndef test_a():\n    pytest.skip('no')\n    assert 1 == 1\n"),
    ("importorskip_added",
     "import pytest\ndef test_a():\n    assert 1 == 1\n",
     "import pytest\ndef test_a():\n    pytest.importorskip('torch')\n    assert 1 == 1\n"),
    ("module_pytestmark_added",
     "def test_a():\n    assert 1 == 1\n",
     "import pytest\npytestmark = pytest.mark.skip\ndef test_a():\n    assert 1 == 1\n"),
    ("net_zero_assertion_swap",
     "def test_a():\n    assert compute() == 3\n",
     "def test_a():\n    assert True\n"),
    ("assertion_removed",
     "def test_a():\n    assert 1 == 1\n    assert 2 == 2\n",
     "def test_a():\n    assert 1 == 1\n"),
    ("decorator_changed",
     "import pytest\n@pytest.mark.covers('T1')\ndef test_a():\n    assert 1 == 1\n",
     "import pytest\n@pytest.mark.covers('T2')\ndef test_a():\n    assert 1 == 1\n"),
    ("test_function_removed",
     "def test_a():\n    assert 1 == 1\ndef test_b():\n    assert 2 == 2\n",
     "def test_a():\n    assert 1 == 1\n"),
    ("method_assertion_weakened",
     "class TestK:\n    def test_a(self):\n        assert abs(1 - 1) < 1e-9\n",
     "class TestK:\n    def test_a(self):\n        assert abs(1 - 1) < 1e-3\n"),
]


@pytest.mark.parametrize("case_id, base_source, weakened_source", WEAKENING_CASES,
                         ids=[case[0] for case in WEAKENING_CASES])
def test_ast_comparison_flags_each_weakening(case_id, base_source, weakened_source):
    """Every weakening pattern yields at least one human-readable reason."""
    assert reasons_of(base_source, weakened_source), case_id


# (case id, base source, benign edit): none of these may be flagged (growth, comments, renames of locals).
BENIGN_CASES = [
    ("identical", "def test_a():\n    assert 1 == 1\n", "def test_a():\n    assert 1 == 1\n"),
    ("comment_only", "def test_a():\n    assert 1 == 1\n", "def test_a():\n    # why\n    assert 1 == 1  # ok\n"),
    ("assertion_added", "def test_a():\n    assert 1 == 1\n", "def test_a():\n    assert 1 == 1\n    assert 2 == 2\n"),
    ("new_test_added", "def test_a():\n    assert 1 == 1\n",
     "def test_a():\n    assert 1 == 1\ndef test_new():\n    assert True\n"),
    ("parametrize_row_appended",
     "import pytest\n@pytest.mark.parametrize('value', [1, 2])\ndef test_a(value):\n    assert value > 0\n",
     "import pytest\n@pytest.mark.parametrize('value', [1, 2, 3])\ndef test_a(value):\n    assert value > 0\n"),
    ("extra_tolerance_assertion_added",
     "def test_a():\n    assert 1 == 1\n", "def test_a():\n    assert 1 == 1\n    assert abs(0) < 1e-12\n"),
]


@pytest.mark.parametrize("case_id, base_source, edited_source", BENIGN_CASES, ids=[case[0] for case in BENIGN_CASES])
def test_ast_comparison_ignores_benign_edits(case_id, base_source, edited_source):
    """Growth, comments and new tests are not reported."""
    assert reasons_of(base_source, edited_source) == [], case_id


def test_ast_comparison_falls_back_when_the_new_source_does_not_parse():
    """An unparsable file is reported (regex fallback) rather than silently accepted."""
    reasons = reasons_of("def test_a():\n    assert 1 == 1\n", "def test_a(:\n    assert 1 == 1\n")
    assert reasons and "parse" in reasons[0].lower()


def test_audit_flags_bare_numeric_loosening_without_any_comment(tmp_path):
    """`< 1e-9` -> `< 1e-3` with no 'tolerance' word on the line is still suspicious."""
    selector = load_selector_module()
    commit_files(tmp_path, {"tests/test_c.py": "def test_c():\n    assert abs(1.0 - 1.0) < 1e-9\n"})
    (tmp_path / "tests" / "test_c.py").write_text("def test_c():\n    assert abs(1.0 - 1.0) < 1e-3\n")
    (finding,) = selector.audit_modified_tests("HEAD", project_root=tmp_path)
    assert finding["suspicious"] is True and finding["reasons"]


# Git detects a rename only above a similarity threshold, so renamed fixtures carry padding lines.
PADDING_LINES = "".join(f"    assert {value} == {value}\n" for value in range(12))


def test_audit_follows_rename_with_weakening(tmp_path):
    """A renamed file is compared with its OLD path's base version."""
    selector = load_selector_module()
    base_text = "def test_x():\n    assert abs(1 - 1) < 1e-9\n" + PADDING_LINES
    commit_files(tmp_path, {"tests/test_old.py": base_text})
    run_git(tmp_path, "mv", "tests/test_old.py", "tests/test_new_name.py")
    (tmp_path / "tests" / "test_new_name.py").write_text(base_text.replace("1e-9", "1e-3"))
    run_git(tmp_path, "add", "-A")
    findings = selector.audit_modified_tests("HEAD", project_root=tmp_path)
    assert [finding["status"][0] for finding in findings] == ["R"]
    assert findings[0]["path"] == "tests/test_new_name.py" and findings[0]["suspicious"] is True


def test_audit_pure_rename_is_not_suspicious(tmp_path):
    """Renaming without content change is reported but not suspicious."""
    selector = load_selector_module()
    commit_files(tmp_path, {"tests/test_old.py": "def test_x():\n    assert 1 == 1\n" + PADDING_LINES})
    run_git(tmp_path, "mv", "tests/test_old.py", "tests/test_renamed.py")
    (finding,) = selector.audit_modified_tests("HEAD", project_root=tmp_path)
    assert finding["status"][0] == "R" and finding["suspicious"] is False


@pytest.mark.parametrize("protected_path", ["tests/conftest.py", "tests/unit/conftest.py", "conftest.py"])
def test_audit_flags_any_conftest_diff_including_untracked(tmp_path, protected_path):
    """Any change to, or new, conftest.py is suspicious (hooks can skip or deselect tests)."""
    selector = load_selector_module()
    commit_files(tmp_path, {"tests/test_a.py": "def test_a():\n    assert 1 == 1\n"})
    target = tmp_path / protected_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("# hook\n")
    findings = {finding["path"]: finding for finding in selector.audit_modified_tests("HEAD", project_root=tmp_path)}
    assert findings[protected_path]["suspicious"] is True


def test_audit_flags_pytest_section_of_pyproject_only(tmp_path):
    """Editing [tool.pytest.*] is suspicious; editing another table is not reported."""
    selector = load_selector_module()
    pyproject = '[project]\nname = "x"\n\n[tool.pytest.ini_options]\naddopts = "-m a"\n'
    commit_files(tmp_path, {"pyproject.toml": pyproject, "tests/test_a.py": "def test_a():\n    assert 1 == 1\n"})
    (tmp_path / "pyproject.toml").write_text(pyproject.replace('name = "x"', 'name = "y"'))
    assert selector.audit_modified_tests("HEAD", project_root=tmp_path) == []
    (tmp_path / "pyproject.toml").write_text(pyproject.replace('"-m a"', '"-m b"'))
    (finding,) = selector.audit_modified_tests("HEAD", project_root=tmp_path)
    assert finding["suspicious"] is True


def test_default_base_is_merge_base_with_main_so_committed_weakening_is_seen(tmp_path):
    """After committing the weakening on a branch, the default base still exposes it."""
    selector = load_selector_module()
    commit_files(tmp_path, {"tests/test_a.py": "def test_a():\n    assert abs(1 - 1) < 1e-9\n"})
    run_git(tmp_path, "checkout", "-q", "-b", "feature")
    (tmp_path / "tests" / "test_a.py").write_text("def test_a():\n    assert abs(1 - 1) < 1e-3\n")
    run_git(tmp_path, "commit", "-q", "-am", "weaken")
    base_ref = selector.default_base_ref(tmp_path)
    assert base_ref != "HEAD"
    (finding,) = selector.audit_modified_tests(base_ref, project_root=tmp_path)
    assert finding["suspicious"] is True


def test_default_base_falls_back_to_head_without_main(tmp_path):
    """No main and no origin/main -> HEAD."""
    selector = load_selector_module()
    run_git(tmp_path, "init", "-q", "-b", "trunk")
    (tmp_path / "file.txt").write_text("x")
    run_git(tmp_path, "add", "-A")
    run_git(tmp_path, "commit", "-q", "-m", "base")
    assert selector.default_base_ref(tmp_path) == "HEAD"


def run_cli(repo: Path, *cli_arguments: str) -> subprocess.CompletedProcess:
    """Run the selector's main() against a synthetic repo and capture its exit code and output."""
    runner = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('affected_tests', {str(PROJECT_ROOT / 'scripts' / 'affected_tests.py')!r})\n"
        "module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)\n"
        f"sys.exit(module.main({list(cli_arguments)!r}, project_root=__import__('pathlib').Path({str(repo)!r})))\n"
    )
    return subprocess.run([sys.executable, "-c", runner], capture_output=True, text=True)


def make_repo_with_weakened_test(repo: Path) -> None:
    """Commit a passing test, then loosen its tolerance in the working tree."""
    commit_files(repo, {
        "tests/test_traceability.py": "def test_ok():\n    assert abs(1 - 1) < 1e-9\n",
    })
    (repo / "tests" / "test_traceability.py").write_text("def test_ok():\n    assert abs(1 - 1) < 1e-3\n")


def test_cli_exits_nonzero_on_suspicious_hit_and_allow_flag_opts_out(tmp_path):
    """Gate: exit 3 on a SUSPICIOUS finding, 0 with --allow-suspicious."""
    make_repo_with_weakened_test(tmp_path)
    blocked = run_cli(tmp_path)
    assert blocked.returncode == 3 and "SUSPICIOUS" in blocked.stdout
    assert run_cli(tmp_path, "--allow-suspicious").returncode == 0


def test_cli_gate_also_applies_in_run_mode(tmp_path):
    """--run still runs the tests (they pass) but the exit code reports the suspicious edit."""
    make_repo_with_weakened_test(tmp_path)
    assert run_cli(tmp_path, "--run").returncode == 3
    assert run_cli(tmp_path, "--run", "--allow-suspicious").returncode == 0


def test_cli_clean_repo_exits_zero(tmp_path):
    """No edits -> exit 0."""
    commit_files(tmp_path, {"tests/test_traceability.py": "def test_ok():\n    assert 1 == 1\n"})
    assert run_cli(tmp_path).returncode == 0


def test_relative_imports_are_resolved_through_the_file_package(tmp_path):
    """`from . import base` and `from .base import VALUE` inside pkg link mid to base."""
    selector = load_selector_module()
    project = build_tiny_project(tmp_path)
    (project / "src" / "pkg" / "mid.py").write_text("from . import base\n")
    (project / "src" / "pkg" / "deep.py").write_text("from .base import VALUE\n")
    selection = selector.select_tests(
        [project / "src" / "pkg" / "base.py"], tests_root=project / "tests", source_root=project / "src"
    )
    assert "pkg.mid" in selector.reverse_dependency_closure({"pkg.base"}, project / "src")
    assert "pkg.deep" in selector.reverse_dependency_closure({"pkg.base"}, project / "src")
    assert "tests/test_mid.py" in selection["test_files"]


def test_script_change_selects_the_tests_that_load_it_by_path(tmp_path):
    """A test that spec-loads scripts/tool.py is selected when that script changes; others are not."""
    selector = load_selector_module()
    project = build_tiny_project(tmp_path)
    (project / "scripts").mkdir()
    (project / "scripts" / "tool.py").write_text("X = 1\n")
    (project / "tests" / "test_tool.py").write_text(
        "from pathlib import Path\nTOOL = Path('x') / 'scripts' / 'tool.py'\ndef test_t(): pass\n"
    )
    selection = selector.select_tests(
        [project / "scripts" / "tool.py"], tests_root=project / "tests", source_root=project / "src"
    )
    assert selection["full_suite"] is False
    assert "tests/test_tool.py" in selection["test_files"]
    assert "tests/test_other.py" not in selection["test_files"]


def test_script_without_a_linked_test_selects_the_full_suite(tmp_path):
    """Mapping unreliable (no test names the script) -> full suite, so nothing is missed."""
    selector = load_selector_module()
    project = build_tiny_project(tmp_path)
    (project / "viz").mkdir()
    (project / "viz" / "orphan.py").write_text("X = 1\n")
    selection = selector.select_tests(
        [project / "viz" / "orphan.py"], tests_root=project / "tests", source_root=project / "src"
    )
    assert selection["full_suite"] is True


def test_covered_transform_ids_uses_the_shared_traceability_matcher(tmp_path):
    """Non-string covers arguments are ignored by the selector exactly as the shared matcher decides."""
    selector = load_selector_module()
    test_file = tmp_path / "test_x.py"
    test_file.write_text(
        "import pytest\nID = 'T9'\n"
        "@pytest.mark.covers('T1', 'T2')\ndef test_a():\n    assert 1\n"
        "@pytest.mark.covers(ID)\ndef test_b():\n    assert 1\n"
    )
    assert selector.covered_transform_ids(test_file) == {"T1", "T2"}
