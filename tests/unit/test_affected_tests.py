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
