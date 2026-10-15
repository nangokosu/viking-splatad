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
