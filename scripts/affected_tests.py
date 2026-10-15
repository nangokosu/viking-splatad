"""Select (and optionally run) only the tests affected by the current code changes.

Used by the read-only `transform-test-runner` agent so it does not need to run the
whole suite after every edit.

Selection algorithm:
  1. Changed files = `git diff --name-only <base>` plus untracked files.
  2. Build the import graph between project modules (AST scan of src/).
  3. Take the reverse-dependency closure of every changed module: a change to module A
     also affects every module that imports A, directly or transitively
     (cost O(modules * edges), trivial at this size).
  4. Affected tests = changed test files + test files importing any module in the closure
     + the traceability meta-tests (always cheap, always relevant).
  5. A change to non-Python inputs (pyproject, configs) or to conftest selects the full suite.
Transform ids in scope are reported by reading the `@pytest.mark.covers` markers of the
selected test files.
"""

import argparse
import ast
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Set

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_ROOT = PROJECT_ROOT / "src"
TESTS_ROOT = PROJECT_ROOT / "tests"
# Meta-test that guards the one-test-per-transform rule; cheap, so always included.
TRACEABILITY_TEST = "tests/test_traceability.py"
# Changes to these paths can affect any test, so they select the full suite.
FULL_SUITE_TRIGGER_NAMES = {"pyproject.toml", "conftest.py"}
FULL_SUITE_TRIGGER_DIRS = {"configs"}


def module_name_of(path: Path, source_root: Path = SOURCE_ROOT) -> str:
    """Convert a file under the source root into its dotted module name.

    Args:
        path: Absolute path of a Python file inside source_root.
        source_root: Directory that holds the top-level package.

    Returns:
        Dotted module name, e.g. "viking_splatad.extract.pose".
    """
    relative_parts = path.relative_to(source_root).with_suffix("").parts
    # A package's __init__ is addressed by the package name itself.
    if relative_parts[-1] == "__init__":
        relative_parts = relative_parts[:-1]
    return ".".join(relative_parts)


def imported_module_names(python_file: Path) -> Set[str]:
    """Collect every dotted name a file imports (including `from X import Y` as X and X.Y).

    Args:
        python_file: Python source file to scan.

    Returns:
        Set of imported dotted names, over-approximating submodule imports.
    """
    imported: Set[str] = set()
    for node in ast.walk(ast.parse(python_file.read_text())):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
            # `from pkg import module` may import a submodule, so also record pkg.module.
            imported.update(f"{node.module}.{alias.name}" for alias in node.names)
    return imported


def changed_paths(base_ref: str) -> List[Path]:
    """List files changed relative to a git ref, plus untracked files.

    Args:
        base_ref: Git ref to diff the working tree against (e.g. "HEAD").

    Returns:
        Absolute paths of changed or new files that still exist.
    """
    def git_lines(arguments: List[str]) -> List[str]:
        """Run git in the project root and return non-empty output lines."""
        completed = subprocess.run(
            ["git", *arguments], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
        )
        return [line for line in completed.stdout.splitlines() if line]

    changed_relative = git_lines(["diff", "--name-only", base_ref])
    changed_relative += git_lines(["ls-files", "--others", "--exclude-standard"])
    absolute_paths = {PROJECT_ROOT / relative for relative in changed_relative}
    # Deleted files cannot be imported or run, so drop them.
    return sorted(path for path in absolute_paths if path.exists())


def reverse_dependency_closure(changed_modules: Set[str], source_root: Path = SOURCE_ROOT) -> Set[str]:
    """Expand changed modules with every module that (transitively) imports them.

    Args:
        changed_modules: Dotted names of modules whose files changed.
        source_root: Directory holding the package sources.

    Returns:
        Changed modules plus all their transitive importers.
    """
    # Step 1: forward edges module -> set of project modules it imports.
    module_files = {module_name_of(path, source_root): path for path in source_root.rglob("*.py")}
    imports_of = {name: imported_module_names(path) for name, path in module_files.items()}
    # Step 2: fixed-point expansion: add any module importing something already affected.
    affected = set(changed_modules)
    grew = True
    while grew:
        grew = False
        for module_name, imported in imports_of.items():
            if module_name not in affected and imported & affected:
                affected.add(module_name)
                grew = True
    return affected


def covered_transform_ids(test_file: Path) -> Set[str]:
    """Read `@pytest.mark.covers("T..")` ids from a test file without importing it.

    Args:
        test_file: Test file to scan.

    Returns:
        Set of transform ids the file claims to cover.
    """
    covered: Set[str] = set()
    for node in ast.walk(ast.parse(test_file.read_text())):
        # Match the call form covers("T1", ...) regardless of how the marker is spelled.
        if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "covers":
            covered.update(arg.value for arg in node.args if isinstance(arg, ast.Constant))
    return covered


def select_tests(changed_files: Iterable[Path], tests_root: Path = TESTS_ROOT,
                 source_root: Path = SOURCE_ROOT) -> Dict[str, object]:
    """Choose the tests affected by a set of changed files.

    Args:
        changed_files: Absolute paths of changed files.
        tests_root: Directory containing the tests.
        source_root: Directory containing package sources.

    Returns:
        Dict with keys: "full_suite" (bool), "test_files" (sorted repo-relative paths),
        "transform_ids" (sorted ids in scope), "changed_modules" (sorted names).
    """
    changed_files = list(changed_files)
    project_root = tests_root.parent
    all_tests = sorted(tests_root.rglob("test_*.py"))

    # Step 1: non-code inputs and shared fixtures can affect anything -> everything runs.
    triggers_full_suite = any(
        path.name in FULL_SUITE_TRIGGER_NAMES
        or (path.relative_to(project_root).parts[0] in FULL_SUITE_TRIGGER_DIRS)
        for path in changed_files
    )

    # Step 2: map changed source files to modules and expand to importers.
    changed_modules = {
        module_name_of(path, source_root) for path in changed_files
        if path.suffix == ".py" and source_root in path.parents
    }
    affected_modules = reverse_dependency_closure(changed_modules, source_root)

    # Step 3: a test is affected when it is itself changed or imports an affected module.
    selected: Set[Path] = {path for path in changed_files if path in all_tests}
    for test_file in all_tests:
        if imported_module_names(test_file) & affected_modules:
            selected.add(test_file)
    selected.add(project_root / TRACEABILITY_TEST)
    if triggers_full_suite:
        selected = set(all_tests)

    existing_selected = sorted(path for path in selected if path.exists())
    transform_ids: Set[str] = set()
    for test_file in existing_selected:
        transform_ids |= covered_transform_ids(test_file)
    return {
        "full_suite": triggers_full_suite,
        "test_files": [str(path.relative_to(project_root)) for path in existing_selected],
        "transform_ids": sorted(transform_ids),
        "changed_modules": sorted(changed_modules),
    }


# Substrings that indicate an assertion or a numeric tolerance on a diff line.
ASSERTION_PATTERN = re.compile(r"\bassert\b|pytest\.approx|allclose|atol|rtol|tolerance|abs=|rel=")
TOLERANCE_PATTERN = re.compile(r"atol|rtol|tolerance|abs=|rel=|pytest\.approx")


def audit_modified_tests(base_ref: str, project_root: Path = PROJECT_ROOT) -> List[Dict[str, object]]:
    """Report edits to tests that already existed at `base_ref` (tests are the spec).

    A test file counts as an edit when git reports it as modified (M) or deleted (D); newly
    added tests are not flagged. For each edited file the diff is scanned for weakening signals:
    removed assertion lines not matched by added ones, and any changed tolerance expression.

    Args:
        base_ref: Git ref holding the trusted version of the tests (e.g. "HEAD").
        project_root: Repository root.

    Returns:
        One dict per edited test file with keys: "path", "status", "removed_assertions",
        "added_assertions", "tolerance_lines_changed", "suspicious" (bool).
    """
    def git_output(arguments: List[str]) -> str:
        """Run git in the project root and return stdout."""
        return subprocess.run(
            ["git", *arguments], cwd=project_root, capture_output=True, text=True, check=True
        ).stdout

    findings: List[Dict[str, object]] = []
    for status_line in git_output(["diff", "--name-status", base_ref, "--", "tests"]).splitlines():
        status, _, relative_path = status_line.partition("\t")
        # Only modified/deleted files existed before; additions are new tests and are fine.
        if status not in ("M", "D"):
            continue
        # -U0 keeps only changed lines so counting +/- lines is exact.
        diff_lines = git_output(["diff", "-U0", base_ref, "--", relative_path]).splitlines()
        removed = [line[1:] for line in diff_lines if line.startswith("-") and not line.startswith("---")]
        added = [line[1:] for line in diff_lines if line.startswith("+") and not line.startswith("+++")]
        removed_assertions = sum(1 for line in removed if ASSERTION_PATTERN.search(line))
        added_assertions = sum(1 for line in added if ASSERTION_PATTERN.search(line))
        # Only REMOVED lines matter: loosening a tolerance means editing/removing an existing line;
        # purely added tolerance expressions belong to new tests.
        tolerance_changed = sum(1 for line in removed if TOLERANCE_PATTERN.search(line))
        findings.append({
            "path": relative_path,
            "status": status,
            "removed_assertions": removed_assertions,
            "added_assertions": added_assertions,
            "tolerance_lines_changed": tolerance_changed,
            # Deleted files, fewer assertions afterwards, or any tolerance edit all need human scrutiny.
            "suspicious": status == "D" or removed_assertions > added_assertions or tolerance_changed > 0,
        })
    return findings


def main() -> int:
    """CLI entry point: print the selection and optionally run it.

    Returns:
        Process exit code (pytest's when --run is given, else 0).
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="HEAD", help="git ref to diff against")
    parser.add_argument("--run", action="store_true", help="run the selected tests with pytest")
    parser.add_argument("--all", action="store_true", help="ignore the diff and select the full suite")
    arguments = parser.parse_args()

    if arguments.all:
        selection = select_tests([PROJECT_ROOT / "pyproject.toml"])
    else:
        selection = select_tests(changed_paths(arguments.base))
    print(f"changed modules : {selection['changed_modules']}")
    print(f"full suite      : {selection['full_suite']}")
    print(f"transform ids   : {selection['transform_ids']}")
    print("test files      :")
    for test_file in selection["test_files"]:
        print(f"  {test_file}")
    # Tests are the spec: surface any edit to a pre-existing test so a reviewer can challenge it.
    edited_tests = audit_modified_tests(arguments.base)
    print("existing tests modified:" + (" none" if not edited_tests else ""))
    for finding in edited_tests:
        print(f"  {'SUSPICIOUS ' if finding['suspicious'] else ''}{finding['status']} {finding['path']}: "
              f"-{finding['removed_assertions']}/+{finding['added_assertions']} assertion lines, "
              f"{finding['tolerance_lines_changed']} tolerance lines changed")
    if not arguments.run:
        return 0
    # `-p no:cacheprovider` keeps the read-only agent from writing .pytest_cache into the repo.
    return subprocess.call(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *selection["test_files"]],
        cwd=PROJECT_ROOT,
    )


if __name__ == "__main__":
    sys.exit(main())
