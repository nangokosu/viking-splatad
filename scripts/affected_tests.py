"""Select (and optionally run) only the tests affected by the current code changes.

Used by the read-only `transform-test-runner` agent so it does not need to run the
whole suite after every edit.

Selection algorithm:
  1. Changed files = `git diff --name-only <base>` plus untracked files.
  2. Build the import graph between project modules (AST scan of src/; relative imports are
     resolved through the importing file's package).
  3. Take the reverse-dependency closure of every changed module: a change to module A
     also affects every module that imports A, directly or transitively
     (cost O(modules * edges), trivial at this size).
  4. Affected tests = changed test files + test files importing any module in the closure
     + tests that load a changed scripts/, viz/ or .claude/ file (by file name or import)
     + the traceability meta-tests (always cheap, always relevant).
  5. A change to non-Python inputs (pyproject, configs) or to conftest selects the full suite,
     and so does a changed scripts/viz file that no test names (the mapping is then unreliable).
Transform ids in scope are reported by reading the `@pytest.mark.covers` markers of the
selected test files (same matcher as tests/test_traceability.py: scripts/covers_rules.py).

Audit ("existing tests modified"): every test file, conftest.py, pyproject pytest section and
enforcement fixture that differs from the base is compared per function with the AST
(changed constants, tolerance keywords, expected values, parametrize rows, removed or altered
assertions, changed decorators, added skip / xfail / gpu / real_data gates). The base defaults to
the merge-base with `main` (else `origin/main`, else HEAD) so committed weakening is still seen.
Any SUSPICIOUS finding makes the exit code 3 (also with --run) unless `--allow-suspicious` is
given; only the main agent may pass it, after stating the reason to the user.
"""

import argparse
import ast
import collections
import copy
import difflib
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

# covers_rules lives next to this script; make it importable however this file was loaded.
SCRIPTS_DIRECTORY = Path(__file__).resolve().parent
if str(SCRIPTS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIRECTORY))
import covers_rules  # noqa: E402  (needs the sys.path entry above)

PROJECT_ROOT = SCRIPTS_DIRECTORY.parent
SOURCE_ROOT = PROJECT_ROOT / "src"
TESTS_ROOT = PROJECT_ROOT / "tests"
# Meta-test that guards the one-test-per-transform rule; cheap, so always included.
TRACEABILITY_TEST = "tests/test_traceability.py"
# Changes to these paths can affect any test, so they select the full suite.
FULL_SUITE_TRIGGER_NAMES = {"pyproject.toml", "conftest.py"}
FULL_SUITE_TRIGGER_DIRS = {"configs"}
# Top-level directories whose files tests load by path (spec_from_file_location) or import.
LINKED_FILE_DIRS = {"scripts", "viz", ".claude"}
# Exit code of a run whose audit has a SUSPICIOUS finding; 3 is distinct from pytest's 0-5 failure
# codes that matter here (1 failed, 2 interrupted, 4 usage, 5 nothing collected) and from argparse's 2.
SUSPICIOUS_EXIT_CODE = 3
# Enforcement data that must not change silently (see tests/test_traceability.py, tests/conftest.py).
PROTECTED_DATA_FILES = {
    "tests/fixtures/transform_manifest.json",
    "tests/fixtures/test_exclusions_allowlist.json",
}
# Names of expected-value variables / constants whose edit changes what a test accepts.
EXPECTED_NAME_PATTERN = re.compile(r"expect|golden|truth|reference|tol|eps", re.IGNORECASE)
# Keyword names that carry a numeric tolerance (atol, rtol, abs_tol, tolerance, eps, ...).
TOLERANCE_KEYWORD_PATTERN = re.compile(r"tol|eps", re.IGNORECASE)
# approx() takes its tolerances as abs= / rel=.
APPROX_KEYWORDS = {"abs", "rel"}
# Calls that compare numbers or assert, beyond assert statements (their arguments are the spec).
NUMERIC_CHECK_NAMES = {"allclose", "isclose", "approx", "raises", "warns"}


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


def imported_module_names(python_file: Path, package_root: Optional[Path] = None) -> Set[str]:
    """Collect every dotted name a file imports (including `from X import Y` as X and X.Y).

    Relative imports (`from . import x`, `from ..pkg import y`) are resolved to absolute names
    with the file's own package: for `a/b/c.py` the package is `a.b`, and `level` dots strip
    `level - 1` trailing package parts before the module part is appended.

    Args:
        python_file: Python source file to scan.
        package_root: Directory the dotted package path starts at (e.g. src/); when None,
            relative imports cannot be resolved and are skipped.

    Returns:
        Set of imported dotted names, over-approximating submodule imports.
    """
    package_parts: Optional[List[str]] = None
    if package_root is not None:
        # Step 1: the file's package = its dotted path minus the module part (works for __init__ too).
        package_parts = list(python_file.relative_to(package_root).with_suffix("").parts[:-1])
    imported: Set[str] = set()
    for ast_node in ast.walk(ast.parse(python_file.read_text())):
        if isinstance(ast_node, ast.Import):
            imported.update(alias.name for alias in ast_node.names)
        elif isinstance(ast_node, ast.ImportFrom):
            # Step 2: absolute name of the module part (None if a relative import cannot be resolved).
            if ast_node.level == 0:
                resolved_module = ast_node.module
            elif package_parts is not None and ast_node.level - 1 <= len(package_parts):
                kept_parts = package_parts[: len(package_parts) - (ast_node.level - 1)]
                resolved_module = ".".join(kept_parts + ([ast_node.module] if ast_node.module else []))
            else:
                resolved_module = None
            if not resolved_module:
                continue
            imported.add(resolved_module)
            # `from pkg import module` may import a submodule, so also record pkg.module.
            imported.update(f"{resolved_module}.{alias.name}" for alias in ast_node.names)
    return imported


def changed_paths(base_ref: str, project_root: Path = PROJECT_ROOT) -> List[Path]:
    """List files changed relative to a git ref, plus untracked files.

    Args:
        base_ref: Git ref to diff the working tree against (e.g. "HEAD").
        project_root: Repository root.

    Returns:
        Absolute paths of changed, new or deleted files (deleted ones still name a changed module).
    """
    def git_lines(arguments: List[str]) -> List[str]:
        """Run git in the project root and return non-empty output lines.

        Args:
            arguments: git arguments after ``git``.

        Returns:
            Non-empty stdout lines (raises CalledProcessError on a non-zero exit).
        """
        completed = subprocess.run(
            ["git", *arguments], cwd=project_root, capture_output=True, text=True, check=True
        )
        return [line for line in completed.stdout.splitlines() if line]

    # "--" ends the revision list so a ref can never be read as a path (or an option).
    changed_relative = git_lines(["diff", "--name-only", base_ref, "--"])
    changed_relative += git_lines(["ls-files", "--others", "--exclude-standard"])
    return sorted({project_root / relative for relative in changed_relative})


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
    imports_of = {name: imported_module_names(path, source_root) for name, path in module_files.items()}
    # Step 2: fixed-point expansion: add any module importing something already affected.
    # Each pass adds at least one module or stops, so at most (modules + 1) passes.
    affected = set(changed_modules)
    closure_grew = True
    while closure_grew:
        closure_grew = False
        for module_name, imported in imports_of.items():
            if module_name not in affected and imported & affected:
                affected.add(module_name)
                closure_grew = True
    return affected


def covered_transform_ids(test_file: Path) -> Set[str]:
    """Read `@pytest.mark.covers("T..")` ids from a test file without importing it.

    Uses the same matcher as tests/test_traceability.py (scripts/covers_rules.py).

    Args:
        test_file: Test file to scan.

    Returns:
        Set of transform ids the file claims to cover.
    """
    return set(covers_rules.covered_ids_by_function(ast.parse(test_file.read_text())))


def tests_linked_to_file(linked_file: Path, test_files: List[Path]) -> Set[Path]:
    """Find tests that load or import a script / viz / hook file.

    A test is linked when its source text contains the file's name (the path constants and
    spec_from_file_location strings of the existing tests) or imports its stem as a module.

    Args:
        linked_file: Changed file outside src/.
        test_files: Candidate test files.

    Returns:
        The linked test files.
    """
    linked: Set[Path] = set()
    for test_file in test_files:
        source_text = test_file.read_text()
        if linked_file.name in source_text:
            linked.add(test_file)
        elif linked_file.suffix == ".py" and linked_file.stem in {
            name.split(".")[-1] for name in imported_module_names(test_file)
        }:
            linked.add(test_file)
    return linked


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

    def top_level_directory(path: Path) -> str:
        """Return the first path component below the project root ('' outside it)."""
        try:
            return path.relative_to(project_root).parts[0]
        except (ValueError, IndexError):
            return ""

    # Step 1: non-code inputs and shared fixtures can affect anything -> everything runs.
    triggers_full_suite = any(
        path.name in FULL_SUITE_TRIGGER_NAMES or top_level_directory(path) in FULL_SUITE_TRIGGER_DIRS
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
        if imported_module_names(test_file, project_root) & affected_modules:
            selected.add(test_file)
    # Step 3b: files outside src/ (scripts, viz, hooks, test helpers/fixtures) reach tests by name;
    # if nothing names a changed one the mapping is unreliable, so run everything.
    for path in changed_files:
        is_support_file = top_level_directory(path) in LINKED_FILE_DIRS or (
            top_level_directory(path) == tests_root.name and path not in all_tests
        )
        if not is_support_file or path.name in FULL_SUITE_TRIGGER_NAMES:
            continue
        linked_tests = tests_linked_to_file(path, all_tests)
        if linked_tests:
            selected |= linked_tests
        else:
            triggers_full_suite = True
    selected.add(project_root / TRACEABILITY_TEST)
    if triggers_full_suite:
        selected = set(all_tests)

    # Step 4: keep tests that exist on disk, and collect the transform ids they cover.
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


# Diff-line regexes (fallback when the AST cannot be parsed): ASSERTION_PATTERN matches any check
# (assert or tolerance call), TOLERANCE_PATTERN only the numeric-tolerance expressions.
ASSERTION_PATTERN = re.compile(r"\bassert\b|pytest\.approx|allclose|atol|rtol|tolerance|abs=|rel=")
TOLERANCE_PATTERN = re.compile(r"atol|rtol|tolerance|abs=|rel=|pytest\.approx")


def callee_name(call_node: ast.Call) -> str:
    """Return the last name segment of a call's function (``np.allclose`` -> ``allclose``)."""
    function = call_node.func
    return function.attr if isinstance(function, ast.Attribute) else getattr(function, "id", "")


def mask_numeric_constants(node: ast.AST) -> str:
    """Unparse a node with every numeric literal replaced by 0 (to tell constant-only edits apart).

    Args:
        node: Expression or statement node.

    Returns:
        Source text with numbers masked.
    """
    masked = copy.deepcopy(node)
    for inner in ast.walk(masked):
        if isinstance(inner, ast.Constant) and isinstance(inner.value, (int, float, complex)) \
                and not isinstance(inner.value, bool):
            inner.value = 0
    return ast.unparse(masked)


def check_items(function_node: ast.AST) -> List[ast.AST]:
    """List a function's checks: assert tests and numeric/exception check calls.

    Args:
        function_node: Function definition.

    Returns:
        Nodes whose unparsed text is the test's accepted behaviour.
    """
    items: List[ast.AST] = []
    for inner in ast.walk(function_node):
        if isinstance(inner, ast.Assert):
            items.append(inner.test)
        elif isinstance(inner, ast.Call):
            name = callee_name(inner)
            if name in NUMERIC_CHECK_NAMES or name.startswith("assert"):
                items.append(inner)
    return items


def tolerance_items(function_node: ast.AST) -> List[str]:
    """List ``callee(keyword=value)`` strings for every tolerance-like keyword in a function.

    Args:
        function_node: Function definition.

    Returns:
        One string per tolerance keyword (names containing tol/eps, or approx's abs=/rel=).
    """
    items: List[str] = []
    for inner in ast.walk(function_node):
        if not isinstance(inner, ast.Call):
            continue
        name = callee_name(inner)
        for keyword in inner.keywords:
            if keyword.arg and (TOLERANCE_KEYWORD_PATTERN.search(keyword.arg)
                                or (name == "approx" and keyword.arg in APPROX_KEYWORDS)):
                items.append(f"{name}({keyword.arg}={ast.unparse(keyword.value)})")
    return items


def expected_assignments(scope_node: ast.AST) -> List[str]:
    """List unparsed assignments whose target name looks like an expected value or tolerance.

    Args:
        scope_node: Function or module node.

    Returns:
        Source of each such assignment.
    """
    items: List[str] = []
    for inner in ast.walk(scope_node):
        if isinstance(inner, (ast.Assign, ast.AnnAssign)):
            targets = inner.targets if isinstance(inner, ast.Assign) else [inner.target]
            target_names = " ".join(
                node.id for target in targets for node in ast.walk(target) if isinstance(node, ast.Name)
            )
            if EXPECTED_NAME_PATTERN.search(target_names):
                items.append(ast.unparse(inner))
    return items


def non_running_calls(function_node: ast.AST) -> List[str]:
    """List call sources that skip or xfail at run time (pytest.skip, importorskip, pytest.xfail)."""
    return [
        ast.unparse(inner) for inner in ast.walk(function_node)
        if isinstance(inner, ast.Call) and covers_rules.is_non_running_marker(inner.func)
    ]


def compare_decorators(label: str, base_decorators: List[ast.AST], new_decorators: List[ast.AST]) -> List[str]:
    """Compare two decorator lists: removed / changed ones and newly added non-running gates.

    A parametrize decorator may grow (extra rows) but not lose or change rows or argument names.

    Args:
        label: Function or class name for the messages.
        base_decorators: Decorators before.
        new_decorators: Decorators after.

    Returns:
        Reason strings.
    """
    reasons: List[str] = []
    new_texts = [ast.unparse(decorator) for decorator in new_decorators]
    for base_decorator in base_decorators:
        base_text = ast.unparse(base_decorator)
        if base_text in new_texts:
            continue
        is_parametrize = isinstance(base_decorator, ast.Call) and callee_name(base_decorator) == "parametrize"
        if is_parametrize and len(base_decorator.args) >= 2:
            base_names = ast.unparse(base_decorator.args[0])
            base_rows = rows_of(base_decorator.args[1])
            # A new parametrize with the same argument names and a superset of rows is growth.
            grown = any(
                isinstance(candidate, ast.Call) and callee_name(candidate) == "parametrize"
                and len(candidate.args) >= 2 and ast.unparse(candidate.args[0]) == base_names
                and base_rows is not None and rows_of(candidate.args[1]) is not None
                and set(base_rows) <= set(rows_of(candidate.args[1]))
                for candidate in new_decorators
            )
            if not grown:
                reasons.append(f"{label}: parametrize argument list changed or removed: {base_text}")
        else:
            reasons.append(f"{label}: decorator changed or removed: {base_text}")
    base_texts = {ast.unparse(decorator) for decorator in base_decorators}
    for new_decorator in new_decorators:
        new_text = ast.unparse(new_decorator)
        if new_text not in base_texts and covers_rules.is_non_running_marker(new_decorator):
            reasons.append(f"{label}: skip/xfail/gpu/real_data gate added: {new_text}")
    return reasons


def rows_of(argvalues_node: ast.AST) -> Optional[List[str]]:
    """Return the unparsed rows of a literal list/tuple parametrize argument, else None."""
    if isinstance(argvalues_node, (ast.List, ast.Tuple)):
        return [ast.unparse(element) for element in argvalues_node.elts]
    return None


def definition_index(syntax_tree: ast.Module) -> Dict[str, ast.AST]:
    """Map qualified names (``Class.method``) to function and class definitions of a module.

    Args:
        syntax_tree: Parsed module.

    Returns:
        Dict name -> node; functions nested inside functions are covered by their parent.
    """
    index: Dict[str, ast.AST] = {}

    def visit(scope_node: ast.AST, prefix: str) -> None:
        """Index definitions directly inside a module or class body, recursing into classes."""
        for child in ast.iter_child_nodes(scope_node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                index[prefix + child.name] = child
                if isinstance(child, ast.ClassDef):
                    visit(child, prefix + child.name + ".")

    visit(syntax_tree, "")
    return index


def multiset_difference(base_items: List[str], new_items: List[str]) -> List[str]:
    """Items present in the base more often than in the new list (multiset subtraction)."""
    return list((collections.Counter(base_items) - collections.Counter(new_items)).elements())


def compare_test_sources(base_source: str, new_source: str) -> List[str]:
    """Compare two versions of a test module and list every weakening signal.

    Signals (any one makes the edit suspicious): a function or class removed; a check (assert
    test, approx / allclose / raises / assert_* call) removed or altered, reported as a constant
    edit when only numeric literals differ; a tolerance keyword (anything containing tol/eps,
    approx's abs=/rel=) changed; an expected-value assignment changed; a parametrize list
    changed or shrunk; a decorator changed or removed; a skip / xfail / importorskip / gpu /
    real_data gate added (decorator, call or module pytestmark). Pure growth (new tests, extra
    assertions, extra parametrize rows, comments) is not reported.

    Args:
        base_source: Module text at the base.
        new_source: Module text now.

    Returns:
        Reason strings (empty = nothing suspicious). If either text does not parse, a single
        reason says so and the caller falls back to the line-regex heuristics.
    """
    # Step 1: parse both sides; a failure cannot be compared structurally.
    try:
        base_tree, new_tree = ast.parse(base_source), ast.parse(new_source)
    except SyntaxError as error:
        return [f"AST parse failed ({error.msg}); regex fallback applies"]
    base_index, new_index = definition_index(base_tree), definition_index(new_tree)
    reasons: List[str] = []
    # Step 2: module-level constants (expected values / tolerances) and newly added gates.
    for lost in multiset_difference(expected_assignments_top_level(base_tree), expected_assignments_top_level(new_tree)):
        reasons.append(f"module-level expected/tolerance constant changed or removed: {lost}")
    base_gates = {ast.unparse(statement) for statement in base_tree.body if covers_rules.module_level_non_running(
        ast.Module(body=[statement], type_ignores=[]))}
    for statement in new_tree.body:
        gate_text = ast.unparse(statement)
        if gate_text not in base_gates and covers_rules.module_level_non_running(
                ast.Module(body=[statement], type_ignores=[])):
            reasons.append(f"module-level skip/gate added: {gate_text}")
    # Step 3: per definition, compare decorators, checks, tolerances, expected values and skip calls.
    for qualified_name, base_node in base_index.items():
        new_node = new_index.get(qualified_name)
        if new_node is None:
            reasons.append(f"{qualified_name}: definition removed")
            continue
        reasons += compare_decorators(qualified_name, base_node.decorator_list, new_node.decorator_list)
        if isinstance(base_node, ast.ClassDef):
            continue
        base_checks, new_checks = check_items(base_node), check_items(new_node)
        new_masked = [mask_numeric_constants(node) for node in new_checks]
        new_texts = [ast.unparse(node) for node in new_checks]
        for lost_text in multiset_difference([ast.unparse(node) for node in base_checks], new_texts):
            lost_node = next(node for node in base_checks if ast.unparse(node) == lost_text)
            if mask_numeric_constants(lost_node) in new_masked:
                reasons.append(f"{qualified_name}: numeric constant changed in a check: {lost_text}")
            else:
                reasons.append(f"{qualified_name}: check removed or altered: {lost_text}")
        for lost in multiset_difference(tolerance_items(base_node), tolerance_items(new_node)):
            reasons.append(f"{qualified_name}: tolerance keyword changed or removed: {lost}")
        for lost in multiset_difference(expected_assignments(base_node), expected_assignments(new_node)):
            reasons.append(f"{qualified_name}: expected-value assignment changed or removed: {lost}")
        added_gates = multiset_difference(non_running_calls(new_node), non_running_calls(base_node))
        reasons += [f"{qualified_name}: skip/xfail call added: {gate}" for gate in added_gates]
    return reasons


def expected_assignments_top_level(syntax_tree: ast.Module) -> List[str]:
    """Expected-value / tolerance assignments written directly at module level."""
    return expected_assignments(ast.Module(
        body=[statement for statement in syntax_tree.body if isinstance(statement, (ast.Assign, ast.AnnAssign))],
        type_ignores=[],
    ))


def pytest_config_section(pyproject_text: str) -> str:
    """Extract the ``[tool.pytest...]`` tables of a pyproject.toml verbatim.

    Args:
        pyproject_text: Full file text.

    Returns:
        The concatenated pytest tables (header line to the next table header).
    """
    sections: List[str] = []
    capturing = False
    for line in pyproject_text.splitlines():
        is_header = bool(re.match(r"^\[[^\]\n]+\]\s*$", line))
        if is_header:
            capturing = line.strip().startswith("[tool.pytest")
        if capturing:
            sections.append(line.rstrip())
    return "\n".join(sections)


def default_base_ref(project_root: Path = PROJECT_ROOT) -> str:
    """Pick the trusted ref for the audit: merge-base with main, else origin/main, else HEAD.

    The merge-base keeps a weakening that was already committed on the branch visible.

    Args:
        project_root: Repository root.

    Returns:
        A commit hash, or "HEAD" when neither branch exists.
    """
    for candidate in ("main", "origin/main"):
        completed = subprocess.run(
            ["git", "merge-base", "HEAD", candidate], cwd=project_root, capture_output=True, text=True
        )
        if completed.returncode == 0 and completed.stdout.strip():
            return completed.stdout.strip()
    return "HEAD"


def audit_modified_tests(base_ref: str, project_root: Path = PROJECT_ROOT) -> List[Dict[str, object]]:
    """Report edits to tests that already existed at `base_ref` (tests are the spec).

    Audited paths: every .py under tests/, any conftest.py, the pytest tables of pyproject.toml
    and the enforcement fixtures. A file counts when git reports it as modified, deleted,
    renamed, copied or type-changed (renames / copies are compared with their OLD path's base
    text); newly added tests are not flagged, but a new or untracked conftest.py is.

    Args:
        base_ref: Git ref holding the trusted version of the tests.
        project_root: Repository root.

    Returns:
        One dict per audited file with keys: "path", "status", "reasons" (AST findings),
        "removed_assertions" / "added_assertions" / "tolerance_lines_changed" (line-regex counts),
        "suspicious" (bool).
    """
    def git_output(arguments: List[str], must_succeed: bool = True) -> str:
        """Run git in the project root and return stdout ('' on a tolerated failure).

        Args:
            arguments: git arguments after ``git``.
            must_succeed: Raise on a non-zero exit when True.

        Returns:
            Full stdout text.
        """
        completed = subprocess.run(["git", *arguments], cwd=project_root, capture_output=True, text=True)
        if completed.returncode != 0 and must_succeed:
            raise subprocess.CalledProcessError(completed.returncode, arguments, completed.stdout, completed.stderr)
        return completed.stdout if completed.returncode == 0 else ""

    def is_audited(relative_path: str) -> bool:
        """Tell whether a path is a test file, conftest, pyproject or protected fixture."""
        name = relative_path.rsplit("/", 1)[-1]
        return (relative_path.startswith("tests/") and relative_path.endswith(".py")) \
            or name == "conftest.py" or relative_path == "pyproject.toml" \
            or relative_path in PROTECTED_DATA_FILES

    # Step 1: tracked changes with rename / copy detection, plus untracked conftest files.
    entries = []  # (status, old path, new path)
    for status_line in git_output(["diff", "--name-status", "-M", "-C", base_ref, "--"]).splitlines():
        fields = status_line.split("\t")
        entries.append((fields[0], fields[1], fields[-1]))
    for untracked in git_output(["ls-files", "--others", "--exclude-standard"]).splitlines():
        if untracked.rsplit("/", 1)[-1] == "conftest.py":
            entries.append(("A", untracked, untracked))

    findings: List[Dict[str, object]] = []
    for status, old_path, new_path in entries:
        if not is_audited(new_path) and not is_audited(old_path):
            continue
        name = new_path.rsplit("/", 1)[-1]
        # Newly added files are fine, except conftest.py which can inject skips / deselection.
        if status[0] == "A" and name != "conftest.py" and new_path not in PROTECTED_DATA_FILES:
            continue
        base_text = git_output(["show", f"{base_ref}:{old_path}"], must_succeed=False) if status[0] != "A" else ""
        new_file = project_root / new_path
        new_text = new_file.read_text() if new_file.exists() else ""
        reasons: List[str] = []
        parse_failed = False
        if name == "conftest.py":
            if status[0] == "D" or base_text != new_text:
                reasons.append("conftest.py changed (hooks can skip, deselect or relax tests)")
        elif new_path == "pyproject.toml":
            if pytest_config_section(base_text) == pytest_config_section(new_text):
                continue
            reasons.append("pyproject.toml [tool.pytest] section changed")
        elif new_path in PROTECTED_DATA_FILES:
            if base_text != new_text:
                reasons.append("enforcement fixture changed (manifest / exclusions allowlist)")
        elif status[0] == "D":
            reasons.append("test file deleted")
        else:
            reasons = compare_test_sources(base_text, new_text)
            parse_failed = any(reason.startswith("AST parse failed") for reason in reasons)
        # Step 2: line-regex counts (always reported; decisive only if the AST could not be used).
        diff_lines = list(difflib.unified_diff(base_text.splitlines(), new_text.splitlines(), n=0, lineterm=""))
        removed = [line[1:] for line in diff_lines if line.startswith("-") and not line.startswith("---")]
        added = [line[1:] for line in diff_lines if line.startswith("+") and not line.startswith("+++")]
        removed_assertions = sum(1 for line in removed if ASSERTION_PATTERN.search(line))
        added_assertions = sum(1 for line in added if ASSERTION_PATTERN.search(line))
        tolerance_changed = sum(1 for line in removed if TOLERANCE_PATTERN.search(line))
        findings.append({
            "path": new_path,
            "status": status,
            "reasons": reasons,
            "removed_assertions": removed_assertions,
            "added_assertions": added_assertions,
            "tolerance_lines_changed": tolerance_changed,
            # With a usable AST its reasons decide; an unparsable file is always suspicious (the
            # regex counts above are then the only evidence a reviewer gets).
            "suspicious": parse_failed or bool(reasons),
        })
    return findings


def main(argument_list: Optional[List[str]] = None, project_root: Path = PROJECT_ROOT) -> int:
    """CLI entry point: print the selection and audit, optionally run the tests.

    Args:
        argument_list: Command-line arguments (None = sys.argv[1:]).
        project_root: Repository root (overridable for tests).

    Returns:
        pytest's exit code with --run (else 0), or SUSPICIOUS_EXIT_CODE when the audit has a
        SUSPICIOUS finding and --allow-suspicious was not given and the run itself was clean.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default=None,
                        help="git ref to diff against (default: merge-base with main, else origin/main, else HEAD)")
    parser.add_argument("--run", action="store_true", help="run the selected tests with pytest")
    parser.add_argument("--all", action="store_true", help="ignore the diff and select the full suite")
    parser.add_argument("--allow-suspicious", action="store_true",
                        help="do not fail on SUSPICIOUS test edits (main agent only, after stating the reason)")
    arguments = parser.parse_args(argument_list)
    # A ref starting with '-' could be read by git as an option.
    if arguments.base is not None and arguments.base.startswith("-"):
        parser.error("--base must be a git ref, not an option")
    base_ref = arguments.base or default_base_ref(project_root)

    # Step 1: select tests; --all fakes a pyproject.toml change, which triggers the full suite.
    tests_root, source_root = project_root / "tests", project_root / "src"
    if arguments.all:
        selection = select_tests([project_root / "pyproject.toml"], tests_root, source_root)
    else:
        selection = select_tests(changed_paths(base_ref, project_root), tests_root, source_root)
    print(f"base ref        : {base_ref}")
    print(f"changed modules : {selection['changed_modules']}")
    print(f"full suite      : {selection['full_suite']}")
    print(f"transform ids   : {selection['transform_ids']}")
    print("test files      :")
    for test_file in selection["test_files"]:
        print(f"  {test_file}")
    # Step 2: tests are the spec; surface (and gate on) any weakening-looking edit.
    edited_tests = audit_modified_tests(base_ref, project_root)
    print("existing tests modified:" + (" none" if not edited_tests else ""))
    for finding in edited_tests:
        print(f"  {'SUSPICIOUS ' if finding['suspicious'] else ''}{finding['status']} {finding['path']}: "
              f"-{finding['removed_assertions']}/+{finding['added_assertions']} assertion lines, "
              f"{finding['tolerance_lines_changed']} tolerance lines changed")
        for reason in finding["reasons"]:
            print(f"      - {reason}")
    has_suspicious = any(finding["suspicious"] for finding in edited_tests)
    gate_code = SUSPICIOUS_EXIT_CODE if has_suspicious and not arguments.allow_suspicious else 0
    if has_suspicious and gate_code:
        print(f"audit gate: exit {SUSPICIOUS_EXIT_CODE} (pass --allow-suspicious only after stating the reason)")
    if not arguments.run:
        return gate_code
    # Step 3: `-p no:cacheprovider` keeps the read-only agent from writing .pytest_cache into the repo.
    pytest_code = subprocess.call(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *selection["test_files"]],
        cwd=project_root,
    )
    # A failing pytest run wins; otherwise a suspicious audit still fails the command.
    return pytest_code or gate_code


if __name__ == "__main__":
    sys.exit(main())
