"""Shared static rules for `@pytest.mark.covers("T<n>")` markers.

Used by tests/test_traceability.py (the enforcing meta-test) and scripts/affected_tests.py (the
selector), so both read a covers marker identically. Everything here is pure AST analysis: no test
file is imported or run.

A covering test counts only when it is a genuine test of the registered function:
  * named test_* and top level, or a method of a Test* class (pytest would collect it);
  * not skipped, xfailed or gated by the gpu / real_data markers (decorators, class decorators or
    module-level pytestmark / pytest.skip / pytest.importorskip);
  * contains a real check (assert, pytest.raises/warns/approx or an assert_* call);
  * references the registered function by name, directly or through helper functions of the same
    test file (a marker on a test that never touches the function proves nothing);
  * every covers() argument is a string literal.
"""

import ast
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

# Decorator / call spellings that stop a test from running (or run it only elsewhere).
NON_RUNNING_PATTERN = re.compile(r"\b(skip|skipif|xfail|importorskip|gpu|real_data)\b")
# Call names that count as a real check inside a test body.
CHECK_CALL_NAMES = {"raises", "warns", "approx", "deprecated_call"}


@dataclass
class CoverMarker:
    """One `@pytest.mark.covers(...)` decorator found on a function.

    Attributes:
        function_node: The decorated function definition.
        owner_class_node: Enclosing class definition, or None for a module-level function.
        nested_in_function: True when the function is defined inside another function.
        transform_ids: The string-literal arguments of the marker.
        has_non_string_argument: True if any argument is not a plain string literal.
    """

    function_node: ast.AST
    owner_class_node: Optional[ast.ClassDef]
    nested_in_function: bool
    transform_ids: List[str]
    has_non_string_argument: bool


def is_covers_decorator(decorator: ast.AST) -> bool:
    """Tell whether a decorator is a call spelled ``<anything>.covers(...)``.

    Args:
        decorator: A decorator expression node.

    Returns:
        True for ``@pytest.mark.covers(...)``-shaped decorators.
    """
    return (
        isinstance(decorator, ast.Call)
        and isinstance(decorator.func, ast.Attribute)
        and decorator.func.attr == "covers"
    )


def find_cover_markers(syntax_tree: ast.AST) -> List[CoverMarker]:
    """Collect every covers marker in a module with its placement.

    Args:
        syntax_tree: Parsed test module.

    Returns:
        One CoverMarker per covers decorator, in source order.
    """
    markers: List[CoverMarker] = []

    def visit(node: ast.AST, owner_class: Optional[ast.ClassDef], inside_function: bool) -> None:
        """Walk children, tracking the enclosing class and whether we are inside a function.

        Args:
            node: Node whose children are visited.
            owner_class: Innermost enclosing class definition, if any.
            inside_function: True once the walk entered a function body.
        """
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for decorator in child.decorator_list:
                    if is_covers_decorator(decorator):
                        string_ids = [
                            argument.value for argument in decorator.args
                            if isinstance(argument, ast.Constant) and isinstance(argument.value, str)
                        ]
                        markers.append(CoverMarker(
                            function_node=child, owner_class_node=owner_class,
                            nested_in_function=inside_function, transform_ids=string_ids,
                            has_non_string_argument=len(string_ids) != len(decorator.args) or bool(decorator.keywords),
                        ))
                visit(child, owner_class, True)
            elif isinstance(child, ast.ClassDef):
                visit(child, child, inside_function)
            else:
                visit(child, owner_class, inside_function)

    visit(syntax_tree, None, False)
    return markers


def covered_ids_by_function(syntax_tree: ast.AST) -> Dict[str, List[str]]:
    """Map each string transform id to the functions that carry its covers marker.

    Args:
        syntax_tree: Parsed test module.

    Returns:
        Mapping transform id -> function names (non-string arguments are ignored here and
        reported by ``covering_violations``).
    """
    covered: Dict[str, List[str]] = {}
    for marker in find_cover_markers(syntax_tree):
        for transform_id in marker.transform_ids:
            covered.setdefault(transform_id, []).append(marker.function_node.name)
    return covered


def referenced_names(node: ast.AST) -> Set[str]:
    """Collect every identifier and attribute name used under a node.

    Args:
        node: Any AST node.

    Returns:
        Names (``Name.id``, ``Attribute.attr``, import ``name`` / ``asname`` last segments).
    """
    names: Set[str] = set()
    for inner in ast.walk(node):
        if isinstance(inner, ast.Name):
            names.add(inner.id)
        elif isinstance(inner, ast.Attribute):
            names.add(inner.attr)
        elif isinstance(inner, (ast.Import, ast.ImportFrom)):
            for alias in inner.names:
                names.add(alias.name.split(".")[-1])
    return names


def contains_real_check(function_node: ast.AST) -> bool:
    """Tell whether a function body asserts something.

    Args:
        function_node: Function definition.

    Returns:
        True if it holds an ``assert`` statement or a pytest.raises/warns/approx or ``assert_*`` call.
    """
    for inner in ast.walk(function_node):
        if isinstance(inner, ast.Assert):
            return True
        if isinstance(inner, ast.Call):
            callee = inner.func
            callee_name = callee.attr if isinstance(callee, ast.Attribute) else getattr(callee, "id", "")
            if callee_name in CHECK_CALL_NAMES or callee_name.startswith("assert"):
                return True
    return False


def references_function(function_node: ast.AST, syntax_tree: ast.AST, short_name: str) -> bool:
    """Tell whether a test reaches a registered function directly or through same-file helpers.

    Breadth-first over names: the test body's names, then the bodies of same-file helper
    functions those names denote. Each name is expanded once, so the cost is linear in the
    number of distinct names. A test that only calls some *caller* of the function does not count.

    Args:
        function_node: The covering test.
        syntax_tree: Its module.
        short_name: Last segment of the registered qualified name.

    Returns:
        True if ``short_name`` is reached.
    """
    # Step 1: index every function / method of the test file by simple name.
    definitions = {
        node.name: node for node in ast.walk(syntax_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    # Step 2: expand reachable names until no new one appears.
    pending_names = set(referenced_names(function_node))
    expanded_names: Set[str] = set()
    while pending_names:
        name = pending_names.pop()
        if name == short_name:
            return True
        if name in expanded_names:
            continue
        expanded_names.add(name)
        if name in definitions:
            pending_names |= referenced_names(definitions[name])
    return False


def is_non_running_marker(node: ast.AST) -> bool:
    """Tell whether a decorator / expression spells skip, xfail, gpu or real_data.

    Args:
        node: Decorator or expression node.

    Returns:
        True if its source matches NON_RUNNING_PATTERN.
    """
    return bool(NON_RUNNING_PATTERN.search(ast.unparse(node)))


def module_level_non_running(syntax_tree: ast.Module) -> bool:
    """Tell whether the module as a whole is skipped or gated (pytestmark / skip / importorskip).

    Args:
        syntax_tree: Parsed test module.

    Returns:
        True if a top-level statement skips the module or marks it gpu / real_data.
    """
    for statement in syntax_tree.body:
        is_marker_assignment = (
            isinstance(statement, ast.Assign)
            and any(getattr(target, "id", "") == "pytestmark" for target in statement.targets)
        )
        is_skip_call = isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call)
        if (is_marker_assignment or is_skip_call) and is_non_running_marker(statement):
            return True
    return False


def covering_violations(marker: CoverMarker, syntax_tree: ast.Module, qualified_name: Optional[str]) -> List[str]:
    """List every rule a covering test breaks (empty = genuine covering test).

    Args:
        marker: The covers marker under inspection.
        syntax_tree: Module containing it.
        qualified_name: Registered ``module.function`` of the covered transform, or None when the
            id is not registered (the reference rule is then skipped).

    Returns:
        Human-readable violations.
    """
    function_node = marker.function_node
    violations: List[str] = []
    # Rule: pytest must collect it.
    if not function_node.name.startswith("test_"):
        violations.append("name does not start with test_")
    owner = marker.owner_class_node
    if marker.nested_in_function or (owner is not None and not owner.name.startswith("Test")):
        violations.append("not top-level or a method of a Test* class")
    # Rule: it must actually run in the default suite.
    non_running = [d for d in function_node.decorator_list if not is_covers_decorator(d) and is_non_running_marker(d)]
    if owner is not None:
        non_running += [d for d in owner.decorator_list if is_non_running_marker(d)]
    if non_running:
        violations.append("skipped / xfail / gpu / real_data decorator")
    if module_level_non_running(syntax_tree):
        violations.append("module-level skip / pytestmark gate")
    # Rule: it must assert something.
    if not contains_real_check(function_node):
        violations.append("contains no assert / pytest.raises / approx / assert_* call")
    # Rule: it must touch the registered function.
    if qualified_name is not None:
        short_name = qualified_name.split(".")[-1]
        if not references_function(function_node, syntax_tree, short_name):
            violations.append(f"never references registered function {short_name!r}")
    if marker.has_non_string_argument:
        violations.append("covers() has a non-string-literal argument")
    return violations
