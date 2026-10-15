"""Meta-tests: every registered transform must have a covering unit test."""

import ast
import importlib
import os
import pkgutil
from pathlib import Path

import pytest

import viking_splatad
from viking_splatad.registry import PLANNED_TRANSFORM_IDS, TRANSFORM_REGISTRY

TESTS_ROOT = Path(__file__).parent


def import_all_package_modules() -> None:
    """Import every submodule so all ``@transform`` decorators run."""
    for module_info in pkgutil.walk_packages(
        viking_splatad.__path__, prefix="viking_splatad."
    ):
        importlib.import_module(module_info.name)


def collect_covered_transform_ids() -> dict:
    """Statically scan test files for ``covers("T<n>")`` markers.

    Static AST scanning keeps the check independent of which tests pytest selected.

    Returns:
        Mapping transform id -> list of "file::test_function" that cover it.
    """
    covered_by_id: dict = {}
    for test_file in TESTS_ROOT.rglob("test_*.py"):
        syntax_tree = ast.parse(test_file.read_text())
        for node in ast.walk(syntax_tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for decorator in node.decorator_list:
                # Match @pytest.mark.covers("T1", "T2", ...)
                if (
                    isinstance(decorator, ast.Call)
                    and isinstance(decorator.func, ast.Attribute)
                    and decorator.func.attr == "covers"
                ):
                    for argument in decorator.args:
                        if isinstance(argument, ast.Constant):
                            covered_by_id.setdefault(argument.value, []).append(
                                f"{test_file.name}::{node.name}"
                            )
    return covered_by_id


def test_every_registered_transform_has_a_covering_test():
    """Registered transform without ``@pytest.mark.covers`` test -> fail."""
    import_all_package_modules()
    covered_by_id = collect_covered_transform_ids()
    untested_ids = sorted(set(TRANSFORM_REGISTRY) - set(covered_by_id))
    assert not untested_ids, f"Transforms without a unit test: {untested_ids}"


def test_tests_only_cover_known_transform_ids():
    """A ``covers`` marker naming an id outside the plan is a typo."""
    unknown_ids = sorted(set(collect_covered_transform_ids()) - PLANNED_TRANSFORM_IDS)
    assert not unknown_ids, f"covers() names unknown transform ids: {unknown_ids}"


def test_all_planned_transforms_implemented():
    """Completeness gate; only enforced when VIKING_REQUIRE_COMPLETE=1."""
    import_all_package_modules()
    pending_ids = sorted(PLANNED_TRANSFORM_IDS - set(TRANSFORM_REGISTRY))
    if os.environ.get("VIKING_REQUIRE_COMPLETE") != "1":
        pytest.skip(f"{len(pending_ids)} transforms pending: {pending_ids}")
    assert not pending_ids, f"Planned transforms not implemented: {pending_ids}"
