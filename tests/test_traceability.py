"""Meta-tests: every registered transform must have a covering unit test."""

import ast
import importlib
import json
import os
import pkgutil
import sys
from pathlib import Path

import pytest

import viking_splatad
from viking_splatad.registry import PLANNED_TRANSFORM_IDS, TRANSFORM_REGISTRY

TESTS_ROOT = Path(__file__).parent
# Golden record of the registry; changing it is itself audited by scripts/affected_tests.py.
MANIFEST_PATH = TESTS_ROOT / "fixtures" / "transform_manifest.json"

# covers_rules (shared with scripts/affected_tests.py) lives in scripts/, which is not a package.
SCRIPTS_DIRECTORY = TESTS_ROOT.parent / "scripts"
if str(SCRIPTS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIRECTORY))
import covers_rules  # noqa: E402  (needs the sys.path entry above)


def import_all_package_modules() -> None:
    """Import every submodule so all ``@transform`` decorators run."""
    for module_info in pkgutil.walk_packages(
        viking_splatad.__path__, prefix="viking_splatad."
    ):
        importlib.import_module(module_info.name)


def collect_covered_transform_ids() -> dict:
    """Statically scan test files for ``covers("T<n>")`` markers.

    Static AST scanning keeps the check independent of which tests pytest selected. The matcher
    is shared with scripts/affected_tests.py (scripts/covers_rules.py).

    Returns:
        Mapping transform id -> list of "file::test_function" that cover it.
    """
    covered_by_id: dict = {}
    for test_file in TESTS_ROOT.rglob("test_*.py"):
        syntax_tree = ast.parse(test_file.read_text())
        for transform_id, function_names in covers_rules.covered_ids_by_function(syntax_tree).items():
            covered_by_id.setdefault(transform_id, []).extend(f"{test_file.name}::{name}" for name in function_names)
    return covered_by_id


def collect_covering_violations(registry: dict, tests_root: Path = TESTS_ROOT) -> list:
    """Apply the covering-test rules (scripts/covers_rules.py) to every registered transform.

    A transform passes when AT LEAST ONE of its covers markers is a genuine test (a gated or
    indirect extra test carrying the same id is tolerated); a marker with a non-string argument
    fails regardless, because it hides which id it claims.

    Args:
        registry: Mapping transform id -> TransformSpec (or anything with ``qualified_name``).
        tests_root: Directory scanned for test files.

    Returns:
        Violation strings; empty when every registered transform has a genuine covering test.
    """
    violations_by_id: dict = {}
    genuine_ids = set()
    problems = []
    for test_file in sorted(tests_root.rglob("test_*.py")):
        syntax_tree = ast.parse(test_file.read_text())
        for marker in covers_rules.find_cover_markers(syntax_tree):
            if marker.has_non_string_argument:
                problems.append(f"{test_file.name}::{marker.function_node.name}: covers() has a non-string-literal argument")
            for transform_id in marker.transform_ids:
                qualified_name = registry[transform_id].qualified_name if transform_id in registry else None
                found = covers_rules.covering_violations(marker, syntax_tree, qualified_name)
                # The non-string rule is reported above; here only whether this marker is genuine.
                found = [violation for violation in found if "non-string" not in violation]
                if not found:
                    genuine_ids.add(transform_id)
                violations_by_id.setdefault(transform_id, []).extend(
                    f"{test_file.name}::{marker.function_node.name}: {violation}" for violation in found
                )
    for transform_id in sorted(set(registry) - genuine_ids):
        details = violations_by_id.get(transform_id, ["no covers marker at all"])
        problems.append(f"{transform_id}: no genuine covering test; " + " | ".join(details))
    return problems


def manifest_differences(current_manifest: dict, golden_manifest: dict) -> list:
    """Describe every difference between the live registry manifest and the committed one.

    Args:
        current_manifest: {"planned_ids": [...], "transforms": [{"id", "qualified_name"}, ...]}.
        golden_manifest: Same shape, loaded from MANIFEST_PATH.

    Returns:
        Human-readable differences (empty = identical).
    """
    differences = []
    current = {entry["id"]: entry["qualified_name"] for entry in current_manifest["transforms"]}
    golden = {entry["id"]: entry["qualified_name"] for entry in golden_manifest["transforms"]}
    for transform_id in sorted(set(golden) - set(current)):
        differences.append(f"{transform_id} is no longer registered (was {golden[transform_id]})")
    for transform_id in sorted(set(current) - set(golden)):
        differences.append(f"{transform_id} is newly registered ({current[transform_id]}) - add it to the manifest")
    for transform_id in sorted(set(current) & set(golden)):
        if current[transform_id] != golden[transform_id]:
            differences.append(f"{transform_id} moved: {golden[transform_id]} -> {current[transform_id]}")
    current_planned, golden_planned = set(current_manifest["planned_ids"]), set(golden_manifest["planned_ids"])
    for transform_id in sorted(golden_planned - current_planned):
        differences.append(f"{transform_id} removed from PLANNED_TRANSFORM_IDS")
    for transform_id in sorted(current_planned - golden_planned):
        differences.append(f"{transform_id} added to PLANNED_TRANSFORM_IDS")
    return differences


def live_manifest() -> dict:
    """Build the manifest of the live registry (after importing every module)."""
    import_all_package_modules()
    return {
        "planned_ids": sorted(PLANNED_TRANSFORM_IDS),
        "transforms": [
            {"id": transform_id, "qualified_name": TRANSFORM_REGISTRY[transform_id].qualified_name}
            for transform_id in sorted(TRANSFORM_REGISTRY)
        ],
    }


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


def test_covering_tests_are_genuine_tests_of_the_registered_function():
    """Every covers marker sits on a collected, running, asserting test that touches the function."""
    import_all_package_modules()
    problems = collect_covering_violations(TRANSFORM_REGISTRY)
    assert not problems, "covering tests that do not really test the transform:\n" + "\n".join(problems)


def test_registry_matches_the_committed_manifest():
    """Removing a @transform or editing PLANNED_TRANSFORM_IDS shows up as a diff of the manifest.

    Regenerate after an intended change (in the reference image) by dumping ``live_manifest()``
    with ``json.dumps(..., indent=2, sort_keys=True)`` into tests/fixtures/transform_manifest.json;
    scripts/affected_tests.py flags that file as SUSPICIOUS, so the edit is reviewed.
    """
    golden_manifest = json.loads(MANIFEST_PATH.read_text())
    differences = manifest_differences(live_manifest(), golden_manifest)
    assert not differences, "registry differs from tests/fixtures/transform_manifest.json:\n" + "\n".join(differences)


def test_manifest_comparison_reports_removed_added_moved_and_planned_changes():
    """Known-answer check of the diff logic itself (synthetic manifests)."""
    golden = {"planned_ids": ["T1", "T2"], "transforms": [
        {"id": "T1", "qualified_name": "pkg.a.f"}, {"id": "T2", "qualified_name": "pkg.b.g"}]}
    assert manifest_differences(golden, golden) == []
    dropped = {"planned_ids": ["T1", "T2"], "transforms": [{"id": "T1", "qualified_name": "pkg.a.f"}]}
    assert manifest_differences(dropped, golden) == ["T2 is no longer registered (was pkg.b.g)"]
    moved = {"planned_ids": ["T1", "T2"], "transforms": [
        {"id": "T1", "qualified_name": "pkg.a.f2"}, {"id": "T2", "qualified_name": "pkg.b.g"}]}
    assert manifest_differences(moved, golden) == ["T1 moved: pkg.a.f -> pkg.a.f2"]
    edited_plan = {"planned_ids": ["T1"], "transforms": golden["transforms"]}
    assert manifest_differences(edited_plan, golden) == ["T2 removed from PLANNED_TRANSFORM_IDS"]
    extra = {"planned_ids": ["T1", "T2", "T3"], "transforms": golden["transforms"] + [
        {"id": "T3", "qualified_name": "pkg.c.h"}]}
    assert len(manifest_differences(extra, golden)) == 2


class FakeSpec:
    """Stand-in for TransformSpec carrying only the qualified name."""

    def __init__(self, qualified_name: str):
        """Store the qualified name the reference rule looks up."""
        self.qualified_name = qualified_name


def test_per_transform_rule_needs_one_genuine_marker_and_rejects_non_string_ids(tmp_path):
    """Synthetic tests dir: a fake-only id fails, a real+fake id passes, a non-string id fails."""
    (tmp_path / "test_fake.py").write_text(
        "import pytest\n"
        "@pytest.mark.covers('T1')\n"
        "def test_never_calls_it():\n    assert 1 == 1\n"
    )
    (tmp_path / "test_mixed.py").write_text(
        "import pytest\nfrom pkg.mod import convert\n"
        "@pytest.mark.covers('T2')\n"
        "def test_real():\n    assert convert(1) == 1\n"
        "@pytest.mark.covers('T2')\n"
        "@pytest.mark.gpu\n"
        "def test_gated_extra():\n    assert convert(1) == 1\n"
        "ID = 'T3'\n"
        "@pytest.mark.covers(ID)\n"
        "def test_non_string():\n    assert convert(1) == 1\n"
    )
    registry = {"T1": FakeSpec("pkg.mod.convert"), "T2": FakeSpec("pkg.mod.convert")}
    problems = collect_covering_violations(registry, tests_root=tmp_path)
    assert any(problem.startswith("T1: no genuine covering test") for problem in problems)
    assert not any(problem.startswith("T2:") for problem in problems)
    assert any("non-string" in problem for problem in problems)
    # An id with no marker at all is also reported.
    assert any("no covers marker" in problem for problem in collect_covering_violations(
        {"T9": FakeSpec("pkg.mod.convert")}, tests_root=tmp_path))
