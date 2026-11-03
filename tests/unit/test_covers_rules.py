"""Proof that scripts/covers_rules.py rejects every kind of fake covering test (synthetic sources)."""

import ast
import sys
import textwrap
from pathlib import Path

import pytest

SCRIPTS_DIRECTORY = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIRECTORY))
import covers_rules  # noqa: E402  (needs the sys.path entry above)

# Registered function used by every synthetic case.
QUALIFIED_NAME = "pkg.module.convert_point"

GENUINE_SOURCE = '''
import pytest
from pkg.module import convert_point

@pytest.mark.covers("T1")
def test_convert_point():
    assert convert_point(1) == 2
'''


def violations_of(source: str, marker_index: int = 0) -> list:
    """Return the rule violations of the marker_index-th covers marker in a synthetic module."""
    syntax_tree = ast.parse(textwrap.dedent(source))
    markers = covers_rules.find_cover_markers(syntax_tree)
    return covers_rules.covering_violations(markers[marker_index], syntax_tree, QUALIFIED_NAME)


def test_genuine_covering_test_has_no_violation():
    """Control: a top-level test_* that asserts on a call of the registered function passes."""
    assert violations_of(GENUINE_SOURCE) == []


def test_method_of_a_test_class_is_a_valid_placement():
    """Methods of Test* classes are collected by pytest, so they may cover."""
    source = '''
    import pytest
    from pkg.module import convert_point
    class TestPoint:
        @pytest.mark.covers("T1")
        def test_it(self):
            assert convert_point(1) == 2
    '''
    assert violations_of(source) == []


# (case id, source, fragment expected in the violations)
BROKEN_CASES = [
    ("helper_not_named_test",
     GENUINE_SOURCE.replace("def test_convert_point", "def check_convert_point"), "test_"),
    ("nested_function",
     '''
     import pytest
     from pkg.module import convert_point
     def test_outer():
         @pytest.mark.covers("T1")
         def test_inner():
             assert convert_point(1) == 2
         assert True
     ''', "top-level"),
    ("method_of_non_test_class",
     '''
     import pytest
     from pkg.module import convert_point
     class Helpers:
         @pytest.mark.covers("T1")
         def test_it(self):
             assert convert_point(1) == 2
     ''', "top-level"),
    ("skipped", GENUINE_SOURCE.replace('@pytest.mark.covers("T1")', '@pytest.mark.covers("T1")\n@pytest.mark.skip'), "skipped"),
    ("skipif", GENUINE_SOURCE.replace('@pytest.mark.covers("T1")', '@pytest.mark.covers("T1")\n@pytest.mark.skipif(True, reason="x")'), "skipped"),
    ("xfail", GENUINE_SOURCE.replace('@pytest.mark.covers("T1")', '@pytest.mark.covers("T1")\n@pytest.mark.xfail'), "skipped"),
    ("gpu", GENUINE_SOURCE.replace('@pytest.mark.covers("T1")', '@pytest.mark.covers("T1")\n@pytest.mark.gpu'), "skipped"),
    ("real_data", GENUINE_SOURCE.replace('@pytest.mark.covers("T1")', '@pytest.mark.covers("T1")\n@pytest.mark.real_data'), "skipped"),
    ("class_level_gpu",
     '''
     import pytest
     from pkg.module import convert_point
     @pytest.mark.gpu
     class TestPoint:
         @pytest.mark.covers("T1")
         def test_it(self):
             assert convert_point(1) == 2
     ''', "skipped"),
    ("module_pytestmark_skip", "import pytest\npytestmark = pytest.mark.skip\n" + GENUINE_SOURCE, "module-level"),
    ("module_importorskip", 'import pytest\npytest.importorskip("torch")\n' + GENUINE_SOURCE, "module-level"),
    ("no_assertion",
     GENUINE_SOURCE.replace("assert convert_point(1) == 2", "convert_point(1)"), "no assert"),
    ("never_references_registered_function",
     GENUINE_SOURCE.replace("assert convert_point(1) == 2", "assert 1 + 1 == 2"), "never references"),
    ("non_string_argument",
     'import pytest\nfrom pkg.module import convert_point\nID = "T1"\n@pytest.mark.covers(ID)\n'
     "def test_it():\n    assert convert_point(1) == 2\n", "non-string"),
]


@pytest.mark.parametrize("case_id, source, expected_fragment", BROKEN_CASES, ids=[case[0] for case in BROKEN_CASES])
def test_each_rule_rejects_its_fake_covering_test(case_id, source, expected_fragment):
    """Each broken pattern is reported with a message naming the broken rule."""
    violations = violations_of(source)
    assert any(expected_fragment in violation for violation in violations), (case_id, violations)


def test_reference_through_a_same_file_helper_is_accepted():
    """A test may reach the registered function through a helper defined in the same file."""
    source = '''
    import pytest
    from pkg.module import convert_point
    def run_conversion(value):
        return convert_point(value)
    @pytest.mark.covers("T1")
    def test_it():
        assert run_conversion(1) == 2
    '''
    assert violations_of(source) == []


def test_pytest_raises_and_approx_count_as_checks():
    """A raises block or an approx comparison is a real check even without `assert`."""
    source = '''
    import pytest
    from pkg.module import convert_point
    @pytest.mark.covers("T1")
    def test_it():
        with pytest.raises(ValueError):
            convert_point(-1)
    '''
    assert violations_of(source) == []


def test_shared_matcher_ignores_non_string_ids_and_lists_functions():
    """covered_ids_by_function is what both traceability and the selector use."""
    syntax_tree = ast.parse(textwrap.dedent('''
    import pytest
    ID = "T9"
    @pytest.mark.covers("T1", "T2")
    def test_a():
        assert 1
    @pytest.mark.covers(ID)
    def test_b():
        assert 1
    '''))
    assert covers_rules.covered_ids_by_function(syntax_tree) == {"T1": ["test_a"], "T2": ["test_a"]}
