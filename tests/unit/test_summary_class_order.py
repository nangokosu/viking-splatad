"""scripts/20 restores the report order of classes lost by scripts/19's sorted JSON keys."""

import importlib.util
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "20_summarize_gaussian_occupancy.py"


def load_summary_script():
    """Load the numbered script as a module."""
    spec = importlib.util.spec_from_file_location("summary_script_for_order", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_groups_first_then_classes_in_group_id_order():
    """Alphabetical JSON order -> obstacles, tree_trunk (1), rock (3), bush (4), then an ungrouped class."""
    script = load_summary_script()
    classes = {
        "bush_or_small_tree": {"class_id": 4},
        "fence": {"class_id": 9},
        "obstacles": {"class_id": None, "class_ids": [1, 3, 4]},
        "rock": {"class_id": 3},
        "tree_trunk": {"class_id": 1},
    }
    assert script.report_class_order(classes) == ["obstacles", "tree_trunk", "rock", "bush_or_small_tree", "fence"]
