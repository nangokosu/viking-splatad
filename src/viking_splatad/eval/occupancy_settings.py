"""Validated settings of the occupancy evaluation (the ``eval`` section of configs/eval.yaml).

The boundary parser turns the raw YAML dict into one frozen record, so every consumer (scripts/13,
19, 20, 24) reads the same checked values and a bad config fails here, naming the offending key.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Tuple

# Key of the eval section's scored_cells block (the scored range setting is read from it).
SCORED_CELLS_KEY = "scored_cells"


@dataclass(frozen=True)
class OccupancyEvalSettings:
    """Checked settings for P2 thresholds, class groups and the scored range.

    Attributes:
        binding_threshold: The binding occupancy cut (= render_accumulation_threshold).
        p2_thresholds: P2 thresholds, sorted ascending and unique; contains binding_threshold.
        class_groups: Group name -> scored class ids.
        scored_range_quantile: Share of all real returns within the scored range limit, in (0, 1].
    """

    binding_threshold: float
    p2_thresholds: Tuple[float, ...]
    class_groups: Dict[str, Tuple[int, ...]]
    scored_range_quantile: float


def _require_number(value: Any, key: str) -> float:
    """Return ``value`` as a float if it is a real number (not bool).

    Args:
        value: Raw config value.
        key: Config key name used in the error message.

    Returns:
        The value as a float.

    Raises:
        ValueError: If the value is not a number.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a number, got {value!r}")
    return float(value)


def parse_occupancy_settings(eval_section: Mapping[str, Any]) -> OccupancyEvalSettings:
    """Validate the ``eval`` section and build the settings record.

    Args:
        eval_section: The dict under ``eval:`` in configs/eval.yaml.

    Returns:
        The frozen, validated settings.

    Raises:
        ValueError: With the offending key in the message, if any rule is violated or a key is missing.
    """
    # Step 1: required keys, so a missing key is reported by name rather than as a KeyError.
    for required_key in ("render_accumulation_threshold", "p2_thresholds", "scored_class_ids", "class_groups", SCORED_CELLS_KEY):
        if required_key not in eval_section:
            raise ValueError(f"{required_key} is missing from the eval section")
    scored_cells_section = eval_section[SCORED_CELLS_KEY]
    if not isinstance(scored_cells_section, Mapping):
        raise ValueError(f"{SCORED_CELLS_KEY} must be a mapping")
    if "scored_range_quantile" not in scored_cells_section:
        raise ValueError(f"{SCORED_CELLS_KEY}.scored_range_quantile is missing")

    # Step 2: P2 thresholds lie in (0, 1], are strictly ascending (sorted, unique) and contain the binding cut.
    binding_threshold = _require_number(eval_section["render_accumulation_threshold"], "render_accumulation_threshold")
    raw_thresholds = eval_section["p2_thresholds"]
    if not isinstance(raw_thresholds, (list, tuple)) or len(raw_thresholds) == 0:
        raise ValueError("p2_thresholds must be a non-empty list")
    thresholds = tuple(_require_number(value, "p2_thresholds") for value in raw_thresholds)
    if any(not 0.0 < value <= 1.0 for value in thresholds):
        raise ValueError(f"p2_thresholds must lie in (0, 1], got {list(thresholds)}")
    if any(later <= earlier for earlier, later in zip(thresholds, thresholds[1:])):
        raise ValueError(f"p2_thresholds must be sorted ascending and unique, got {list(thresholds)}")
    if binding_threshold not in thresholds:
        raise ValueError(
            f"p2_thresholds must contain render_accumulation_threshold ({binding_threshold}), got {list(thresholds)}"
        )

    # Step 3: class groups only use scored ids and never reuse a scored class name.
    scored_class_ids = eval_section["scored_class_ids"]
    valid_class_ids = set(scored_class_ids.values())
    class_groups: Dict[str, Tuple[int, ...]] = {}
    for group_name, group_ids in eval_section["class_groups"].items():
        if group_name in scored_class_ids:
            raise ValueError(f"class_groups name {group_name!r} collides with a scored class name")
        if not isinstance(group_ids, (list, tuple)) or len(group_ids) == 0:
            raise ValueError(f"class_groups.{group_name} must be a non-empty list of class ids")
        unknown_ids = [class_id for class_id in group_ids if class_id not in valid_class_ids]
        if unknown_ids:
            raise ValueError(f"class_groups.{group_name} has ids {unknown_ids} not in scored_class_ids")
        class_groups[str(group_name)] = tuple(int(class_id) for class_id in group_ids)

    # Step 4: scored range quantile.
    scored_range_quantile = _require_number(
        scored_cells_section["scored_range_quantile"], f"{SCORED_CELLS_KEY}.scored_range_quantile")
    if not 0.0 < scored_range_quantile <= 1.0:
        raise ValueError(f"{SCORED_CELLS_KEY}.scored_range_quantile must lie in (0, 1], got {scored_range_quantile}")

    return OccupancyEvalSettings(
        binding_threshold=binding_threshold,
        p2_thresholds=thresholds,
        class_groups=class_groups,
        scored_range_quantile=scored_range_quantile,
    )


def class_group_names_ordered(settings: OccupancyEvalSettings, scored_class_ids: Mapping[str, int]) -> List[str]:
    """Report order of class rows: groups first, then the scored classes.

    Args:
        settings: Validated settings (supplies the group names, in config order).
        scored_class_ids: Scored class name -> id mapping from the eval section.

    Returns:
        Group names followed by scored class names.
    """
    # Groups lead because the headline claim is on the group; singles follow for detail.
    return list(settings.class_groups) + list(scored_class_ids)
