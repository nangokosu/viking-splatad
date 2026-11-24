"""Validated settings of the hidden-voxel evaluation (the ``eval`` section of configs/eval.yaml).

The boundary parser turns the raw YAML dict into one frozen record, so every consumer (scripts/13,
19, 20, 24) reads the same checked values and a bad config fails here, naming the offending key.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

# Key of the eval section's hidden_space block (the margin and range settings below are read from it).
HIDDEN_SPACE_KEY = "hidden_space"


@dataclass(frozen=True)
class HiddenEvalSettings:
    """Checked settings for P2 thresholds, class groups and the hidden-space margin.

    Attributes:
        binding_threshold: The binding occupancy cut (= render_accumulation_threshold).
        p2_thresholds: P2 thresholds, sorted ascending and unique; contains binding_threshold.
        class_groups: Group name -> scored class ids.
        margin_quantile: Quantile of cross-sweep residuals taken as the range margin, in (0, 1).
        margin_sweep_gap: Sweeps between the two sweeps of a residual pair (positive).
        margin_sweep_stride: Sweeps between first sweeps of pairs (positive).
        margin_neighbour_count: Neighbours per plane fit (at least 3).
        margin_planarity_min: Minimum planarity of a neighbourhood, in [0, 1].
        margin_max_range_m: Largest lidar range (m) used for the margin (positive).
        scored_range_quantile: Share of all real returns within the scored range limit, in (0, 1].
        near_band_m: Horizontal distance (m) from the driven path bounding the headline H_near (positive).
        worker_count: CPU workers; None = all cores.
    """

    binding_threshold: float
    p2_thresholds: Tuple[float, ...]
    class_groups: Dict[str, Tuple[int, ...]]
    margin_quantile: float
    margin_sweep_gap: int
    margin_sweep_stride: int
    margin_neighbour_count: int
    margin_planarity_min: float
    margin_max_range_m: float
    scored_range_quantile: float
    near_band_m: float
    worker_count: Optional[int]


def _require_positive_int(value: Any, key: str, minimum: int = 1) -> int:
    """Return ``value`` if it is an int (not bool) of at least ``minimum``.

    Args:
        value: Raw config value.
        key: Config key name used in the error message.
        minimum: Smallest accepted value.

    Returns:
        The value as an int.

    Raises:
        ValueError: If the value is not an int or is below ``minimum``.
    """
    # bool is an int subclass in Python; a YAML `true` must not pass as a count.
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{key} must be an int >= {minimum}, got {value!r}")
    return value


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


def parse_hidden_settings(eval_section: Mapping[str, Any]) -> HiddenEvalSettings:
    """Validate the ``eval`` section and build the settings record.

    Args:
        eval_section: The dict under ``eval:`` in configs/eval.yaml.

    Returns:
        The frozen, validated settings.

    Raises:
        ValueError: With the offending key in the message, if any rule is violated or a key is missing.
    """
    # Step 1: required keys, so a missing key is reported by name rather than as a KeyError.
    for required_key in ("render_accumulation_threshold", "p2_thresholds", "scored_class_ids", "class_groups", HIDDEN_SPACE_KEY):
        if required_key not in eval_section:
            raise ValueError(f"{required_key} is missing from the eval section")
    hidden_section = eval_section[HIDDEN_SPACE_KEY]
    if not isinstance(hidden_section, Mapping):
        raise ValueError(f"{HIDDEN_SPACE_KEY} must be a mapping")
    for required_key in (
        "margin_quantile", "margin_sweep_gap", "margin_sweep_stride", "margin_neighbour_count",
        "margin_planarity_min", "margin_max_range_m", "scored_range_quantile", "near_band_m", "worker_count",
    ):
        if required_key not in hidden_section:
            raise ValueError(f"{HIDDEN_SPACE_KEY}.{required_key} is missing")

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

    # Step 4: hidden-space margin settings.
    margin_quantile = _require_number(hidden_section["margin_quantile"], f"{HIDDEN_SPACE_KEY}.margin_quantile")
    if not 0.0 < margin_quantile < 1.0:
        raise ValueError(f"{HIDDEN_SPACE_KEY}.margin_quantile must lie in (0, 1), got {margin_quantile}")
    margin_planarity_min = _require_number(hidden_section["margin_planarity_min"], f"{HIDDEN_SPACE_KEY}.margin_planarity_min")
    if not 0.0 <= margin_planarity_min <= 1.0:
        raise ValueError(f"{HIDDEN_SPACE_KEY}.margin_planarity_min must lie in [0, 1], got {margin_planarity_min}")
    margin_max_range_m = _require_number(hidden_section["margin_max_range_m"], f"{HIDDEN_SPACE_KEY}.margin_max_range_m")
    if margin_max_range_m <= 0.0:
        raise ValueError(f"{HIDDEN_SPACE_KEY}.margin_max_range_m must be > 0, got {margin_max_range_m}")
    scored_range_quantile = _require_number(
        hidden_section["scored_range_quantile"], f"{HIDDEN_SPACE_KEY}.scored_range_quantile")
    if not 0.0 < scored_range_quantile <= 1.0:
        raise ValueError(f"{HIDDEN_SPACE_KEY}.scored_range_quantile must lie in (0, 1], got {scored_range_quantile}")
    near_band_m = _require_number(hidden_section["near_band_m"], f"{HIDDEN_SPACE_KEY}.near_band_m")
    if near_band_m <= 0.0:
        raise ValueError(f"{HIDDEN_SPACE_KEY}.near_band_m must be > 0, got {near_band_m}")
    raw_worker_count = hidden_section["worker_count"]
    worker_count = None if raw_worker_count is None else _require_positive_int(raw_worker_count, f"{HIDDEN_SPACE_KEY}.worker_count")

    return HiddenEvalSettings(
        binding_threshold=binding_threshold,
        p2_thresholds=thresholds,
        class_groups=class_groups,
        margin_quantile=margin_quantile,
        margin_sweep_gap=_require_positive_int(hidden_section["margin_sweep_gap"], f"{HIDDEN_SPACE_KEY}.margin_sweep_gap"),
        margin_sweep_stride=_require_positive_int(hidden_section["margin_sweep_stride"], f"{HIDDEN_SPACE_KEY}.margin_sweep_stride"),
        margin_neighbour_count=_require_positive_int(
            hidden_section["margin_neighbour_count"], f"{HIDDEN_SPACE_KEY}.margin_neighbour_count", minimum=3
        ),
        margin_planarity_min=margin_planarity_min,
        margin_max_range_m=margin_max_range_m,
        scored_range_quantile=scored_range_quantile,
        near_band_m=near_band_m,
        worker_count=worker_count,
    )


def class_group_names_ordered(settings: HiddenEvalSettings, scored_class_ids: Mapping[str, int]) -> List[str]:
    """Report order of class rows: groups first, then the scored classes.

    Args:
        settings: Validated settings (supplies the group names, in config order).
        scored_class_ids: Scored class name -> id mapping from the eval section.

    Returns:
        Group names followed by scored class names.
    """
    # Groups lead because the headline claim is on the group; singles follow for detail.
    return list(settings.class_groups) + list(scored_class_ids)
