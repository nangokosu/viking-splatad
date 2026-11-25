"""Figure-suite settings and run resolution (which run folders each figure draws).

Settings come from the ``eval.figures`` block of configs/eval.yaml and are validated once, here.
Runs are found from rows of the submission ledger (runs/submitted.tsv, parsed by
``read_ledger`` in scripts/slurm/submit_run.py) and the local run folders runs/<run_id>/.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from viking_splatad.viz.pyvista_scene import MAX_CHANNEL_VALUE

# Colours are RGB triples with this many channels, each within [0, MAX_CHANNEL_VALUE].
RGB_CHANNEL_COUNT = 3
# Fallback point count of a threshold-curve grid for settings built directly in code (the committed
# configs/eval.yaml value is the source of truth and is required by ``load_figure_settings``): 21
# points over [0, 1] give a 0.05 step that contains the P2 thresholds 0.5 and 0.8.
DEFAULT_THRESHOLD_CURVE_POINTS = 21
# A curve grid needs at least its two end points.
MINIMUM_THRESHOLD_CURVE_POINTS = 2


@dataclass(frozen=True)
class FigureSettings:
    """Validated figure settings (the ``eval.figures`` block).

    Attributes:
        arm_order: Arm keys, left to right.
        arm_labels: Display text per arm.
        campaign_profiles: session -> arm -> 20k campaign profile name.
        section_trunk_count: Trunk sections per figure.
        rows_per_page: Tile rows per page.
        dpi: Figure resolution.
        tile_px: Tile side in pixels.
        bev_bin_m: Bird's-eye bin edge in metres, None = session voxel size.
        difference_cmap: Colormap of difference maps.
        occupancy_cmap: Colormap of occupancy maps.
        colours: name -> (R, G, B) in 0-255.
        footer_note: Text printed under every figure.
        representative_metric: Flat metric key (``p2_repeats.flatten_hidden_metrics``) whose median
            repeat represents an arm in the figures.
        threshold_curve_points: Points of the evenly spaced [0, 1] occupancy-threshold grid of the
            threshold-curve figures (ends included).
    """

    arm_order: Tuple[str, ...]
    arm_labels: Dict[str, str]
    campaign_profiles: Dict[str, Dict[str, str]]
    section_trunk_count: int
    rows_per_page: int
    dpi: int
    tile_px: int
    bev_bin_m: Optional[float]
    difference_cmap: str
    occupancy_cmap: str
    colours: Dict[str, Tuple[int, int, int]]
    footer_note: str
    representative_metric: str
    threshold_curve_points: int = DEFAULT_THRESHOLD_CURVE_POINTS


def _require_positive_int(figures_block: Mapping[str, Any], key: str) -> int:
    """Return ``figures_block[key]`` after checking it is a positive int (not a bool).

    Args:
        figures_block: The ``figures`` mapping.
        key: Entry to read.

    Returns:
        The integer value.

    Raises:
        ValueError: When the entry is missing, not an int, or not positive.
    """
    # Step 1: bool is an int subclass, so exclude it explicitly.
    value = figures_block.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"figures.{key} must be a positive integer, got {value!r}")
    return value


def _require_profile_map(figures_block: Mapping[str, Any], key: str,
                         arm_order: Sequence[str]) -> Dict[str, Dict[str, str]]:
    """Validate a session -> arm -> profile mapping against the arm order.

    Args:
        figures_block: The ``figures`` mapping.
        key: Entry holding the mapping.
        arm_order: Arm keys every session must map exactly.

    Returns:
        A plain nested dict copy.

    Raises:
        ValueError: When a session's arms differ from ``arm_order`` or a profile is not a string.
    """
    session_map = figures_block.get(key)
    if not isinstance(session_map, Mapping) or not session_map:
        raise ValueError(f"figures.{key} must be a non-empty mapping of session -> arm -> profile")
    validated: Dict[str, Dict[str, str]] = {}
    for session, profile_by_arm in session_map.items():
        # Step 1: the arm keys must equal arm_order as sets.
        if not isinstance(profile_by_arm, Mapping) or set(profile_by_arm) != set(arm_order):
            raise ValueError(
                f"figures.{key}.{session} arms {sorted(profile_by_arm) if isinstance(profile_by_arm, Mapping) else profile_by_arm!r}"
                f" do not match arm_order {list(arm_order)}")
        # Step 2: each profile is a non-empty string.
        for arm, profile in profile_by_arm.items():
            if not isinstance(profile, str) or not profile:
                raise ValueError(f"figures.{key}.{session}.{arm} must be a profile name, got {profile!r}")
        validated[session] = dict(profile_by_arm)
    return validated


def load_figure_settings(eval_config: Mapping[str, Any]) -> FigureSettings:
    """Parse and validate the figure settings.

    Args:
        eval_config: The ``eval`` section of configs/eval.yaml (the dict holding ``figures``).

    Returns:
        The frozen FigureSettings.

    Raises:
        ValueError: On any missing or inconsistent entry (message names the entry).
    """
    # Step 1: locate the block.
    figures_block = eval_config.get("figures")
    if not isinstance(figures_block, Mapping):
        raise ValueError("eval.figures block is missing")
    # Step 2: arm order and labels must agree.
    arm_order = figures_block.get("arm_order")
    if not isinstance(arm_order, list) or not arm_order or len(set(arm_order)) != len(arm_order):
        raise ValueError(f"figures.arm_order must be a non-empty list of unique arms, got {arm_order!r}")
    arm_labels = figures_block.get("arm_labels")
    if not isinstance(arm_labels, Mapping) or set(arm_labels) != set(arm_order):
        raise ValueError(f"figures.arm_labels keys must equal arm_order {arm_order}, got {arm_labels!r}")
    # Step 3: profile maps.
    campaign_profiles = _require_profile_map(figures_block, "campaign_profiles", arm_order)
    # Step 4: colours are RGB triples within range.
    colour_block = figures_block.get("colours")
    if not isinstance(colour_block, Mapping) or not colour_block:
        raise ValueError("figures.colours must be a non-empty mapping")
    colours: Dict[str, Tuple[int, int, int]] = {}
    for colour_name, channels in colour_block.items():
        channels_valid = (isinstance(channels, (list, tuple)) and len(channels) == RGB_CHANNEL_COUNT
                          and all(isinstance(channel, int) and not isinstance(channel, bool)
                                  and 0 <= channel <= MAX_CHANNEL_VALUE for channel in channels))
        if not channels_valid:
            raise ValueError(f"figures.colours.{colour_name} must be {RGB_CHANNEL_COUNT} ints in "
                             f"0-{MAX_CHANNEL_VALUE}, got {channels!r}")
        colours[colour_name] = tuple(channels)
    # Step 5: optional bev bin, strings.
    bev_bin_m = figures_block.get("bev_bin_m")
    if bev_bin_m is not None and (isinstance(bev_bin_m, bool) or not isinstance(bev_bin_m, (int, float))
                                  or bev_bin_m <= 0):
        raise ValueError(f"figures.bev_bin_m must be null or a positive number, got {bev_bin_m!r}")
    text_values = {}
    for text_key in ("difference_cmap", "occupancy_cmap", "footer_note", "representative_metric"):
        text_value = figures_block.get(text_key)
        if not isinstance(text_value, str) or not text_value:
            raise ValueError(f"figures.{text_key} must be a non-empty string, got {text_value!r}")
        text_values[text_key] = text_value
    # Step 6: the threshold grid has at least its two end points.
    threshold_curve_points = _require_positive_int(figures_block, "threshold_curve_points")
    if threshold_curve_points < MINIMUM_THRESHOLD_CURVE_POINTS:
        raise ValueError(f"figures.threshold_curve_points must be at least {MINIMUM_THRESHOLD_CURVE_POINTS}, "
                         f"got {threshold_curve_points}")
    return FigureSettings(
        arm_order=tuple(arm_order),
        arm_labels=dict(arm_labels),
        campaign_profiles=campaign_profiles,
        section_trunk_count=_require_positive_int(figures_block, "section_trunk_count"),
        rows_per_page=_require_positive_int(figures_block, "rows_per_page"),
        dpi=_require_positive_int(figures_block, "dpi"),
        tile_px=_require_positive_int(figures_block, "tile_px"),
        bev_bin_m=bev_bin_m,
        difference_cmap=text_values["difference_cmap"],
        occupancy_cmap=text_values["occupancy_cmap"],
        colours=colours,
        footer_note=text_values["footer_note"],
        representative_metric=text_values["representative_metric"],
        threshold_curve_points=threshold_curve_points,
    )


def resolve_arm_runs(session: str, profile_by_arm: Mapping[str, str],
                     submitted_rows: Sequence[Mapping[str, str]], runs_dir: Path) -> Dict[str, List[Path]]:
    """Find the local run folders of each arm.

    Args:
        session: Session name; ledger rows of other sessions are ignored.
        profile_by_arm: arm -> Slurm profile name.
        submitted_rows: Ledger rows (dicts with run_id, session, profile), e.g. from ``read_ledger``.
        runs_dir: Folder holding runs/<run_id>/.

    Returns:
        arm -> run folders (existing only), sorted by run id; an arm without runs maps to [].
    """
    runs_by_arm: Dict[str, List[Path]] = {}
    for arm, profile in profile_by_arm.items():
        # Step 1: ledger rows of this session and profile, unique run ids, sorted for determinism.
        matching_run_ids = sorted({row["run_id"] for row in submitted_rows
                                   if row.get("session") == session and row.get("profile") == profile})
        # Step 2: keep the ones fetched to disk.
        run_folders = [runs_dir / run_id for run_id in matching_run_ids]
        runs_by_arm[arm] = [folder for folder in run_folders if folder.is_dir()]
    return runs_by_arm


def median_repeat(metric_by_run: Mapping[Path, Optional[float]]) -> Path:
    """The run folder whose metric value is the median of its arm's repeats.

    Sorting the defined values ascending (ties broken by run folder name, so the choice is
    deterministic) and taking position (n - 1) // 2 gives the exact median for odd n (the
    campaign's 5 repeats: the 3rd smallest) and the lower median for even n. Runs whose metric
    is undefined (None) are left out.

    Args:
        metric_by_run: run folder -> metric value (None when undefined).

    Returns:
        The median run folder.

    Raises:
        ValueError: When no run has a defined value.
    """
    # Step 1: keep the defined values; an arm with none cannot be represented.
    defined_runs = [(value, run_folder.name, run_folder) for run_folder, value in metric_by_run.items()
                    if value is not None]
    if not defined_runs:
        raise ValueError(f"no run has a defined value among {sorted(str(folder) for folder in metric_by_run)}")
    # Step 2: ascending by (value, name); the lower-median position.
    defined_runs.sort(key=lambda entry: (entry[0], entry[1]))
    median_position = (len(defined_runs) - 1) // 2
    return defined_runs[median_position][2]
