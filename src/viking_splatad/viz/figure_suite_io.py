"""Run loading, cross-run checks, hashing, failure bookkeeping and manifests of the occupancy figure suite.

Everything the figure script needs before and after drawing, with no PyVista or matplotlib import: which run
each arm draws (``select_arm_runs``), whether the runs share one cell grid (``check_runs_consistent``,
``check_binding_threshold``), the sha256 of every input file (``HashRecorder``), per-figure success / skip /
failure records (``FigureRecorder``) and the ``figures_manifest.json`` written next to the pictures.
"""

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from viking_splatad.eval.p2_repeats import flatten_hidden_metrics
from viking_splatad.eval.run_context import file_sha256
from viking_splatad.viz.figure_runs import median_repeat

# Repository root: this file is <root>/src/viking_splatad/viz/figure_suite_io.py.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_SCHEMA = "occupancy-figures/1"
MANIFEST_FILE_NAME = "figures_manifest.json"
CELLS_RELATIVE_PATH = Path("eval") / "gaussian_occupancy_cells.npz"
REPORT_RELATIVE_PATH = Path("eval") / "gaussian_occupancy.json"
# Per-figure statuses recorded in the manifest: drawn, skipped for a stated reason (not an error), or failed.
STATUS_WRITTEN, STATUS_SKIPPED, STATUS_FAILED = "written", "skipped", "failed"
# Keys of the archive that do not depend on the model (must agree across runs of a session). ``threshold`` and
# ``p2_thresholds`` fix the cut every arm's outcomes use; ``evidence`` is the shared hidden-space labelling.
MODEL_INDEPENDENT_KEYS = ("cell_indices", "centres", "hidden", "hidden_near", "grid_origin", "voxel_size",
                          "threshold", "p2_thresholds", "evidence", "object_box_ids", "object_class_ids",
                          "object_cell_offsets", "object_cells", "object_inside")
MODEL_INDEPENDENT_PREFIXES = ("neighbourhood_", "truth_")
# Archive keys of the binding threshold and of the evaluated thresholds.
BINDING_THRESHOLD_KEY, P2_THRESHOLDS_KEY = "threshold", "p2_thresholds"


class FigureSkipped(Exception):
    """Raised by a figure function that has nothing to draw; the recorder stores its message as the reason."""


@dataclass
class ArmRun:
    """One run drawn for (or counted in) an arm, with its cells archive.

    Attributes:
        arm: Arm the run is drawn for or counted in (an override keeps the arm it stands in for).
        run_dir: Run folder.
        cells: Cells archive arrays (``load_cells``).
        profile: Slurm profile of the run (ledger, else the arm's campaign profile).
    """

    arm: str
    run_dir: Path
    cells: Dict[str, np.ndarray]
    profile: str

    @property
    def run_id(self) -> str:
        """Run folder name."""
        return self.run_dir.name

    @property
    def cells_path(self) -> Path:
        """Cells archive file."""
        return self.run_dir / CELLS_RELATIVE_PATH


@dataclass
class RunSelection:
    """Result of ``select_arm_runs``.

    Attributes:
        drawn: arm -> the run drawn for the arm (its ``arm`` is the arm it stands in for).
        repeats: arm -> the arm's campaign runs ([] when the arm has none and an override stands in).
        arms_info: arm -> manifest entry (profile of the drawn run, campaign profile, how it was drawn ...).
        cells_by_run: run folder name -> cells archive, every run loaded exactly once.
        binding_threshold: The occupancy cut every loaded run was archived at (``p2_thresholds[0]``).
    """

    drawn: Dict[str, ArmRun]
    repeats: Dict[str, List[ArmRun]]
    arms_info: Dict[str, Any]
    cells_by_run: Dict[str, Dict[str, np.ndarray]]
    binding_threshold: float


def parse_run_overrides(run_specs: Sequence[str], arm_order: Sequence[str]) -> Dict[str, Path]:
    """Parse repeated ``--run arm=<dir>`` options.

    Args:
        run_specs: Raw option values.
        arm_order: Valid arm keys.

    Returns:
        arm -> run folder.

    Raises:
        ValueError: On a value without ``=``, an unknown arm, or an arm given twice.
    """
    overrides: Dict[str, Path] = {}
    for run_spec in run_specs:
        # Step 1: split at the first "=" so folder names may contain "=" themselves.
        if "=" not in run_spec:
            raise ValueError(f"--run expects arm=<dir>, got {run_spec!r}")
        arm, directory = run_spec.split("=", 1)
        # Step 2: the arm must exist and appear once.
        if arm not in arm_order:
            raise ValueError(f"--run names unknown arm {arm!r}; arms are {list(arm_order)}")
        if arm in overrides:
            raise ValueError(f"--run names arm {arm!r} twice")
        overrides[arm] = Path(directory)
    return overrides


def choose_drawn_run(metric_by_run: Mapping[Path, Optional[float]], override: Optional[Path]) -> Path:
    """The run drawn for an arm: the override, or the median repeat of the representative metric.

    Args:
        metric_by_run: Campaign run folder -> representative metric (None = undefined).
        override: Folder from ``--run``, or None.

    Returns:
        The run folder to draw.
    """
    return override if override is not None else median_repeat(metric_by_run)


def check_runs_consistent(cells_by_run: Mapping[str, Mapping[str, np.ndarray]]) -> None:
    """Require every run's model-independent arrays to be identical to the first run's.

    The figures draw several runs on one cell grid, which is valid only when the grid, hidden flags,
    evidence labels, objects, neighbourhoods, truth masks and the binding threshold do not depend on the run.

    Args:
        cells_by_run: run label -> cells archive arrays.

    Raises:
        ValueError: Naming the first run and key that differ (or a key missing from a run).
    """
    labels = list(cells_by_run)
    if not labels:
        raise ValueError("no run to compare")
    reference = cells_by_run[labels[0]]
    # Step 1: the keys to compare are the fixed ones plus every neighbourhood_/truth_ array of the reference.
    keys = [key for key in MODEL_INDEPENDENT_KEYS if key in reference]
    keys += [key for key in reference if key.startswith(MODEL_INDEPENDENT_PREFIXES)]
    # Step 2: exact equality with the reference; the archives are integer/bool grids, or identical floats.
    for label in labels[1:]:
        for key in keys:
            if key not in cells_by_run[label] or not np.array_equal(reference[key], cells_by_run[label][key]):
                raise ValueError(f"runs are not comparable: '{key}' of {label} differs from {labels[0]}")


def check_binding_threshold(cells_by_run: Mapping[str, Mapping[str, np.ndarray]]) -> float:
    """Require every run's archived ``threshold`` to equal ``p2_thresholds[0]``; return that binding value.

    ``p2_cell_outcomes`` cuts every arm at the archived ``threshold`` and the figures name ``p2_thresholds[0]``,
    so a run archived at a different cut would silently be drawn at a different occupancy level.

    Args:
        cells_by_run: run label -> cells archive arrays.

    Returns:
        The binding threshold ``p2_thresholds[0]`` of the first run.

    Raises:
        ValueError: When no run is given, or a run's ``threshold`` differs from its ``p2_thresholds[0]``.
    """
    if not cells_by_run:
        raise ValueError("no run to check")
    binding_threshold = None
    for label, cells in cells_by_run.items():
        # Step 1: the first evaluated threshold is the binding one by construction of scripts/19.
        first_p2_threshold = float(np.asarray(cells[P2_THRESHOLDS_KEY], dtype=float).reshape(-1)[0])
        archived_threshold = float(cells[BINDING_THRESHOLD_KEY])
        if not np.isclose(archived_threshold, first_p2_threshold):
            raise ValueError(f"archive threshold {archived_threshold} of {label} differs from p2_thresholds[0] "
                             f"{first_p2_threshold}")
        binding_threshold = first_p2_threshold if binding_threshold is None else binding_threshold
    return binding_threshold


def load_cells(run_dir: Path) -> Dict[str, np.ndarray]:
    """Load a run's cells archive fully (small: per-cell arrays).

    Args:
        run_dir: Run folder.

    Returns:
        key -> array.
    """
    with np.load(run_dir / CELLS_RELATIVE_PATH, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def representative_value(run_dir: Path, metric_key: str) -> Optional[float]:
    """The representative metric of a run, from its ``gaussian_occupancy.json``.

    Args:
        run_dir: Run folder.
        metric_key: Flat metric key (``flatten_hidden_metrics``).

    Returns:
        The value (None when undefined for the run).

    Raises:
        KeyError: When the key is not a metric of the report.
    """
    report = json.loads((run_dir / REPORT_RELATIVE_PATH).read_text())
    return flatten_hidden_metrics(report)[metric_key]


def ledger_profile_by_run(rows: Sequence[Mapping[str, str]]) -> Dict[str, str]:
    """run_id -> profile from ledger rows.

    Args:
        rows: Ledger rows.

    Returns:
        Mapping.
    """
    return {row["run_id"]: row["profile"] for row in rows}


def select_arm_runs(arm_order: Sequence[str], campaign_profile_by_arm: Mapping[str, str],
                    runs_by_arm: Mapping[str, Sequence[Path]], overrides: Mapping[str, Path],
                    profile_by_run: Mapping[str, str], representative_metric: str, *,
                    cells_loader: Callable[[Path], Dict[str, np.ndarray]] = load_cells,
                    metric_reader: Callable[[Path, str], Optional[float]] = representative_value
                    ) -> RunSelection:
    """Load every campaign run and override once, check they share one grid, and pick the run drawn per arm.

    A ``--run arm=<dir>`` override may name a run of another arm (or of no campaign): the drawn ``ArmRun`` then
    carries the override's arm and the profile the ledger records for that folder. An arm without campaign
    runs is allowed when an override stands in for it; it then has no spread band.

    Args:
        arm_order: Arm keys, left to right.
        campaign_profile_by_arm: arm -> campaign profile name.
        runs_by_arm: arm -> campaign run folders (may be empty).
        overrides: arm -> run folder from ``--run``.
        profile_by_run: run folder name -> profile (ledger); a folder not in it falls back to the arm's
            campaign profile.
        representative_metric: Flat metric key whose median repeat is drawn.
        cells_loader: Reads a run's cells archive (injectable for tests).
        metric_reader: Reads the representative metric of a run (injectable for tests).

    Returns:
        The ``RunSelection``.

    Raises:
        ValueError: On an arm with neither campaign runs nor an override, non-comparable runs, or an archive
            whose binding threshold disagrees with ``p2_thresholds[0]``.
    """
    # Step 1: every arm needs campaign runs or an override that stands in for it.
    empty_arms = [arm for arm in arm_order if not runs_by_arm.get(arm) and arm not in overrides]
    if empty_arms:
        raise ValueError(f"no fetched campaign runs for arms {empty_arms}; pass --run <arm>=<dir> to stand in")
    # Step 2: load each folder once (cells) and build one ArmRun per (arm, folder): the same folder drawn for
    # another arm keeps that arm's name.
    cells_by_dir: Dict[Path, Dict[str, np.ndarray]] = {}
    arm_runs: Dict[Tuple[str, Path], ArmRun] = {}
    for arm in arm_order:
        for run_dir in list(runs_by_arm.get(arm, [])) + ([overrides[arm]] if arm in overrides else []):
            if run_dir not in cells_by_dir:
                cells_by_dir[run_dir] = cells_loader(run_dir)
            if (arm, run_dir) not in arm_runs:
                profile = profile_by_run.get(run_dir.name, campaign_profile_by_arm[arm])
                arm_runs[(arm, run_dir)] = ArmRun(arm, run_dir, cells_by_dir[run_dir], profile)
    # Step 3: one grid for all of them, cut at one archived threshold.
    cells_by_run = {run_dir.name: cells for run_dir, cells in cells_by_dir.items()}
    check_runs_consistent(cells_by_run)
    binding_threshold = check_binding_threshold(cells_by_run)
    # Step 4: representative metric of every campaign repeat, then the drawn run per arm.
    drawn: Dict[str, ArmRun] = {}
    repeats: Dict[str, List[ArmRun]] = {}
    arms_info: Dict[str, Any] = {}
    for arm in arm_order:
        campaign_dirs = list(runs_by_arm.get(arm, []))
        values = {run_dir: metric_reader(run_dir, representative_metric) for run_dir in campaign_dirs}
        drawn_dir = choose_drawn_run(values, overrides.get(arm))
        drawn[arm] = arm_runs[(arm, drawn_dir)]
        repeats[arm] = [arm_runs[(arm, run_dir)] for run_dir in campaign_dirs]
        arms_info[arm] = {
            "profile": drawn[arm].profile, "campaign_profile": campaign_profile_by_arm[arm],
            "drawn_run_id": drawn_dir.name, "drawn_by": "override" if arm in overrides else "median repeat",
            "campaign_run_count": len(campaign_dirs), "campaign_band": bool(campaign_dirs),
            "campaign_band_note": None if campaign_dirs else
            "no campaign runs for this arm: the override stands in and no spread band is drawn",
            "repeat_metric_values": {run_dir.name: value for run_dir, value in values.items()}}
    return RunSelection(drawn=drawn, repeats=repeats, arms_info=arms_info, cells_by_run=cells_by_run,
                        binding_threshold=binding_threshold)


def display_path(path: Path) -> str:
    """Repo-relative text of a path inside the project, else the absolute text.

    Args:
        path: Any path.

    Returns:
        The text recorded in manifests.
    """
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


class FigureRecorder:
    """Runs figure-drawing callables, catching and recording each failure or skip (never silent)."""

    def __init__(self) -> None:
        """Start with no figure recorded."""
        self.figures: Dict[str, Dict[str, Any]] = {}

    def record(self, name: str, output_path: Path, status: str, error: Optional[str] = None) -> None:
        """Store one figure's outcome.

        Args:
            name: Figure name.
            output_path: File (or folder) the figure writes.
            status: STATUS_WRITTEN, STATUS_SKIPPED or STATUS_FAILED.
            error: Error text, or the skip reason (None when written).
        """
        self.figures[name] = {"status": status, "path": display_path(output_path), "error": error}

    def guard(self, name: str, output_path: Path, action: Callable[[], Any]) -> Tuple[bool, Any]:
        """Call ``action``; on ``FigureSkipped`` record "skipped", on any other exception record "failed".

        Success is NOT recorded here (use ``attempt`` for that), so a preparation step can be guarded without
        adding a "written" entry of its own. Failures are also printed to stderr.

        Args:
            name: Figure name.
            output_path: File (or folder) recorded with a skip or failure.
            action: Zero-argument callable.

        Returns:
            (True, its result) on success, (False, None) after a skip or failure.
        """
        try:
            return True, action()
        except FigureSkipped as skip:
            print(f"figure {name} skipped: {skip}", file=sys.stderr)
            self.record(name, output_path, STATUS_SKIPPED, str(skip))
        except Exception as error:  # noqa: BLE001 - one failing figure must not stop the others
            message = f"{type(error).__name__}: {error}"
            print(f"figure {name} FAILED: {message}", file=sys.stderr)
            self.record(name, output_path, STATUS_FAILED, message)
        return False, None

    def attempt(self, name: str, output_path: Path, draw: Callable[[], Any]) -> bool:
        """Call ``draw``; record "written", "skipped" (``FigureSkipped``) or "failed" with the error.

        Args:
            name: Figure name.
            output_path: File the figure writes (recorded).
            draw: Zero-argument callable that draws and saves.

        Returns:
            True when it succeeded.
        """
        succeeded, _ = self.guard(name, output_path, draw)
        if succeeded:
            self.record(name, output_path, STATUS_WRITTEN)
        return succeeded

    def any_failed(self) -> bool:
        """Whether any recorded figure failed (a skip is not a failure).

        Returns:
            True when at least one status is "failed".
        """
        return any(entry["status"] == STATUS_FAILED for entry in self.figures.values())


class HashRecorder:
    """sha256 of every file read, cached per file and collected per scope (one manifest each)."""

    def __init__(self) -> None:
        """Start with an empty cache and scope."""
        self.cache: Dict[Path, str] = {}
        self.scope: Dict[str, str] = {}

    def begin_scope(self) -> None:
        """Forget the current scope's entries (the digest cache stays)."""
        self.scope = {}

    def record(self, path: Path) -> None:
        """Hash a file (once) and add it to the current scope.

        Args:
            path: File that was read.
        """
        resolved = Path(path).resolve()
        if resolved not in self.cache:
            self.cache[resolved] = file_sha256(resolved)
        self.scope[display_path(resolved)] = self.cache[resolved]


def git_state(project_root: Path) -> Dict[str, Any]:
    """HEAD commit and dirty flag of the repository (``git rev-parse``, ``git status --porcelain``).

    Args:
        project_root: Repository root.

    Returns:
        {"head_sha", "dirty"} (None values and an "error" text when git fails).
    """
    try:
        head_sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=project_root, capture_output=True, text=True,
                                  check=True).stdout.strip()
        status_text = subprocess.run(["git", "status", "--porcelain"], cwd=project_root, capture_output=True,
                                     text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        return {"head_sha": None, "dirty": None, "error": str(error)}
    return {"head_sha": head_sha, "dirty": bool(status_text.strip())}


def json_number_list(values: Any) -> List[Optional[float]]:
    """A numeric array as a JSON-ready list in which NaN (and any non-finite value) becomes None.

    Args:
        values: Array-like of numbers.

    Returns:
        Python floats and None, in order.
    """
    return [float(value) if np.isfinite(value) else None for value in np.asarray(values, dtype=float).reshape(-1)]


def build_manifest(*, session: str, scope_label: str, git: Mapping[str, Any], class_name: str,
                   binding_threshold: float, p2_thresholds: Sequence[float], representative_metric: str,
                   arms_info: Mapping[str, Any], figures: Mapping[str, Any], hashes: Mapping[str, str],
                   reported_numbers: Mapping[str, Any]) -> Dict[str, Any]:
    """Assemble ``figures_manifest.json``.

    Args:
        session: Session name.
        scope_label: "arms" or "run:<arm>".
        git: ``git_state`` result.
        class_name: Headline class (group) name.
        binding_threshold: The occupancy cut of the figures.
        p2_thresholds: Every evaluated threshold.
        representative_metric: Flat key that picked the drawn runs.
        arms_info: arm -> how its run was chosen.
        figures: name -> {"status", "path", "error"}.
        hashes: file -> sha256 of the files this scope read.
        reported_numbers: Every number printed in or behind the figures of this scope.

    Returns:
        JSON-serialisable dict.
    """
    return {
        "schema": MANIFEST_SCHEMA, "session": session, "scope": scope_label, "git": dict(git),
        "class": class_name, "binding_threshold": binding_threshold,
        "p2_thresholds": [float(value) for value in p2_thresholds],
        "representative_metric": representative_metric,
        "arms": dict(arms_info), "input_sha256": dict(hashes),
        "reported_numbers": dict(reported_numbers), "figures": dict(figures),
    }


def write_manifest(output_dir: Path, manifest: Mapping[str, Any]) -> Path:
    """Write the manifest JSON into an output folder.

    Args:
        output_dir: Folder.
        manifest: Manifest dict.

    Returns:
        The file written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / MANIFEST_FILE_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest_path
