"""Registry of every Viking Hill -> SplatAD transformation / calibration step.

Each conversion function is tagged with ``@transform("T<n>", description)``. The
traceability test (tests/test_traceability.py) fails if a registered transform has
no unit test marked ``@pytest.mark.covers("T<n>")``, so no transform can be added
without a test.
"""

from dataclasses import dataclass
from typing import Callable, Dict

# Every transform id promised by the approved plan. A registered id must be in this
# set (typo guard); ids not yet registered are reported as pending implementation.
PLANNED_TRANSFORM_IDS = frozenset(
    {
        "T1", "T2", "T3", "T4", "T5", "T6",
        "T7a", "T7b", "T7c",
        "T8", "T9a", "T9b", "T10",
        "T11", "T12", "T13", "T14", "T15", "T16", "T17", "T18", "T19",
        "T20", "T21", "T22", "T23", "T24", "T25", "T26", "T27", "T28",
        "T29", "T30", "T31", "T32", "T33",
        # Sweep period covering every extracted sweep (fix for the June 20-22 contract failure).
        "T49",
        # Rendered lidar scans: dataparser-frame pose -> map-frame pose (scripts/08_*).
        "T50",
        # Phase B, option C: Ouster no-returns as a per-sweep sidecar (extract/no_return.py) and
        # their 2000 m placeholders in the dataparser (splatad/viking_conversions.py).
        "T51", "T52", "T53", "T54", "T55",
        # P2: a run's checkpoint Gaussians in the map frame (eval/gaussian_occupancy.py).
        "T56",
        # A run's exact train/test frames rebuilt without torch (dataset/run_split.py,
        # scripts/09_extract_run_split.py): neurad's time window and per-sensor LINSPACE split.
        "T58", "T59",
        # Middle-block test split shared by every sensor (dataset/run_split.py, used by
        # VikingHillDataParser's test_block_count and scripts/09).
        "T60",
        # Elevation mapping = far-field beam elevations of the T51 fit (extract/no_return.py),
        # replacing T16's parallax-biased medians in the manifest.
        "T61",
        # Test block cores (dataset/run_split.py, VikingHillDataParser.test_block_core_fraction):
        # only the central part of each T60 block is test and rendered; the rest is guard.
        "T62",
        # Eval cadence (eval/training_curves.py): largest interval giving >= P all-images evals.
        "T63",
        # Train/test metric pairing from TensorBoard tags (eval/training_curves.py).
        "T64",
    }
)


@dataclass(frozen=True)
class TransformSpec:
    """Metadata for one registered transformation.

    Attributes:
        transform_id: Plan id such as "T9b".
        description: One-line statement of what the transform does.
        qualified_name: ``module.function`` that implements it.
    """

    transform_id: str
    description: str
    qualified_name: str


# Registered transforms keyed by id; filled as modules are imported.
TRANSFORM_REGISTRY: Dict[str, TransformSpec] = {}


def transform(transform_id: str, description: str) -> Callable:
    """Decorator registering a function as a plan transformation.

    Args:
        transform_id: Id from PLANNED_TRANSFORM_IDS.
        description: One-line description of the transformation.

    Returns:
        Decorator returning the function unchanged (registration is a side effect).

    Raises:
        ValueError: If the id is unknown or already registered by another function.
    """

    def register(function: Callable) -> Callable:
        # Guard against typos: only ids promised by the plan may be registered.
        if transform_id not in PLANNED_TRANSFORM_IDS:
            raise ValueError(f"Unknown transform id {transform_id!r}")
        qualified_name = f"{function.__module__}.{function.__qualname__}"
        existing_spec = TRANSFORM_REGISTRY.get(transform_id)
        # One id <-> one implementation, so a test maps unambiguously to code.
        if existing_spec is not None and existing_spec.qualified_name != qualified_name:
            raise ValueError(
                f"{transform_id} already registered by {existing_spec.qualified_name}"
            )
        TRANSFORM_REGISTRY[transform_id] = TransformSpec(
            transform_id, description, qualified_name
        )
        return function

    return register
