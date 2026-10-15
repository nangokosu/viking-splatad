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
