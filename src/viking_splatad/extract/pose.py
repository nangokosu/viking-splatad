"""Trajectory interpolation (T3) and /tf chain composition + z-up check (T4)."""

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from viking_splatad.registry import transform

# Body-frame up axis (+z) used when no other is given; z-up is the convention being checked.
DEFAULT_BODY_UP_AXIS = (0.0, 0.0, 1.0)


@dataclass
class Trajectory:
    """Time-stamped SE(3) samples T_world_body.

    Attributes:
        timestamps: (N,) strictly increasing float64 seconds.
        positions: (N,3) translations.
        quaternions_xyzw: (N,4) orientations (x,y,z,w).
    """

    timestamps: np.ndarray
    positions: np.ndarray
    quaternions_xyzw: np.ndarray

    def __post_init__(self) -> None:
        """Validate shapes and monotonic timestamps; coerce to float64 arrays."""
        self.timestamps = np.asarray(self.timestamps, dtype=np.float64)
        self.positions = np.asarray(self.positions, dtype=np.float64)
        self.quaternions_xyzw = np.asarray(self.quaternions_xyzw, dtype=np.float64)
        sample_count = len(self.timestamps)
        # Slerp needs >= 2 keyframes.
        if sample_count < 2:
            raise ValueError("Trajectory needs at least two samples")
        if self.positions.shape != (sample_count, 3) or self.quaternions_xyzw.shape != (sample_count, 4):
            raise ValueError("positions must be (N,3) and quaternions (N,4) matching timestamps")
        if not np.all(np.diff(self.timestamps) > 0):
            raise ValueError("timestamps must be strictly increasing")


@transform("T3", "Interpolate SE3 poses from a trajectory (lerp translation, slerp rotation)")
def interpolate_pose_se3(
    trajectory: Trajectory,
    query_times: np.ndarray,
    allow_extrapolation: bool = False,
) -> np.ndarray:
    """Interpolate T_world_body at query times.

    Translation: p(t) = (1-a) p0 + a p1, a = (t-t0)/(t1-t0).
    Rotation: slerp, q(t) = q0 (q0^-1 q1)^a (constant angular velocity on the segment).

    Args:
        trajectory: Source samples.
        query_times: (M,) seconds on the trajectory clock.
        allow_extrapolation: If True, out-of-range queries clamp to the nearest end pose;
            otherwise they raise.

    Returns:
        (M,4,4) float64 poses.

    Raises:
        ValueError: If a query is outside the span and extrapolation is not allowed.
    """
    query_times = np.atleast_1d(np.asarray(query_times, dtype=np.float64))
    first_time, last_time = trajectory.timestamps[0], trajectory.timestamps[-1]
    out_of_range = (query_times < first_time) | (query_times > last_time)
    if out_of_range.any() and not allow_extrapolation:
        raise ValueError(
            f"{int(out_of_range.sum())} query times outside [{first_time}, {last_time}]"
        )
    # Clamp (no-op for in-range queries) so lerp/slerp stay inside the keyframes.
    clamped_times = np.clip(query_times, first_time, last_time)
    # Right-neighbour keyframe for each query, kept in [1, N-1] so a left neighbour exists.
    right_index = np.clip(
        np.searchsorted(trajectory.timestamps, clamped_times, side="right"),
        1,
        len(trajectory.timestamps) - 1,
    )
    left_index = right_index - 1
    segment_duration = trajectory.timestamps[right_index] - trajectory.timestamps[left_index]
    blend_fraction = (clamped_times - trajectory.timestamps[left_index]) / segment_duration
    # Linear translation.
    positions = (
        (1.0 - blend_fraction)[:, None] * trajectory.positions[left_index]
        + blend_fraction[:, None] * trajectory.positions[right_index]
    )
    # Spherical rotation interpolation over all keyframes.
    slerp = Slerp(trajectory.timestamps, Rotation.from_quat(trajectory.quaternions_xyzw))
    rotation_matrices = slerp(clamped_times).as_matrix()
    # Assemble homogeneous matrices.
    poses = np.tile(np.eye(4), (len(query_times), 1, 1))
    poses[:, :3, :3] = rotation_matrices
    poses[:, :3, 3] = positions
    return poses


@dataclass
class TfEdge:
    """One /tf transform parent->child.

    Attributes:
        translation: (3,) child origin in parent frame.
        quaternion_xyzw: (4,) child orientation in parent frame.
        invert: If True the edge is traversed child->parent, so its inverse is used.
    """

    translation: np.ndarray
    quaternion_xyzw: np.ndarray
    invert: bool = False


@transform("T4", "Compose a /tf chain (with inverse edges) into T_parent_child")
def compose_tf_chain(edges: Sequence[TfEdge]) -> np.ndarray:
    """Compose edges ordered from root to leaf: T = E1 @ E2 @ ... @ En.

    An inverted edge contributes E^-1 = [R^T, -R^T t].

    Args:
        edges: Ordered chain, root first.

    Returns:
        4x4 float64 T_root_leaf.
    """
    composed = np.eye(4)
    for edge in edges:
        edge_matrix = np.eye(4)
        edge_matrix[:3, :3] = Rotation.from_quat(edge.quaternion_xyzw).as_matrix()
        edge_matrix[:3, 3] = edge.translation
        # Rigid inverse is exact and better conditioned than a general inverse.
        if edge.invert:
            rotation_transposed = edge_matrix[:3, :3].T
            edge_matrix = np.eye(4)
            edge_matrix[:3, :3] = rotation_transposed
            edge_matrix[:3, 3] = -rotation_transposed @ np.asarray(edge.translation, dtype=np.float64)
        composed = composed @ edge_matrix
    return composed


def mean_up_vector_from_orientations(
    quaternions_xyzw: np.ndarray, body_up_axis: Sequence[float] = DEFAULT_BODY_UP_AXIS
) -> np.ndarray:
    """Mean unit up-vector in the world frame from body orientations.

    Each up_i = R_i @ body_up_axis; the mean is re-normalised to unit length.

    Args:
        quaternions_xyzw: (N,4) body->world orientations.
        body_up_axis: Up direction in the body frame.

    Returns:
        (3,) unit vector.
    """
    # Rotate the body up axis into the world for every sample, then average.
    up_vectors = Rotation.from_quat(quaternions_xyzw).apply(np.asarray(body_up_axis, dtype=np.float64))
    mean_up = up_vectors.mean(axis=0)
    return mean_up / np.linalg.norm(mean_up)


def check_world_z_up(up_vector_world: np.ndarray, tolerance_radians: float) -> float:
    """Return the tilt of an up-vector from world +z, raising if above tolerance.

    tilt = arccos(clip(u_hat . z_hat, -1, 1)).

    Args:
        up_vector_world: Estimated up (or negative gravity) direction in the world frame.
        tolerance_radians: Maximum allowed tilt.

    Returns:
        Tilt in radians.

    Raises:
        ValueError: If tilt exceeds the tolerance.
    """
    unit_up = np.asarray(up_vector_world, dtype=np.float64)
    unit_up = unit_up / np.linalg.norm(unit_up)
    world_z_axis = np.array([0.0, 0.0, 1.0])
    # Clip guards arccos against rounding just outside [-1, 1].
    tilt_radians = float(np.arccos(np.clip(unit_up @ world_z_axis, -1.0, 1.0)))
    if tilt_radians > tolerance_radians:
        raise ValueError(
            f"World is not z-up: tilt {tilt_radians:.4f} rad exceeds tolerance {tolerance_radians:.4f} rad"
        )
    return tilt_radians
