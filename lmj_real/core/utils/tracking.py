"""Shared math for motion tracking.

The anchor re-framing below lives in one place and is used by all three consumers -- the goal observation, the reward, and the
terminal-state handler -- rather than being reimplemented three times.

**What re-anchoring does.** The robot is not compared to the reference in world
coordinates. Every step the reference is moved so that its anchor body (the pelvis)
sits at the robot's *current* xy and heading, while keeping the reference's own height,
roll and pitch::

    delta_pos = robot_anchor_pos            with z taken from ref_anchor_pos
    delta_rot = yaw_only(R_robot_anchor @ R_ref_anchor^T)
    ref_body_pos_rel = delta_pos + delta_rot @ (ref_body_pos - ref_anchor_pos)
    ref_body_rot_rel = delta_rot @ ref_body_rot

The consequence is that limb tracking is invariant to global xy drift and heading
drift: those are policed only by the separate, lower-weighted global anchor reward
terms. This is why the terminations are all z-only -- a robot that has drifted a
metre downrange but is still doing the right motion should not be killed.
"""

from __future__ import annotations

from types import ModuleType

import jax.numpy as jnp
import numpy as np
from jax.scipy.spatial.transform import Rotation as jnp_R
from loco_mujoco.core.utils.math import quat_scalarfirst2scalarlast, transform_motion
from scipy.spatial.transform import Rotation as np_R


def _rotation(backend: ModuleType):
    return np_R if backend is np else jnp_R


def mat_from_quat(quats, backend: ModuleType):
    """Scalar-first quaternions -> rotation matrices.

    Args:
        quats: Shape (..., 4), MuJoCo's (w, x, y, z) convention.
        backend: ``np`` or ``jnp``.

    Returns:
        Rotation matrices of shape (..., 3, 3).
    """
    flat = backend.reshape(quats, (-1, 4))
    mats = _rotation(backend).from_quat(quat_scalarfirst2scalarlast(flat)).as_matrix()
    return backend.reshape(mats, quats.shape[:-1] + (3, 3))


def yaw_angle(mat, backend: ModuleType):
    """Yaw of a rotation matrix, ``atan2(R[1,0], R[0,0])``.

    Args:
        mat: Rotation matrix, shape (3, 3).
        backend: ``np`` or ``jnp``.
    """
    return backend.arctan2(mat[1, 0], mat[0, 0])


def rot_z(angle, backend: ModuleType):
    """Rotation matrix about the world z axis.

    Args:
        angle: Scalar angle in radians.
        backend: ``np`` or ``jnp``.

    Returns:
        Rotation matrix, shape (3, 3).
    """
    c, s = backend.cos(angle), backend.sin(angle)
    zero, one = backend.zeros_like(c), backend.ones_like(c)
    return backend.stack(
        [
            backend.stack([c, -s, zero]),
            backend.stack([s, c, zero]),
            backend.stack([zero, zero, one]),
        ]
    )


def yaw_rot(mat, backend: ModuleType):
    """Yaw-only part of a rotation matrix."""
    return rot_z(yaw_angle(mat, backend), backend)


def heading_rot_inv(mat, backend: ModuleType):
    """Inverse heading rotation, as a matrix.

    Equivalent to ``loco_mujoco...visualizer.extract_z_rotation`` inverted for
    |pitch| < 90 deg, but written out to avoid a quaternion/Euler round trip per step
    and to stay well defined at the Euler singularity.
    """
    return rot_z(-yaw_angle(mat, backend), backend)


def quat_to_tan_norm_from_mat(mats, backend: ModuleType):
    """6D tangent/normal rotation encoding.

    The reference x and z axes rotated by the orientation; for a rotation matrix those
    are its first and third columns.

    Args:
        mats: Rotation matrices, shape (n, 3, 3).
        backend: ``np`` or ``jnp``.

    Returns:
        Shape (n, 6): ``[tan_xyz, norm_xyz]`` per rotation.
    """
    return backend.concatenate([mats[..., :, 0], mats[..., :, 2]], axis=-1)


def rotation_angle(mat_a, mat_b, backend: ModuleType):
    """Geodesic angle between two rotations, in radians.

    The norm of the axis-angle difference, same metric as
    ``loco_mujoco...math.quaternion_angular_distance``; taken from
    matrices here because that is what the callers already hold, avoiding two
    quaternion conversions per body per step.

    Args:
        mat_a: Rotation matrices, shape (..., 3, 3).
        mat_b: Rotation matrices, shape (..., 3, 3).
        backend: ``np`` or ``jnp``.

    Returns:
        Angles of shape (...).
    """
    rel = backend.einsum("...ji,...jk->...ik", mat_a, mat_b)
    trace = rel[..., 0, 0] + rel[..., 1, 1] + rel[..., 2, 2]
    # atan2(2 sin, 2 cos) rather than arccos(cos): arccos has an infinite derivative at
    # an angle of zero, which is where a perfectly tracked body sits, so differentiating
    # a reward through it gives NaN there. The inner ``where`` keeps the sqrt away from
    # zero for the same reason.
    axis = backend.stack(
        [
            rel[..., 2, 1] - rel[..., 1, 2],
            rel[..., 0, 2] - rel[..., 2, 0],
            rel[..., 1, 0] - rel[..., 0, 1],
        ],
        axis=-1,
    )
    sq = backend.sum(axis * axis, axis=-1)
    sin2 = backend.where(sq > 0.0, backend.sqrt(backend.where(sq > 0.0, sq, 1.0)), 0.0)
    return backend.arctan2(sin2, trace - 1.0)


def mat_to_6d_cols01(mats, backend: ModuleType):
    """6D rotation encoding as the **first two columns** of a rotation matrix.

    This is the convention the tracking *observations* use, flattened row-major to ``[R00, R01, R10, R11, R20, R21]``.

    Distinct from :func:`quat_to_tan_norm_from_mat`, which takes columns 0 and **2**
    and is what the privileged all-body observations use. Both conventions coexist, so
    they are kept as separate functions.

    Args:
        mats: Rotation matrices, shape (..., 3, 3).
        backend: ``np`` or ``jnp``.

    Returns:
        Shape (..., 6).
    """
    return backend.reshape(mats[..., :, :2], mats.shape[:-2] + (6,))


def body_spatial_velocities(xpos, xmat, root_subtree_com, cvel, backend: ModuleType):
    """Spatial velocities at body frame origins, from raw fields.

    ``loco_mujoco.core.utils.math.calc_body_velocities`` does this for a live ``data``
    object, but trajectory data carries ``xquat`` rather than ``xmat``, so this variant
    takes the pieces explicitly. Same convention as MuJoCo's ``mj_objectVelocity``:
    **rotational in ``[:3]``, linear in ``[3:]``**, world frame.

    Args:
        xpos: Body frame origins, shape (n, 3).
        xmat: Body rotation matrices, shape (n, 3, 3).
        root_subtree_com: Subtree CoM of each body's root, shape (n, 3).
        cvel: MuJoCo ``cvel`` for those bodies, shape (n, 6).
        backend: ``np`` or ``jnp``.

    Returns:
        Shape (n, 6) spatial velocities.
    """
    return transform_motion(cvel, xpos, root_subtree_com, xmat, backend, False)


def reference_body_frames(traj_data_single, body_ids, body_rootid, backend: ModuleType):
    """Extract reference body poses and velocities for one trajectory frame.

    Uses the ``xpos`` / ``xquat`` / ``cvel`` / ``subtree_com`` fields that
    ``ExtendTrajData`` populates when a trajectory is built, so no forward kinematics
    is needed at run time.

    Args:
        traj_data_single: One frame of ``TrajectoryData``.
        body_ids: Body ids to extract, shape (n,).
        body_rootid: ``model.body_rootid``.
        backend: ``np`` or ``jnp``.

    Returns:
        ``(pos, mat, vel)`` with shapes (n, 3), (n, 3, 3), (n, 6).
    """
    pos = traj_data_single.xpos[body_ids]
    mat = mat_from_quat(traj_data_single.xquat[body_ids], backend)
    vel = body_spatial_velocities(
        pos, mat, traj_data_single.subtree_com[body_rootid[body_ids]],
        traj_data_single.cvel[body_ids], backend,
    )
    return pos, mat, vel


def anchor_delta(robot_anchor_pos, robot_anchor_mat, ref_anchor_pos, ref_anchor_mat,
                 backend: ModuleType):
    """The re-anchoring transform: robot xy + heading, reference z + roll/pitch.

    Args:
        robot_anchor_pos: Robot anchor body position, shape (3,).
        robot_anchor_mat: Robot anchor body rotation, shape (3, 3).
        ref_anchor_pos: Reference anchor position, shape (3,).
        ref_anchor_mat: Reference anchor rotation, shape (3, 3).
        backend: ``np`` or ``jnp``.

    Returns:
        ``(delta_pos, delta_rot)`` with shapes (3,) and (3, 3).
    """
    delta_pos = backend.concatenate([robot_anchor_pos[:2], ref_anchor_pos[2:3]])
    delta_rot = yaw_rot(robot_anchor_mat @ ref_anchor_mat.T, backend)
    return delta_pos, delta_rot


def reanchor(ref_pos, ref_mat, ref_anchor_pos, delta_pos, delta_rot, backend: ModuleType):
    """Apply :func:`anchor_delta` to a set of reference body poses.

    Args:
        ref_pos: Reference body positions, shape (n, 3).
        ref_mat: Reference body rotations, shape (n, 3, 3).
        ref_anchor_pos: Reference anchor position, shape (3,).
        delta_pos: Translation from :func:`anchor_delta`, shape (3,).
        delta_rot: Rotation from :func:`anchor_delta`, shape (3, 3).
        backend: ``np`` or ``jnp``.

    Returns:
        ``(pos_rel, mat_rel)`` with shapes (n, 3), (n, 3, 3).
    """
    pos_rel = delta_pos + (ref_pos - ref_anchor_pos) @ delta_rot.T
    mat_rel = backend.einsum("ij,bjk->bik", delta_rot, ref_mat)
    return pos_rel, mat_rel


def to_local(vectors, rot_inv, backend: ModuleType):
    """Rotate world-frame row vectors into a local frame.

    Args:
        vectors: Shape (n, 3).
        rot_inv: Inverse rotation of the target frame, shape (3, 3).
        backend: ``np`` or ``jnp``.
    """
    return vectors @ rot_inv.T
