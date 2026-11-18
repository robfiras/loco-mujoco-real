"""Terminal-state handler for motion tracking.

* ``bad_anchor_pos_z_only``     -- anchor body height differs from the reference by
                                   more than 0.25 m
* ``bad_anchor_ori``            -- projected-gravity z differs by more than 0.8
* ``bad_motion_body_pos_z_only`` -- any tracked end effector's height differs from the
                                   **re-anchored** reference by more than 0.25 m

**Every condition is z-only, and that is the point.** The reference is re-anchored to the
robot's live xy and heading each step (see :mod:`lmj_real.core.utils.tracking`), so
horizontal and heading drift are deliberately not failures -- they are penalised, and
only mildly, by the two global anchor terms in the reward. A robot that is a metre
downrange but still performing the motion should keep going.

Episode timeout is not handled here; it comes from the environment's ``horizon``
(500 steps at 50 Hz).
"""

from __future__ import annotations

from types import ModuleType
from typing import Any, List

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from loco_mujoco.core.terminal_state_handler.base import TerminalStateHandler
from loco_mujoco.core.trajectory import Trajectory
from loco_mujoco.core.utils.backend import assert_backend_is_supported
from mujoco import MjData
from mujoco.mjx import Data

from lmj_real.core.utils.tracking import anchor_delta, mat_from_quat, reanchor


class G1RealTrackingTerminalStateHandler(TerminalStateHandler):
    """Terminate on z-only deviation from the reference motion."""

    def __init__(
        self,
        env: Any,
        anchor_pos_z_threshold: float | None = 0.25,
        anchor_ori_threshold: float | None = 0.8,
        ee_body_pos_z_threshold: float | None = 0.25,
        body_names: List[str] | None = None,
        anchor_body_name: str | None = None,
        end_effector_body_names: List[str] | None = None,
    ):
        """
        Args:
            env: Environment instance.
            anchor_pos_z_threshold: Metres of anchor height deviation allowed. None
                disables the condition.
            anchor_ori_threshold: Allowed difference in projected-gravity z. None
                disables the condition.
            ee_body_pos_z_threshold: Metres of end-effector height deviation allowed.
                None disables the condition.
            body_names: Tracked bodies, anchor first. Defaults to the env's
                ``tracked_body_names``.
            anchor_body_name: Defaults to the env's ``anchor_body_name``.
            end_effector_body_names: Defaults to the env's
                ``end_effector_body_names``. Must be a subset of ``body_names``.
        """
        super().__init__(env)

        self._anchor_pos_z_threshold = anchor_pos_z_threshold
        self._anchor_ori_threshold = anchor_ori_threshold
        self._ee_body_pos_z_threshold = ee_body_pos_z_threshold

        model = env._model
        props = self._info_props

        names = list(props["tracked_body_names"]) if body_names is None else list(body_names)
        anchor = props["anchor_body_name"] if anchor_body_name is None else anchor_body_name
        ee = (
            list(props["end_effector_body_names"])
            if end_effector_body_names is None
            else list(end_effector_body_names)
        )
        missing = [n for n in ee if n not in names]
        if missing:
            raise ValueError(
                f"end-effector bodies must be tracked bodies; {missing} are not in {names}"
            )

        self._body_ids = np.array(
            [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in names]
        )
        self._anchor_row = names.index(anchor)
        self._anchor_body_id = int(self._body_ids[self._anchor_row])
        self._ee_rows = np.array([names.index(n) for n in ee])
        self._body_rootid = model.body_rootid

    def reset(
        self,
        env: Any,
        model: Any,
        data: MjData | Data,
        carry: Any,
        backend: ModuleType,
        traj: Trajectory | None = None,
    ):
        assert_backend_is_supported(backend)
        return data, carry

    def is_absorbing(self, env, obs, info, data, carry, traj=None):
        return self._is_absorbing_compat(env, obs, info, data, carry, np, traj)

    def mjx_is_absorbing(self, env, obs, info, data, carry, traj=None):
        return self._is_absorbing_compat(env, obs, info, data, carry, jnp, traj)

    def mjx_absorbing_prob(self, env, obs, info, data, carry, traj=None):
        """Smooth termination probability of the anchor height and tilt conditions.

        For algorithms that differentiate through a rollout and need a soft version of
        :meth:`mjx_is_absorbing`. Returns None when the end-effector condition is active,
        which has no smooth counterpart here.
        """
        if self._ee_body_pos_z_threshold is not None:
            return None
        ref = traj.data.get(carry.traj_state.traj_no, carry.traj_state.subtraj_step_no, jnp)
        robot_pos = data.xpos[self._anchor_body_id]
        robot_mat = data.xmat[self._anchor_body_id].reshape(3, 3)
        ref_mat = mat_from_quat(ref.xquat[self._anchor_body_id], jnp)
        alive = jnp.array(1.0)
        if self._anchor_pos_z_threshold is not None:
            dz = jnp.abs(ref.xpos[self._anchor_body_id][2] - robot_pos[2])
            alive = alive * (1.0 - jax.nn.sigmoid((dz - self._anchor_pos_z_threshold) / 0.02))
        if self._anchor_ori_threshold is not None:
            d_grav = jnp.abs(ref_mat[2, 2] - robot_mat[2, 2])
            alive = alive * (1.0 - jax.nn.sigmoid((d_grav - self._anchor_ori_threshold) / 0.04))
        return 1.0 - alive

    def _is_absorbing_compat(
        self,
        env: Any,
        obs: np.ndarray | jnp.ndarray,
        info: dict[str, Any],
        data: MjData | Data,
        carry: Any,
        backend: ModuleType,
        traj: Trajectory | None = None,
    ):
        assert_backend_is_supported(backend)
        if traj is None:
            raise ValueError("traj is required for G1RealTrackingTerminalStateHandler.")

        ref = traj.data.get(carry.traj_state.traj_no, carry.traj_state.subtraj_step_no, backend)

        robot_pos = data.xpos[self._body_ids]
        robot_mat = data.xmat[self._body_ids].reshape(-1, 3, 3)
        robot_anchor_pos = robot_pos[self._anchor_row]
        robot_anchor_mat = robot_mat[self._anchor_row]

        ref_pos = ref.xpos[self._body_ids]
        ref_mat = mat_from_quat(ref.xquat[self._body_ids], backend)
        ref_anchor_pos = ref_pos[self._anchor_row]
        ref_anchor_mat = ref_mat[self._anchor_row]

        is_absorbing = backend.array(False)

        if self._anchor_pos_z_threshold is not None:
            dz = backend.abs(ref_anchor_pos[2] - robot_anchor_pos[2])
            is_absorbing = backend.logical_or(
                is_absorbing, backend.greater(dz, self._anchor_pos_z_threshold)
            )

        if self._anchor_ori_threshold is not None:
            # Compare projected gravity z for reference and robot. With gravity (0, 0, -1)
            # that z component is just -R[2, 2], so the difference of the two is
            # R_ref[2,2] - R_robot[2,2].
            d_grav = backend.abs(ref_anchor_mat[2, 2] - robot_anchor_mat[2, 2])
            is_absorbing = backend.logical_or(
                is_absorbing, backend.greater(d_grav, self._anchor_ori_threshold)
            )

        if self._ee_body_pos_z_threshold is not None:
            delta_pos, delta_rot = anchor_delta(
                robot_anchor_pos, robot_anchor_mat, ref_anchor_pos, ref_anchor_mat, backend
            )
            ref_pos_rel, _ = reanchor(
                ref_pos, ref_mat, ref_anchor_pos, delta_pos, delta_rot, backend
            )
            dz = backend.abs(ref_pos_rel[self._ee_rows, 2] - robot_pos[self._ee_rows, 2])
            is_absorbing = backend.logical_or(
                is_absorbing,
                backend.any(backend.greater(dz, self._ee_body_pos_z_threshold)),
            )

        return is_absorbing, carry
