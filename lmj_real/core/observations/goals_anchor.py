"""Anchored tracking goal.

The ``tracking_goal`` observation group::

    command             -> reference joint positions and velocities   (29 + 29)
    motion_anchor_pos_b -> reference anchor position, in robot anchor frame     (3)
                           (only with ``use_anchor_pos``, off in the training config)
    motion_anchor_ori_b -> reference anchor orientation, 6D, robot anchor frame (6)
    robot_body_pos_b    -> robot's tracked bodies, in robot anchor frame  (14 * 3)
    robot_body_ori_b    -> robot's tracked bodies, 6D, robot anchor frame (14 * 6)

Two things are worth knowing.

First, the reference enters the *observation* only through the anchor offset and the
reference joint angles. The re-anchored reference body poses are consumed by the reward
and the terminal handler, not fed to the policy. The actor gets the anchor delta plus its
own body poses in its own anchor frame and infers the error.

Second, everything here is expressed in the *robot's* anchor frame (full orientation),
which is not the same as the heading-only frame used by the privileged all-body
observations in :mod:`lmj_real.core.observations.g1_real_obs`.

The reference is read in the world frame of the episode: when the environment starts
episodes at the origin (``recenter_root_xy``), ``env.get_traj_offset_xy`` gives the xy shift
from clip coordinates to the world, and it is added to the reference anchor here, as it is
in ``G1RealTrackingReward``. Robot and reference therefore start on the same spot.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any, List

import mujoco
import numpy as np
from loco_mujoco.core.observations import Goal
from loco_mujoco.core.trajectory import Trajectory
from loco_mujoco.core.utils import mj_jntid2qposid, mj_jntid2qvelid
from mujoco import MjData, MjModel
from mujoco.mjx import Data, Model

from lmj_real.core.utils.tracking import (
    anchor_delta,
    mat_from_quat,
    mat_to_6d_cols01,
    reanchor,
)


class GoalTrajMimicAnchor(Goal):
    """Anchored tracking goal."""

    def __init__(self, info_props: dict, body_names: List[str] | None = None,
                 anchor_body_name: str | None = None,
                 visualize_raw_reference: bool = False,
                 use_anchor: bool = True, use_anchor_pos: bool = True, **kwargs):
        """
        Args:
            info_props: Environment info properties.
            body_names: Tracked bodies, anchor first. Defaults to the env's
                ``tracked_body_names``.
            anchor_body_name: Anchor body. Defaults to the env's ``anchor_body_name``.
            visualize_raw_reference: With ``visualize_goal``, also draw the *unanchored*
                world-frame reference (red), alongside the re-anchored one (green). The
                re-anchored markers are pinned to the robot's xy and heading by
                construction, so they cannot show global drift; the raw markers can.
            use_anchor: When False, drop ``motion_anchor_pos_b`` (3) and
                ``motion_anchor_ori_b`` (6) from the observation, taking it from 193 to
                184 dims. Those nine numbers are the ONLY place the reference's spatial
                pose enters this observation -- the reference joint angles are
                frame-free and the body terms describe the robot's own state in its own
                frame -- so removing them leaves the policy with no notion of where the
                reference body is, only what pose it holds. Pair with
                ``G1RealTrackingReward(use_anchor=False)``, which drops the matching
                re-anchoring from the reward. Checkpoints are not interchangeable
                between the two settings: the observation dimension differs.
            use_anchor_pos: When False, drop only ``motion_anchor_pos_b`` (3), taking the
                observation from 193 to 190 dims. It is the one term a real robot cannot
                sense: it needs a world position, whereas ``motion_anchor_ori_b`` only
                needs a yaw relative to the start of the motion. The reward is unchanged,
                so keep ``G1RealTrackingReward(use_anchor=True)``. Ignored when
                ``use_anchor`` is False. Defaults True so that checkpoints whose saved
                config predates this option rebuild with the observation they trained on.
        """
        # n_visual_geoms must be known at construction (StatefulObject allocates the
        # slots), so resolve the body list here rather than in _init_from_mj.
        names = list(body_names) if body_names is not None else list(
            info_props["tracked_body_names"]
        )
        self._visualize_raw = bool(visualize_raw_reference)
        self._use_anchor = bool(use_anchor)
        self._use_anchor_pos = self._use_anchor and bool(use_anchor_pos)
        n_visual = len(names) * (2 if self._visualize_raw else 1) \
            if kwargs.get("visualize_goal") else 0
        super().__init__(info_props, n_visual_geoms=n_visual, **kwargs)
        self._body_names = names
        self._anchor_body_name = anchor_body_name
        self.allow_randomization = False

        self._dim = None
        self._body_ids = None
        self._anchor_body_id = None
        self._anchor_row = None
        self._qpos_ind = None
        self._qvel_ind = None

    def _init_from_mj(self, env, model, data, current_obs_size):
        props = self._info_props
        if self._anchor_body_name is None:
            self._anchor_body_name = props["anchor_body_name"]
        if self._anchor_body_name not in self._body_names:
            raise ValueError(
                f"anchor body {self._anchor_body_name!r} must be one of the tracked bodies"
            )

        self._body_ids = np.array(
            [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in self._body_names]
        )
        if np.any(self._body_ids < 0):
            missing = [n for n, i in zip(self._body_names, self._body_ids) if i < 0]
            raise ValueError(f"unknown tracked bodies: {missing}")
        self._anchor_row = self._body_names.index(self._anchor_body_name)
        self._anchor_body_id = int(self._body_ids[self._anchor_row])

        # Reference joint angles/velocities: every non-free joint.
        qpos_ind, qvel_ind = [], []
        for i in range(model.njnt):
            if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE:
                qpos_ind.append(mj_jntid2qposid(i, model))
                qvel_ind.append(mj_jntid2qvelid(i, model))
        self._qpos_ind = np.concatenate(qpos_ind)
        self._qvel_ind = np.concatenate(qvel_ind)

        n_bodies = len(self._body_ids)
        self._dim = (
            len(self._qpos_ind)      # reference joint pos
            + len(self._qvel_ind)    # reference joint vel
            + (3 if self._use_anchor_pos else 0)   # motion_anchor_pos_b
            + (6 if self._use_anchor else 0)       # motion_anchor_ori_b
            + n_bodies * 3           # robot_body_pos_b
            + n_bodies * 6           # robot_body_ori_b
        )
        self.min = [-np.inf] * self._dim
        self.max = [np.inf] * self._dim
        self.data_type_ind = np.array(list(range(data.userdata.size)))
        self.obs_ind = np.array(list(range(current_obs_size, current_obs_size + self._dim)))
        self._initialized_from_mj = True

    def init_from_traj(self, traj_handler: Any):
        assert traj_handler is not None, (
            f"{type(self).__name__} requires a trajectory, but none was loaded."
        )
        self._initialized_from_traj = True

    def get_obs_and_update_state(
        self,
        env: Any,
        model: MjModel | Model,
        data: MjData | Data,
        carry: Any,
        backend: ModuleType,
        traj: Trajectory | None = None,
    ):
        if traj is None:
            raise ValueError("traj is required for GoalTrajMimicAnchor.")
        traj_state = carry.traj_state
        ref = traj.data.get(traj_state.traj_no, traj_state.subtraj_step_no, backend)

        # Robot anchor frame.
        robot_anchor_pos = data.xpos[self._anchor_body_id]
        robot_anchor_mat = data.xmat[self._anchor_body_id].reshape(3, 3)
        robot_anchor_inv = robot_anchor_mat.T

        # Reference anchor pose, expressed in the robot's anchor frame.
        # (clip coordinates -> world: the robot may have been moved to the origin at reset)
        offset = backend.concatenate([env.get_traj_offset_xy(carry, backend, traj), backend.zeros(1)])
        ref_anchor_pos = ref.xpos[self._anchor_body_id] + offset
        ref_anchor_mat = mat_from_quat(ref.xquat[self._anchor_body_id], backend)
        anchor_pos_b = robot_anchor_inv @ (ref_anchor_pos - robot_anchor_pos)
        anchor_ori_b = mat_to_6d_cols01(robot_anchor_inv @ ref_anchor_mat, backend)

        # The robot's own tracked bodies, in its anchor frame.
        body_pos_b = (data.xpos[self._body_ids] - robot_anchor_pos) @ robot_anchor_inv.T
        body_mats = data.xmat[self._body_ids].reshape(-1, 3, 3)
        body_ori_b = mat_to_6d_cols01(
            backend.einsum("ij,bjk->bik", robot_anchor_inv, body_mats), backend
        )

        blocks = [ref.qpos[self._qpos_ind], ref.qvel[self._qvel_ind]]
        if self._use_anchor_pos:
            blocks.append(anchor_pos_b)
        if self._use_anchor:
            blocks.append(anchor_ori_b)
        blocks += [backend.ravel(body_pos_b), backend.ravel(body_ori_b)]
        obs = backend.concatenate(blocks)

        if self.visualize_goal:
            carry = self._draw_reanchored_reference(
                robot_anchor_pos, robot_anchor_mat, ref, carry, backend
            )

        return obs, carry

    def _draw_reanchored_reference(self, robot_anchor_pos, robot_anchor_mat, ref, carry,
                                   backend: ModuleType):
        """Draw the **re-anchored** reference body poses as boxes.

        Deliberately the re-anchored reference rather than the raw one: that is the target
        the reward actually scores against (:mod:`lmj_real.core.utils.tracking`), so the
        markers show where the policy is being asked to put each body given its current
        xy and heading. Drawing the raw reference would drift away from the robot and look
        like a tracking failure even when tracking is perfect.
        """
        ref_pos = ref.xpos[self._body_ids]
        ref_mat = mat_from_quat(ref.xquat[self._body_ids], backend)
        delta_pos, delta_rot = anchor_delta(
            robot_anchor_pos, robot_anchor_mat, ref_pos[self._anchor_row],
            ref_mat[self._anchor_row], backend,
        )
        pos_rel, mat_rel = reanchor(
            ref_pos, ref_mat, ref_pos[self._anchor_row], delta_pos, delta_rot, backend
        )

        n_bodies = len(self._body_ids)
        if self._visualize_raw:
            pos_rel = backend.concatenate([pos_rel, ref_pos], axis=0)
            mat_rel = backend.concatenate([mat_rel, ref_mat], axis=0)
        n = self.n_visual_geoms
        idx = self.visual_geoms_idx
        geoms = carry.user_scene.geoms
        gtype = backend.full(n, int(mujoco.mjtGeom.mjGEOM_BOX), dtype=backend.int32).reshape(-1, 1)
        size = backend.tile(backend.array([0.045, 0.045, 0.045]), (n, 1))
        if self._visualize_raw:
            rgba = backend.concatenate([
                backend.tile(backend.array([0.0, 1.0, 0.0, 0.55]), (n_bodies, 1)),
                backend.tile(backend.array([1.0, 0.2, 0.2, 0.35]), (n_bodies, 1)),
            ], axis=0)
        else:
            rgba = backend.tile(backend.array([0.0, 1.0, 0.0, 0.55]), (n, 1))
        mat = backend.reshape(mat_rel, (n, 9))

        if backend is np:
            geoms.pos[idx] = pos_rel
            geoms.mat[idx] = mat
            geoms.type[idx] = gtype
            geoms.size[idx] = size
            geoms.rgba[idx] = rgba
            new_geoms = geoms
        else:
            new_geoms = geoms.replace(
                pos=geoms.pos.at[idx].set(pos_rel),
                mat=geoms.mat.at[idx].set(mat),
                type=geoms.type.at[idx].set(gtype),
                size=geoms.size.at[idx].set(size),
                rgba=geoms.rgba.at[idx].set(rgba),
            )
        return carry.replace(user_scene=carry.user_scene.replace(geoms=new_geoms))

    @property
    def has_visual(self) -> bool:
        return True

    @property
    def requires_trajectory(self) -> bool:
        return True

    @property
    def dim(self) -> int:
        return self._dim
