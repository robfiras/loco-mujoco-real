"""Motion tracking reward.

Every tracking term is an exponential kernel ``exp(-error / std**2)``:

============================  =====  ======  =====================================
term                          std    weight  error
============================  =====  ======  =====================================
``motion_global_root_pos``    0.3     +0.5   squared distance, anchor body, world
``motion_global_root_ori``    0.4     +0.5   squared geodesic angle, anchor, world
``motion_body_pos``           0.3     +1.0   **re-anchored** body positions
``motion_body_ori``           0.4     +1.0   **re-anchored** body orientations
``motion_body_lin_vel``       1.0     +1.0   body linear velocities, world
``motion_body_ang_vel``       3.14    +1.0   body angular velocities, world
``action_rate_l2``             --     -0.1*s^2  sum of squared action differences
``joint_limit``                --    -10.0   amount outside the soft joint limits
``self_collisions``            --    -10.0   self-collision contacts
============================  =====  ======  =====================================

Maximum achievable reward is therefore ~5.0.

Body position/orientation error is measured against the **re-anchored** reference (see
:mod:`lmj_real.core.utils.tracking`), so xy and heading drift do not pollute limb
tracking, and errors are over *body frames*, not mimic *sites*.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any, List

import jax.numpy as jnp
import mujoco
import numpy as np
from flax import struct
from loco_mujoco.core.reward.trajectory_based import TrajectoryBasedReward, check_traj_provided
from loco_mujoco.core.trajectory import Trajectory
from loco_mujoco.core.utils import mj_jntid2qposid
from loco_mujoco.core.utils.math import calc_body_velocities
from mujoco import MjData, MjModel
from mujoco.mjx import Data, Model

from lmj_real.core.utils.tracking import (
    anchor_delta,
    mat_from_quat,
    reanchor,
    reference_body_frames,
    rotation_angle,
)


#: Largest excess over a soft joint limit that is still penalised, per joint, in radians.
MAX_JOINT_LIMIT_EXCESS = 1.0


@struct.dataclass
class G1RealTrackingRewardState:
    """Reward state: previous action for the action-rate penalty, plus logged parts."""

    last_action: np.ndarray | jnp.ndarray
    reward_components: dict


class G1RealTrackingReward(TrajectoryBasedReward):
    """Exponential-kernel motion tracking reward with action, limit and collision penalties."""

    def __init__(
        self,
        env: Any,
        body_names: List[str] | None = None,
        anchor_body_name: str | None = None,
        # exponential-kernel standard deviations
        root_pos_std: float = 0.3,
        root_ori_std: float = 0.4,
        body_pos_std: float = 0.3,
        body_ori_std: float = 0.4,
        body_lin_vel_std: float = 1.0,
        body_ang_vel_std: float = 3.14,
        # weights
        w_root_pos: float = 0.5,
        w_root_ori: float = 0.5,
        w_body_pos: float = 1.0,
        w_body_ori: float = 1.0,
        w_body_lin_vel: float = 1.0,
        w_body_ang_vel: float = 1.0,
        w_action_rate: float | None = None,
        w_joint_limit: float = -10.0,
        w_self_collision: float = 0.0,
        self_collision_penetration_threshold: float = 0.0,
        action_scale: float = 1.0,
        soft_joint_pos_limit_factor: float = 0.9,
        scale_by_dt: bool = False,
        use_anchor: bool = True,
        **kwargs: Any,
    ):
        """
        Args:
            env: Environment instance.
            body_names: Tracked bodies, anchor first. Defaults to the env's
                ``tracked_body_names``.
            anchor_body_name: Anchor body. Defaults to the env's ``anchor_body_name``.
            w_action_rate: Defaults to ``-0.1 * action_scale**2``.
            w_self_collision: Defaults to 0.0; ``training`` sets -10.0. See
                :meth:`_self_collision_count`.
            self_collision_penetration_threshold: Only contacts penetrating deeper than
                this (metres) count. 0.0 counts any penetration.
            use_anchor: When False, the ``motion_body_pos`` / ``motion_body_ori`` terms
                score the reference in its OWN world frame instead of re-anchored into
                the robot's live xy and heading. Re-anchoring exists so that global
                drift does not pollute limb tracking -- the two global root terms police
                drift separately -- so turning it off makes those two terms measure
                absolute world-frame error and the task strictly harder. Returns are
                therefore NOT comparable to anchored runs. Pair with
                ``GoalTrajMimicAnchor(use_anchor=False)``, which drops the matching nine
                observation dims.
            scale_by_dt: Multiply every term by the control timestep
                (``reward = raw * weight * dt``), so the 5.0-per-step ceiling becomes
                0.1, i.e. 50 over a 500-step episode.

                Turning it on is close to behaviourally neutral in this
                setup -- with ``normalize_env: true`` the running reward std absorbs the
                constant, and the PPO loss standardises advantages per minibatch
                (``gae = (gae - gae.mean()) / (gae.std() + 1e-8)``) -- but it is not
                exactly a no-op: the reward-normaliser warmup sees a different scale for
                the first updates. With ``normalize_env: false`` it is definitely not
                neutral, because the value loss and ``max_grad_norm`` would then see
                50x-smaller targets.
        """
        super().__init__(env, **kwargs)

        self._std = {
            "root_pos": root_pos_std,
            "root_ori": root_ori_std,
            "body_pos": body_pos_std,
            "body_ori": body_ori_std,
            "body_lin_vel": body_lin_vel_std,
            "body_ang_vel": body_ang_vel_std,
        }
        self._w = {
            "root_pos": w_root_pos,
            "root_ori": w_root_ori,
            "body_pos": w_body_pos,
            "body_ori": w_body_ori,
            "body_lin_vel": w_body_lin_vel,
            "body_ang_vel": w_body_ang_vel,
            "action_rate": (
                -0.1 * action_scale**2 if w_action_rate is None else w_action_rate
            ),
            "joint_limit": w_joint_limit,
            "self_collision": w_self_collision,
        }

        self._penetration_threshold = float(self_collision_penetration_threshold)
        self._use_anchor = bool(use_anchor)
        # env.dt is the control timestep (timestep * n_substeps).
        self._reward_scale = float(env.dt) if scale_by_dt else 1.0

        model = env._model
        props = self._info_props

        names = list(props["tracked_body_names"]) if body_names is None else list(body_names)
        anchor = props["anchor_body_name"] if anchor_body_name is None else anchor_body_name
        if anchor not in names:
            raise ValueError(f"anchor body {anchor!r} must be one of the tracked bodies")
        self._body_ids = np.array(
            [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in names]
        )
        if np.any(self._body_ids < 0):
            raise ValueError(
                f"unknown tracked bodies: {[n for n, i in zip(names, self._body_ids) if i < 0]}"
            )
        self._anchor_row = names.index(anchor)
        self._anchor_body_id = int(self._body_ids[self._anchor_row])
        self._body_rootid = model.body_rootid

        # Soft joint limits: shrink about the midpoint.
        hinge_ids = [
            i for i in range(model.njnt) if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE
        ]
        self._jnt_qpos_ind = np.concatenate([mj_jntid2qposid(i, model) for i in hinge_ids])
        ranges = np.asarray(model.jnt_range)[hinge_ids]
        mid = ranges.mean(axis=1)
        half = 0.5 * (ranges[:, 1] - ranges[:, 0]) * soft_joint_pos_limit_factor
        self._soft_low = mid - half
        self._soft_high = mid + half

        # Geoms whose mutual contacts count as self-collision. The feet are excluded so
        # that ground contact is not penalised.
        self._self_collision_geom_ids = np.array(
            [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
                for n in props["self_collision_geom_names"]
            ]
        )
        # Boolean lookup over ALL geoms, so a contact is classified with two array reads
        # instead of a pair search. No pair enumeration is needed: MuJoCo has already
        # applied the <contact><exclude> set and body-adjacency filtering, so any contact
        # between two geoms in this set is a genuine self-collision.
        self._is_self_collision_geom = np.zeros(model.ngeom, dtype=bool)
        self._is_self_collision_geom[self._self_collision_geom_ids] = True

        self._assert_contact_buffer_fits(env)

    @staticmethod
    def _zero_components() -> dict:
        return {
            "tracking/root_pos_reward": 0.0,
            "tracking/root_quat_reward": 0.0,
            "tracking/body_pos_reward": 0.0,
            "tracking/body_ori_reward": 0.0,
            "tracking/body_lin_vel_reward": 0.0,
            "tracking/body_ang_vel_reward": 0.0,
            "penalities/action_rate_reward": 0.0,
            "penalities/joint_position_limit_reward": 0.0,
            "penalities/self_collision_reward": 0.0,
        }

    def init_state(
        self,
        env: Any,
        key: Any,
        model: MjModel | Model,
        data: MjData | Data,
        backend: ModuleType,
        traj: Trajectory | None = None,
    ):
        return G1RealTrackingRewardState(
            last_action=backend.zeros(env.info.action_space.shape[0]),
            reward_components=self._zero_components(),
        )

    def reset(
        self,
        env: Any,
        model: MjModel | Model,
        data: MjData | Data,
        carry: Any,
        backend: ModuleType,
        traj: Trajectory | None = None,
    ):
        carry = carry.replace(reward_state=self.init_state(env, None, model, data, backend))
        return data, carry

    _CONTACTS_PER_ENV_BOUND = 32

    def _assert_contact_buffer_fits(self, env) -> None:
        """Fail at construction if MjWarp's shared contact buffer is too small.

        ``naconmax`` bounds contacts across ALL worlds, not per world. Past that ceiling
        MuJoCo silently drops contacts, and a ``worldid`` count then under-reports with
        no error -- the failure mode this whole module exists to avoid. Measured mean is
        ~12.75 contacts/env on the G1, so the bound below is ~2.5x headroom.
        """
        if not getattr(env, "_use_mjwarp", False):
            return
        first = getattr(env, "_first_data", None)
        impl = getattr(first, "_impl", first)
        worldid = getattr(impl, "contact__worldid", None)
        if worldid is None:
            return
        naconmax = int(np.asarray(worldid).shape[0])
        n_envs = int(getattr(env, "n_envs", 1) or 1)
        needed = n_envs * self._CONTACTS_PER_ENV_BOUND
        if needed > naconmax:
            raise ValueError(
                f"MjWarp contact buffer too small for a per-world self-collision count: "
                f"naconmax={naconmax} but {n_envs} envs x {self._CONTACTS_PER_ENV_BOUND} "
                f"contacts/env = {needed}. Past the ceiling MuJoCo drops contacts and the "
                f"count silently under-reports. Raise the env's `naconmax` to at least "
                f"{needed}, or lower num_envs per process."
            )

    def _self_collision_count(self, data, carry, backend: ModuleType):
        """Number of self-colliding geom pairs, from MuJoCo's own contact list.

        Reads the contact buffer rather than recomputing 154 pairwise capsule/sphere
        distances. Under MjWarp that buffer is a single array shared by every world, so
        the count is masked by ``contact__worldid == carry.env_id``; ``env_id`` is filled
        in from the vmap axis index by ``loco_mujoco.core.mujoco_base.current_world_index``.
        Reading the buffer WITHOUT that mask charges every environment with all worlds'
        contacts, which once cost a 3B-step run (episode return -50,568).

        Two further details the backend forces:

        * the ``slot < nacon`` mask is mandatory -- MjWarp does not clear unused slots
          between steps, so stale contacts linger past the active count;
        * plain MJX batches ``data.contact`` per environment, so there is no ``worldid``
          there and no mask is needed.

        This counts contacts rather than contact *forces*: forces are not reachable
        uniformly across backends, so contact presence stands in for them.

        **Reference data can self-collide.** Motion retargeted without self-collision
        awareness often has hands or wrists inside the hips and thighs as the arms swing.
        Check how often your reference frames collide before choosing a weight: on such
        frames the penalty also applies to the reference motion itself.
        """
        impl = getattr(data, "_impl", data)
        is_self = backend.asarray(self._is_self_collision_geom)

        if hasattr(impl, "contact__geom"):
            # MjWarp: one global buffer, (naconmax, 2). Attribute by worldid.
            geom = impl.contact__geom
            dist = impl.contact__dist
            worldid = impl.contact__worldid
            nacon = getattr(impl, "nacon", None)

            # dist is signed: negative means penetrating, so a positive
            # `penetration_threshold` (metres of depth) maps to `dist < -threshold`.
            both = is_self[geom[:, 0]] & is_self[geom[:, 1]]
            active = both & (dist < -self._penetration_threshold)
            active = active & (worldid == carry.env_id)
            if nacon is not None:
                slot = backend.arange(geom.shape[0])
                active = active & (slot < backend.reshape(nacon, ())[()])
            return backend.sum(active.astype(backend.float32))

        # MJX / CPU: data.contact is already this environment's own.
        contact = data.contact
        geom = backend.asarray(contact.geom)
        dist = backend.asarray(contact.dist)
        both = is_self[geom[:, 0]] & is_self[geom[:, 1]]
        active = both & (dist < -self._penetration_threshold)
        ncon = getattr(data, "ncon", None)
        if ncon is not None and backend is np:
            # CPU MjData keeps a fixed-size contact array; only the first ncon are real.
            active = active & (np.arange(geom.shape[0]) < int(ncon))
        return backend.sum(active.astype(backend.float32))

    @check_traj_provided
    def __call__(
        self,
        state: np.ndarray | jnp.ndarray,
        action: np.ndarray | jnp.ndarray,
        next_state: np.ndarray | jnp.ndarray,
        absorbing: bool,
        info: dict[str, Any],
        env: Any,
        model: MjModel | Model,
        data: MjData | Data,
        carry: Any,
        backend: ModuleType,
        traj: Trajectory | None = None,
    ) -> tuple[float, Any]:
        reward_state = carry.reward_state
        traj_state = carry.traj_state
        ref = traj.data.get(traj_state.traj_no, traj_state.subtraj_step_no, backend)

        # ---- robot quantities ----
        robot_pos = data.xpos[self._body_ids]
        robot_mat = data.xmat[self._body_ids].reshape(-1, 3, 3)
        robot_vel = calc_body_velocities(
            self._body_ids, data, self._body_rootid[self._body_ids], backend
        )
        robot_anchor_pos = robot_pos[self._anchor_row]
        robot_anchor_mat = robot_mat[self._anchor_row]

        # ---- reference quantities ----
        ref_pos, ref_mat, ref_vel = reference_body_frames(
            ref, self._body_ids, self._body_rootid, backend
        )
        # clip coordinates -> world: the robot may have been moved to the origin at reset
        ref_pos = ref_pos + backend.concatenate(
            [env.get_traj_offset_xy(carry, backend, traj), backend.zeros(1)]
        )
        ref_anchor_pos = ref_pos[self._anchor_row]
        ref_anchor_mat = ref_mat[self._anchor_row]

        # ---- global anchor terms (these police xy / heading drift) ----
        root_pos_err = backend.sum(backend.square(ref_anchor_pos - robot_anchor_pos))
        root_pos_reward = backend.exp(-root_pos_err / self._std["root_pos"] ** 2)

        root_ori_err = rotation_angle(ref_anchor_mat, robot_anchor_mat, backend) ** 2
        root_ori_reward = backend.exp(-root_ori_err / self._std["root_ori"] ** 2)

        # ---- body terms ----
        # With use_anchor the reference is re-framed into the robot's live xy/heading,
        # so these are invariant to global drift (which the two root terms above police
        # instead). Without it they are raw world-frame errors.
        if self._use_anchor:
            delta_pos, delta_rot = anchor_delta(
                robot_anchor_pos, robot_anchor_mat, ref_anchor_pos, ref_anchor_mat, backend
            )
            ref_pos_rel, ref_mat_rel = reanchor(
                ref_pos, ref_mat, ref_anchor_pos, delta_pos, delta_rot, backend
            )
        else:
            ref_pos_rel, ref_mat_rel = ref_pos, ref_mat

        body_pos_err = backend.mean(
            backend.sum(backend.square(ref_pos_rel - robot_pos), axis=-1)
        )
        body_pos_reward = backend.exp(-body_pos_err / self._std["body_pos"] ** 2)

        body_ori_err = backend.mean(rotation_angle(ref_mat_rel, robot_mat, backend) ** 2)
        body_ori_reward = backend.exp(-body_ori_err / self._std["body_ori"] ** 2)

        # ---- velocity terms, compared in the world frame ----
        # calc_body_velocities / body_spatial_velocities both return MuJoCo's
        # mj_objectVelocity layout: angular in [:3], linear in [3:].
        ang_err = backend.mean(
            backend.sum(backend.square(ref_vel[:, :3] - robot_vel[:, :3]), axis=-1)
        )
        lin_err = backend.mean(
            backend.sum(backend.square(ref_vel[:, 3:] - robot_vel[:, 3:]), axis=-1)
        )
        body_ang_vel_reward = backend.exp(-ang_err / self._std["body_ang_vel"] ** 2)
        body_lin_vel_reward = backend.exp(-lin_err / self._std["body_lin_vel"] ** 2)

        # ---- penalties ----
        action_rate = backend.sum(backend.square(action - reward_state.last_action))

        qpos = data.qpos[self._jnt_qpos_ind]
        out_of_limits = -backend.minimum(qpos - backend.array(self._soft_low), 0.0)
        out_of_limits = out_of_limits + backend.maximum(
            qpos - backend.array(self._soft_high), 0.0
        )
        # A joint is only far outside its range when the simulation of that world has
        # diverged, and then the excess is unbounded. Capping it per joint keeps the
        # reward finite there.
        joint_limit = backend.sum(backend.minimum(out_of_limits, MAX_JOINT_LIMIT_EXCESS))

        # Skip the contact scan entirely when the term is disabled: it is the only part
        # of this reward that touches backend-specific contact layouts, so the weight-0
        # configuration stays independent of that plumbing.
        if self._w["self_collision"] == 0.0:
            self_collision = 0.0
        else:
            self_collision = self._self_collision_count(data, carry, backend)

        w = self._w
        components = {
            "tracking/root_pos_reward": w["root_pos"] * root_pos_reward,
            "tracking/root_quat_reward": w["root_ori"] * root_ori_reward,
            "tracking/body_pos_reward": w["body_pos"] * body_pos_reward,
            "tracking/body_ori_reward": w["body_ori"] * body_ori_reward,
            "tracking/body_lin_vel_reward": w["body_lin_vel"] * body_lin_vel_reward,
            "tracking/body_ang_vel_reward": w["body_ang_vel"] * body_ang_vel_reward,
            "penalities/action_rate_reward": w["action_rate"] * action_rate,
            "penalities/joint_position_limit_reward": w["joint_limit"] * joint_limit,
            "penalities/self_collision_reward": w["self_collision"] * self_collision,
        }
        if self._reward_scale != 1.0:
            components = {k: v * self._reward_scale for k, v in components.items()}
        total_reward = sum(components.values())
        total_reward = backend.nan_to_num(total_reward, nan=0.0, posinf=0.0, neginf=0.0)

        carry = carry.replace(
            reward_state=reward_state.replace(
                last_action=action, reward_components=components
            )
        )
        return total_reward, carry
