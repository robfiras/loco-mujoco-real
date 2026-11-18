"""Observation types for the G1Real env that ``loco_mujoco`` does not already provide.

======================================  ==========================================
quantity                                type
======================================  ==========================================
joint positions, relative to default    :class:`G1RealJointPosRel`
all-body positions                      :class:`G1RealBodyPosRel`
all-body orientations                   :class:`G1RealBodyOriRel`
all-body linear velocities              :class:`G1RealBodyLinVelRel`
all-body angular velocities             :class:`G1RealBodyAngVelRel`
======================================  ==========================================

These are all ``StatefulObservation`` subclasses. That is deliberate:
``ObservationIndexContainer`` builds its per-type slots from
``ObservationType.list_all_non_stateful()``, so a *non*-stateful extension type
defined outside ``loco_mujoco`` would need ``ObservationType`` monkeypatched.
Stateful observations are resolved per instance through
``get_obs_and_update_state`` and just work.

The remaining terms need no new code:

* joint velocity      -> ``JointVelArray`` (default joint velocity is 0, so relative and
                         absolute coincide)
* projected gravity   -> ``ProjectedGravityVector``
* the IMU gyro        -> ``EntryFromFreeJointVel`` indices 3/4/5. The ``imu_in_pelvis``
  site has an identity quaternion on ``pelvis``, and MuJoCo expresses free-joint angular
  velocity in the local body frame, so the gyro reading and ``qvel[3:6]`` are the same
  vector.
* root height         -> ``EntryFromFreeJointPos`` index 2
* last action         -> ``LastAction``

Observation delay and history are implemented, but neither lives here.

* DELAY: ``G1RealRandomizer.obs_delay_steps``. ``update_observation`` is the only hook
  that sees the assembled observation and owns per-env state across steps, so the ring
  buffer lives there. It delays the sensed terms only -- joint pos/vel, gyro, projected
  gravity -- and leaves the goal (commanded, not measured), ``last_action`` (the
  controller knows what it sent) and the privileged terms alone. Default 0.
* HISTORY: ``NStepWrapper(group=...)`` via ``experiment.len_obs_history`` and
  ``experiment.obs_history_group``. It stacks past frames of the ``history`` group only
  and appends them after the current observation, so the actor grows and the critic does
  not.

The two are separate because history changes the observation *dimension*, which has to
be known when the env is constructed, while ``update_observation`` must return the shape
it was handed. Measured here, a one-step (20 ms) lag costs ~2% and a 4-step history did
not recover it, so both default to off.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any, List

import mujoco
import numpy as np
from loco_mujoco.core.observations import StatefulObservation
from loco_mujoco.core.trajectory import Trajectory
from loco_mujoco.core.utils import mj_jntname2qposid
from loco_mujoco.core.utils.math import calc_body_velocities
from mujoco import MjData, MjModel
from mujoco.mjx import Data, Model

from lmj_real.core.utils.keyframe import default_qpos
from lmj_real.core.utils.tracking import heading_rot_inv, quat_to_tan_norm_from_mat


class _AllBodyObs(StatefulObservation):
    """Shared setup for the all-body observations.

    Every one of them re-expresses a per-body world quantity in the root body's
    heading-corrected frame, so they only differ in what they read and how many
    numbers each body contributes.
    """

    #: numbers contributed per body
    per_body: int = 3
    #: whether the root body's own entry is dropped (it is identically zero for
    #: positions, but meaningful for orientations and velocities)
    drop_root: bool = False

    def __init__(self, obs_name: str, body_names: List[str] | None = None, **kwargs):
        """
        Args:
            obs_name: Observation name.
            body_names: Bodies to include, in order. Defaults to every body in the
                model except ``world``.
        """
        self._body_names = body_names
        self.dim = None
        self._body_ids = None
        self._root_body_id = None
        super().__init__(obs_name, **kwargs)

    def _init_from_mj(self, env, model, data, current_obs_size):
        if self._body_names is None:
            self._body_names = [
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
                for i in range(model.nbody)
                if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) != "world"
            ]
        self._body_ids = np.array(
            [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in self._body_names]
        )
        if np.any(self._body_ids < 0):
            missing = [n for n, i in zip(self._body_names, self._body_ids) if i < 0]
            raise ValueError(f"unknown bodies in {type(self).__name__}: {missing}")

        root_name = env._get_all_info_properties()["root_body_name"]
        self._root_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_name)
        self._body_rootid = model.body_rootid

        n = len(self._body_ids) - (1 if self.drop_root else 0)
        self.dim = n * self.per_body
        self.min, self.max = [-np.inf] * self.dim, [np.inf] * self.dim
        self.obs_ind = np.array(list(range(current_obs_size, current_obs_size + self.dim)))
        self._initialized_from_mj = True

    def _root_frame(self, data, backend: ModuleType):
        root_xmat = data.xmat[self._root_body_id].reshape(3, 3)
        return heading_rot_inv(root_xmat, backend)


class G1RealJointPosRel(StatefulObservation):
    """Joint positions relative to the model's default pose.

    The zero point is the model's ``init_state`` keyframe (the standing pose), the same reference the position controller offsets from, so a zero action
    corresponds to a zero observation. Deliberately *not* ``qpos0``: that is set by
    joint ``ref``, which shifts forward kinematics -- see
    :mod:`lmj_real.core.utils.keyframe`.

    The persistent per-episode encoder bias belongs to the domain randomizer, not to
    this term.
    """

    def __init__(self, obs_name: str, xml_names: List[str], **kwargs):
        self._xml_names = xml_names
        self.dim = None
        self._qpos_ind = None
        self._qpos0 = None
        super().__init__(obs_name, **kwargs)

    def _init_from_mj(self, env, model, data, current_obs_size):
        self._qpos_ind = np.concatenate(
            [mj_jntname2qposid(name, model) for name in self._xml_names]
        )
        self._qpos0 = default_qpos(model)[self._qpos_ind]
        self.dim = len(self._qpos_ind)
        self.min, self.max = [-np.inf] * self.dim, [np.inf] * self.dim
        self.obs_ind = np.array(list(range(current_obs_size, current_obs_size + self.dim)))
        self._initialized_from_mj = True

    def get_obs_and_update_state(
        self,
        env: Any,
        model: MjModel | Model,
        data: MjData | Data,
        carry: Any,
        backend: ModuleType,
        traj: Trajectory | None = None,
    ):
        return data.qpos[self._qpos_ind] - backend.array(self._qpos0), carry


class G1RealBodyPosRel(_AllBodyObs):
    """Body positions in the root's heading frame.

    The root's own entry is dropped, since it is identically zero.
    """

    per_body = 3
    drop_root = True

    def get_obs_and_update_state(self, env, model, data, carry, backend, traj=None):
        rot_inv = self._root_frame(data, backend)
        rel = data.xpos[self._body_ids] - data.xpos[self._root_body_id]
        local = rel @ rot_inv.T
        keep = self._body_ids != self._root_body_id
        return backend.ravel(local[keep]), carry


class G1RealBodyOriRel(_AllBodyObs):
    """Body orientations in the root's heading frame, 6D tan/norm encoded.

    The root entry is kept.
    """

    per_body = 6
    drop_root = False

    def get_obs_and_update_state(self, env, model, data, carry, backend, traj=None):
        rot_inv = self._root_frame(data, backend)
        xmats = data.xmat[self._body_ids].reshape(-1, 3, 3)
        local = backend.einsum("ij,bjk->bik", rot_inv, xmats)
        return backend.ravel(quat_to_tan_norm_from_mat(local, backend)), carry


class _AllBodyVelObs(_AllBodyObs):
    """Body velocities in the root's heading frame.

    ``calc_body_velocities`` returns MuJoCo's ``mj_objectVelocity`` layout at the body
    frame origin: rotational in ``[:3]``, linear in ``[3:]``.
    """

    per_body = 3
    drop_root = False
    #: slice of the 6D spatial velocity this observation exposes
    _vel_slice = slice(0, 3)

    def get_obs_and_update_state(self, env, model, data, carry, backend, traj=None):
        rot_inv = self._root_frame(data, backend)
        vel = calc_body_velocities(
            self._body_ids, data, self._body_rootid[self._body_ids], backend
        )
        local = vel[:, self._vel_slice] @ rot_inv.T
        return backend.ravel(local), carry


class G1RealBodyAngVelRel(_AllBodyVelObs):
    """Body angular velocities in the root's heading frame."""

    _vel_slice = slice(0, 3)


class G1RealBodyLinVelRel(_AllBodyVelObs):
    """Body linear velocities in the root's heading frame."""

    _vel_slice = slice(3, 6)
