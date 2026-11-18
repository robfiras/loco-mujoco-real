"""Joint-position control for the G1.

The G1 is driven by MuJoCo's *builtin* position actuators, so the PD loop lives inside
the physics engine and runs at every simulation step. The control function only has to
turn a policy output into a target angle::

    target = default_joint_pos + action * scale

with ``scale[j] = 0.25 * effort_limit[j] / stiffness[j]``, optionally multiplied by a
global ``action_scale``. This module is that mapping, and nothing more -- there is
deliberately no Python-side PD arithmetic here. The per-joint scale is why the same
action unit means the same fraction of achievable deflection everywhere:

===========================================  ==========
joint group                                  rad / unit
===========================================  ==========
hip pitch/yaw, waist yaw (7520-14)               0.5475
hip roll, knee (7520-22)                         0.3507
waist roll/pitch, ankles (5020 x2)               0.4386
shoulders, elbow, wrist roll (5020)              0.4386
wrist pitch/yaw (4010)                           0.0745
===========================================  ==========
"""

from __future__ import annotations

from types import ModuleType
from typing import Any

import jax
import numpy as np
from flax import struct
from loco_mujoco.core.control_functions import ControlFunction
from loco_mujoco.core.trajectory import Trajectory
from loco_mujoco.core.utils import mj_jntname2qposid
from loco_mujoco.core.utils.backend import assert_backend_is_supported
from mujoco import MjData, MjModel
from mujoco.mjx import Data, Model

from lmj_real.core.utils.keyframe import default_qpos


@struct.dataclass
class G1RealPositionControlState:
    """State a domain randomizer can perturb.

    ``pos_offset`` shifts the commanded target (a *commanded* bias, distinct from
    the randomizer's ``encoder_bias``, which corrupts the joint-position *observation* ). Gain noise is absent on purpose: with builtin position
    actuators the gains live in ``model.actuator_gainprm`` / ``actuator_biasprm``, so a
    randomizer perturbs the model rather than this state.
    """

    pos_offset: np.ndarray | jax.Array


class G1RealPositionControl(ControlFunction):
    """Map a policy action to a joint-position target for MuJoCo position actuators."""

    def __init__(
        self,
        env: Any,
        action_scale: float = 1.0,
        scale: float | np.ndarray | None = None,
        clip_actions: float | None = None,
        clip_target_to_jnt_range: bool = False,
        nominal_joint_positions: np.ndarray | None = None,
        **kwargs: Any,
    ):
        """
        Args:
            env: Environment instance.
            action_scale: Global multiplier on the per-joint scale.
            scale: Overrides the per-joint scale derived from the model. Leave as None
                to use ``0.25 * effort_limit / stiffness``, which stays correct if the
                MJCF changes.
            clip_actions: Clip raw actions to +/- this before scaling. None (default)
                leaves actions unbounded.
            clip_target_to_jnt_range: Clamp the target to the joint range. False
                (default) relies on physical limits plus the joint-limit reward
                penalty instead of clamping.
            nominal_joint_positions: Zero point of the action. Defaults to the
                model's ``init_state`` keyframe (the standing pose), falling back
                to ``qpos0`` if the model has no such keyframe.
        """
        self._action_scale = float(action_scale)
        self._clip_actions = clip_actions
        self._clip_target_to_jnt_range = clip_target_to_jnt_range

        model = env._model
        action_indices = np.asarray(env._action_indices)

        # Joint bookkeeping, ordered to match the action vector.
        jnt_names, jnt_ranges = [], []
        for actuator in env.mjspec.actuators:
            jnt_name = actuator.target
            jnt_names.append(jnt_name)
            jnt_ranges.append(
                next(j.range for j in env.mjspec.joints if j.name == jnt_name)
            )
        self._jnt_names = [jnt_names[i] for i in action_indices]
        self._jnt_ranges = np.concatenate(
            [np.asarray(jnt_ranges[i]).reshape(1, 2) for i in action_indices]
        )
        self._qpos_ids = np.concatenate(
            [mj_jntname2qposid(name, model) for name in self._jnt_names]
        )

        # Per-joint action scale, read from the compiled model so it tracks the MJCF.
        # gainprm[0] is the position-servo stiffness; forcerange[1] the effort limit.
        stiffness = np.asarray(model.actuator_gainprm[action_indices, 0], dtype=float)
        effort = np.asarray(model.actuator_forcerange[action_indices, 1], dtype=float)
        if np.any(stiffness <= 0.0):
            raise ValueError(
                "G1RealPositionControl needs position actuators with positive stiffness; "
                "got actuator_gainprm[:, 0] <= 0. Is this MJCF using torque motors? "
                "This control function needs position servos."
            )
        if np.any(effort <= 0.0):
            raise ValueError(
                "actuator_forcerange upper bound must be positive to derive the action "
                "scale; got a non-positive effort limit."
            )
        derived = 0.25 * effort / stiffness
        self._scale = (derived if scale is None else np.broadcast_to(
            np.asarray(scale, dtype=float), derived.shape
        ).copy()) * self._action_scale

        # Read from the compiled model, not the spec: `MjsActuator.ctrllimited` is an enum
        # that newer MuJoCo versions make truthy for every value, which would clip the
        # target to the unset ctrlrange [0, 0] of an actuator without one.
        ctrl_limited = np.asarray(model.actuator_ctrllimited[action_indices], dtype=bool)
        self._ctrl_ranges = np.where(
            ctrl_limited[:, None],
            np.asarray(model.actuator_ctrlrange[action_indices], dtype=float),
            np.array([-np.inf, np.inf]),
        )

        if nominal_joint_positions is None:
            # The default pose is the `init_state` KEYFRAME, not qpos0. Joint `ref`
            # (which is what sets qpos0) shifts forward kinematics, so it must not be
            # used to carry a default pose -- see lmj_real.core.utils.keyframe.
            self._nominal_joint_positions = np.asarray(
                default_qpos(model)[self._qpos_ids], dtype=float
            )
        else:
            self._nominal_joint_positions = np.asarray(nominal_joint_positions, dtype=float)

        # Action space is normalized; one unit means `self._scale` radians.
        low = -np.ones_like(self._nominal_joint_positions)
        high = np.ones_like(self._nominal_joint_positions)

        super().__init__(env, low, high, **kwargs)

    @property
    def scale(self) -> np.ndarray:
        """Per-joint radians per unit action (``0.25 * effort / stiffness * action_scale``)."""
        return self._scale

    def init_state(
        self,
        env: Any,
        key: Any,
        model: MjModel | Model,
        data: MjData | Data,
        backend: ModuleType,
        traj: Trajectory | None = None,
    ) -> G1RealPositionControlState:
        assert_backend_is_supported(backend)
        return G1RealPositionControlState(
            pos_offset=backend.zeros_like(backend.array(self._nominal_joint_positions))
        )

    def generate_action(
        self,
        env: Any,
        action: np.ndarray | jax.Array,
        model: MjModel | Model,
        data: MjData | Data,
        carry: Any,
        backend: ModuleType,
        traj: Trajectory | None = None,
    ) -> tuple[np.ndarray | jax.Array, Any]:
        """Return joint-position targets for ``data.ctrl``.

        The physics engine closes the loop, so no gains appear here.
        """
        assert_backend_is_supported(backend)

        if self._clip_actions is not None:
            action = backend.clip(action, -self._clip_actions, self._clip_actions)

        offset = carry.control_func_state.pos_offset
        target = self._nominal_joint_positions + action * self._scale + offset

        if self._clip_target_to_jnt_range:
            target = backend.clip(target, self._jnt_ranges[:, 0], self._jnt_ranges[:, 1])

        # Position actuators take the target angle directly as ctrl.
        target = backend.clip(target, self._ctrl_ranges[:, 0], self._ctrl_ranges[:, 1])
        return target, carry

    @property
    def run_with_simulation_frequency(self) -> bool:
        """False: the servo is in the physics, so the target is written once per control step.

        The policy output is held for ``n_substeps`` physics steps while MuJoCo runs the
        PD loop internally.
        """
        return False
