"""mjlab's G1 velocity task, rebuilt from LocoMuJoCo components, for a like-for-like benchmark.

Re-implements ``Mjlab-Velocity-Flat-Unitree-G1`` of mjlab 1.6.1, without its domain randomization events (pushes,
foot friction, encoder bias, COM offset), on mjlab's own model (export_models.py, mjlab_model.py), for all three
LocoMuJoCo backends (CPU, MJX, MjWarp):

* control function ``MLControl``: mjlab's joint-position action (target = default pose + scale * action, in joint
  order, unclipped), applied at every physics substep, where it also does the per-substep bookkeeping of mjlab's
  contact sensors (feet air and contact time, the self-collision force history over the 4 substeps);
* observations ``MLActor`` (mjlab's ``actor`` group, 99 entries, with its uniform noise) and ``MLCritic`` (the
  ``critic`` group, 111 entries, noiseless);
* goal ``MLTwist``: mjlab's UniformVelocityCommand (resampled every 3-8 s; heading-controlled, standing and forward
  environments) with the command_vel curriculum;
* reward ``MLVelocityReward``: the 13 terms of nonzero weight, ``sum(weight * term) * step_dt``, NaN terms set to 0;
* terminal state handler ``MLTermination``: fell over (pelvis tilted beyond 70 degrees); the time limit is the horizon;
* initial state handler ``MLInitState``: mjlab's reset events (base shifted, raised and turned; joints at default).

Timing as in mjlab's step: the contact bookkeeping reads the sensors after every substep; the reward and the
termination read the state the last substep left, whose derived quantities (kinematics, sensors) lag the integration
by one substep; the observations and the command update read a refreshed state (mjlab calls ``forward`` before them),
and the foot heights the reward uses are those of the previous observation (mjlab's ray casts run with the
observations). LocoMuJoCo computes the observation before the reward, so the observation does the last substep's
bookkeeping, the command update and its own ``forward`` on a copy of the data, and the reward reads what mjlab's does.

Differences in how, not in what a step computes:

* LocoMuJoCo computes the observation of an automatically reset environment separately (mjlab runs one ``forward``
  after the resets, for all environments). On MJX and MjWarp it computes it from kinematics only, which gives the same
  values because a reset state has no velocity and no contacts (checked when the task is built), and because MuJoCo
  Warp cannot run inside LocoMuJoCo's automatic reset (a ``lax.cond`` under ``vmap``).
* MJX's JAX implementation has no subtree/body contact sensors, so on MJX they are computed from the contact list
  (mjlab_contacts.py).
* The foot heights come from the foot sites' height above the plane, which is what mjlab's 7 downward rays per foot
  return on its flat terrain (clamped to [0, 1]).
* mjlab caches its contact sensor readings per step, so its critic reads the contact state of the refreshed state when
  some environment was reset in the step and that of the last substep otherwise; LocoMuJoCo reads the refreshed state.
* The world-frame command environments of mjlab's command (probability 0) are left out.
"""
from __future__ import annotations

import math
import re
from typing import Any

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from flax import struct
from loco_mujoco.core import Observation
from loco_mujoco.core.control_functions import ControlFunction
from loco_mujoco.core.initial_state_handler import InitialStateHandler
from loco_mujoco.core.observations.base import StatefulObservation
from loco_mujoco.core.observations.goals import Goal
from loco_mujoco.core.reward import Reward
from loco_mujoco.core.stateful_object import StatefulObject
from loco_mujoco.core.terminal_state_handler import TerminalStateHandler
from lmj_real.core.control_functions.g1_real_position import G1RealPositionControl

from mjlab_contacts import ContactReadings, sensor_slice, strip_contact_sensors
from mjlab_model import load_spec

PHYSICS_DT, DECIMATION = 0.005, 4
STEP_DT = PHYSICS_DT * DECIMATION
NOISE = dict(lin_vel=0.5, ang_vel=0.2, gravity=0.05, joint_pos=0.01, joint_vel=1.5)
# reward terms of nonzero weight, in mjlab's order (air_time has weight 0 and is skipped, as mjlab skips it)
REWARD_WEIGHTS = dict(track_linear_velocity=2.0, track_angular_velocity=2.0, upright=1.0, pose=1.0,
                      body_ang_vel=-0.05, angular_momentum=-0.02, dof_pos_limits=-1.0, action_rate_l2=-0.1,
                      foot_clearance=-2.0, foot_swing_height=-0.25, foot_slip=-0.1, soft_landing=-1e-5,
                      self_collisions=-1.0)
REWARD_TERMS = tuple(REWARD_WEIGHTS)
LIN_STD, ANG_STD, UPRIGHT_STD = math.sqrt(0.25), math.sqrt(0.5), math.sqrt(0.2)
SOFT_LIMIT_FACTOR = 0.9
FOOT_TARGET_HEIGHT, COMMAND_THRESHOLD, SELF_COLLISION_FORCE = 0.1, 0.05, 10.0
WALKING_THRESHOLD, RUNNING_THRESHOLD = 0.05, 1.5
POSE_STD_STANDING = {".*": 0.05}
POSE_STD_WALKING = {r".*hip_pitch.*": 0.3, r".*hip_roll.*": 0.15, r".*hip_yaw.*": 0.15, r".*knee.*": 0.35,
                    r".*ankle_pitch.*": 0.25, r".*ankle_roll.*": 0.1, r".*waist_yaw.*": 0.2, r".*waist_roll.*": 0.08,
                    r".*waist_pitch.*": 0.1, r".*shoulder_pitch.*": 0.15, r".*shoulder_roll.*": 0.15,
                    r".*shoulder_yaw.*": 0.1, r".*elbow.*": 0.15, r".*wrist.*": 0.3}
POSE_STD_RUNNING = {r".*hip_pitch.*": 0.5, r".*hip_roll.*": 0.2, r".*hip_yaw.*": 0.2, r".*knee.*": 0.6,
                    r".*ankle_pitch.*": 0.35, r".*ankle_roll.*": 0.15, r".*waist_yaw.*": 0.3, r".*waist_roll.*": 0.08,
                    r".*waist_pitch.*": 0.2, r".*shoulder_pitch.*": 0.5, r".*shoulder_roll.*": 0.2,
                    r".*shoulder_yaw.*": 0.15, r".*elbow.*": 0.35, r".*wrist.*": 0.3}
# UniformVelocityCommand of the G1 flat task
RESAMPLE_TIME = (3.0, 8.0)
REL_STANDING, REL_HEADING, REL_FORWARD = 0.1, 0.3, 0.2
HEADING_STIFFNESS = 0.5
LIN_VEL_Y, HEADING = (-1.0, 1.0), (-math.pi, math.pi)
# command_vel curriculum: (step, lin_vel_x, ang_vel_z)
CURRICULUM = ((0, (-1.0, 1.0), (-0.5, 0.5)), (5000 * 24, (-1.5, 2.0), (-0.7, 0.7)), (10000 * 24, (-2.0, 3.0), (-0.7, 0.7)))
RESET_XY, RESET_Z, RESET_YAW = (-0.5, 0.5), (0.01, 0.05), (-3.14, 3.14)
FELL_OVER_ANGLE = math.radians(70.0)


# ------------------------------------------------------------------------------------------------- model and helpers
class MLModel:
    """Indices and constants of mjlab's G1 scene."""

    def __init__(self, model: mujoco.MjModel):
        self.root = model.body("robot/pelvis").id
        self.torso = model.body("robot/torso_link").id
        self.feet_sites = np.array([model.site("robot/left_foot").id, model.site("robot/right_foot").id])
        self.feet_site_bodies = model.site_bodyid[self.feet_sites]
        self.imu_lin_vel = sensor_slice(model, "robot/imu_lin_vel")
        self.imu_ang_vel = sensor_slice(model, "robot/imu_ang_vel")
        self.root_angmom = sensor_slice(model, "robot/root_angmom")
        self.joint_names = [model.joint(i).name.removeprefix("robot/") for i in range(1, model.njnt)]
        self.nu = len(self.joint_names)
        key = np.array(model.key("init_state").qpos)
        self.default_root, self.default_pose = key[:7].copy(), key[7:].copy()
        # mjlab computes the soft limits in float32
        lo, hi = model.jnt_range[1:].astype(np.float32).T
        mean, rng = (lo + hi) / 2, hi - lo
        self.soft_lo = (mean - 0.5 * rng * np.float32(SOFT_LIMIT_FACTOR)).astype(np.float64)
        self.soft_hi = (mean + 0.5 * rng * np.float32(SOFT_LIMIT_FACTOR)).astype(np.float64)
        self.pose_std = np.stack([self._per_joint(d) for d in (POSE_STD_STANDING, POSE_STD_WALKING, POSE_STD_RUNNING)])

    def _per_joint(self, patterns):
        out = []
        for name in self.joint_names:
            out.append(next(v for k, v in patterns.items() if re.fullmatch(k, name)))
        return np.array(out, dtype=np.float32).astype(np.float64)


class MLShared:
    """What the components share: the model's indices, the contact readings and the refresh for the backend."""

    def __init__(self, env, backend: str):
        self.m = MLModel(env._model)
        self.contacts = ContactReadings(env._model, native=backend != "mjx")
        self.backend = backend
        self._scratch = mujoco.MjData(env._model) if backend == "cpu" else None
        self.imu_site = env._model.site("robot/imu_in_pelvis").id
        if backend != "cpu":
            # the observation of a reset state is computed inside LocoMuJoCo's automatic reset (a lax.cond under
            # vmap), where MuJoCo Warp cannot run; it needs only kinematics, from the collision-free JAX copy
            self.kin_sys, self.kin_data = env._make_kinematics_copy()
            # ... and has no contacts: the reset poses (default joints, upright, base raised by RESET_Z) touch nothing
            d = mujoco.MjData(env._model)
            d.qpos[:] = np.concatenate([self.m.default_root, self.m.default_pose])
            d.qpos[2] += RESET_Z[0]
            mujoco.mj_forward(env._model, d)
            assert d.ncon == 0, "a reset pose has contacts; its observation needs a forward pass"

    def forward(self, env, model, data, bk):
        """The data refreshed by ``forward``, as mjlab's sim.forward(), leaving ``data`` unchanged."""
        if bk is np:
            s = self._scratch
            s.qpos[:], s.qvel[:], s.ctrl[:], s.qacc_warmstart[:], s.time = (
                data.qpos, data.qvel, data.ctrl, data.qacc_warmstart, data.time)
            mujoco.mj_forward(model, s)
            return s
        from mujoco import mjx
        return mjx.forward(env.sys, data)

    def reset_kinematics(self, data):
        """Kinematics (xquat, site poses, cvel, subtree_com) of a reset state, and its IMU readings."""
        from mujoco import mjx
        kin = self.kin_data.replace(qpos=data.qpos, qvel=data.qvel)
        kin = mjx.com_vel(self.kin_sys, mjx.com_pos(self.kin_sys, mjx.kinematics(self.kin_sys, kin)))
        R, root = kin.site_xmat[self.imu_site], self.m.root
        omega = kin.cvel[root, 0:3]
        # MuJoCo's gyro and velocimeter at the pelvis IMU site
        gyro = R.T @ omega
        velocimeter = R.T @ (kin.cvel[root, 3:6] - jnp.cross(kin.site_xpos[self.imu_site] - kin.subtree_com[root], omega))
        return kin, velocimeter, gyro


def _shared(env) -> MLShared:
    return env._mjlab_task


def _where(bk, cond, a, b):
    """Elementwise select that also takes pytrees (NumPy: a Python branch on a scalar condition)."""
    if bk is np:
        return a if cond else b
    return jax.tree.map(lambda x, y: jnp.where(cond, x, y), a, b)


def quat_apply_inverse(bk, q, v):
    """mjlab's quat_apply_inverse for a (w, x, y, z) quaternion."""
    xyz = q[1:]
    t = 2.0 * bk.cross(xyz, v)
    return v - q[0] * t + bk.cross(xyz, t)


def quat_apply(bk, q, v):
    xyz = q[1:]
    t = 2.0 * bk.cross(xyz, v)
    return v + q[0] * t + bk.cross(xyz, t)


def root_link_vel_b(bk, m: MLModel, data):
    """mjlab's root_link_lin_vel_b and root_link_ang_vel_b, from cvel, subtree_com, xpos and xquat."""
    cvel = data.cvel[m.root]
    q = data.xquat[m.root]
    lin_w = cvel[3:6] - bk.cross(cvel[0:3], data.subtree_com[m.root] - data.xpos[m.root])
    return quat_apply_inverse(bk, q, lin_w), quat_apply_inverse(bk, q, cvel[0:3])


def feet_lin_vel_w(bk, m: MLModel, data):
    """mjlab's site_lin_vel_w of the two foot sites."""
    cvel = data.cvel[m.feet_site_bodies]
    lever = data.subtree_com[m.root][None] - data.site_xpos[m.feet_sites]
    return cvel[:, 3:6] - bk.cross(cvel[:, 0:3], lever)


def _uniform(carry, bk, shape=(), low=0.0, high=1.0):
    """U(low, high) from the carry's key (JAX backends) or NumPy's generator (CPU), and the advanced carry."""
    if bk is np:
        return np.random.uniform(low, high, shape), carry
    key, sub = jax.random.split(carry.key)
    return jax.random.uniform(sub, shape, minval=low, maxval=high), carry.replace(key=key)


def _backend_of(data):
    return np if isinstance(data, mujoco.MjData) else jnp


def _set_obs_state(carry, name, state):
    return carry.replace(observation_states=carry.observation_states.replace(**{name: state}))


# --------------------------------------------------------------------------------------- control and contact tracking
@struct.dataclass
class MLControlState:
    pos_offset: Any
    substep: Any            # calls of generate_action in this step: the first one precedes the step's substeps
    current_air_time: Any   # mjlab's ContactSensor air-time state of the two feet
    last_air_time: Any
    current_contact_time: Any
    last_contact_time: Any
    force_history: Any      # the self-collision force of the last 4 substeps, newest first


def track_contacts(bk, st: MLControlState, feet_found, self_force) -> MLControlState:
    """mjlab's ContactSensor update after a physics substep: air/contact times and the force history."""
    is_contact = feet_found > 0
    first_contact = (st.current_air_time > 0) & is_contact
    first_detached = (st.current_contact_time > 0) & ~is_contact
    return st.replace(
        last_air_time=bk.where(first_contact, st.current_air_time + PHYSICS_DT, st.last_air_time),
        current_air_time=bk.where(~is_contact, st.current_air_time + PHYSICS_DT, 0.0),
        last_contact_time=bk.where(first_detached, st.current_contact_time + PHYSICS_DT, st.last_contact_time),
        current_contact_time=bk.where(is_contact, st.current_contact_time + PHYSICS_DT, 0.0),
        force_history=bk.concatenate([self_force[None], st.force_history[:-1]]))


def first_contact(bk, st: MLControlState):
    """mjlab's compute_first_contact(dt=step_dt): contact that began within the last policy step."""
    return (st.current_contact_time > 0.0) & (st.current_contact_time < STEP_DT + 1.0e-6)


class MLControl(G1RealPositionControl):
    """mjlab's JointPositionAction (default pose + scale * action), written at every physics substep."""

    def init_state(self, env, key, model, data, backend, traj=None):
        z2 = backend.zeros(2)
        return MLControlState(pos_offset=backend.zeros(len(self._nominal_joint_positions)), substep=backend.array(0),
                              current_air_time=z2, last_air_time=z2, current_contact_time=z2, last_contact_time=z2,
                              force_history=backend.zeros((DECIMATION, 3)))

    def reset(self, env, model, data, carry, backend, traj=None):
        # mjlab's contact sensor reset: air-time state and force history to zero
        return data, carry.replace(control_func_state=self.init_state(env, None, model, data, backend))

    def generate_action(self, env, action, model, data, carry, backend, traj=None):
        target, carry = super().generate_action(env, action, model, data, carry, backend, traj)
        st = carry.control_func_state
        # the data before substep k holds the sensors of substep k-1: mjlab's reading after that substep
        feet_found, _, _, self_force = _shared(env).contacts(model, data, backend)
        st = _where(backend, st.substep > 0, track_contacts(backend, st, feet_found, self_force), st)
        st = st.replace(substep=(st.substep + 1) % DECIMATION)
        return target, carry.replace(control_func_state=st)

    @property
    def run_with_simulation_frequency(self) -> bool:
        return True


# ---------------------------------------------------------------------------------------------------- command
@struct.dataclass
class MLTwistState:
    command: Any          # vel_command_b: the command the policy observes
    prev_command: Any     # the command of the previous observation, which mjlab's reward of this step reads
    time_left: Any
    heading_target: Any
    is_heading: Any
    is_standing: Any


def _ranges(bk, global_step):
    """lin_vel_x and ang_vel_z ranges of the command_vel curriculum."""
    lx, az = np.array(CURRICULUM[0][1]), np.array(CURRICULUM[0][2])
    for step, lin_x, ang_z in CURRICULUM[1:]:
        reached = global_step >= step
        lx = _where(bk, reached, np.array(lin_x), lx) if bk is np else jnp.where(reached, np.array(lin_x), lx)
        az = _where(bk, reached, np.array(ang_z), az) if bk is np else jnp.where(reached, np.array(ang_z), az)
    return lx, az


def resample_twist(bk, st: MLTwistState, carry):
    """UniformVelocityCommand._resample: time left, velocities, heading target and environment kind."""
    lx, az = _ranges(bk, carry.global_step)
    u, carry = _uniform(carry, bk, (9,))
    time_left = RESAMPLE_TIME[0] + u[0] * (RESAMPLE_TIME[1] - RESAMPLE_TIME[0])
    vx, vy = lx[0] + u[1] * (lx[1] - lx[0]), LIN_VEL_Y[0] + u[2] * (LIN_VEL_Y[1] - LIN_VEL_Y[0])
    wz = az[0] + u[3] * (az[1] - az[0])
    heading = HEADING[0] + u[4] * (HEADING[1] - HEADING[0])
    is_heading, is_standing, is_forward = u[5] <= REL_HEADING, u[6] <= REL_STANDING, u[7] <= REL_FORWARD
    vx = bk.where(is_forward, bk.maximum(bk.abs(vx), 0.3), vx)
    vy, wz = bk.where(is_forward, 0.0, vy), bk.where(is_forward, 0.0, wz)
    command = bk.stack([vx, vy, wz])
    return st.replace(command=command, time_left=time_left, heading_target=heading, is_heading=is_heading,
                      is_standing=is_standing), carry


def wrap_to_pi(bk, a):
    w = bk.remainder(a + np.pi, 2 * np.pi) - np.pi
    return bk.where((w == -np.pi) & (a > 0), np.pi, w)


def update_twist(bk, st: MLTwistState, fresh, m: MLModel, dt, carry):
    """CommandTerm.compute: time left, resampling, then the heading controller and the standing environments."""
    prev = st.command
    st = st.replace(time_left=st.time_left - dt)
    new, carry = resample_twist(bk, st, carry)
    st = _where(bk, st.time_left <= 0.0, new, st)
    forward = quat_apply(bk, fresh.xquat[m.root], np.array([1.0, 0.0, 0.0]))
    heading_w = bk.arctan2(forward[1], forward[0])
    _, az = _ranges(bk, carry.global_step)
    wz = bk.clip(HEADING_STIFFNESS * wrap_to_pi(bk, st.heading_target - heading_w), az[0], az[1])
    command = bk.where(st.is_heading, bk.concatenate([st.command[:2], wz[None]]), st.command)
    command = bk.where(st.is_standing, 0.0, command)
    return st.replace(command=command, prev_command=prev), carry


class MLTwist(Goal):
    """mjlab's twist command (vel_command_b); updated by MLActor, which runs first in the observation."""

    def __init__(self, info_props, **kwargs):
        super().__init__(info_props, **kwargs)

    @property
    def has_visual(self) -> bool:
        return False

    @property
    def dim(self) -> int:
        return 3

    def _init_from_mj(self, env, model, data, current_obs_size):
        self.obs_ind = np.arange(current_obs_size, current_obs_size + self.dim)
        self.data_type_ind = np.array([])
        self.min, self.max = [-np.inf] * self.dim, [np.inf] * self.dim
        self._initialized_from_mj = True

    def init_state(self, env, key, model, data, backend, traj=None):
        z3 = backend.zeros(3)
        return MLTwistState(command=z3, prev_command=z3, time_left=backend.array(0.0),
                            heading_target=backend.array(0.0), is_heading=backend.array(False),
                            is_standing=backend.array(False))

    def reset_state(self, env, model, data, carry, backend, traj=None):
        bk = _backend_of(data)
        st, carry = resample_twist(bk, getattr(carry.observation_states, self.name), carry)
        return data, _set_obs_state(carry, self.name, st)

    def get_obs_and_update_state(self, env, model, data, carry, backend, traj=None):
        return getattr(carry.observation_states, self.name).command, carry


# ----------------------------------------------------------------------------------------------------- observations
class _MLObservation(StatefulObservation):
    dim = None

    def __init__(self, obs_name: str, group, **kwargs):
        Observation.__init__(self, obs_name, group=group, **kwargs)
        StatefulObject.__init__(self, 0)
        self.m = None

    def _init_from_mj(self, env, model, data, current_obs_size):
        self.m = MLModel(model)
        self.obs_ind = np.arange(current_obs_size, current_obs_size + self.dim)
        self.data_type_ind = np.array([])
        self.min, self.max = [-np.inf] * self.dim, [np.inf] * self.dim
        self._initialized_from_mj = True


@struct.dataclass
class MLActorState:
    heights: Any       # foot heights of this observation (mjlab's ray casts)
    heights_prev: Any  # those of the previous observation, which mjlab's reward of this step reads
    lin_vel: Any       # refreshed readings, shared with MLCritic
    ang_vel: Any
    gravity: Any
    feet_found: Any
    feet_force: Any


class MLActor(_MLObservation):
    """mjlab's ``actor`` group; also does the last substep's contact bookkeeping, the refresh and the command update."""

    dim = 99

    def __init__(self, obs_name: str = "MLActor", group=("actor",), noise: bool = True, **kwargs):
        super().__init__(obs_name, list(group), **kwargs)
        self.noise = noise
        # set by reset_state, which runs right before the observation of a reset state is built (also while tracing)
        self._reset_pending = False
        nu = 29
        self._noise_scale = np.concatenate([np.full(3, NOISE["lin_vel"]), np.full(3, NOISE["ang_vel"]),
                                            np.full(3, NOISE["gravity"]), np.full(nu, NOISE["joint_pos"]),
                                            np.full(nu, NOISE["joint_vel"])])

    def init_state(self, env, key, model, data, backend, traj=None):
        z2, z3 = backend.zeros(2), backend.zeros(3)
        return MLActorState(heights=z2, heights_prev=z2, lin_vel=z3, ang_vel=z3, gravity=z3, feet_found=z2,
                            feet_force=backend.zeros((2, 3)))

    def reset_state(self, env, model, data, carry, backend, traj=None):
        self._reset_pending = True
        return data, carry

    def get_obs_and_update_state(self, env, model, data, carry, backend, traj=None):
        bk, m, sh = backend, self.m, _shared(env)
        st = getattr(carry.observation_states, self.name)
        reset, self._reset_pending = self._reset_pending, False
        if not reset:
            # mjlab's contact reading after the last substep
            feet_found, _, _, self_force = sh.contacts(model, data, bk)
            carry = carry.replace(control_func_state=track_contacts(bk, carry.control_func_state, feet_found, self_force))
        if reset and bk is not np:
            # a reset state: kinematics only (no velocity, no contacts; see MLShared)
            fresh, lin_vel, ang_vel = sh.reset_kinematics(data)
            f_found, f_force = jnp.zeros(2), jnp.zeros((2, 3))
        else:
            # mjlab's sim.forward() before the command update and the observations
            fresh = sh.forward(env, env._model if bk is np else model, data, bk)
            lin_vel, ang_vel = fresh.sensordata[m.imu_lin_vel], fresh.sensordata[m.imu_ang_vel]
            f_found, f_force, _, _ = sh.contacts(model, fresh, bk)
        twist = carry.observation_states.MLTwist
        # reset environments advance their command with dt = 0, as in mjlab
        twist, carry = update_twist(bk, twist, fresh, m, 0.0 if reset else STEP_DT, carry)
        carry = _set_obs_state(carry, "MLTwist", twist)
        gravity = quat_apply_inverse(bk, fresh.xquat[m.root], np.array([0.0, 0.0, -1.0]))
        heights = bk.clip(fresh.site_xpos[m.feet_sites, 2], 0.0, 1.0)
        noisy = bk.concatenate([lin_vel, ang_vel, gravity, data.qpos[7:] - m.default_pose, data.qvel[6:]])
        if self.noise:
            u, carry = _uniform(carry, bk, noisy.shape, -1.0, 1.0)
            noisy = noisy + u * self._noise_scale
        obs = bk.concatenate([noisy, carry.last_action, twist.command])
        st = st.replace(heights_prev=st.heights, heights=heights, lin_vel=lin_vel,
                        ang_vel=ang_vel, gravity=gravity, feet_found=f_found, feet_force=f_force)
        return obs, _set_obs_state(carry, self.name, st)


class MLCritic(_MLObservation):
    """mjlab's ``critic`` group, from MLActor's refreshed readings."""

    dim = 111

    def __init__(self, obs_name: str = "MLCritic", group=("critic",), **kwargs):
        super().__init__(obs_name, list(group), **kwargs)

    def get_obs_and_update_state(self, env, model, data, carry, backend, traj=None):
        bk, m = backend, self.m
        a = carry.observation_states.MLActor
        force = a.feet_force.reshape(-1)
        obs = bk.concatenate([
            a.lin_vel, a.ang_vel, a.gravity, data.qpos[7:] - m.default_pose, data.qvel[6:], carry.last_action,
            carry.observation_states.MLTwist.command, a.heights, carry.control_func_state.current_air_time,
            (a.feet_found > 0).astype(float), bk.sign(force) * bk.log1p(bk.abs(force))])
        return obs, carry


# ---------------------------------------------------------------------------------------------------------- reward
@struct.dataclass
class MLRewardState:
    peak_heights: Any       # feet_swing_height's state (mjlab never resets it)
    prev_action: Any        # the action manager's previous and second previous actions
    prev_prev_action: Any
    terms: Any              # the weighted terms of the last step, in REWARD_TERMS order (mjlab's _step_reward)
    mean_action_acc: Any    # mjlab's metric of the step


class MLVelocityReward(Reward):
    """The reward terms of mjlab's G1 velocity task, on the state the last substep left."""

    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        self.m = MLModel(env._model)
        self._weights = np.array([REWARD_WEIGHTS[k] for k in REWARD_TERMS])

    def init_state(self, env, key, model, data, backend, traj=None):
        nu = self.m.nu
        return MLRewardState(peak_heights=backend.zeros(2), prev_action=backend.zeros(nu),
                             prev_prev_action=backend.zeros(nu), terms=backend.zeros(len(REWARD_TERMS)),
                             mean_action_acc=backend.array(0.0))

    def reset(self, env, model, data, carry, backend, traj=None):
        # mjlab's action manager reset; feet_swing_height keeps its peak heights
        st = carry.reward_state
        st = st.replace(prev_action=backend.zeros_like(st.prev_action),
                        prev_prev_action=backend.zeros_like(st.prev_prev_action))
        return data, carry.replace(reward_state=st)

    def __call__(self, state, action, next_state, absorbing, info, env, model, data, carry, backend, traj=None):
        bk, m, st = backend, self.m, carry.reward_state
        ctrl, actor = carry.control_func_state, carry.observation_states.MLActor
        cmd = carry.observation_states.MLTwist.prev_command
        feet_found, feet_force, _, _ = _shared(env).contacts(model, data, bk)
        lin_b, ang_b = root_link_vel_b(bk, m, data)
        q = data.qpos[7:]
        total_cmd = bk.linalg.norm(cmd[:2]) + bk.abs(cmd[2])
        active = (total_cmd > COMMAND_THRESHOLD).astype(float)

        g_torso = quat_apply_inverse(bk, data.xquat[m.torso], np.array([0.0, 0.0, -1.0]))
        speed = total_cmd
        regime = bk.stack([speed < WALKING_THRESHOLD, (speed >= WALKING_THRESHOLD) & (speed < RUNNING_THRESHOLD),
                           speed >= RUNNING_THRESHOLD]).astype(float)
        std = bk.sum(m.pose_std * regime[:, None], axis=0)
        feet_vel = feet_lin_vel_w(bk, m, data)[:, :2]
        vel_norm = bk.linalg.norm(feet_vel, axis=-1)
        in_air, in_contact = feet_found == 0, feet_found > 0
        first = first_contact(bk, ctrl)
        peak = bk.where(in_air, bk.maximum(st.peak_heights, actor.heights_prev), st.peak_heights)
        hist_hit = bk.linalg.norm(ctrl.force_history, axis=-1) > SELF_COLLISION_FORCE
        terms = dict(
            track_linear_velocity=bk.exp(-(bk.sum(bk.square(cmd[:2] - lin_b[:2])) + bk.square(lin_b[2])) / LIN_STD ** 2),
            track_angular_velocity=bk.exp(-(bk.square(cmd[2] - ang_b[2]) + bk.sum(bk.square(ang_b[:2]))) / ANG_STD ** 2),
            upright=bk.exp(-bk.sum(bk.square(g_torso[:2])) / UPRIGHT_STD ** 2),
            pose=bk.exp(-bk.mean(bk.square(q - m.default_pose) / std ** 2)),
            body_ang_vel=bk.sum(bk.square(data.cvel[m.torso, 0:2])),
            angular_momentum=bk.sum(bk.square(data.sensordata[m.root_angmom])),
            dof_pos_limits=bk.sum(bk.maximum(m.soft_lo - q, 0.0) + bk.maximum(q - m.soft_hi, 0.0)),
            action_rate_l2=bk.sum(bk.square(action - st.prev_action)),
            foot_clearance=bk.sum(bk.abs(actor.heights_prev - FOOT_TARGET_HEIGHT) * vel_norm) * active,
            foot_swing_height=bk.sum(bk.square(peak / FOOT_TARGET_HEIGHT - 1.0) * first) * active,
            foot_slip=bk.sum(bk.square(vel_norm) * in_contact) * active,
            soft_landing=bk.sum(bk.linalg.norm(feet_force, axis=-1) * first) * active,
            self_collisions=bk.sum(hist_hit).astype(float),
        )
        terms = bk.stack([bk.asarray(terms[k], dtype=float) for k in REWARD_TERMS]) * self._weights
        terms = bk.nan_to_num(terms, nan=0.0, posinf=0.0, neginf=0.0)
        reward = bk.sum(terms) * STEP_DT
        acc = bk.mean(bk.abs(action - 2 * st.prev_action + st.prev_prev_action))
        st = st.replace(peak_heights=bk.where(first, 0.0, peak), prev_prev_action=st.prev_action, prev_action=action,
                        terms=terms, mean_action_acc=acc)
        return reward, carry.replace(reward_state=st)


# -------------------------------------------------------------------------------------- termination and reset
class MLTermination(TerminalStateHandler):
    """mjlab's bad_orientation(70 deg) on the pelvis; time_out is the environment's horizon."""

    def __init__(self, env, **handler_config):
        super().__init__(env, **handler_config)
        self.m = MLModel(env._model)

    def reset(self, env, model, data, carry, backend, traj=None):
        return data, carry

    def is_absorbing(self, env, obs, info, data, carry, traj=None):
        return bool(self._fell_over(np, data)), carry

    def mjx_is_absorbing(self, env, obs, info, data, carry, traj=None):
        return self._fell_over(jnp, data), carry

    def _fell_over(self, bk, data):
        g = quat_apply_inverse(bk, data.xquat[self.m.root], np.array([0.0, 0.0, -1.0]))
        return bk.arccos(bk.clip(-g[2], -1.0, 1.0)) > FELL_OVER_ANGLE


class MLInitState(InitialStateHandler):
    """mjlab's reset: reset_root_state_uniform and reset_joints_by_offset (zero offsets), velocities zero."""

    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        self.m = MLModel(env._model)

    def reset(self, env, model, data, carry, backend, traj=None):
        m = self.m
        u, carry = _uniform(carry, backend, (4,))
        x = RESET_XY[0] + u[0] * (RESET_XY[1] - RESET_XY[0])
        y = RESET_XY[0] + u[1] * (RESET_XY[1] - RESET_XY[0])
        z = RESET_Z[0] + u[2] * (RESET_Z[1] - RESET_Z[0])
        yaw = RESET_YAW[0] + u[3] * (RESET_YAW[1] - RESET_YAW[0])
        quat = backend.stack([backend.cos(yaw / 2), backend.zeros(()), backend.zeros(()), backend.sin(yaw / 2)])
        pos = m.default_root[:3] + backend.stack([x, y, z])
        qpos = backend.concatenate([pos, quat, m.default_pose])
        nv = data.qvel.shape[-1]
        if backend is np:
            data.qpos[:], data.qvel[:], data.ctrl[:], data.qacc_warmstart[:] = qpos, 0.0, 0.0, 0.0
        else:
            data = data.replace(qpos=qpos, qvel=jnp.zeros(nv), ctrl=jnp.zeros_like(data.ctrl),
                                qacc_warmstart=jnp.zeros_like(data.qacc_warmstart))
        return data, carry


# ------------------------------------------------------------------------------------------------- registration
def register():
    """Registers the components with LocoMuJoCo (idempotent)."""
    for cls in (MLActor, MLCritic, MLTwist):
        if cls.__name__ not in Observation.registered:
            cls.register()
    for base, cls in ((Reward, MLVelocityReward), (TerminalStateHandler, MLTermination),
                      (InitialStateHandler, MLInitState), (ControlFunction, MLControl)):
        if cls.__name__ not in base.registered:
            cls.register()


def make_env(backend: str, export_dir: str, naconmax: int | None = None, njmax: int | None = None,
             noise: bool = True, horizon: int = 1000, **kwargs):
    """The task on backend ``cpu``, ``mjx`` or ``mjwarp``, on mjlab's model in ``export_dir`` (full or reduced)."""
    import lmj_real  # noqa: F401  registers G1Real, MjxG1Real and G1RealPositionControl
    from lmj_real.environments.g1_real import G1Real
    from lmj_real.environments.g1_real_mjx import MjxG1Real
    register()
    spec = load_spec(export_dir)
    spec.joint("robot/floating_base_joint").name = "root"  # G1Real's name of the free joint
    if backend == "mjx":
        strip_contact_sensors(spec)
    joints = [j.name for j in spec.joints if j.type == mujoco.mjtJoint.mjJNT_HINGE]
    by_joint = {a.target: a.name for a in spec.actuators}
    common = dict(
        spec=spec, observation_spec=[MLActor(noise=noise), MLCritic()], actuation_spec=[by_joint[j] for j in joints],
        horizon=horizon, timestep=PHYSICS_DT, n_substeps=DECIMATION, headless=True, print_obs_summary=False,
        model_option_conf=dict(iterations=10, ls_iterations=20, ccd_iterations=50),
        control_type="MLControl", control_params=dict(action_scale=1.0, clip_actions=None, clip_target_to_jnt_range=False),
        goal_type="MLTwist", goal_params=dict(group="command"), reward_type="MLVelocityReward",
        terminal_state_type="MLTermination", init_state_type="MLInitState")
    common.update(kwargs)
    if backend == "cpu":
        env = G1Real(**common)
    elif backend == "mjx":
        env = MjxG1Real(use_mjwarp=False, **common)
    else:
        env = MjxG1Real(use_mjwarp=True, naconmax=naconmax, njmax=njmax, **common)
    env._mjlab_task = MLShared(env, backend)
    return env
