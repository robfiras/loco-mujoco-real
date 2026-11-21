"""MuJoCo Playground's G1 joystick task, rebuilt from LocoMuJoCo components, for a like-for-like benchmark.

Re-implements ``G1JoystickFlatTerrain`` of MuJoCo Playground 0.2.0 (``mujoco_playground/_src/locomotion/g1/
joystick.py``) on Playground's own model files, for all three LocoMuJoCo backends (CPU, MJX, MjWarp):

* observations ``PGState`` (Playground's ``state``, 103 entries, groups ``actor`` and ``critic``) and
  ``PGPrivileged`` (the rest of ``privileged_state``, 113 entries, group ``critic``), so that the ``critic`` group is
  Playground's ``privileged_state`` (216) in its order;
* goal ``PGCommand``: the joystick command, resampled once its step counter passes 500;
* reward ``PGJoystickReward``: all 24 terms with Playground's scales, ``sum(scale * term) * ctrl_dt``;
* terminal state handler ``PGTermination``: torso upside down, foot-foot or foot-shin contact, or NaN;
* initial state handler ``PGInitState``: Playground's randomized reset;
* domain randomizer ``PGPush``: Playground's random pushes.

What Playground keeps in ``state.info`` (gait phase, last actions, feet air time, last contact, swing peak, the
command's step counter, the push counter) lives in the states of these components and is updated at the same
points of the step. Every quantity is read from the same MJCF sensors and sites as in Playground.

Differences in how, not in what a step computes:

* On an automatic reset, Playground (``wrap_for_brax_training``) restores a cached first state of each
  environment; LocoMuJoCo draws a new initial state, which is more work.
* The observation that LocoMuJoCo's MJX/MjWarp backends return after an automatic reset is computed without a
  forward pass, so its sensor entries are not those of the new state. (Playground returns the cached first
  observation.)
* The 67 observation-noise samples are drawn in one call rather than five; the distribution is the same.
"""
from __future__ import annotations

import os
from typing import Any

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from flax import struct
from loco_mujoco.core import Observation
from loco_mujoco.core.domain_randomizer import DomainRandomizer
from loco_mujoco.core.domain_randomizer.no_randomization import NoDomainRandomization
from loco_mujoco.core.initial_state_handler import InitialStateHandler
from loco_mujoco.core.observations.base import StatefulObservation
from loco_mujoco.core.observations.goals import Goal
from loco_mujoco.core.reward import Reward
from loco_mujoco.core.stateful_object import StatefulObject
from loco_mujoco.core.terminal_state_handler import TerminalStateHandler

KEYFRAME = "init_state"  # Playground's knees_bent keyframe, renamed by load_spec for G1RealPositionControl

# joystick.py default_config()
CTRL_DT = 0.02
ACTION_SCALE = 0.5
NOISE_SCALES = dict(gyro=0.2, gravity=0.05, joint_pos=0.03, joint_vel=1.5, linvel=0.1)
REWARD_SCALES = dict(
    tracking_lin_vel=1.0, tracking_ang_vel=0.75, lin_vel_z=0.0, ang_vel_xy=-0.15, orientation=-2.0,
    base_height=0.0, torques=0.0, action_rate=0.0, energy=0.0, dof_acc=0.0, feet_slip=-0.25, feet_clearance=0.0,
    feet_height=0.0, feet_air_time=2.0, feet_phase=1.0, alive=0.0, termination=-100.0, stand_still=-1.0,
    collision=-0.1, contact_force=-0.01, joint_deviation_hip=-0.25, joint_deviation_knee=-0.1,
    dof_pos_limits=-1.0, pose=-0.1)
REWARD_TERMS = tuple(REWARD_SCALES)  # the order of joystick.py's _get_reward
TRACKING_SIGMA, MAX_FOOT_HEIGHT, BASE_HEIGHT_TARGET, MAX_CONTACT_FORCE = 0.25, 0.15, 0.5, 500.0
SOFT_JOINT_POS_LIMIT_FACTOR = 0.95
PUSH_INTERVAL, PUSH_MAGNITUDE = (5.0, 10.0), (0.1, 2.0)
LIN_VEL_X, LIN_VEL_Y, ANG_VEL_YAW = (-1.0, 1.0), (-0.5, 0.5), (-1.0, 1.0)
COMMAND_RESAMPLE_STEPS = 500

SENSORS = ("upvector_torso", "local_linvel_pelvis", "accelerometer_pelvis", "gyro_pelvis", "global_linvel_torso",
           "global_linvel_pelvis", "global_angvel_torso", "global_angvel_pelvis", "left_foot_global_linvel",
           "right_foot_global_linvel", "left_foot_force", "right_foot_force", "left_foot_floor_found",
           "right_foot_floor_found", "right_foot_left_foot_found", "left_foot_right_shin_found",
           "right_foot_left_shin_found", "left_hand_left_thigh_found", "right_hand_right_thigh_found")


class PGModel:
    """Indices into Playground's G1 model, as joystick.py's ``_post_init`` computes them."""

    def __init__(self, model: mujoco.MjModel):
        def sensor(name):
            i = model.sensor(name).id
            return np.arange(model.sensor_adr[i], model.sensor_adr[i] + model.sensor_dim[i])

        self.s = {name: sensor(name) for name in SENSORS}
        self.feet_linvel = np.concatenate([self.s["left_foot_global_linvel"], self.s["right_foot_global_linvel"]])
        self.feet_floor_found = np.concatenate([self.s["left_foot_floor_found"], self.s["right_foot_floor_found"]])
        self.imu_pelvis = model.site("imu_in_pelvis").id
        self.feet_sites = np.array([model.site("left_foot").id, model.site("right_foot").id])
        self.init_q = np.array(model.key(KEYFRAME).qpos)
        self.default_pose = self.init_q[7:].copy()
        lowers, uppers = model.jnt_range[1:].T
        c, r = (lowers + uppers) / 2, uppers - lowers
        self.soft_lowers = c - 0.5 * r * SOFT_JOINT_POS_LIMIT_FACTOR
        self.soft_uppers = c + 0.5 * r * SOFT_JOINT_POS_LIMIT_FACTOR
        joint = lambda name: int(model.joint(name).qposadr[0]) - 7
        self.hip = np.array([joint(f"{s}_{j}_joint") for s in ("left", "right") for j in ("hip_roll", "hip_yaw")])
        self.knee = np.array([joint(f"{s}_knee_joint") for s in ("left", "right")])
        self.nu, self.nv = model.nu, model.nv


def _backend_of(data):
    """The backend that matches ``data`` (LocoMuJoCo's CPU reset passes jax.numpy to reset_state)."""
    return np if isinstance(data, mujoco.MjData) else jnp


def _uniform(carry, backend, shape=(), low=0.0, high=1.0):
    """U(low, high) from the carry's key (JAX backends) or NumPy's generator (CPU), and the advanced carry."""
    if backend is np:
        return np.random.uniform(low, high, shape), carry
    key, sub = jax.random.split(carry.key)
    return jax.random.uniform(sub, shape, minval=low, maxval=high), carry.replace(key=key)


def sample_command(backend, key=None):
    """joystick.py's sample_command: uniform velocities, all zero with probability 0.1."""
    if backend is np:
        cmd = np.array([np.random.uniform(*LIN_VEL_X), np.random.uniform(*LIN_VEL_Y), np.random.uniform(*ANG_VEL_YAW)])
        return np.zeros(3) if np.random.random() < 0.1 else cmd
    k1, k2, k3, k4 = jax.random.split(key, 4)
    cmd = jnp.hstack([jax.random.uniform(k1, minval=LIN_VEL_X[0], maxval=LIN_VEL_X[1]),
                      jax.random.uniform(k2, minval=LIN_VEL_Y[0], maxval=LIN_VEL_Y[1]),
                      jax.random.uniform(k3, minval=ANG_VEL_YAW[0], maxval=ANG_VEL_YAW[1])])
    return jnp.where(jax.random.bernoulli(k4, p=0.1), jnp.zeros(3), cmd)


def gait_rz(backend, phi, swing_height):
    """gait.get_rz: target foot height over the gait phase (cubic Bezier up and down)."""
    def bezier(y_start, y_end, x):
        return y_start + (y_end - y_start) * (x ** 3 + 3 * (x ** 2 * (1 - x)))
    x = (phi + np.pi) / (2 * np.pi)
    return backend.where(x <= 0.5, bezier(0, swing_height, 2 * x), bezier(swing_height, 0, 2 * x - 1))


def gravity_in_pelvis_imu(backend, m: PGModel, data):
    """``site_xmat[imu_in_pelvis].T @ [0, 0, -1]``, the third row of the site rotation negated."""
    return -backend.reshape(data.site_xmat[m.imu_pelvis], (3, 3))[2]


class _PGObservation(StatefulObservation):
    """A stateful observation of fixed size over Playground's model."""

    dim = None

    def __init__(self, obs_name: str, group, **kwargs):
        Observation.__init__(self, obs_name, group=group, **kwargs)
        StatefulObject.__init__(self, 0)
        self.m = None

    def _init_from_mj(self, env, model, data, current_obs_size):
        self.m = PGModel(model)
        self.obs_ind = np.arange(current_obs_size, current_obs_size + self.dim)
        self.data_type_ind = np.array([])
        self.min, self.max = [-np.inf] * self.dim, [np.inf] * self.dim
        self._initialized_from_mj = True


@struct.dataclass
class PGStateState:
    phase: Any
    phase_dt: Any
    last_act: Any
    last_last_act: Any


class PGState(_PGObservation):
    """Playground's ``state`` observation: noisy IMU, joint readings, command, last action and gait phase."""

    dim = 103

    def __init__(self, obs_name: str = "PGState", group=("actor", "critic"), noise_level: float = 1.0, **kwargs):
        super().__init__(obs_name, list(group), **kwargs)
        self.noise_level = noise_level
        nu = 29
        s = NOISE_SCALES
        # in the order of the readings below: gyro, gravity, joint positions, joint velocities, linear velocity
        self._noise_scale = noise_level * np.concatenate(
            [np.full(3, s["gyro"]), np.full(3, s["gravity"]), np.full(nu, s["joint_pos"]),
             np.full(nu, s["joint_vel"]), np.full(3, s["linvel"])])

    def init_state(self, env, key, model, data, backend, traj=None):
        # gait frequency U(1.25, 1.5) Hz, the two feet half a cycle apart
        freq = np.random.uniform(1.25, 1.5, 1) if backend is np else \
            jax.random.uniform(key, (1,), minval=1.25, maxval=1.5)
        nu = self.m.nu
        return PGStateState(phase=backend.array([0.0, np.pi]), phase_dt=2 * np.pi * env.dt * freq,
                            last_act=backend.zeros(nu), last_last_act=backend.zeros(nu))

    def reset_state(self, env, model, data, carry, backend, traj=None):
        return data, carry  # Playground keeps its info across automatic resets

    def get_obs_and_update_state(self, env, model, data, carry, backend, traj=None):
        m, st = self.m, getattr(carry.observation_states, self.name)
        command = carry.observation_states.PGCommand.command
        sd = data.sensordata
        readings = backend.concatenate([sd[m.s["gyro_pelvis"]], gravity_in_pelvis_imu(backend, m, data),
                                        data.qpos[7:], data.qvel[6:], sd[m.s["local_linvel_pelvis"]]])
        if self.noise_level > 0:
            u, carry = _uniform(carry, backend, readings.shape)
            readings = readings + (2 * u - 1) * self._noise_scale
        nu = m.nu
        gyro, gravity, q, dq, linvel = (readings[0:3], readings[3:6], readings[6:6 + nu],
                                        readings[6 + nu:6 + 2 * nu], readings[6 + 2 * nu:])
        obs = backend.concatenate([linvel, gyro, gravity, command, q - m.default_pose, dq, st.last_act,
                                   backend.cos(st.phase), backend.sin(st.phase)])
        return obs, carry


@struct.dataclass
class PGPrivilegedState:
    feet_air_time: Any
    last_contact: Any
    swing_peak: Any
    contact: Any
    first_contact: Any
    skip_update: Any  # set by reset_state: the observation of a reset state does no bookkeeping


class PGPrivileged(_PGObservation):
    """The noiseless part of Playground's ``privileged_state``, and the feet contact bookkeeping of its step."""

    dim = 113

    def __init__(self, obs_name: str = "PGPrivileged", group=("critic",), **kwargs):
        super().__init__(obs_name, list(group), **kwargs)

    def init_state(self, env, key, model, data, backend, traj=None):
        return PGPrivilegedState(feet_air_time=backend.zeros(2), last_contact=backend.zeros(2, dtype=bool),
                                 swing_peak=backend.zeros(2), contact=backend.zeros(2, dtype=bool),
                                 first_contact=backend.zeros(2, dtype=bool), skip_update=backend.array(False))

    def reset_state(self, env, model, data, carry, backend, traj=None):
        backend = _backend_of(data)
        st = getattr(carry.observation_states, self.name).replace(skip_update=backend.array(True))
        return data, carry.replace(observation_states=carry.observation_states.replace(**{self.name: st}))

    def get_obs_and_update_state(self, env, model, data, carry, backend, traj=None):
        m, st = self.m, getattr(carry.observation_states, self.name)
        sd = data.sensordata
        # Playground's step, between the physics and the observation
        contact = sd[m.feet_floor_found] > 0
        contact_filt = contact | st.last_contact
        first_contact = (st.feet_air_time > 0.0) & contact_filt
        air_time = st.feet_air_time + env.dt
        swing_peak = backend.maximum(st.swing_peak, data.site_xpos[m.feet_sites, 2])
        if backend is np:
            if st.skip_update:
                air_time, swing_peak, first_contact = st.feet_air_time, st.swing_peak, st.first_contact
        else:
            air_time = jnp.where(st.skip_update, st.feet_air_time, air_time)
            swing_peak = jnp.where(st.skip_update, st.swing_peak, swing_peak)
            first_contact = jnp.where(st.skip_update, st.first_contact, first_contact)
        st = st.replace(feet_air_time=air_time, swing_peak=swing_peak, contact=contact, first_contact=first_contact,
                        skip_update=backend.array(False))
        carry = carry.replace(observation_states=carry.observation_states.replace(**{self.name: st}))
        obs = backend.concatenate([
            sd[m.s["gyro_pelvis"]], sd[m.s["accelerometer_pelvis"]], gravity_in_pelvis_imu(backend, m, data),
            sd[m.s["local_linvel_pelvis"]], sd[m.s["global_angvel_pelvis"]], data.qpos[7:] - m.default_pose,
            data.qvel[6:], data.qpos[2:3], data.actuator_force, contact.astype(float), sd[m.feet_linvel], air_time])
        return obs, carry


@struct.dataclass
class PGCommandState:
    command: Any
    step: Any


class PGCommand(Goal):
    """Playground's joystick command (lin_vel_x, lin_vel_y, ang_vel_yaw); the reward advances its step counter."""

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
        return PGCommandState(command=sample_command(backend, key), step=backend.array(0))

    def get_obs_and_update_state(self, env, model, data, carry, backend, traj=None):
        return getattr(carry.observation_states, self.name).command, carry


@struct.dataclass
class PGRewardState:
    terms: Any  # the unscaled reward terms of the last step, in the order of REWARD_TERMS (Playground's metrics)


class PGJoystickReward(Reward):
    """joystick.py's ``_get_reward`` and the bookkeeping at the end of its ``step``."""

    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        self.m = PGModel(env._model)
        self._scales = np.array([REWARD_SCALES[k] for k in REWARD_TERMS])
        self._dt = env.dt

    def init_state(self, env, key, model, data, backend, traj=None):
        return PGRewardState(terms=backend.zeros(len(REWARD_TERMS)))

    def __call__(self, state, action, next_state, absorbing, info, env, model, data, carry, backend, traj=None):
        bk, m = backend, self.m
        obs_states = carry.observation_states
        st, priv, goal = obs_states.PGState, obs_states.PGPrivileged, obs_states.PGCommand
        cmd, sd, q = goal.command, data.sensordata, data.qpos[7:]
        local_linvel, gyro = sd[m.s["local_linvel_pelvis"]], sd[m.s["gyro_pelvis"]]
        linvel_pelvis, linvel_torso = sd[m.s["global_linvel_pelvis"]], sd[m.s["global_linvel_torso"]]
        angvel_torso, angvel_pelvis = sd[m.s["global_angvel_torso"]], sd[m.s["global_angvel_pelvis"]]
        force, foot_z = data.actuator_force, data.site_xpos[m.feet_sites, 2]
        feet_vel = bk.reshape(sd[m.feet_linvel], (2, 3))
        contact, first_contact = priv.contact, priv.first_contact
        cmd_norm = bk.linalg.norm(cmd)

        rz = gait_rz(bk, st.phase, MAX_FOOT_HEIGHT)
        moving = bk.logical_or(bk.linalg.norm(linvel_pelvis[:2]) > 0.1, bk.abs(angvel_pelvis[2]) > 0.1)
        feet_phase = bk.exp(-bk.sum(bk.square(foot_z - rz)) / 0.01) * bk.logical_or(moving, cmd_norm > 0.01)
        hip_weight = bk.where(cmd[1] > 0.1, np.array([0.0, 1.0, 0.0, 1.0]), np.ones(4))
        feet_force_z = bk.abs(bk.stack([sd[m.s["left_foot_force"]][2], sd[m.s["right_foot_force"]][2]]))
        terms = dict(
            tracking_lin_vel=bk.exp(-bk.sum(bk.square(cmd[:2] - local_linvel[:2])) / TRACKING_SIGMA),
            tracking_ang_vel=bk.exp(-bk.square(cmd[2] - gyro[2]) / TRACKING_SIGMA),
            lin_vel_z=bk.square(linvel_pelvis[2]) + bk.square(linvel_torso[2]),
            ang_vel_xy=bk.sum(bk.square(angvel_torso[:2])),
            orientation=bk.sum(bk.square(sd[m.s["upvector_torso"]] - np.array([0.073, 0.0, 1.0]))),
            base_height=bk.square(data.qpos[2] - BASE_HEIGHT_TARGET),
            torques=bk.sum(bk.abs(force)),
            action_rate=bk.sum(bk.square(action - st.last_act)),
            energy=bk.sum(bk.abs(data.qvel[6:]) * bk.abs(force)),
            dof_acc=bk.sum(bk.square(data.qacc[6:])),
            feet_slip=bk.sum(bk.linalg.norm(linvel_pelvis[:2]) * contact),
            feet_clearance=bk.sum(bk.abs(foot_z - MAX_FOOT_HEIGHT)
                                  * bk.sqrt(bk.linalg.norm(feet_vel[:, :2], axis=-1))),
            feet_height=bk.sum(bk.square(priv.swing_peak / MAX_FOOT_HEIGHT - 1.0) * first_contact),
            feet_air_time=bk.sum(bk.minimum((priv.feet_air_time - 0.2) * first_contact, 0.5 - 0.2)),
            feet_phase=feet_phase,
            alive=bk.ones(()),
            termination=bk.asarray(absorbing, dtype=float),
            stand_still=bk.sum(bk.abs(q - m.default_pose)) * (cmd_norm < 0.01),
            collision=bk.logical_or(sd[m.s["left_hand_left_thigh_found"]][0] > 0,
                                    sd[m.s["right_hand_right_thigh_found"]][0] > 0),
            contact_force=bk.sum(bk.maximum(feet_force_z - MAX_CONTACT_FORCE, 0.0)),
            # joystick.py's hip indices have shape (4, 1), so its weights broadcast to a 4 x 4 product:
            # sum(|error|) * sum(weight). Reproduced as computed there.
            joint_deviation_hip=bk.sum(bk.abs(q[m.hip] - m.default_pose[m.hip])) * bk.sum(hip_weight),
            joint_deviation_knee=bk.sum(bk.abs(q[m.knee] - m.default_pose[m.knee])),
            dof_pos_limits=bk.sum(-bk.minimum(q - m.soft_lowers, 0.0) + bk.maximum(q - m.soft_uppers, 0.0)),
            pose=bk.sum(bk.square(q - m.default_pose)),
        )
        terms = bk.stack([bk.asarray(terms[k], dtype=float) for k in REWARD_TERMS])
        reward = bk.sum(terms * self._scales) * self._dt

        # the end of Playground's step: phase, last actions, command and its counter, feet bookkeeping
        step = goal.step + 1
        phase = bk.fmod(st.phase + st.phase_dt + np.pi, 2 * np.pi) - np.pi
        if bk is np:
            if step > COMMAND_RESAMPLE_STEPS:
                cmd = sample_command(np)
            step = 0 if (absorbing or step > COMMAND_RESAMPLE_STEPS) else step
        else:
            key, sub = jax.random.split(carry.key)
            carry = carry.replace(key=key)
            cmd = jnp.where(step > COMMAND_RESAMPLE_STEPS, sample_command(jnp, sub), cmd)
            step = jnp.where(absorbing | (step > COMMAND_RESAMPLE_STEPS), 0, step)
        no_contact = ~contact
        obs_states = obs_states.replace(
            PGState=st.replace(phase=phase, last_last_act=st.last_act, last_act=action),
            PGPrivileged=priv.replace(feet_air_time=priv.feet_air_time * no_contact, last_contact=contact,
                                      swing_peak=priv.swing_peak * no_contact),
            PGCommand=goal.replace(command=cmd, step=step))
        carry = carry.replace(observation_states=obs_states, reward_state=PGRewardState(terms=terms))
        return reward, carry


class PGTermination(TerminalStateHandler):
    """joystick.py's ``_get_termination``: torso upside down, foot-foot or foot-shin contact, or NaN."""

    def __init__(self, env, **handler_config):
        super().__init__(env, **handler_config)
        self.m = PGModel(env._model)

    def reset(self, env, model, data, carry, backend, traj=None):
        return data, carry

    def is_absorbing(self, env, obs, info, data, carry, traj=None):
        return bool(self._absorbing(data, np)), carry

    def mjx_is_absorbing(self, env, obs, info, data, carry, traj=None):
        return self._absorbing(data, jnp), carry

    def _absorbing(self, data, bk):
        m, sd = self.m, data.sensordata
        fall = sd[m.s["upvector_torso"]][2] < 0.0
        contact = (sd[m.s["right_foot_left_foot_found"]][0] > 0) | (sd[m.s["left_foot_right_shin_found"]][0] > 0) \
            | (sd[m.s["right_foot_left_shin_found"]][0] > 0)
        return fall | contact | bk.isnan(data.qpos).any() | bk.isnan(data.qvel).any()


class PGInitState(InitialStateHandler):
    """joystick.py's ``reset``: keyframe pose shifted and turned, joints scaled, random base velocity."""

    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        self.m = PGModel(env._model)

    def reset(self, env, model, data, carry, backend, traj=None):
        m = self.m
        if backend is np:
            dxy, yaw = np.random.uniform(-0.5, 0.5, 2), np.random.uniform(-3.14, 3.14, 1)
            joint_scale, base_vel = np.random.uniform(0.5, 1.5, m.nu), np.random.uniform(-0.5, 0.5, 6)
        else:
            key, k1, k2, k3, k4 = jax.random.split(carry.key, 5)
            carry = carry.replace(key=key)
            dxy, yaw = jax.random.uniform(k1, (2,), minval=-0.5, maxval=0.5), \
                jax.random.uniform(k2, (1,), minval=-3.14, maxval=3.14)
            joint_scale = jax.random.uniform(k3, (m.nu,), minval=0.5, maxval=1.5)
            base_vel = jax.random.uniform(k4, (6,), minval=-0.5, maxval=0.5)
        q0 = m.init_q
        # quat_mul(q0, axis_angle_to_quat(z, yaw)), with q0's rotation and a yaw about z
        w1, x1, y1, z1 = q0[3:7]
        cy, sy = backend.cos(yaw[0] / 2), backend.sin(yaw[0] / 2)
        quat = backend.stack([w1 * cy - z1 * sy, x1 * cy + y1 * sy, y1 * cy - x1 * sy, z1 * cy + w1 * sy])
        joints = q0[7:] * joint_scale
        qpos = backend.concatenate([q0[0:2] + dxy, q0[2:3], quat, joints])
        qvel = backend.concatenate([base_vel, backend.zeros(m.nv - 6)])
        if backend is np:
            data.qpos[:], data.qvel[:], data.ctrl[:] = qpos, qvel, joints
        else:
            data = data.replace(qpos=qpos, qvel=qvel, ctrl=joints)
        return data, carry


@struct.dataclass
class PGPushState:
    push_step: Any
    push_interval_steps: Any


class PGPush(NoDomainRandomization):
    """joystick.py's pushes: every 5 to 10 s (drawn per episode), a 0.1 to 2 m/s kick of the base in xy."""

    def __init__(self, env, enable: bool = True, **kwargs):
        super().__init__(env, **kwargs)
        self.enable = enable

    def init_state(self, env, key, model, data, backend, traj=None):
        interval = np.random.uniform(*PUSH_INTERVAL) if backend is np else \
            jax.random.uniform(key, minval=PUSH_INTERVAL[0], maxval=PUSH_INTERVAL[1])
        steps = backend.round(interval / env.dt).astype(backend.int32)
        return PGPushState(push_step=backend.array(0), push_interval_steps=steps)

    def update(self, env, model, data, carry, backend, traj=None):
        st = carry.domain_randomizer_state
        if self.enable:
            theta, carry = _uniform(carry, backend, (), 0.0, 2 * np.pi)
            magnitude, carry = _uniform(carry, backend, (), *PUSH_MAGNITUDE)
            push = backend.stack([backend.cos(theta), backend.sin(theta)])
            push = push * (backend.mod(st.push_step + 1, st.push_interval_steps) == 0)
            qvel_xy = data.qvel[:2] + push * magnitude
            if backend is np:
                data.qvel[:2] = qvel_xy
            else:
                data = data.replace(qvel=data.qvel.at[:2].set(qvel_xy))
        carry = carry.replace(domain_randomizer_state=st.replace(push_step=st.push_step + 1))
        return model, data, carry


def register():
    """Registers the components with LocoMuJoCo (idempotent)."""
    for cls in (PGState, PGPrivileged, PGCommand):
        if cls.__name__ not in Observation.registered:
            cls.register()
    for base, cls in ((Reward, PGJoystickReward), (TerminalStateHandler, PGTermination),
                      (InitialStateHandler, PGInitState), (DomainRandomizer, PGPush)):
        if cls.__name__ not in base.registered:
            cls.register()


def load_spec(assets_dir: str) -> mujoco.MjSpec:
    """Playground's G1 scene (scene_mjx_feetonly_flat_terrain.xml) from its model files, named for G1Real."""
    assets = {f: open(os.path.join(assets_dir, f), "rb").read() for f in os.listdir(assets_dir)}
    spec = mujoco.MjSpec.from_string(assets["scene_mjx_feetonly_flat_terrain.xml"].decode(), assets=assets,
                                     include={k: v for k, v in assets.items() if k.endswith(".xml")})
    spec.joint("floating_base_joint").name = "root"  # G1Real's name of the free joint
    spec.key("knees_bent").name = KEYFRAME          # the default pose of G1RealPositionControl
    for a in spec.actuators:
        # Playground's servos have no effort limit; G1RealPositionControl requires one to derive its scale, which
        # is overridden anyway. A placeholder range with forcelimited off leaves the forces unclamped.
        a.forcelimited = mujoco.mjtLimited.mjLIMITED_FALSE
        a.forcerange = [-1.0, 1.0]
    return spec


def make_env(backend: str, assets_dir: str, naconmax: int | None = None, njmax: int | None = None,
             noise_level: float = 1.0, push: bool = True, horizon: int = 1000, **kwargs):
    """The task on backend ``cpu``, ``mjx`` or ``mjwarp``, with Playground's settings at 0.005 s x 4 substeps."""
    register()
    import lmj_real  # noqa: F401  registers G1Real, MjxG1Real and G1RealPositionControl
    from lmj_real.environments.g1_real import G1Real
    from lmj_real.environments.g1_real_mjx import MjxG1Real

    common = dict(
        spec=load_spec(assets_dir), observation_spec=[PGState(noise_level=noise_level), PGPrivileged()],
        horizon=horizon, timestep=0.005, n_substeps=4, headless=True, print_obs_summary=False,
        model_option_conf=dict(iterations=3, ls_iterations=5, ccd_iterations=35),
        control_type="G1RealPositionControl",
        control_params=dict(action_scale=1.0, scale=ACTION_SCALE, clip_actions=None, clip_target_to_jnt_range=False),
        goal_type="PGCommand", goal_params=dict(group="command"), reward_type="PGJoystickReward",
        terminal_state_type="PGTermination", init_state_type="PGInitState",
        domain_randomization_type="PGPush", domain_randomization_params=dict(enable=push))
    common.update(kwargs)
    if backend == "cpu":
        return G1Real(**common)
    if backend == "mjx":
        return MjxG1Real(use_mjwarp=False, **common)
    return MjxG1Real(use_mjwarp=True, naconmax=naconmax, njmax=njmax, **common)
