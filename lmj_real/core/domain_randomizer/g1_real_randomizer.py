"""Domain randomization for sim-to-real transfer: sensing and disturbances, not the plant.

``DefaultRandomizer`` perturbs the *plant* (gravity, link masses, inertias, joint
damping/armature, PD gains). This module perturbs *sensing and disturbances*:

==================  =========================================================
event               behaviour
==================  =========================================================
``push_robot``      every U(1, 3) s, add a uniform impulse to the root velocity
``encoder_bias``    persistent per-episode +/-0.02 rad joint-position bias
``foot_friction``   U(0.3, 1.2), one shared draw across all foot geoms (or all
                    colliding robot geoms)
``base_com``        pelvis CoM offset, per-axis +/-(0.025, 0.05, 0.05)
==================  =========================================================

plus per-term observation noise (joint pos +/-0.01, joint vel +/-0.5, base angular
velocity +/-0.2). Projected gravity is *not* corrupted, and neither is the privileged
group.

**On pushes.** :meth:`G1RealRandomizer.update` hangs off ``DomainRandomizer.update``,
which is called every step and may return modified ``data`` -- the natural home for a
mid-episode event.

This subclasses :class:`DefaultRandomizer` so that its plant-randomization knobs stay
*available* for experiments, but every one of them defaults to **off**. Note the
inherited PD-gain noise never engages regardless: ``DefaultRandomizer.reset`` gates it
on ``isinstance(env._control_func, PDControl)``, and this env uses
:class:`~lmj_real.core.control_functions.g1_real_position.G1RealPositionControl`, whose
gains live in the model rather than in Python.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any, List

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from flax import struct
from loco_mujoco.core.domain_randomizer.default import DefaultRandomizer, DefaultRandomizerState
from loco_mujoco.core.utils.backend import assert_backend_is_supported

from lmj_real.core.observations.g1_real_obs import G1RealJointPosRel

# Everything DefaultRandomizer would otherwise do, switched off.
_PLANT_RANDOMIZATION_OFF: dict[str, Any] = {
    "randomize_gravity": False,
    "gravity_range": [9.81, 9.81],
    "randomize_geom_friction_tangential": False,
    "geom_friction_tangential_range": [1.0, 1.0],
    "randomize_geom_friction_torsional": False,
    "geom_friction_torsional_range": [0.005, 0.005],
    "randomize_geom_friction_rolling": False,
    "geom_friction_rolling_range": [0.0001, 0.0001],
    "randomize_geom_damping": False,
    "geom_damping_range": [0.0, 0.0],
    "randomize_geom_stiffness": False,
    "geom_stiffness_range": [0.0, 0.0],
    "randomize_joint_damping": False,
    "joint_damping_range": [0.0, 0.0],
    "randomize_joint_friction_loss": False,
    "joint_friction_loss_range": [0.0, 0.0],
    "randomize_joint_armature": False,
    "joint_armature_range": [0.0, 0.0],
    "randomize_base_mass": False,
    "base_mass_to_add_range": [0.0, 0.0],
    "randomize_com_displacement": False,
    "com_displacement_range": [0.0, 0.0],
    "randomize_link_mass": False,
    "link_mass_multiplier_range": {"root_body": [1.0, 1.0], "other_bodies": [1.0, 1.0]},
    "add_p_gains_noise": False,
    "p_gains_noise_scale": 0.0,
    "add_d_gains_noise": False,
    "d_gains_noise_scale": 0.0,
    # Observation noise is applied by this class, not the parent, because the G1Real
    # observation terms are StatefulObservations and so carry no entries in
    # `env._obs_indices.JointPos` / `.JointVel` / etc.
    "add_joint_pos_noise": False,
    "joint_pos_noise_scale": 0.0,
    "add_joint_vel_noise": False,
    "joint_vel_noise_scale": 0.0,
    "add_gravity_noise": False,
    "gravity_noise_scale": 0.0,
    "add_free_joint_lin_vel_noise": False,
    "lin_vel_noise_scale": 0.0,
    "add_free_joint_ang_vel_noise": False,
    "ang_vel_noise_scale": 0.0,
}

# Ordered like MuJoCo's free-joint qvel: linear xyz, angular xyz.
PUSH_VELOCITY_RANGE = ((-0.5, 0.5), (-0.5, 0.5), (-0.2, 0.2),
                       (-0.52, 0.52), (-0.52, 0.52), (-0.78, 0.78))


@struct.dataclass
class G1RealRandomizerState(DefaultRandomizerState):
    """``DefaultRandomizerState`` plus the push countdown and the encoder bias."""

    #: control steps until the next push
    push_countdown: np.ndarray | jax.Array
    #: persistent per-episode joint-position sensing bias
    encoder_bias: np.ndarray | jax.Array
    #: ring buffer of past *sensed* observation entries, shape (delay+1, n_delayed).
    #: Slot -1 is the newest, slot 0 is what the policy is handed.
    obs_delay_buffer: np.ndarray | jax.Array
    #: False until the buffer has been seeded with the episode's first observation.
    #: Without this the first `delay` steps would feed zeros -- a policy trained on that
    #: learns from garbage transitions at every episode start.
    obs_delay_primed: np.ndarray | jax.Array


class G1RealRandomizer(DefaultRandomizer):
    """Pushes, encoder bias, foot friction, CoM offset and observation noise."""

    def __init__(
        self,
        env: Any,
        enable_push: bool = True,
        push_interval_range_s: tuple[float, float] = (1.0, 3.0),
        push_velocity_range: tuple[tuple[float, float], ...] = PUSH_VELOCITY_RANGE,
        enable_encoder_bias: bool = True,
        encoder_bias_range: tuple[float, float] = (-0.02, 0.02),
        enable_foot_friction: bool = True,
        foot_friction_range: tuple[float, float] = (0.3, 1.2),
        enable_base_com: bool = True,
        base_com_range: tuple[float, float, float] = (0.025, 0.05, 0.05),
        friction_all_geoms: bool = False,
        ground_friction: float | None = None,
        joint_pos_obs_noise: float = 0.01,
        joint_vel_obs_noise: float = 0.5,
        base_ang_vel_obs_noise: float = 0.2,
        noisy_obs_group: str = "history",
        obs_delay_steps: int = 0,
        **kwargs: Any,
    ):
        """
        Args:
            env: Environment instance.
            enable_push: Periodic pushes on the root velocity.
            push_interval_range_s: Seconds between pushes, sampled uniformly.
            push_velocity_range: Per-DoF impulse bounds, ordered ``(x, y, z, rx, ry, rz)``.
            enable_encoder_bias: Persistent per-episode joint-position sensing bias.
            enable_foot_friction: Shared friction draw across all foot geoms.
            enable_base_com: Pelvis CoM offset.
            base_com_range: Half-widths per axis.
            friction_all_geoms: Apply the friction draw to every colliding robot geom,
                not only the feet.
            ground_friction: Tangential friction of the world geoms. MuJoCo takes the
                larger of a pair, so with the floor at its default 1.0 any draw below 1.0
                does nothing. Setting this to the bottom of the range lets the draw win.
            joint_pos_obs_noise: Uniform noise half-width on ``joint_pos``.
            joint_vel_obs_noise: ... on ``joint_vel``.
            base_ang_vel_obs_noise: ... on the IMU gyro.
            noisy_obs_group: Only observations in this group get noise, so the
                critic's ``privileged_state`` stays clean.
            **kwargs: Passed to :class:`DefaultRandomizer`; any inherited knob set here
                overrides the all-off defaults.
        """
        conf = dict(_PLANT_RANDOMIZATION_OFF)
        conf.update(kwargs)

        self._enable_push = enable_push
        self._push_interval_range_s = push_interval_range_s
        self._push_velocity_range = np.asarray(push_velocity_range, dtype=float)
        self._enable_encoder_bias = enable_encoder_bias
        self._encoder_bias_range = encoder_bias_range
        self._enable_foot_friction = enable_foot_friction
        self._foot_friction_range = foot_friction_range
        self._enable_base_com = enable_base_com
        self._base_com_range = np.asarray(base_com_range, dtype=float)
        self._ground_friction = ground_friction
        self._joint_pos_obs_noise = joint_pos_obs_noise
        self._joint_vel_obs_noise = joint_vel_obs_noise
        self._base_ang_vel_obs_noise = base_ang_vel_obs_noise

        super().__init__(env, **conf)

        model = env._model
        props = env._get_all_info_properties()

        self._root_free_joint_qvel_ind = np.arange(6)
        self._n_actuated = len(env._action_indices)

        self._foot_geom_ids = np.array(
            [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
                for n in props["foot_geom_names"]
            ]
        )
        self._pelvis_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, props["root_body_name"]
        )
        colliding = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
        self._ground_geom_ids = np.where(model.geom_bodyid == 0)[0]
        self._friction_geom_ids = (np.where(colliding & (model.geom_bodyid != 0))[0]
                                   if friction_all_geoms else self._foot_geom_ids)

        # Push interval in control steps. `env.dt` is the control timestep.
        self._push_interval_steps = (
            max(1, int(round(push_interval_range_s[0] / env.dt))),
            max(1, int(round(push_interval_range_s[1] / env.dt))),
        )

        # Observation slices to corrupt. Resolved by observation *type*, restricted to
        # the actor group, because the G1Real observation terms are stateful and so are
        # invisible to the parent's index-container based noising.
        # Defence in depth. Group membership already keeps the clean `privileged_state`
        # duplicates out of both resolvers, but allow_randomization is the flag a reader
        # will assume protects an observation -- DefaultRandomizer honours it and ours
        # did not. Intersecting here makes `allow_randomization: false` actually mean
        # something on this path, so a term added to the corrupted group later cannot be
        # silently noised against its own declaration.
        self._randomizable = set(
            np.asarray(env.obs_container.get_randomizable_obs_indices()).tolist())
        self._obs_noise_slices = [
            (ind, hw, is_jp)
            for ind, hw, is_jp in self._resolve_noise_slices(env, noisy_obs_group)
            if set(np.asarray(ind).tolist()) <= self._randomizable
        ]
        # Default 0: the buffer is never consulted.
        #
        # A delay makes the MDP partially observed and a memoryless policy cannot
        # compensate. Use experiment.len_obs_history to supply a history; delaying without
        # it measures how much latency costs, not whether the policy can handle it.
        self._obs_delay_steps = int(obs_delay_steps)
        if self._obs_delay_steps < 0:
            raise ValueError(f"obs_delay_steps must be >= 0, got {obs_delay_steps}")
        self._delay_obs_ind = self._resolve_delay_indices(env, noisy_obs_group)

    def _resolve_noise_slices(self, env: Any, group: str) -> list[tuple[np.ndarray, float, bool]]:
        """Find ``(obs_indices, noise_half_width, is_joint_pos)`` per noisy term."""
        from loco_mujoco.core.observations.base import EntryFromFreeJointVel, JointVelArray

        out: list[tuple[np.ndarray, float, bool]] = []
        for obs in env.obs_container.values():
            # `group` is a str, list, or omegaconf ListConfig depending on how the
            # observation was declared. Normalise on str instead of enumerating container
            # types: a ListConfig is neither list nor tuple.
            groups = [obs.group] if isinstance(obs.group, str) else list(obs.group)
            if group not in groups:
                continue
            if isinstance(obs, G1RealJointPosRel):
                out.append((np.asarray(obs.obs_ind), self._joint_pos_obs_noise, True))
            elif isinstance(obs, JointVelArray):
                out.append((np.asarray(obs.obs_ind), self._joint_vel_obs_noise, False))
            elif isinstance(obs, EntryFromFreeJointVel):
                # Only the angular entries stand in for the IMU gyro.
                if np.all(np.asarray(obs.data_type_ind) >= 3):
                    out.append((np.asarray(obs.obs_ind), self._base_ang_vel_obs_noise, False))
        return out

    def _resolve_delay_indices(self, env: Any, group: str) -> np.ndarray:
        """Observation indices that represent a *sensor reading*, hence can be stale.

        Joint positions, joint velocities, the IMU gyro and projected gravity. Explicitly
        NOT included:

        * the tracking goal -- the reference is commanded, not measured, so delaying it
          models nothing physical;
        * ``last_action`` -- the controller knows exactly what it just sent;
        * the privileged critic terms -- they are not sensed at all, and staleness there
          would only handicap the value function.

        Resolved from ``env.obs_container`` by type rather than from
        ``env._obs_indices``: the G1Real observation types are ``StatefulObservation``
        subclasses, so they never appear in the per-type index container.
        """
        from loco_mujoco.core.observations.base import (EntryFromFreeJointVel,
                                                        JointVelArray,
                                                        ProjectedGravityVector)

        ind: list[np.ndarray] = []
        for obs in env.obs_container.values():
            # `group` is a str, list, or omegaconf ListConfig depending on how the
            # observation was declared. Normalise on str instead of enumerating container
            # types: a ListConfig is neither list nor tuple.
            groups = [obs.group] if isinstance(obs.group, str) else list(obs.group)
            if group not in groups:
                continue
            if isinstance(obs, (G1RealJointPosRel, JointVelArray, ProjectedGravityVector)):
                ind.append(np.asarray(obs.obs_ind))
            elif isinstance(obs, EntryFromFreeJointVel):
                # Angular entries only: those are the gyro. The linear ones are not
                # directly sensed on the real robot anyway.
                if np.all(np.asarray(obs.data_type_ind) >= 3):
                    ind.append(np.asarray(obs.obs_ind))
        if not ind:
            return np.zeros(0, dtype=np.int64)
        return np.unique(np.concatenate(ind)).astype(np.int64)

    def _delay_observation(self, obs, carry, backend: ModuleType):
        """Return the sensed entries as they were ``obs_delay_steps`` control steps ago.

        Push the current reading into the ring buffer and hand back the oldest slot. The
        delay is applied to the CLEAN reading and the noise is added afterwards by the
        caller, which is the physical order: the sensor sampled at ``t - delay`` and the
        measurement noise happens at read time, so it should be fresh each step rather
        than carried along in the buffer.
        """
        state = carry.domain_randomizer_state
        ind = self._delay_obs_ind
        cur = obs[ind]
        buf = state.obs_delay_buffer

        if backend == jnp:
            # On the first observation of an episode every slot becomes the current
            # reading, so the policy sees a consistent (if not yet stale) history.
            buf = jnp.where(state.obs_delay_primed, buf,
                            jnp.broadcast_to(cur, buf.shape))
            buf = jnp.roll(buf, -1, axis=0).at[-1].set(cur)
            obs = obs.at[ind].set(buf[0])
            primed = jnp.array(True)
        else:
            buf = np.array(buf)
            if not bool(state.obs_delay_primed):
                buf[:] = cur
            buf = np.roll(buf, -1, axis=0)
            buf[-1] = cur
            obs = np.array(obs)
            obs[ind] = buf[0]
            primed = np.array(True)

        carry = carry.replace(domain_randomizer_state=state.replace(
            obs_delay_buffer=buf, obs_delay_primed=primed))
        return obs, carry

    def _uniform(self, shape, low, high, carry, backend: ModuleType):
        """Uniform sample, threading ``carry.key`` on the JAX path."""
        if backend == jnp:
            key, subkey = jax.random.split(carry.key)
            carry = carry.replace(key=key)
            return jax.random.uniform(subkey, shape=shape, minval=low, maxval=high), carry
        return np.random.uniform(low=low, high=high, size=shape), carry

    def init_state(self, env, key, model, data, backend, traj=None):
        assert_backend_is_supported(backend)
        parent = super().init_state(env, key, model, data, backend, traj)
        return G1RealRandomizerState(
            **{f: getattr(parent, f) for f in parent.__dataclass_fields__},
            push_countdown=backend.array(self._push_interval_steps[1], dtype=backend.int32),
            encoder_bias=backend.zeros(self._n_actuated),
            obs_delay_buffer=backend.zeros(
                (self._obs_delay_steps + 1, len(self._delay_obs_ind))),
            obs_delay_primed=backend.array(False),
        )

    def reset(self, env, model, data, carry, backend, traj=None):
        assert_backend_is_supported(backend)
        data, carry = super().reset(env, model, data, carry, backend, traj)
        state = carry.domain_randomizer_state

        if self._enable_encoder_bias:
            bias, carry = self._uniform(
                (self._n_actuated,), *self._encoder_bias_range, carry, backend
            )
        else:
            bias = backend.zeros(self._n_actuated)

        countdown, carry = self._uniform(
            (), self._push_interval_steps[0], self._push_interval_steps[1] + 1, carry, backend
        )

        geom_friction = state.geom_friction
        if self._enable_foot_friction:
            # One shared draw for every friction geom.
            shared, carry = self._uniform((), *self._foot_friction_range, carry, backend)
            if backend == jnp:
                geom_friction = geom_friction.at[self._friction_geom_ids, 0].set(shared)
                if self._ground_friction is not None:
                    geom_friction = geom_friction.at[self._ground_geom_ids, 0].set(
                        self._ground_friction)
            else:
                geom_friction = np.array(geom_friction)
                geom_friction[self._friction_geom_ids, 0] = shared
                if self._ground_friction is not None:
                    geom_friction[self._ground_geom_ids, 0] = self._ground_friction

        com = state.com_displacement
        if self._enable_base_com:
            offset, carry = self._uniform(
                (3,), -self._base_com_range, self._base_com_range, carry, backend
            )
            com = offset

        carry = carry.replace(
            domain_randomizer_state=state.replace(
                encoder_bias=bias,
                push_countdown=countdown.astype(backend.int32),
                geom_friction=geom_friction,
                com_displacement=com,
            )
        )
        return data, carry

    def update(self, env, model, data, carry, backend, traj=None):
        """Apply the parent's model updates and the friction and CoM draws, then fire
        pushes when the timer expires.

        The impulse is added to the root free joint's ``qvel``. Note MuJoCo stores
        free-joint angular velocity in the **body** frame, so the angular part of the
        push is applied in that frame.
        """
        assert_backend_is_supported(backend)
        model, data, carry = super().update(env, model, data, carry, backend, traj)

        # The parent only writes these under its own flags, which are off here.
        state = carry.domain_randomizer_state
        if self._enable_foot_friction:
            model = self._set_attribute_in_model(model, "geom_friction", state.geom_friction,
                                                 backend)
        if self._enable_base_com:
            if backend == jnp:
                body_ipos = model.body_ipos.at[self._pelvis_body_id].add(state.com_displacement)
            else:
                body_ipos = self._init_body_ipos.copy()
                body_ipos[self._pelvis_body_id] += state.com_displacement
            model = self._set_attribute_in_model(model, "body_ipos", body_ipos, backend)

        if not self._enable_push:
            return model, data, carry

        state = carry.domain_randomizer_state
        countdown = state.push_countdown - 1
        fire = countdown <= 0

        impulse, carry = self._uniform(
            (6,), self._push_velocity_range[:, 0], self._push_velocity_range[:, 1],
            carry, backend,
        )
        next_countdown, carry = self._uniform(
            (), self._push_interval_steps[0], self._push_interval_steps[1] + 1, carry, backend
        )

        ind = self._root_free_joint_qvel_ind
        if backend == jnp:
            qvel = data.qvel.at[ind].add(backend.where(fire, impulse, 0.0))
            data = data.replace(qvel=qvel)
            countdown = jnp.where(fire, next_countdown.astype(jnp.int32), countdown)
        else:
            if fire:
                data.qvel[ind] += impulse
                countdown = np.int32(next_countdown)

        carry = carry.replace(
            domain_randomizer_state=carry.domain_randomizer_state.replace(
                push_countdown=countdown
            )
        )
        return model, data, carry

    def update_observation(self, env, obs, model, data, carry, backend, traj=None):
        """Delay the sensed entries, then add the noise and the encoder bias."""
        assert_backend_is_supported(backend)
        if self._obs_delay_steps > 0 and len(self._delay_obs_ind) > 0:
            obs, carry = self._delay_observation(obs, carry, backend)
        state = carry.domain_randomizer_state

        for obs_ind, half_width, is_joint_pos in self._obs_noise_slices:
            n = len(obs_ind)
            if half_width > 0.0:
                noise, carry = self._uniform((n,), -half_width, half_width, carry, backend)
            else:
                noise = backend.zeros(n)
            if is_joint_pos and self._enable_encoder_bias:
                noise = noise + state.encoder_bias[:n]
            if backend == jnp:
                obs = obs.at[obs_ind].add(noise)
            else:
                obs = np.array(obs)
                obs[obs_ind] += noise

        return obs, carry
