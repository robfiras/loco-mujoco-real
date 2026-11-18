"""Assert an exported bundle reproduces the training environment, term by term.

The rollout deliberately reimplements the observation in numpy instead of calling
back into ``loco_mujoco``; this script is what makes that reimplementation trustworthy.

Two phases.

**Teacher-forced parity.** The real ``lmj_real`` environment is stepped with the jax
policy's own action, PPO's or SuperMimic's. SuperMimic's runs in float32 here, as the
bundle's does; the exporter reports what that changes against its training dtype. At
every step, and *from the same* ``MjData``, the bundle's
:class:`~lmj_deploy.ObservationBuilder` rebuilds the actor observation and the numpy actor
produces an action. Both are diffed against the environment's, per observation
term, so a mismatch points at the term that is wrong rather than at a 283-long vector.
The joint-position target from :class:`~lmj_deploy.PositionActionMapper` is checked
against the environment's control function too.

**Closed-loop divergence.** A standalone plain-MuJoCo copy is reset to the same state
and run on its own, off the bundle alone, and its ``qpos`` is compared with the
environment's rollout step by step. Some divergence is expected and healthy -- both
integrate the same equations, so it is floating-point noise amplified by contact -- but
it should grow slowly rather than jump on step one.

Usage::

    python deployment/validate_bundle.py \\
        --bundle deploy_bundles/g1_real \\
        --checkpoint outputs/.../PPOJax_saved.pkl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_bundle import build_env_from_checkpoint, require_body_poses  # noqa: E402
from lmj_deploy import (  # noqa: E402
    ObservationBuilder,
    PolicySpec,
    PositionActionMapper,
    apply_model_contact,
    apply_model_options,
    load_actor,
    training_model_xml,
)

# Tolerances. Observations are pure float arithmetic on the same MjData, so they should
# agree to float32 round-off. Actions come from a 3-layer MLP in float64 (numpy) vs
# float32 (jax), hence the looser action budget.
OBS_ATOL = 1e-5
ACTION_ATOL = 1e-3
TARGET_ATOL = 1e-4


def deploy_reference(env, traj, carry, builder: ObservationBuilder) -> dict[str, np.ndarray]:
    """The reference frame the environment is currently pointing at, in bundle shape.

    Body positions are moved from clip coordinates to the episode's world frame, as the
    training goal does: the environment may have moved the robot to the origin at reset.
    """
    ts = carry.traj_state
    clip, step = int(ts.traj_no), int(ts.subtraj_step_no)
    ref = traj.data.get(clip, step, np)
    offset_xy = np.asarray(env.get_traj_offset_xy(carry, np, traj))
    out = {
        "joint_pos": np.asarray(ref.qpos)[builder.qpos_adr],
        "joint_vel": np.asarray(ref.qvel)[builder.dof_adr],
        "body_pos_w": np.asarray(ref.xpos)[builder.body_ids] + np.append(offset_xy, 0.0),
        "body_quat_w": np.asarray(ref.xquat)[builder.body_ids],
        "qpos": np.asarray(ref.qpos),
        "qvel": np.asarray(ref.qvel),
    }
    n_future = builder.spec.n_future
    if n_future:
        split = np.asarray(traj.data.split_points)
        last = int(split[clip + 1] - split[clip]) - 1
        frames = [traj.data.get(clip, min(step + k, last), np) for k in range(n_future + 1)]
        out["qpos_future"] = np.stack([np.asarray(f.qpos) for f in frames])
        out["qvel_future"] = np.stack([np.asarray(f.qvel) for f in frames])
    return out


class JaxPolicy:
    """The checkpoint's own deterministic policy, on the env's full observation."""

    def __init__(self, agent_conf, agent_state, algorithm: str, seed: int):
        import jax

        self.algorithm = algorithm
        config = agent_conf.config
        if algorithm == "SuperMimicJax":
            import dataclasses

            from export_bundle import _supermimic_params
            from loco_mujoco.algorithms import SuperMimicJax

            net = agent_conf.actor_net.clone(compute_dtype="float32")
            self._act = jax.jit(SuperMimicJax.policy_fn(
                dataclasses.replace(agent_conf, actor_net=net)))
            self.params, self.norm = _supermimic_params(agent_state, seed)
            self.hist_ind = np.asarray(agent_conf.history_ind)
            self.n_frames = int(config.experiment.history_len)
            self.history = None
            return

        from loco_mujoco.algorithms.common.policies import PPOPolicy

        # Deterministic: a -inf log_std collapses the Gaussian to its mean.
        policy = PPOPolicy(agent_conf.network)
        train_state = agent_state.train_state
        n_seeds = int(getattr(config.experiment, "n_seeds", 1) or 1)
        if n_seeds > 1:
            train_state = jax.tree.map(lambda x: x[seed], train_state)
        from export_bundle import frozen_run_stats

        # Frozen normalizer statistics, as in the bundle (see frozen_run_stats).
        self.train_state = train_state.replace(
            run_stats=frozen_run_stats(train_state.run_stats),
            params={**train_state.params,
                    "log_std": np.full_like(np.asarray(train_state.params["log_std"]),
                                            -np.inf)}
        )
        self._ppo = jax.jit(lambda ts, obs, key: policy.get_action_and_value(obs, ts, key))
        self.rng = jax.random.key(seed)

    def reset(self) -> None:
        if self.algorithm == "SuperMimicJax":
            self.history = None

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        import jax
        import jax.numpy as jnp

        obs = jnp.asarray(obs)[None]
        if self.algorithm == "SuperMimicJax":
            frame = obs[:, None, self.hist_ind]
            self.history = (jnp.repeat(frame, self.n_frames, axis=1) if self.history is None
                            else jnp.concatenate([self.history[:, 1:], frame], axis=1))
            return np.asarray(self._act(self.params, self.norm, obs, self.history)).reshape(-1)
        self.rng, key = jax.random.split(self.rng)
        a, _, _, self.train_state = self._ppo(self.train_state, obs, key)
        return np.asarray(a).reshape(-1)


def bind_traj(env, traj) -> None:
    """Thread ``traj`` into ``_create_observation``.

    ``Mujoco.reset`` builds the first observation without passing the trajectory
    through, which ``GoalTrajMimicAnchor`` hard-requires. ``eval.py`` patches the same
    method for its ``--use_mujoco`` path; this is that workaround, scoped to the
    validator.
    """
    env._create_observation = (
        lambda model, data, carry, _t=traj:
        env._create_observation_compat(model, data, carry, np, _t)
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n-steps", type=int, default=100)
    ap.add_argument("--train-state-seed", type=int, default=0)
    ap.add_argument("--closed-loop-steps", type=int, default=50)
    args = ap.parse_args()

    spec = PolicySpec.load(args.bundle)
    actor = load_actor(args.bundle, spec)

    env, traj, agent_conf, agent_state, algorithm = build_env_from_checkpoint(
        args.checkpoint)
    require_body_poses(traj)
    bind_traj(env, traj)

    # The env's columns for each bundle term, in bundle order. Per-term comparisons only
    # read the terms' own slices, so a stacked history after them does not matter.
    actor_ind = np.concatenate(
        [np.asarray(env.obs_container[t["name"]].obs_ind) for t in spec.obs_terms])
    builder = ObservationBuilder(spec, env._model)
    mapper = PositionActionMapper(spec, env._model)

    if env._n_intermediate_steps * env._n_substeps != spec.n_substeps:
        print(f"[validate] WARNING: env runs {env._n_intermediate_steps} x "
              f"{env._n_substeps} physics steps per control step, bundle says "
              f"{spec.n_substeps}")

    policy = JaxPolicy(agent_conf, agent_state, algorithm, args.train_state_seed)

    # -- phase 1: teacher-forced parity ----------------------------------------------
    obs = env.reset(traj=traj)
    policy.reset()
    actor.reset()
    last_action = np.zeros(spec.action_dim)

    per_term = {t["name"]: 0.0 for t in spec.obs_terms}
    worst = {"obs": 0.0, "action": 0.0, "target": 0.0}
    worst_at = {"obs": (-1, ""), "action": -1, "target": -1}
    n = 0

    for step in range(args.n_steps):
        env_obs = np.asarray(obs)[actor_ind]
        ref = deploy_reference(env, traj, env._additional_carry, builder)
        dep_obs = builder.build(env._data, ref, last_action)

        for t in spec.obs_terms:
            sl = slice(t["start"], t["start"] + t["dim"])
            d = float(np.abs(env_obs[sl] - dep_obs[sl]).max())
            per_term[t["name"]] = max(per_term[t["name"]], d)
            if d > worst["obs"]:
                worst["obs"], worst_at["obs"] = d, (step, t["name"])

        jax_action = policy(obs)
        np_action = actor.act(dep_obs)
        d = float(np.abs(jax_action - np_action).max())
        if d > worst["action"]:
            worst["action"], worst_at["action"] = d, step

        # Control mapping: the env's own control function vs the bundle's.
        env_target, _ = env._control_func.generate_action(
            env, jax_action, env._model, env._data, env._additional_carry, np, traj
        )
        d = float(np.abs(np.asarray(env_target) - mapper.target(jax_action)).max())
        if d > worst["target"]:
            worst["target"], worst_at["target"] = d, step

        last_action = jax_action.astype(np.float64)
        n += 1
        obs, _, absorbing, done, _ = env.step(jax_action, traj=traj)
        if absorbing or done:
            print(f"[validate] episode ended at step {step} "
                  f"(absorbing={bool(absorbing)}, done={bool(done)})")
            break

    print(f"\n[validate] teacher-forced parity over {n} steps")
    print(f"{'term':<24s} {'max |deploy - env|':>20s}")
    for name, d in per_term.items():
        flag = "  <-- FAIL" if d > OBS_ATOL else ""
        print(f"{name:<24s} {d:>20.3e}{flag}")
    print(f"\n  observation   max {worst['obs']:.3e} "
          f"(step {worst_at['obs'][0]}, {worst_at['obs'][1]})  tol {OBS_ATOL:.0e}")
    print(f"  action        max {worst['action']:.3e} "
          f"(step {worst_at['action']})  tol {ACTION_ATOL:.0e}")
    print(f"  ctrl target   max {worst['target']:.3e} "
          f"(step {worst_at['target']})  tol {TARGET_ATOL:.0e}")

    failures = []
    if worst["obs"] > OBS_ATOL:
        failures.append("observation")
    if worst["action"] > ACTION_ATOL:
        failures.append("action")
    if worst["target"] > TARGET_ATOL:
        failures.append("ctrl target")

    # -- phase 2: closed-loop divergence ---------------------------------------------
    # Both sides run their own policy from the same initial state: the environment on
    # jax, a standalone MjData on the bundle alone. The reference cursor is read off the
    # environment, which is the one thing the standalone rig gets from a manifest at
    # deploy time rather than computing.
    n_cl = args.closed_loop_steps
    if n_cl > 1:
        model = mujoco.MjModel.from_xml_path(str(training_model_xml(spec)))
        apply_model_options(model, spec)
        apply_model_contact(model, spec)
        data = mujoco.MjData(model)
        cl_builder = ObservationBuilder(spec, model)
        cl_mapper = PositionActionMapper(spec, model)

        obs = env.reset(traj=traj)
        # `Mujoco.reset` runs mj_forward *before* the initial-state handler writes qpos,
        # so env._data.xpos is left describing the previous pose and the episode's first
        # observation is built on stale body poses (~1 cm on the G1). Refresh here, and
        # recompute the first observation from the refreshed state, so the two sides
        # start from the same kinematics -- otherwise this phase would be measuring that
        # backbone quirk rather than the bundle.
        mujoco.mj_forward(env._model, env._data)
        obs, env._additional_carry = env._create_observation(
            env._model, env._data, env._additional_carry
        )
        data.qpos[:] = env._data.qpos
        data.qvel[:] = env._data.qvel
        mujoco.mj_forward(model, data)

        la = np.zeros(spec.action_dim)
        policy.reset()
        actor.reset()
        divergence = []
        for step in range(n_cl):
            ref = deploy_reference(env, traj, env._additional_carry, cl_builder)
            a = actor.act(cl_builder.build(data, ref, la))
            la = a.astype(np.float64)
            target = cl_mapper.target(a)
            for _ in range(spec.n_substeps):
                cl_mapper.apply(data, target)
                mujoco.mj_step(model, data)

            obs, _, absorbing, done, _ = env.step(policy(obs), traj=traj)
            diff = np.abs(np.asarray(data.qpos) - env._data.qpos)
            divergence.append(float(diff.max()))
            if step == 0:
                first_worst = int(diff.argmax())
            if absorbing or done:
                break

        n_cl = len(divergence)
        print(f"\n[validate] closed-loop |qpos| divergence over {n_cl} steps: "
              f"step1 {divergence[0]:.2e}  step{max(n_cl // 2, 1)} "
              f"{divergence[max(n_cl // 2 - 1, 0)]:.2e}  step{n_cl} {divergence[-1]:.2e}")
        if divergence[0] > 1e-4:
            failures.append("closed-loop step-1 divergence")
            print(f"  largest step-1 difference is qpos[{first_worst}]")
            print("  <-- FAIL: the very first standalone step already differs; the "
                  "actuation or reset path does not match")

    if failures:
        raise SystemExit(f"\n[validate] FAIL: {', '.join(failures)}")
    print("\n[validate] PASS: the bundle reproduces the training env's observation, "
          "action and control target.")


if __name__ == "__main__":
    main()
