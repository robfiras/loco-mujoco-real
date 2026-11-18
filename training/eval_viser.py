"""Stream a trained PPO checkpoint to a viser viewer in the browser.

The viser counterpart of ``eval.py``: same checkpoint/config reconstruction, but the
rollout calls ``env.mjx_render_viser(state)`` instead of the OpenGL ``mjx_render``, so it
needs no local framebuffer and works over SSH. Lays the environments out on a grid.

    python training/eval_viser.py \
        --path outputs/<run>/PPOJax_saved.pkl \
        --n-envs 4 --port 8080

Then open http://localhost:8080 (forward the port if you are on a remote machine).

Notes:

* Defaults to ``use_mjwarp=False``. The checkpoints here were trained with MjWarp, but
  the viewer only needs ``qpos``, and MJX/JAX keeps the maintained rendering path; pass
  ``--mjwarp`` to view on the training backend instead.
* ``port`` reaches ``ViserViewer`` through ``Mujoco.__init__``'s ``**viewer_params``
  catch-all, which is what that mechanism is actually for.
* Requires the viser extra: ``pip install loco-mujoco[viser]``.
"""

from __future__ import annotations

import argparse
import os

import jax
import jax.numpy as jnp
import numpy as np
from loco_mujoco import TaskFactory
from loco_mujoco.algorithms import PPOJax
from loco_mujoco.algorithms.common.policies import PPOPolicy
from loco_mujoco.algorithms.experimental.td3_jax import TD3Jax
from omegaconf import OmegaConf

import lmj_real  # noqa: F401  (registers the env + components)

os.environ.setdefault("XLA_FLAGS", "--xla_gpu_triton_gemm_any=True ")

# Namespaces / agent-side params that are not env constructor kwargs. If they leak
# through they end up in _viewer_params and break the viewer constructor.
NON_ENV_KEYS = {"robot", "actor_obs_group", "critic_obs_group"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--path", required=True, help="PPOJax_saved.pkl / TD3Jax_saved.pkl")
    ap.add_argument("--algo", choices=("ppo", "td3"), default=None,
                    help="defaults to td3 if the filename contains 'td3' (any case), else ppo")
    ap.add_argument("--n-envs", type=int, default=4, help="environments to show on the grid")
    ap.add_argument("--port", type=int, default=8080, help="viser port")
    ap.add_argument("--n-steps", type=int, default=0, help="0 = run until interrupted")
    ap.add_argument("--stochastic", action="store_true",
                    help="sample from the policy instead of taking the mean action")
    ap.add_argument("--mjwarp", action="store_true",
                    help="view on the MjWarp backend (as trained) rather than MJX")
    ap.add_argument("--visualize-goal", action="store_true",
                    help="draw the re-anchored reference body poses as green boxes")
    ap.add_argument("--raw-goal", action="store_true",
                    help="with --visualize-goal, also draw the unanchored world-frame "
                         "reference in red")
    ap.add_argument("--dr", action="store_true",
                    help="keep domain randomization on (pushes, encoder bias, foot "
                         "friction, CoM offset, obs noise). Off by default.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-state-seed", type=int, default=0,
                    help="which training seed to replay when the checkpoint holds several")
    args = ap.parse_args()

    algo = args.algo or ("td3" if "td3" in os.path.basename(args.path).lower() else "ppo")
    loader = TD3Jax if algo == "td3" else PPOJax
    agent_conf, agent_state = loader.load_agent(args.path)
    config = agent_conf.config
    print(f"[viser] algo={algo}", flush=True)
    OmegaConf.set_struct(config, False)

    env_params = {
        k: v for k, v in config.experiment.env_params.items() if k not in NON_ENV_KEYS
    }
    env_params["headless"] = True          # viser renders in the browser, no GL window
    env_params["use_mjwarp"] = bool(args.mjwarp)
    env_params["port"] = args.port
    if args.visualize_goal:
        # Visual geoms live in the user scene, not the observation, so switching this on
        # does not change the observation dim and stays checkpoint-compatible.
        goal_params = dict(env_params.get("goal_params") or {})
        goal_params["visualize_goal"] = True
        goal_params["visualize_raw_reference"] = bool(args.raw_goal)
        env_params["goal_params"] = goal_params
        print("[viser] goal visualization ON (green boxes = re-anchored reference bodies)",
              flush=True)

    if not args.dr:
        # Switch off pushes, CoM offset, encoder bias, foot friction and observation noise,
        # so what you see is the policy, not the disturbances.
        dr = dict(env_params.get("domain_randomization_params") or {})
        dr.update(enable_push=False, enable_encoder_bias=False, enable_foot_friction=False,
                  enable_base_com=False, joint_pos_obs_noise=0.0, joint_vel_obs_noise=0.0,
                  base_ang_vel_obs_noise=0.0)
        env_params["domain_randomization_params"] = dr
        print("[viser] domain randomization OFF (pass --dr to keep it)", flush=True)

    factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
    factory_params = OmegaConf.to_container(
        config.experiment.task_factory.params, resolve=True
    )

    # The config carries a list of .npz paths; resolve it exactly as the training script
    # does (CustomDatasetConf only accepts a Trajectory).
    custom = factory_params.get("custom_dataset_conf")
    if isinstance(custom, dict) and "paths" in custom:
        from loco_mujoco.core.trajectory import Trajectory
        from loco_mujoco.task_factories.dataset_confs import CustomDatasetConf

        paths = list(custom["paths"])
        print(f"[viser] loading {len(paths)} trajectories", flush=True)
        loaded = [Trajectory.load(p) for p in paths]
        factory_params["custom_dataset_conf"] = CustomDatasetConf(
            traj=Trajectory.concatenate(loaded) if len(loaded) > 1 else loaded[0]
        )

    env, traj = factory.make(**env_params, **factory_params)
    print(f"[viser] env={type(env).__name__} backend={'MjWarp' if args.mjwarp else 'MJX'} "
          f"obs={env.info.observation_space.shape} act={env.info.action_space.shape}", flush=True)

    # Action function, mirroring each algorithm's own play_policy.
    if algo == "ppo":
        policy = PPOPolicy(agent_conf.network)
        train_state = agent_state.train_state
        if not args.stochastic:
            # A -inf log_std makes the Gaussian collapse to its mean.
            train_state.params["log_std"] = np.ones_like(train_state.params["log_std"]) * -np.inf
        if config.experiment.n_seeds > 1:
            train_state = jax.tree.map(lambda x: x[args.train_state_seed], train_state)

        @jax.jit
        def act(ts, obs, key):
            a, _, _, ts = policy.get_action_and_value(obs, ts, key)
            return a, ts
    else:
        # TD3's actor is deterministic; exploration noise is added only when
        # deterministic=False, and extra_state is unused in that branch, so None is safe.
        train_state = agent_state.actor_state
        if config.experiment.n_seeds > 1:
            train_state = jax.tree.map(lambda x: x[args.train_state_seed], train_state)
        exp = config.experiment
        deterministic = not args.stochastic

        @jax.jit
        def act(ts, obs, key):
            return TD3Jax._select_action(ts, obs, key, exp, None,
                                         deterministic=deterministic)

    vec_env = PPOJax._wrap_env(env, config.experiment)

    rng = jax.random.key(args.seed)
    keys = jax.random.split(rng, args.n_envs + 1)
    rng, env_keys = keys[0], keys[1:]
    obs, env_state = vec_env.reset(env_keys, traj)

    print(f"[viser] serving {args.n_envs} env(s) on http://localhost:{args.port} "
          f"-- forward the port if you are remote. Ctrl-C to stop.", flush=True)

    step = 0
    limit = args.n_steps if args.n_steps > 0 else np.iinfo(np.int32).max
    try:
        while step < limit:
            rng, key = jax.random.split(rng)
            action, train_state = act(train_state, obs, key)
            obs, reward, absorbing, done, info, env_state = vec_env.step(
                env_state, jnp.atleast_2d(action), traj
            )
            vec_env.mjx_render_viser(env_state)
            step += 1
            if step % 100 == 0:
                print(f"[viser] step {step}  mean reward {float(jnp.mean(reward)):+.3f}",
                      flush=True)
    except KeyboardInterrupt:
        print("\n[viser] stopped", flush=True)


if __name__ == "__main__":
    main()
