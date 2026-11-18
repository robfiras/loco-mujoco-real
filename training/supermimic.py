"""
SuperMimic training in resumable chunks.

Training runs in chunks of ``experiment.iterations_per_chunk`` iterations. Every chunk overwrites
``SuperMimicJax_saved.pkl`` in the output directory and appends its log rows to ``metrics.csv``, and
rerunning in the same output directory resumes from that checkpoint. The checkpoint with the highest
training return so far is kept as ``SuperMimicJax_best.pkl``, and ``experiment.keep_every_steps`` keeps
numbered copies along the way.

The environment is the robot's tracking environment, the one PPO trains on, with two changes that
``supermimic.yaml`` states: the observations SuperMimic reads in addition to the environment's own are put
in front of the observation spec, and a few environment settings are overridden. The merged, resolved
config is written to ``config.yaml`` in the output directory; it is also stored in every checkpoint, which
is what the deployment tools read.

Usage
-----
  python training/x2/train_supermimic.py
  python training/g1/train_supermimic.py experiment.seed=1 hydra.run.dir=outputs/g1_s1

There is no wandb logging: progress is printed and written to ``metrics.csv``.
"""
import csv
import os
import shutil
import time
from pathlib import Path

import jax
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict

import lmj_real  # noqa: F401  registers the real-robot environments and their components
from loco_mujoco import TaskFactory
from loco_mujoco.algorithms import SuperMimicJax
from loco_mujoco.core.trajectory import Trajectory
from loco_mujoco.task_factories.dataset_confs import CustomDatasetConf

LOG_COLUMNS = ["max_timestep", "mean_episode_return", "mean_episode_length", "mean_wm_nse_1step",
               "mean_wm_nse_kstep", "mean_term_acc", "mean_imag_return", "mean_imag_survival",
               "mean_imag_alive", "mean_critic_loss", "mean_actor_grad_norm", "num_updates"]


def _resolve(config: DictConfig) -> DictConfig:
    """Resolves the config and moves the training-loop settings out of the algorithm's view."""
    OmegaConf.resolve(config)
    exp = config.experiment
    with open_dict(exp):
        # the observations SuperMimic needs go in front of the environment's own
        exp.env_params.observation_spec = list(exp.pop("goal_observation_spec")) + list(exp.env_params.observation_spec)
        exp.iterations_per_call = int(exp.pop("iterations_per_chunk"))
    return config


def _make_env(exp: DictConfig):
    factory = TaskFactory.get_factory_cls(exp.task_factory.name)
    factory_params = OmegaConf.to_container(exp.task_factory.params, resolve=True)
    custom_conf = factory_params.get("custom_dataset_conf")
    if isinstance(custom_conf, dict) and "paths" in custom_conf:
        loaded = [Trajectory.load(path) for path in custom_conf["paths"]]
        if not loaded:
            raise ValueError("custom_dataset_conf.paths is empty.")
        factory_params["custom_dataset_conf"] = CustomDatasetConf(
            traj=Trajectory.concatenate(loaded) if len(loaded) > 1 else loaded[0])
    # `robot` under env_params is a namespace of shared constants, not an argument of the environment
    env_params = {k: v for k, v in exp.env_params.items() if k != "robot"}
    return factory.make(**env_params, **factory_params)


def experiment(config: DictConfig):
    out = Path(HydraConfig.get().runtime.output_dir)
    max_iterations = int(config.experiment.max_iterations)
    keep_every = int(float(config.experiment.keep_every_steps))
    with open_dict(config.experiment):
        del config.experiment["max_iterations"], config.experiment["keep_every_steps"]
    config = _resolve(config)
    exp = config.experiment
    chunk = int(exp.iterations_per_call)
    if not (out / "config.yaml").exists():
        OmegaConf.save(config, out / "config.yaml")

    env, traj = _make_env(exp)
    agent_conf = SuperMimicJax.init_agent_conf(env, config)
    e = agent_conf.config.experiment
    print(f"obs {e.obs_dim} act {e.action_dim} iterations {e.num_iterations} (stopping at {max_iterations}) "
          f"chunk {chunk} devices {jax.devices()}", flush=True)

    rng = jax.random.PRNGKey(int(exp.get("seed", 0)))
    checkpoint = out / "SuperMimicJax_saved.pkl"
    if checkpoint.exists():
        _, state = SuperMimicJax.load_agent(checkpoint)
        print(f"resumed at iteration {int(state.iteration)} ({int(state.num_updates)} updates)", flush=True)
    else:
        state = SuperMimicJax.init_agent_state(env, agent_conf, rng)
    rng = jax.random.fold_in(rng, int(state.iteration))  # a resumed run continues with fresh randomness
    if state.env_state is None:
        rng, init_key = jax.random.split(rng)
        state = jax.jit(SuperMimicJax.build_init_fn(env, agent_conf))(init_key, state, traj)
    # donating the state lets the output reuse its buffers, which are several GB
    train = jax.jit(SuperMimicJax.build_train_fn(env, agent_conf), donate_argnums=(1,))

    def save(agent_state):
        tmp = out / "tmp"
        tmp.mkdir(exist_ok=True)
        os.replace(SuperMimicJax.save_agent(tmp, agent_conf, agent_state), checkpoint)
        step = int(agent_state.iteration) * int(e.num_envs)
        if keep_every and step // keep_every > (step - chunk * int(e.num_envs)) // keep_every:
            shutil.copy(checkpoint, out / f"SuperMimicJax_{step // 1_000_000:05d}M.pkl")

    csv_path = out / "metrics.csv"
    new_csv = not csv_path.exists()
    best_file = out / "best_return.txt"
    while int(state.iteration) < min(int(e.num_iterations), max_iterations):
        t0 = time.time()
        rng, key = jax.random.split(rng)
        result = train(key, state, traj)
        jax.block_until_ready(result["training_metrics"])
        state, metrics = result["agent_state"], result["training_metrics"]
        seconds = time.time() - t0

        rows = [[float(np.asarray(getattr(metrics, c))[i]) for c in LOG_COLUMNS]
                for i in range(len(metrics.max_timestep))]
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if new_csv:
                writer.writerow([c.replace("mean_", "") for c in LOG_COLUMNS])
                new_csv = False
            writer.writerows(rows)
        last = dict(zip(LOG_COLUMNS, rows[-1]))
        finite = all(bool(jax.numpy.all(jax.numpy.isfinite(x)))
                     for name in ("actor_state", "critic_state", "wm_state")
                     for x in jax.tree.leaves(getattr(state, name).params))
        print(f"it {int(state.iteration):7d}  steps {last['max_timestep'] / 1e6:8.1f}M  "
              f"return {last['mean_episode_return']:7.2f}  length {last['mean_episode_length']:6.1f}  "
              f"model error {last['mean_wm_nse_1step']:.3f}  imagined survival {last['mean_imag_survival']:.3f}  "
              f"{chunk * int(e.num_envs) / seconds:,.0f} steps/s  {seconds:.0f}s", flush=True)

        save(state)
        best = float(best_file.read_text()) if best_file.exists() else -np.inf
        if last["mean_episode_return"] > best:
            shutil.copy(checkpoint, out / "SuperMimicJax_best.pkl")
            best_file.write_text(str(last["mean_episode_return"]))
        if not finite:
            raise RuntimeError("Non-finite network parameters; stopping.")
        # the optimiser drops non-finite gradients, so finite parameters alone do not show the actor learns
        grad_norms = np.asarray(metrics.mean_actor_grad_norm)
        if int(last["num_updates"]) > 0 and not np.isfinite(grad_norms).any():
            raise RuntimeError("The actor gradient was non-finite in every log row of this chunk; stopping.")
    print(f"finished: {checkpoint}", flush=True)
