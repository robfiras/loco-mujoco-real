"""
Data-parallel PPO across all GPUs visible to this process, in resumable chunks.

The train function is wrapped in ``jax.pmap`` so N GPUs each step ``num_envs / N``
environments and average their gradients every minibatch. With one GPU it is plain
PPO. Training runs in chunks; each chunk overwrites the checkpoint, and rerunning in
the same output directory resumes from it.

Why pmap and not automatic sharding
-----------------------------------
The env step enters through a ``mujoco_warp`` FFI custom call, which does not
participate in XLA's ``NamedSharding`` propagation -- annotating the arrays and
letting the compiler partition would not work. ``pmap`` gives each device its
own program instance over its own env batch, which the custom call handles
correctly because it is exactly the single-device case repeated N times.

This also sidesteps the MjWarp global contact buffer entirely: ``naconmax`` is
allocated per process-device, so splitting envs across devices *reduces*
per-device contact pressure rather than aggravating it.

Semantics
---------
``experiment.num_envs`` remains the GLOBAL environment count. ``init_agent_conf``
divides it by ``n_devices`` and records the global value as ``num_envs_total``;
``num_updates`` is computed from the global count, so a given
``total_timesteps`` means the same amount of experience regardless of how many
GPUs are used -- only the wall-clock changes.

Gradients are ``pmean``-ed inside the minibatch update, so all replicas hold
identical parameters at every step and the run is mathematically a single
larger-batch PPO run, not N independent ones.

Equivalence caveats, stated rather than hidden:
  * Advantage normalisation stays per-device (``gae - gae.mean()`` inside
    ``_loss_fn``). With num_envs/N still in the hundreds this is a small
    difference from global normalisation, but it is a difference.
  * Reward-normalisation statistics are per-device and not synced.
  * Each replica gets its OWN rng, so the N env batches are genuinely different
    -- that is the point -- but a run is therefore not bitwise reproducible
    against a 1-GPU run of the same seed.

Usage
-----
  # all visible GPUs; each robot folder has its own train.py and configs
  python training/g1/train.py experiment.num_envs=16384

``n_devices`` is detected from ``jax.local_device_count()`` unless
``experiment.n_devices`` is set explicitly.
"""
import json
import os
import secrets
import sys
import time
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import wandb
from dataclasses import fields
from omegaconf import DictConfig, OmegaConf, open_dict
import lmj_real  # registers G1Real / MjxG1Real and their components
from loco_mujoco import TaskFactory
from loco_mujoco.core.trajectory import Trajectory
from loco_mujoco.task_factories.dataset_confs import CustomDatasetConf
from loco_mujoco.algorithms import PPOJax
from loco_mujoco.utils import MetricsHandler

import hydra
from hydra.core.hydra_config import HydraConfig
import traceback

PMAP_AXIS = "devices"


def _device_axis_sharding(n):
    """NamedSharding that splits a leading axis of size n across n devices."""
    mesh = Mesh(np.array(jax.local_devices()[:n]), (PMAP_AXIS,))
    return NamedSharding(mesh, P(PMAP_AXIS))


def _replicate(tree, n, sharding):
    """Stack a pytree n times along a new leading axis, one slice per device.

    Two non-obvious requirements, both learned the hard way:

    * jnp.stack, not jax.device_put_replicated -- the latter is removed in
      recent JAX and raises AttributeError.
    * The result must then be explicitly device_put with a P(PMAP_AXIS)
      sharding. A bare jnp.stack lands as an unsharded array (spec P()), which
      pmap accepts on the FIRST call while it is still tracing, then rejects on
      the second with "Sharding passed to jit does not match the sharding on the
      respective arg" once the executable is cached. That failure mode is
      chunk-2-only, so it does not show up in a single-chunk test.
    """
    stacked = jax.tree.map(lambda x: jnp.stack([x] * n), tree)
    return jax.tree.map(lambda x: jax.device_put(x, sharding), stacked)


def _unreplicate(tree):
    """Take replica 0. Valid only because pmean keeps every replica identical."""
    return jax.tree.map(lambda x: x[0], tree)


def experiment(config: DictConfig):
    try:
        os.environ['XLA_FLAGS'] = ('--xla_gpu_triton_gemm_any=True ')

        result_dir = HydraConfig.get().runtime.output_dir

        n_devices = int(OmegaConf.select(config, "experiment.n_devices") or 0)
        if n_devices <= 0:
            n_devices = jax.local_device_count()
        if n_devices > jax.local_device_count():
            raise ValueError(
                f"experiment.n_devices={n_devices} but only "
                f"{jax.local_device_count()} local device(s) visible: "
                f"{jax.local_devices()}"
            )
        print(f"[dist] {n_devices} device(s): {jax.local_devices()[:n_devices]}",
              flush=True)

        with open_dict(config.experiment):
            config.experiment.n_devices = n_devices
            # Read by PPOJax._train_fn to decide whether to emit collectives.
            config.experiment.pmap_axis_name = PMAP_AXIS if n_devices > 1 else None

        wandb.login()
        config_dict = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
        # Stable run id across restarts, so a resumed run continues the same wandb curve.
        # The id is persisted next to the checkpoint and reused with resume="allow".
        # Logged steps stay monotonic because global_step_offset is restored below,
        # which is what wandb requires of a resumed run.
        wandb_id_file = os.path.join(result_dir, "wandb_run_id.txt")
        if os.path.exists(wandb_id_file):
            with open(wandb_id_file) as f:
                wandb_id = f.read().strip()
            print(f"[dist] resuming wandb run {wandb_id}", flush=True)
        else:
            wandb_id = secrets.token_hex(4)  # wandb accepts any id; its own helper moved between versions
            os.makedirs(result_dir, exist_ok=True)
            with open(wandb_id_file, "w") as f:
                f.write(wandb_id)
        run = wandb.init(project=config.wandb.project, config=config_dict,
                         id=wandb_id, resume="allow")

        factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
        factory_params = OmegaConf.to_container(
            config.experiment.task_factory.params, resolve=True
        )

        custom_conf = factory_params.get("custom_dataset_conf")
        if isinstance(custom_conf, dict) and "paths" in custom_conf:
            paths = list(custom_conf["paths"])
            if not paths:
                raise ValueError("custom_dataset_conf.paths is empty.")
            print(f"[dist] loading {len(paths)} trajectories:", flush=True)
            for path in paths:
                print(f"  - {path}", flush=True)
            loaded = [Trajectory.load(path) for path in paths]
            factory_params["custom_dataset_conf"] = CustomDatasetConf(
                traj=Trajectory.concatenate(loaded) if len(loaded) > 1 else loaded[0]
            )

        # `robot` under env_params is a shared-constants namespace referenced by
        # interpolation, not a make() kwarg. Mujoco.__init__ swallows unknown kwargs
        # into **viewer_params and only fails much later, so strip it here.
        env_params = {k: v for k, v in config.experiment.env_params.items() if k != "robot"}
        env, traj = factory.make(**env_params, **factory_params)

        total_timesteps = int(config.experiment.total_timesteps)
        timesteps_per_chunk = int(config.experiment.timesteps_per_chunk)
        num_envs_global = int(config.experiment.num_envs)

        # Per-chunk total so init_agent_conf sizes num_updates for one chunk.
        with open_dict(config.experiment):
            config.experiment.total_timesteps = timesteps_per_chunk

        # NOTE: this rewrites config.experiment.num_envs to the PER-DEVICE count
        # and records num_envs_total. Everything after this point that needs the
        # global figure must use num_envs_global, captured above.
        agent_conf = PPOJax.init_agent_conf(env, config)
        per_device_envs = int(agent_conf.config.experiment.num_envs)
        print(f"[dist] num_envs global={num_envs_global} "
              f"per-device={per_device_envs}", flush=True)

        updates_per_chunk = int(agent_conf.config.experiment.num_updates)
        if updates_per_chunk < 1:
            raise ValueError(
                f"timesteps_per_chunk={timesteps_per_chunk} is smaller than one "
                f"update ({config.experiment.num_steps} x {num_envs_global} = "
                f"{int(config.experiment.num_steps) * num_envs_global} steps). "
                "Raise timesteps_per_chunk or lower num_envs."
            )
        total_updates = total_timesteps // int(config.experiment.num_steps) // num_envs_global
        n_chunks = max(1, total_updates // updates_per_chunk)
        mh = MetricsHandler(config, env) if config.experiment.validation.active else None

        if int(config.experiment.n_seeds) > 1:
            raise ValueError(
                "n_seeds > 1 is not supported by the distributed driver: the "
                "device axis is already used for data parallelism. Launch "
                "separate jobs per seed instead."
            )

        # One independent rng per device, so the replicas explore different env
        # batches. Params stay identical regardless, because gradients are pmean-ed.
        base_rng = jax.random.PRNGKey(int(OmegaConf.select(config, "experiment.seed") or 0))
        rng_init, rng_train = jax.random.split(base_rng)
        device_rngs = jax.random.split(rng_train, n_devices)
        # Same sharding requirement as _replicate: the rng array must be
        # split across the device axis, or chunk 2 hits a cached-executable
        # sharding mismatch.
        dev_sharding = _device_axis_sharding(n_devices) if n_devices > 1 else None
        if dev_sharding is not None:
            device_rngs = jax.device_put(device_rngs, dev_sharding)

        # --- restart / preemption handling -------------------------------------
        # A preemptible job can be killed and requeued at any time, so this has
        # to be idempotent: rerunning the same command must CONTINUE rather than restart.
        # Three things are needed, and all three fail silently if missing --
        #   * pick up result_dir's own checkpoint when no explicit resume_path is given,
        #     otherwise a requeue trains from scratch;
        #   * skip the chunks already done, otherwise a requeue runs the FULL chunk count
        #     again and overshoots total_timesteps;
        #   * restore global_step_offset, otherwise the wandb x-axis restarts at 0.
        # progress.json carries the last two. It is written after the checkpoint each
        # chunk, so if the job dies between the two the worst case is one chunk redone.
        progress_file = os.path.join(result_dir, "progress.json")
        chunks_done, global_step_offset = 0, 0
        if os.path.exists(progress_file):
            try:
                with open(progress_file) as f:
                    prog = json.load(f)
                chunks_done = int(prog.get("chunks_done", 0))
                global_step_offset = int(prog.get("global_step", 0))
            except (ValueError, OSError) as exc:
                print(f"[dist] WARNING: unreadable {progress_file} ({exc}); "
                      "starting chunk count from 0", flush=True)

        resume_path = OmegaConf.select(config, "resume_path")
        own_ckpt = os.path.join(result_dir, "PPOJax_saved.pkl")
        if not (resume_path and os.path.exists(str(resume_path))) and os.path.exists(own_ckpt):
            resume_path = own_ckpt
            print(f"[dist] restarting: found own checkpoint after {chunks_done} chunk(s), "
                  f"global_step={global_step_offset:,}", flush=True)

        if resume_path and os.path.exists(str(resume_path)):
            print(f"[dist] loading expert from {resume_path}", flush=True)
            _, agent_state = PPOJax.load_agent(resume_path)
        else:
            print("[dist] resume_path not set / missing; initialising fresh agent",
                  flush=True)
            agent_state = PPOJax.init_agent_state(env, agent_conf, rng_init)
        print(f"[dist] running {n_chunks} chunks x {timesteps_per_chunk:,} steps "
              f"= {total_timesteps:,} env steps across {n_devices} device(s)",
              flush=True)

        train_fn = PPOJax.build_train_fn(env, agent_conf, mh=mh)
        if n_devices > 1:
            # in_axes: rng per device, agent_state per device (replicated, kept
            # identical by pmean), traj broadcast. traj is read-only reference
            # data, so each device holding a copy costs memory but not correctness.
            train_fn = jax.pmap(train_fn, axis_name=PMAP_AXIS,
                                in_axes=(0, 0, None))
        else:
            train_fn = jax.jit(train_fn)

        save_path = None
        # Steps actually simulated per chunk, GLOBAL across devices. Derived from
        # the update count rather than timesteps_per_chunk because num_updates is
        # floor-divided, so the real figure is usually a little lower.
        steps_per_chunk = updates_per_chunk * int(config.experiment.num_steps) * num_envs_global
        chunk_times = []
        if chunks_done >= n_chunks:
            print(f"[dist] already at {chunks_done}/{n_chunks} chunks; nothing to do",
                  flush=True)
        for chunk in range(chunks_done, n_chunks):
            print(f"[dist] chunk {chunk + 1}/{n_chunks}", flush=True)

            # Fresh env reset each chunk so env state aligns with the trajectory.
            if agent_state is not None:
                agent_state = agent_state.replace(env_state=None, last_obs=None)

            t0 = time.perf_counter()
            if n_devices > 1:
                out = train_fn(device_rngs,
                               _replicate(agent_state, n_devices, dev_sharding),
                               traj)
                # Replicas are identical after pmean, so replica 0 is the model.
                agent_state = _unreplicate(out["agent_state"])
                training_metrics = _unreplicate(out["training_metrics"]) \
                    if not config.experiment.debug else None
            else:
                out = train_fn(device_rngs[0], agent_state, traj)
                agent_state = out["agent_state"]
                training_metrics = out["training_metrics"] \
                    if not config.experiment.debug else None
            # JAX is async: without block_until_ready the timer would measure
            # dispatch, not execution. Blocking on the agent state is enough --
            # it depends on the whole chunk's computation.
            jax.block_until_ready(jax.tree.leaves(agent_state))
            dt = time.perf_counter() - t0
            chunk_times.append(dt)
            print(f"[bench] chunk {chunk + 1} wall={dt:.1f}s "
                  f"steps={steps_per_chunk:,} "
                  f"throughput={steps_per_chunk / dt:,.0f} steps/s "
                  f"({'includes compile' if chunk == 0 else 'steady state'})",
                  flush=True)

            if training_metrics is not None:
                training_metrics = jax.tree.map(
                    lambda x: jnp.mean(jnp.atleast_2d(x), axis=0), training_metrics
                )
                for i in range(len(training_metrics.mean_episode_return)):
                    # max_timestep counts PER-DEVICE env steps (see
                    # ppo_jax._train_fn, which scales by the per-device
                    # num_envs), so scale back up to global experience or the
                    # x-axis would under-report by a factor of n_devices and
                    # these curves would not line up with the 1-GPU runs.
                    step = global_step_offset + int(training_metrics.max_timestep[i]) * n_devices
                    metrics_to_log = {
                        f"Training/{field.name}": getattr(training_metrics, field.name)[i]
                        for field in fields(training_metrics) if field.name != "max_timestep"
                    }
                    metrics_to_log["Training/chunk"] = chunk
                    metrics_to_log["Training/n_devices"] = n_devices
                    run.log(metrics_to_log, step=step)
                global_step_offset += int(training_metrics.max_timestep[-1]) * n_devices

            save_path = PPOJax.save_agent(result_dir, agent_conf, agent_state)
            # Written AFTER the checkpoint, and via a temp file + rename so a kill
            # mid-write cannot leave a truncated progress.json that reads as chunk 0.
            tmp = progress_file + ".tmp"
            with open(tmp, "w") as f:
                json.dump({"chunks_done": chunk + 1, "global_step": global_step_offset,
                           "n_chunks": n_chunks}, f)
            os.replace(tmp, progress_file)
            print(f"[dist] chunk {chunk + 1}: overwrote checkpoint -> {save_path} "
                  f"(global_step={global_step_offset:,})", flush=True)

        # Chunk 1 carries MJX/warp compilation, so the steady-state number is
        # the mean over the rest. With only one chunk there is nothing to
        # report that is not compile-dominated.
        if len(chunk_times) > 1:
            steady = sum(chunk_times[1:]) / len(chunk_times[1:])
            print(f"[bench] SUMMARY n_devices={n_devices} "
                  f"num_envs_global={num_envs_global} "
                  f"per_device={per_device_envs} "
                  f"compile_chunk={chunk_times[0]:.1f}s "
                  f"steady_chunk={steady:.1f}s "
                  f"throughput={steps_per_chunk / steady:,.0f} steps/s",
                  flush=True)
            run.summary["bench/steady_steps_per_s"] = steps_per_chunk / steady
            run.summary["bench/n_devices"] = n_devices

        if save_path is not None:
            run.config.update({"agent_save_path": str(save_path)})

        wandb.finish()

    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise



