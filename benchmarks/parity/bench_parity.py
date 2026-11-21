"""Throughput of LocoMuJoCo's three backends on the matched tasks: MuJoCo Playground's G1 joystick task
(playground_g1.py) and mjlab's G1 velocity task (mjlab_g1.py), each on the other framework's own model.

GPU backends (mjwarp, mjx): per repeat, reset every env, 50 warm-up env steps, time `steps` env steps (env SPS).
CPU backend: one environment per process, `procs` processes started together behind a barrier; per repeat, reset,
50 warm-up steps, time `cpu_steps` steps; throughput is procs * cpu_steps / slowest process. Repeat 0 is discarded.

Zero actions, as in the other frameworks' benchmark scripts. The sources are taken from LMJ_SRC and LMJ_REAL_SRC, the
models from PG_G1_ASSETS (Playground's model files) and MJLAB_G1_EXPORT (export_models.py). Prints one JSON line
tagged RESULT.
"""
import argparse
import importlib.metadata as md
import json
import os
import sys
import time

LMJ_SRC, LMJ_REAL_SRC = os.environ["LMJ_SRC"], os.environ["LMJ_REAL_SRC"]
sys.path[:0] = [LMJ_SRC, LMJ_REAL_SRC, os.path.dirname(os.path.abspath(__file__))]

p = argparse.ArgumentParser()
p.add_argument("--backend", choices=["mjwarp", "mjx", "cpu"], required=True)
p.add_argument("--num-envs", type=int, required=True, help="parallel envs (GPU) or processes (CPU)")
p.add_argument("--task", choices=["playground", "mjlab"], required=True)
p.add_argument("--njmax", type=int, required=True, help="mjwarp: constraints per world")
p.add_argument("--ncon-per-env", type=int, required=True, help="mjwarp: contacts per env")
p.add_argument("--warmup", type=int, default=50)
p.add_argument("--steps", type=int, default=200)
p.add_argument("--cpu-steps", type=int, default=2000)
p.add_argument("--repeats", type=int, default=5)
p.add_argument("--contacts", choices=["full", "reduced"], default="full",
               help="mjlab task: mjlab's full self-collision or its feet-only collision set")
p.add_argument("--memory", action="store_true",
               help="memory mode: one jitted env step in a Python loop (warmup + steps), no timing repeats")
p.add_argument("--donate-state", action="store_true",
               help="memory mode: donate the state to the jitted step (its buffers are reused for the next state, "
                    "as in a compiled training loop)")
p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
args = p.parse_args()
N = args.num_envs
if args.backend == "cpu":
    os.environ["JAX_PLATFORMS"] = "cpu"  # keep the CPU processes off the GPU


def make_env(backend, n):
    """The matched task on ``backend``; on MjWarp with the buffers given by --ncon-per-env and --njmax."""
    buffers = dict(naconmax=args.ncon_per_env * n, njmax=args.njmax) if backend == "mjwarp" else {}
    if args.task == "playground":
        import playground_g1
        assert args.contacts == "full", "Playground's G1 has its own five contact pairs"
        return playground_g1.make_env(backend, os.environ["PG_G1_ASSETS"], **buffers)
    import mjlab_g1
    return mjlab_g1.make_env(backend, os.path.join(os.environ["MJLAB_G1_EXPORT"], args.contacts), **buffers)


def describe(env):
    import mujoco
    m, o = env._model, env._model.opt
    sub = env._n_intermediate_steps * env._n_substeps
    assert sub == round(env.dt / o.timestep), (sub, env.dt, o.timestep)
    coll = [i for i in range(m.ngeom) if m.geom_contype[i] or m.geom_conaffinity[i]]
    return dict(dt=float(env.dt), timestep=float(o.timestep), substeps=int(sub), contacts=args.contacts,
                colliding_geoms=len(coll),
                integrator=mujoco.mjtIntegrator(o.integrator).name, cone=mujoco.mjtCone(o.cone).name,
                solver=mujoco.mjtSolver(o.solver).name, iterations=int(o.iterations), ls_iterations=int(o.ls_iterations),
                ccd_iterations=int(o.ccd_iterations), ngeom=int(m.ngeom), nu=int(m.nu))


def versions():
    import loco_mujoco, lmj_real
    v = {k: md.version(k) for k in ("jax", "mujoco", "mujoco-mjx", "mujoco-warp", "warp-lang")}
    v.update(loco_mujoco_file=loco_mujoco.__file__, lmj_real_file=lmj_real.__file__)
    return v


# ----------------------------------------------------------------------------------------------------------- CPU
def cpu_process(barrier, queue):
    try:
        import numpy as np
        env = make_env("cpu", 1)
        zero = np.zeros(env.info.action_space.shape)
        times, resets, z = [], [], []
        for r in range(args.repeats + 1):
            env.reset()
            for _ in range(args.warmup):
                if any(env.step(zero)[2:4]):
                    env.reset()
            barrier.wait()
            n_reset, t = 0, time.perf_counter()
            for _ in range(args.cpu_steps):
                _, _, absorbing, done, _ = env.step(zero)
                if absorbing or done:
                    env.reset(); n_reset += 1
            times.append(time.perf_counter() - t); resets.append(n_reset / args.cpu_steps); z.append(float(env._data.qpos[2]))
        queue.put(dict(times=times, resets=resets, z=z, info=describe(env)))
    except Exception as e:  # noqa: BLE001
        barrier.abort()
        queue.put(dict(error=f"{type(e).__name__}: {e}"))


def run_cpu():
    import multiprocessing as mp
    import numpy as np
    ctx = mp.get_context("spawn")
    barrier, queue = ctx.Barrier(N), ctx.Queue()
    procs = [ctx.Process(target=cpu_process, args=(barrier, queue)) for _ in range(N)]
    for pr in procs:
        pr.start()
    out = [queue.get(timeout=3600) for _ in procs]
    for pr in procs:
        pr.join()
    errors = [o["error"] for o in out if "error" in o]
    if errors:
        raise RuntimeError(errors[0])
    wall = np.max(np.array([o["times"] for o in out]), axis=0)[1:]  # slowest process per repeat, cold repeat dropped
    sps = N * args.cpu_steps / wall
    return dict(out[0]["info"], env_sps_mean=float(sps.mean()), env_sps_std=float(sps.std()), env_sps_all=sps.round(1).tolist(),
                termination_per_step=float(np.mean([o["resets"][1:] for o in out])),
                root_z_mean=float(np.mean([o["z"][1:] for o in out])), timed_steps=args.cpu_steps)


# ----------------------------------------------------------------------------------------------------------- GPU
def run_gpu():
    import jax
    import jax.numpy as jnp
    import numpy as np
    from mujoco import mjx
    env = make_env(args.backend, N)
    info = describe(env)
    if args.memory:
        act_dim = int(env.info.action_space.shape[0])
        reset_all = jax.jit(jax.vmap(env.mjx_reset))
        step = jax.jit(jax.vmap(env.mjx_step), donate_argnums=0 if args.donate_state else ())
        zero = jnp.zeros((N, act_dim))
        s = reset_all(jax.random.split(jax.random.key(0), N))
        for _ in range(args.warmup + args.steps):
            s = step(s, zero)
        jax.block_until_ready(s)
        stats = jax.devices()[0].memory_stats() or {}
        info.update(memory_mode=True, donate_state=args.donate_state, steps_run=args.warmup + args.steps,
                    jax_peak_bytes=stats.get("peak_bytes_in_use"), jax_bytes_in_use=stats.get("bytes_in_use"),
                    root_z_mean=float(np.asarray(s.data.qpos[:, 2]).mean()), termination_per_step=float("nan"))
        return info
    sub = info["substeps"]
    act_dim = int(env.info.action_space.shape[0])
    if args.backend == "mjwarp":
        info.update(naconmax_alloc=int(env._first_data._impl.naconmax), njmax_alloc=int(env._first_data._impl.njmax))
    key = jax.random.key(0)
    reset_all = jax.jit(jax.vmap(env.mjx_reset))
    zero = jnp.zeros((N, act_dim))

    def env_step(state, _):
        state = jax.vmap(env.mjx_step)(state, zero)
        return state, state.done.mean()

    env_rollout = jax.jit(lambda s: jax.lax.scan(env_step, s, None, length=args.steps))
    warmup_rollout = jax.jit(lambda s: jax.lax.scan(env_step, s, None, length=args.warmup))

    block = jax.block_until_ready
    t = time.perf_counter()
    s = block(warmup_rollout(block(reset_all(jax.random.split(key, N))))[0])
    s = block(env_rollout(s)[0])
    info["compile_s"] = time.perf_counter() - t

    env_sps, term, z, bad_quat, nan, nacon, nefc = ([] for _ in range(7))
    for r in range(args.repeats + 1):  # repeat 0 is discarded (cold start)
        s = block(reset_all(jax.random.split(jax.random.fold_in(key, r), N)))
        s = block(warmup_rollout(s)[0])
        t = time.perf_counter()
        s, done = env_rollout(s)
        block((s, done))
        env_sps.append(args.steps * N / (time.perf_counter() - t))
        q = np.asarray(s.data.qpos)
        term.append(float(np.asarray(done).mean())); z.append(float(q[:, 2].mean()))
        bad_quat.append(float((np.abs(np.linalg.norm(q[:, 3:7], axis=1) - 1) > 1e-3).mean()))
        nan.append(int(np.isnan(q).any(axis=1).sum()))
        if args.backend == "mjwarp":
            nacon.append(int(np.asarray(s.data._impl.nacon).ravel()[0])); nefc.append(int(np.asarray(s.data._impl.nefc).max()))
    env_sps = np.array(env_sps[1:])
    term, z, bad_quat, nan, nacon, nefc = (x[1:] for x in (term, z, bad_quat, nan, nacon, nefc))
    info.update(env_sps_mean=float(env_sps.mean()), env_sps_std=float(env_sps.std()), env_sps_all=env_sps.round(1).tolist(),
                termination_per_step=float(np.mean(term)), root_z_mean=float(np.mean(z)),
                frac_bad_quat=float(np.mean(bad_quat)), envs_with_nan=int(max(nan)), jax_backend=jax.default_backend())
    if args.backend == "mjwarp":
        info.update(nacon_max=max(nacon), nefc_max_per_world=max(nefc),
                    contact_overflow=bool(max(nacon) >= info["naconmax_alloc"]),
                    constraint_overflow=bool(max(nefc) >= info["njmax_alloc"]))
    return info


if __name__ == "__main__":
    res = run_cpu() if args.backend == "cpu" else run_gpu()
    res.update(task=args.task, backend=args.backend, num_envs=N, lmj_src=LMJ_SRC, lmj_real_src=LMJ_REAL_SRC, versions=versions())
    print("RESULT " + json.dumps(res), flush=True)
