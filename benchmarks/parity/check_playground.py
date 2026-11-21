"""Checks the LocoMuJoCo Playground task (playground_g1.py) against a recording of Playground's own task.

    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python check_playground.py RECORDING cpu|mjx     (float64 recording)
    python check_playground.py RECORDING mjwarp                                        (float32 recording, GPU)

with LMJ_SRC, LMJ_REAL_SRC and PG_G1_ASSETS set as for bench_parity.py; RECORDING is the OUT of record_playground.py.

Per recorded step: LocoMuJoCo's state is set to Playground's state before the step (data and task state), the same
action is taken, and the simulation is replaced by Playground's own result of that step (qpos, qvel, qacc, sensors,
sites, actuator forces), injected through the step's ``physics_fn`` hook (MJX/MjWarp) or in place of ``mj_step``
(CPU). Everything the task computes from it is compared: both observation groups, the 24 reward terms, the reward,
the termination and the task state after the step. The control target and the compiled model are compared too.

The physics itself is not compared over a free rollout: with 3 solver iterations, the solver's early-stop test
makes the result of a step depend on rounding, so two separately compiled programs diverge at some steps even in
float64. The compiled models are compared field by field instead, and both are stepped by the same mjx.step.
"""
import os, sys
sys.path[:0] = [os.environ["LMJ_SRC"], os.environ["LMJ_REAL_SRC"], os.path.join(os.environ["LMJ_REAL_SRC"], "benchmarks", "parity")]
import numpy as np
import mujoco
import playground_g1 as pg

rec_path, backend = sys.argv[1], sys.argv[2]
R = np.load(rec_path + ".npz")
T = len(R["actions"])
env = pg.make_env(backend, os.environ["PG_G1_ASSETS"], noise_level=0.0, push=False, naconmax=64, njmax=90)
oc = env.obs_container
actor, critic = oc.get_obs_ind_by_group("actor"), oc.get_obs_ind_by_group("critic")
rec_names = [str(k) for k in R["term_names"]]  # alphabetical, as Playground's config iterates
REC_TERMS = R["terms"][:, [rec_names.index(k) for k in pg.REWARD_TERMS]]
scales = np.array([pg.REWARD_SCALES[k] for k in pg.REWARD_TERMS])
POST = ("qpos", "qvel", "qacc", "sensordata", "site_xpos", "site_xmat", "actuator_force", "ctrl", "qacc_warmstart")

# ---- the compiled model: every numeric field and option
pm, lm = mujoco.MjModel.from_binary_path(rec_path + ".mjb"), env._model
diff = []
for obj_p, obj_l, prefix in ((pm, lm, ""), (pm.opt, lm.opt, "opt.")):
    for name in dir(obj_p):
        if name.startswith("_"):
            continue
        try:
            a, b = getattr(obj_p, name), getattr(obj_l, name)
        except Exception:
            continue
        if isinstance(a, (np.ndarray, int, float)) and not np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True):
            diff.append(prefix + name)
expected = {"actuator_forcerange", "names_map", "nnames", "signature"}  # placeholder effort range; renamed root/keyframe
diff = [d for d in diff if d not in expected and not d.startswith("name_")]
print(f"[{backend}] model fields that differ (besides names and the unused force range):", diff or "none")

# ---- the control target: LMJ's control function vs Playground's motor targets, both clamped to the ctrl range
cf, lo, hi = env._control_func, pm.actuator_ctrlrange[:, 0], pm.actuator_ctrlrange[:, 1]
ctrl_err = 0.0
for t in range(T):
    target = cf._nominal_joint_positions + R["actions"][t].astype(np.float64) * cf._scale
    ctrl_err = max(ctrl_err, np.abs(np.clip(target, lo, hi) - np.clip(R["post_ctrl"][t], lo, hi)).max())


def to_states(t, xp):
    f = lambda k: xp.asarray(R["pre_" + k][t])
    return dict(phase=f("phase"), phase_dt=f("phase_dt"), last_act=f("last_act"), last_last_act=f("last_last_act"),
                air=f("feet_air_time"), last_contact=f("last_contact"), peak=f("swing_peak"), command=f("command"),
                step=xp.asarray(R["pre_step"][t], dtype=xp.int32))


def set_task_state(c, st, xp):
    os_ = c.observation_states
    return c.replace(cur_step_in_episode=xp.ones_like(xp.asarray(c.cur_step_in_episode)), observation_states=os_.replace(
        PGState=os_.PGState.replace(phase=st["phase"], phase_dt=st["phase_dt"], last_act=st["last_act"],
                                    last_last_act=st["last_last_act"]),
        PGPrivileged=os_.PGPrivileged.replace(feet_air_time=st["air"], last_contact=st["last_contact"],
                                              swing_peak=st["peak"], skip_update=xp.zeros_like(os_.PGPrivileged.skip_update)),
        PGCommand=os_.PGCommand.replace(command=st["command"], step=st["step"])))


OUTS = []
if backend == "cpu":
    import loco_mujoco.core.mujoco_base as base
    env.reset()
    post = {}

    def injected_step(model, data, nstep=1):
        for k in POST:
            getattr(data, k)[:] = post[k].reshape(getattr(data, k).shape)

    real_step, base.mujoco.mj_step = base.mujoco.mj_step, injected_step
    try:
        for t in range(T):
            d = env._data
            d.qpos[:], d.qvel[:], d.ctrl[:] = R["pre_qpos"][t], R["pre_qvel"][t], R["pre_ctrl"][t]
            env._additional_carry, env._cur_step_in_episode = set_task_state(env._additional_carry, to_states(t, np), np), 0
            post = {k: R["post_" + k][t] for k in POST}
            obs, reward, absorbing, done, _ = env.step(R["actions"][t].astype(np.float64))
            c = env._additional_carry
            OUTS.append(dict(obs=obs, reward=reward, absorbing=absorbing, terms=c.reward_state.terms, os=c.observation_states))
    finally:
        base.mujoco.mj_step = real_step
else:
    import jax
    import jax.numpy as jnp
    if jax.config.jax_enable_x64 and backend == "mjx":
        from mujoco import mjx  # test only: mjx.put_data makes int32 contact indices, which a float64 step makes int64
        env._first_data = mjx.make_data(env._model)
    s0 = jax.jit(jax.vmap(env.mjx_reset))(jax.random.split(jax.random.key(0), 1))

    @jax.jit
    def step(s, a, pre, st, post):
        def physics_fn(sys, data, carry, action):
            return data.replace(**{k: post[k].reshape(getattr(data, k).shape) for k in POST}), carry

        def one(s, a, pre, st, post):
            d = s.data.replace(**pre)
            s = s.replace(data=d, additional_carry=set_task_state(s.additional_carry, st, jnp))
            return env.mjx_step(s, a, auto_reset=False, physics_fn=physics_fn)
        return jax.vmap(one)(s, a, pre, st, post)

    for t in range(T):
        b = lambda x: jax.tree.map(lambda y: jnp.asarray(y)[None], x)
        pre = {k: R["pre_" + k][t] for k in ("qpos", "qvel", "ctrl", "qacc_warmstart")}
        post = {k: R["post_" + k][t] for k in POST}
        s1 = step(s0, b(R["actions"][t]), b(pre), b(to_states(t, np)), b(post))
        c = s1.additional_carry  # MjWarp's shared buffers in s1.data are not batched; take only what is compared
        OUTS.append(dict(obs=np.asarray(s1.observation[0]), reward=float(s1.reward[0]), absorbing=bool(s1.absorbing[0]),
                         terms=np.asarray(c.reward_state.terms[0]),
                         os=jax.tree.map(lambda x: np.asarray(x[0]), c.observation_states)))

errs, done_mismatch = {}, 0


def compare(name, a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    assert a.shape == b.shape, (name, a.shape, b.shape)
    errs[name] = max(errs.get(name, 0.0), float(np.max(np.abs(a - b) / (1.0 + np.abs(b)))) if a.size else 0.0)


for t, o in enumerate(OUTS):
    compare("obs actor = Playground state (103)", o["obs"][actor], R["obs_state"][t])
    compare("obs critic = Playground privileged_state (216)", o["obs"][critic], R["obs_priv"][t])
    compare("reward terms (24)", o["terms"], REC_TERMS[t])
    compare("reward", o["reward"], np.sum(scales * REC_TERMS[t]) * 0.02)
    st, pr, cm = o["os"].PGState, o["os"].PGPrivileged, o["os"].PGCommand
    for name, a, k in (("phase", st.phase, "phase"), ("last_act", st.last_act, "last_act"),
                       ("last_last_act", st.last_last_act, "last_last_act"), ("feet_air_time", pr.feet_air_time, "feet_air_time"),
                       ("swing_peak", pr.swing_peak, "swing_peak"), ("last_contact", pr.last_contact, "last_contact"),
                       ("command", cm.command, "command"), ("command step counter", cm.step, "step")):
        compare("task state after step: " + name, a, R["post_" + k][t])
    done_mismatch += int(bool(o["absorbing"]) != bool(R["done"][t]))
worst_term = max(range(len(pg.REWARD_TERMS)), key=lambda i: max(abs(o["terms"][i] - REC_TERMS[t][i]) / (1 + abs(REC_TERMS[t][i])) for t, o in enumerate(OUTS)))
print(f"[{backend}] {T} steps, {int(R['done'].sum())} terminations in Playground, termination mismatches: {done_mismatch}")
print(f"  max |control target difference| (clamped to ctrlrange) {ctrl_err:.2e}")
for k, v in errs.items():
    print(f"  max rel. error {k:50s} {v:.2e}")
print(f"  largest-error reward term: {pg.REWARD_TERMS[worst_term]}")
