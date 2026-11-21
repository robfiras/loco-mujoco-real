"""Checks the LocoMuJoCo mjlab task (mjlab_g1.py, CPU backend) against a recording of mjlab's own task.

    JAX_PLATFORMS=cpu python check_mjlab.py RECORDING full|reduced cpu|mjx
    python check_mjlab.py RECORDING full|reduced mjwarp                       (GPU)

with LMJ_SRC, LMJ_REAL_SRC and MJLAB_G1_EXPORT set as for bench_parity.py; RECORDING is the OUT of record_mjlab.py.

Per recorded step, LocoMuJoCo's state is set to mjlab's state before the step (data and task state), the same action
is taken, and the physics is replaced by mjlab's own: each of the 4 substeps writes mjlab's data after that substep
(CPU: in place of ``mj_step``; MJX/MjWarp: through the step's ``physics_fn`` hook, which runs LocoMuJoCo's control
function before each substep as its own loop does), and the refresh before the observations writes mjlab's data after
its ``forward``. On MJX, whose model has no contact sensors, the contact readings are decoded from mjlab's sensors
(the MJX computation of the readings is checked against MuJoCo's own sensors by check_mjx_contacts.py). Everything the task
computes from it is compared: the control targets, both observation groups, the 13 weighted reward terms, the reward,
the termination, and the task state after the step (air and contact times, force history, swing peak heights,
command). A command resampled during the step takes mjlab's draw. Steps that end in mjlab's reset are compared up to
the reset (reward, termination); the observation of the reset state is not compared.

mjlab caches its contact readings per step, so its critic reads the contacts of the last substep unless some
environment was reset in the step; LocoMuJoCo reads the refreshed state. The 8 contact entries of the critic are
therefore checked against the refreshed state, and their agreement with mjlab's cached values is reported separately.
"""
import os, sys
sys.path[:0] = [os.environ["LMJ_SRC"], os.environ["LMJ_REAL_SRC"], os.path.dirname(os.path.abspath(__file__))]
import numpy as np
import mujoco
import mjlab_g1 as ml
from mjlab_contacts import ContactReadings

rec_path, contacts, backend = sys.argv[1], sys.argv[2], sys.argv[3]
R = np.load(rec_path + ".npz")
T = len(R["actions"])
export = os.path.join(os.environ["MJLAB_G1_EXPORT"], contacts)
env = ml.make_env(backend, export, noise=False, naconmax=48, njmax=300)
sh, m = env._mjlab_task, env._mjlab_task.m
oc = env.obs_container
actor, critic = oc.get_obs_ind_by_group("actor"), oc.get_obs_ind_by_group("critic")
names = [str(n) for n in R["term_names"]]
perm = [names.index(k) for k in ml.REWARD_TERMS]
assert np.all(R["terms"][:, names.index("air_time")] == 0)  # weight 0: skipped by mjlab, left out here

# ---- the compiled model against mjlab's runtime model
rm, lm = mujoco.MjModel.from_binary_path(os.path.join(export, "runtime_model.mjb")), env._model
diff = [n for n in dir(rm) if not n.startswith("_") and isinstance(getattr(rm, n, None), np.ndarray)
        and getattr(rm, n).dtype != object and not np.array_equal(getattr(rm, n), getattr(lm, n), equal_nan=True)]
diff += ["opt." + n for n in dir(rm.opt) if not n.startswith("_")
         and not np.array_equal(np.asarray(getattr(rm.opt, n)), np.asarray(getattr(lm.opt, n)))]
diff = [d for d in diff if not d.startswith("name") and d not in ("signature",)]
if backend == "mjx":  # its model has the contact sensors removed
    diff = [d for d in diff if not d.startswith("sensor_") and d not in ("nsensor", "nsensordata")]
print(f"[{contacts}, {backend}] model fields that differ from mjlab's runtime model (besides names):", diff or "none")

FIELDS = ("qpos", "qvel", "ctrl", "qacc_warmstart", "sensordata", "xpos", "xquat", "cvel", "subtree_com", "site_xpos",
          "site_xmat")


def write(d, prefix, t):
    for f in FIELDS:
        getattr(d, f)[:] = R[prefix + f][t].reshape(getattr(d, f).shape)


FIELDS = ("qpos", "qvel", "ctrl", "qacc_warmstart", "sensordata", "xpos", "xquat", "cvel", "subtree_com", "site_xpos",
          "site_xmat")
# mjlab's sensordata entries in the order of this model's sensors (MJX: without the contact sensors)
sd_index = np.concatenate([np.arange(rm.sensor_adr[rm.sensor(lm.sensor(i).name).id],
                                     rm.sensor_adr[rm.sensor(lm.sensor(i).name).id] + lm.sensor_dim[i])
                           for i in range(lm.nsensor)])
native = ContactReadings(rm, native=True)  # mjlab's contact sensors, read from its sensordata
read_native = ContactReadings.__call__


def recorded(prefix, t):
    out = {f: R[prefix + f][t] for f in FIELDS}
    out["sensordata"] = out["sensordata"][sd_index]
    return out


def pre_task_state(c, t, xp):
    pre = lambda k: R["pretask_" + k][t]
    f = lambda k: xp.asarray(pre(k), dtype=float)
    os_ = c.observation_states
    return c.replace(
        global_step=xp.zeros_like(xp.asarray(c.global_step)), cur_step_in_episode=xp.asarray(int(pre("ep_len")) + 1),
        control_func_state=c.control_func_state.replace(
            substep=xp.zeros_like(xp.asarray(c.control_func_state.substep)), current_air_time=f("cur_air"),
            last_air_time=f("last_air"), current_contact_time=f("cur_contact"), last_contact_time=f("last_contact"),
            force_history=f("force_hist")),
        reward_state=c.reward_state.replace(peak_heights=f("peak"), prev_action=f("action"), prev_prev_action=f("prev_action")),
        observation_states=os_.replace(
            MLActor=os_.MLActor.replace(heights=f("heights")),
            MLTwist=os_.MLTwist.replace(command=f("command"), time_left=f("time_left"), heading_target=f("heading_target"),
                                        is_heading=xp.asarray(bool(pre("is_heading"))),
                                        is_standing=xp.asarray(bool(pre("is_standing"))))))


def draw(t):
    """mjlab's command draw of step t (its command state after the step)."""
    p = lambda k: R["posttask_" + k][t]
    return dict(command=p("command").astype(float), time_left=p("time_left").astype(float),
                heading_target=p("heading_target").astype(float), is_heading=p("is_heading").astype(bool),
                is_standing=p("is_standing").astype(bool))


def injected_resample(bk, st, carry):  # mjlab's draw of this step (set per step, or passed into the jitted step)
    return st.replace(**{k: bk.asarray(v) for k, v in STEP["draw"].items()}), carry


STEP = dict(t=0, draw=draw(0))
ml.resample_twist = injected_resample
OUTS = []
if backend == "cpu":
    import loco_mujoco.core.mujoco_base as base
    fresh = mujoco.MjData(lm)

    def write(d, values):
        for k, v in values.items():
            getattr(d, k)[:] = v.reshape(getattr(d, k).shape)

    def injected_step(model, data, nstep=1):
        STEP["ctrl"].append(data.ctrl.copy())
        write(data, recorded(f"sub{STEP['k']}_", STEP["t"]))
        STEP["k"] += 1

    def injected_forward(self, env_, model, data, bk):
        write(fresh, recorded("fresh_", STEP["t"]))
        return fresh

    base.mujoco.mj_step, ml.MLShared.forward = injected_step, injected_forward
    env.reset()
    for t in range(T):
        STEP.update(t=t, k=0, ctrl=[], draw=draw(t))
        write(env._data, recorded("pre_", t))
        env._additional_carry = pre_task_state(env._additional_carry, t, np)
        env._cur_step_in_episode = int(R["pretask_ep_len"][t])
        obs, reward, absorbing, done, _ = env.step(R["actions"][t].astype(np.float64))
        OUTS.append(dict(obs=obs, reward=reward, absorbing=absorbing, carry=env._additional_carry, ctrl=STEP["ctrl"]))
else:
    import jax
    import jax.numpy as jnp
    s0 = jax.jit(jax.vmap(env.mjx_reset))(jax.random.split(jax.random.key(0), 1))
    contact_calls = []

    def patched_contacts(self, model, data, bk):  # MJX: mjlab's readings for the data of this call, in call order
        sd = CONTACT_SD[len(contact_calls)]
        contact_calls.append(1)
        return read_native(native, None, type("D", (), {"sensordata": sd})(), jnp)

    @jax.jit
    def step(s, a, t_pre, subs, fresh, contact_sd, drawn):
        global CONTACT_SD
        CONTACT_SD = contact_sd
        STEP["draw"] = drawn
        contact_calls.clear()
        if backend == "mjx":
            ContactReadings.__call__ = patched_contacts
        ml.MLShared.forward = lambda self, env_, model, data, bk: data.replace(
            **{k: v.reshape(getattr(data, k).shape) for k, v in fresh.items()})

        def physics_fn(sys, data, carry, action):
            for k in range(4):
                target, carry = env._mjx_compute_action(action, env._model, data, carry, None)
                data = data.replace(ctrl=data.ctrl.at[np.asarray(env._action_indices)].set(target))
                ctrl = data.ctrl
                data = data.replace(**{f: v.reshape(getattr(data, f).shape) for f, v in subs[k].items()}).replace(ctrl=ctrl)
            return data, carry

        def one(s, a, t_pre, subs, fresh):
            s = s.replace(data=s.data.replace(**{k: v.reshape(getattr(s.data, k).shape) for k, v in t_pre["data"].items()}),
                          additional_carry=t_pre["carry"])
            return env.mjx_step(s, a, auto_reset=False, physics_fn=physics_fn)
        return jax.vmap(one, in_axes=(0, 0, 0, None, None))(s, a, t_pre, subs, fresh)

    for t in range(T):
        STEP["t"] = t
        c0 = jax.tree.map(lambda x: x[0], s0.additional_carry)
        t_pre = dict(data=recorded("pre_", t), carry=pre_task_state(c0, t, jnp))
        t_pre = jax.tree.map(lambda x: jnp.asarray(x)[None], t_pre)
        subs = [recorded(f"sub{k}_", t) for k in range(4)]
        fresh = recorded("fresh_", t)
        # the order in which the step reads contacts: before substeps 1-4 (data before each), the last substep's
        # bookkeeping (data after substep 4), the refreshed state, and the reward (data after substep 4)
        contact_sd = [R["pre_sensordata"][t]] + [R[f"sub{k}_sensordata"][t] for k in range(3)] + \
                     [R["sub3_sensordata"][t], R["fresh_sensordata"][t], R["sub3_sensordata"][t]]
        s1 = step(s0, jnp.asarray(R["actions"][t][None]), t_pre, subs, fresh, contact_sd, draw(t))
        c = s1.additional_carry.replace(key=None, user_scene=None)  # the parts compared, as NumPy arrays of env 0
        c = jax.tree.map(lambda x: np.asarray(x[0]), c)
        ctrl4 = np.asarray(s1.data.ctrl[0])
        OUTS.append(dict(obs=np.asarray(s1.observation[0]), reward=float(s1.reward[0]), absorbing=bool(s1.absorbing[0]),
                         carry=c, ctrl=[ctrl4] * 4))

errs, n_term_mismatch, n_compared, cached_agree = {}, 0, 0, 0


def compare(name, a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    assert a.shape == b.shape, (name, a.shape, b.shape)
    errs[name] = max(errs.get(name, 0.0), float(np.max(np.abs(a - b) / (1.0 + np.abs(b)))) if a.size else 0.0)


for t, o in enumerate(OUTS):
    c, obs = o["carry"], o["obs"]
    for k in range(4):
        compare("control targets (ctrl, actuator order)", o["ctrl"][k], R[f"sub{k}_ctrl"][t])
    compare("reward terms (13, weighted)", c.reward_state.terms, R["terms"][t][perm])
    compare("reward", o["reward"], R["reward"][t])
    n_term_mismatch += int(bool(o["absorbing"]) != bool(R["terminated"][t]))
    if R["terminated"][t] or R["truncated"][t]:
        continue
    n_compared += 1
    compare("obs actor (99)", obs[actor], R["obs_actor"][t])
    compare("obs critic (111) without the contact entries", obs[critic][:-8], R["obs_critic"][t][:-8])
    ff, fF, _, _ = read_native(native, None, type("D", (), {"sensordata": R["fresh_sensordata"][t]})(), np)
    f = fF.reshape(-1)
    compare("obs critic contact entries (8), refreshed state", obs[critic][-8:],
            np.concatenate([(ff > 0).astype(float), np.sign(f) * np.log1p(np.abs(f))]))
    cached_agree += int(np.allclose(obs[critic][-8:], R["obs_critic"][t][-8:], rtol=1e-4, atol=1e-4))
    ct, rs, tw = c.control_func_state, c.reward_state, c.observation_states.MLTwist
    post = lambda k: R["posttask_" + k][t]
    for name, a, b in (("air time", ct.current_air_time, post("cur_air")), ("last air time", ct.last_air_time, post("last_air")),
                       ("contact time", ct.current_contact_time, post("cur_contact")),
                       ("last contact time", ct.last_contact_time, post("last_contact")),
                       ("force history", ct.force_history, post("force_hist")), ("swing peak heights", rs.peak_heights, post("peak")),
                       ("command", tw.command, post("command")), ("command time left", tw.time_left, post("time_left")),
                       ("foot heights (ray casts)", c.observation_states.MLActor.heights, post("heights"))):
        compare("task state after step: " + name, a, b)
if backend != "cpu":
    # the observation of a reset state (computed from kinematics on MJX/MjWarp): LocoMuJoCo's automatic reset with
    # mjlab's reset pose and command draw, against mjlab's observation after its reset
    reset_pose = {}

    def injected_init(self, env_, model, data, carry, backend_, traj=None):
        q = reset_pose["qpos"]
        return data.replace(qpos=q, qvel=jnp.zeros_like(data.qvel), ctrl=jnp.zeros_like(data.ctrl),
                            qacc_warmstart=jnp.zeros_like(data.qacc_warmstart)), carry

    ml.MLInitState.reset = injected_init

    @jax.jit
    def reset_obs(s, qpos, drawn):
        reset_pose["qpos"], STEP["draw"] = qpos, drawn
        return jax.vmap(lambda st: env._mjx_reset_in_step(st))(s).observation

    for t in np.nonzero(R["terminated"] | R["truncated"])[0]:
        obs = np.asarray(reset_obs(s0, jnp.asarray(R["fresh_qpos"][t]), draw(t))[0])
        compare("obs actor (99) of a reset state", obs[actor], R["obs_actor"][t])
        compare("obs critic (111) of a reset state", obs[critic], R["obs_critic"][t])

print(f"[{contacts}, {backend}] {T} steps ({n_compared} without a reset), {int(R['terminated'].sum())} terminations in mjlab, "
      f"termination mismatches: {n_term_mismatch}")
for k, v in errs.items():
    print(f"  max rel. error {k:62s} {v:.2e}")
print(f"  critic contact entries equal to mjlab's cached ones in {cached_agree}/{n_compared} steps")
