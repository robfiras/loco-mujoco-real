"""Records MuJoCo Playground's G1JoystickFlatTerrain (MJX, JAX implementation) step by step, for check_playground.py.

Run in an environment with playground 0.2.0 (the JAX release), on the CPU:

    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python record_playground.py OUT SEED STEPS [ACTION_SCALE]

Noise off, pushes off, every reward scale 1 (so that Playground's metrics are the unscaled terms), 0.005 s x 4
substeps; uniform random actions in [-ACTION_SCALE, ACTION_SCALE]. Playground's step continues after a termination
(no wrapper), so terminated states are recorded too. Writes OUT.mjb (the compiled model) and OUT.npz: per step the
data and task state before it, the action, and the data, task state, observations, reward terms, reward and
termination after it.
"""
import sys
import jax
import jax.numpy as jp
import mujoco
import numpy as np
from mujoco_playground import registry
from mujoco_playground._src.locomotion.g1 import joystick

out, seed, T = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
act_scale = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
scales = joystick.default_config().reward_config.scales
over = {"impl": "jax", "sim_dt": 0.005, "noise_config.level": 0.0, "push_config.enable": False}
over.update({f"reward_config.scales.{k}": 1.0 for k in scales})
env = registry.load("G1JoystickFlatTerrain", config_overrides=over)
mujoco.mj_saveModel(env.mj_model, out + ".mjb", None)
reset, step = jax.jit(env.reset), jax.jit(env.step)
state = reset(jax.random.PRNGKey(seed))
actions = np.random.default_rng(seed).uniform(-1, 1, (T, env.action_size)).astype(np.float32) * np.float32(act_scale)
DATA = ("qpos", "qvel", "ctrl", "qacc_warmstart", "time", "qacc", "sensordata", "site_xpos", "site_xmat", "actuator_force")
INFO = ("command", "step", "last_act", "last_last_act", "feet_air_time", "last_contact", "swing_peak", "phase_dt",
        "phase", "push_step", "push_interval_steps")
rec = {}
def add(k, v):
    rec.setdefault(k, []).append(np.asarray(v))
add("reset_obs_state", state.obs["state"]); add("reset_obs_priv", state.obs["privileged_state"])
for t in range(T):
    for k in DATA:
        add("pre_" + k, getattr(state.data, k))
    for k in INFO:
        add("pre_" + k, state.info[k])
    state = step(state, jp.asarray(actions[t]))
    for k in DATA:
        add("post_" + k, getattr(state.data, k))
    for k in INFO:
        add("post_" + k, state.info[k])
    add("obs_state", state.obs["state"]); add("obs_priv", state.obs["privileged_state"])
    add("reward", state.reward); add("done", state.done)
    add("terms", np.array([state.metrics[f"reward/{k}"] for k in scales]))
rec = {k: np.stack(v) for k, v in rec.items()}
np.savez(out + ".npz", actions=actions, term_names=np.array(list(scales)),
         true_scales=np.array([scales[k] for k in scales]), **rec)
print("recorded", T, "steps; dones", int(rec["done"].sum()), "; final z", float(rec["post_qpos"][-1, 2]))
