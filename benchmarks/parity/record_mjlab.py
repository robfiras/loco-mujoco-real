"""Records mjlab's G1 velocity task step by step, for check_mjlab.py.

Run in an environment with mjlab 1.6.1 and a GPU:

    python record_mjlab.py OUT SEED STEPS ACTION_SCALE [full|reduced [RESAMPLE_MIN RESAMPLE_MAX]]

One environment, the benchmark's settings (no domain randomization events, Euler), observation noise off, uniform
random actions in [-ACTION_SCALE, ACTION_SCALE]. Per step it saves the data and the task state before the step
(command, contact-sensor air times and force history, swing peak heights, foot heights, actions), the data after each
of the 4 physics substeps and after mjlab's forward, and the observations, weighted reward terms, reward and
termination flags after the step. Writes OUT.npz. RESAMPLE_MIN/MAX shorten the command's resampling interval (3-8 s),
so that a short recording also covers resampling by the timer (the check takes mjlab's draws either way).
"""
import sys

import numpy as np
import torch

from export_models import build

FIELDS = ("qpos", "qvel", "ctrl", "qacc_warmstart", "sensordata", "xpos", "xquat", "cvel", "subtree_com",
          "site_xpos", "site_xmat", "time")

out, seed, T, act_scale = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
contacts = sys.argv[5] if len(sys.argv) > 5 else "full"
torch.manual_seed(seed)
env = build(contacts, corruption=False)
wd = env.sim.wp_data


def snap():
    return {f: getattr(wd, f).numpy()[0].copy() for f in FIELDS}


cur = {"sub": [], "fresh": None}
orig_step, orig_forward = env.sim.step, env.sim.forward


def step_hook():
    orig_step()
    cur["sub"].append(snap())


def forward_hook():
    orig_forward()
    cur["fresh"] = snap()


env.sim.step, env.sim.forward = step_hook, forward_hook
cmd = env.command_manager._terms["twist"]
if len(sys.argv) > 7:
    cmd.cfg.resampling_time_range = (float(sys.argv[6]), float(sys.argv[7]))
feet, own = env.scene["feet_ground_contact"], env.scene["self_collision"]
swing = env.reward_manager.get_term_cfg("foot_swing_height").func
heights = env.scene["foot_height_scan"]
am = env.action_manager


def task():
    a = feet._air_time_state
    t = dict(command=cmd.vel_command_b[0], time_left=cmd.time_left[0], heading_target=cmd.heading_target[0],
             is_heading=cmd.is_heading_env[0], is_standing=cmd.is_standing_env[0], cur_air=a.current_air_time[0],
             last_air=a.last_air_time[0], cur_contact=a.current_contact_time[0], last_contact=a.last_contact_time[0],
             force_hist=own._history_state["force"][0, 0], peak=swing.peak_heights[0], heights=heights.data.heights[0],
             action=am.action[0], prev_action=am.prev_action[0], ep_len=env.episode_length_buf[0])
    return {k: v.detach().cpu().numpy().copy() for k, v in t.items()}


env.reset()
actions = np.random.default_rng(seed).uniform(-1, 1, (T, 29)).astype(np.float32) * np.float32(act_scale)
rec = {}


def add(prefix, d):
    for k, v in d.items():
        rec.setdefault(prefix + k, []).append(np.asarray(v))


for t in range(T):
    add("pre_", snap())
    add("pretask_", task())
    cur["sub"], cur["fresh"] = [], None
    obs, rew, terminated, truncated, _ = env.step(torch.as_tensor(actions[t], device=env.device)[None])
    assert len(cur["sub"]) == 4 and cur["fresh"] is not None
    for k, s in enumerate(cur["sub"]):
        add(f"sub{k}_", s)
    add("fresh_", cur["fresh"])
    add("posttask_", task())
    add("", dict(obs_actor=obs["actor"][0].cpu().numpy(), obs_critic=obs["critic"][0].cpu().numpy(),
                 reward=rew[0].item(), terms=env.reward_manager._step_reward[0].cpu().numpy(),
                 terminated=bool(terminated[0]), truncated=bool(truncated[0])))
rec = {k: np.stack(v) for k, v in rec.items()}
np.savez(out + ".npz", actions=actions, term_names=np.array(env.reward_manager._term_names), **rec)
print("recorded", T, "steps; terminations", int(rec["terminated"].sum()), "; command resamples",
      int((np.diff(rec["posttask_time_left"]) > 0).sum()))
