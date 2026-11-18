# Deployment reference

Background for [README.md](README.md): what is inside a bundle, how the observation is laid
out, and what to know before reading rollout results. You do not need any of this to export
and run a policy.

## The simulated robot in a rollout (`--plant`)

"Plant" is control-engineering shorthand for the thing being controlled: here, the simulated
robot that `rollout_mujoco.py` and `rollout_viser.py` drive. `--plant` chooses which one.

| value | actuation | what it isolates |
|---|---|---|
| `servo` (default) | MuJoCo position servos, PD inside the physics | nothing — same plant as training. A failure here is a pipeline bug. |
| `torque` | torque motors, PD in Python at the sim rate | the deploy actuation path. Measured to reproduce the servo to `1.1e-15` in `qpos` over 20 control steps, so it is an equivalence check, not a gap measurement. |
| any path | whatever that MJCF has | bring your own, e.g. the stock Unitree MJCF for the full plant gap: joint damping and frictionloss, vendor armature, mesh collision geometry. Read as a robustness probe. |

All of them are put on the training model's compiled `mj_option` (timestep, solver,
iteration counts) unless `--no-match-solver`, so the comparison is not confounded by the
integrator.

## Two orderings, and why mixing them is silent

| list | order | used by |
|---|---|---|
| `spec.joint_names` | MuJoCo joint order (legs first) — same as `robot.joints_to_consider` | `joint_pos_rel`, `joint_vel`, reference `command` |
| `spec.actuator_joint_names` | actuator order (shoulders first) | the action vector, `last_action`, `action_scale`, `kp`/`kd`, `ctrl` |

Both are exported. Everything in `lmj_deploy` resolves joints and bodies by *name*
against whatever model it is handed, so a differently-ordered deploy MJCF is fine.

## SuperMimic bundles

SuperMimic's actor is an encoder and a decoder. The encoder reads `goal_obs_group`,
usually `TrajFutureFrames`: the next 10 reference frames relative to the current one, from
the reference alone. The decoder reads the last `history_len` frames of
`history_obs_group`, oldest first and ending with the current one, which
`EncoderDecoderActor` keeps itself; `reset()` it at every episode start. The bundle stores
the two slices back to back, so `obs_dim` is one encoder input plus one history frame.

Nothing in that observation needs a world position, so these bundles run over DDS too.
Training may run the matmuls in bfloat16; the bundle runs them in float32, and the
exporter prints the action difference that makes.

The actor is about 60 times larger than the PPO one: about 5 ms per step alone on an
M-series Mac, 12 to 22 ms with the bridge running alongside. Capping numpy at two BLAS
threads (`VECLIB_MAXIMUM_THREADS=2` on macOS, `OPENBLAS_NUM_THREADS=2` elsewhere) kept
ticks on time in one run where the default did not. Measure on the robot's computer
before trusting either.

## Observation layout (PPO: 283 dims, the `policy` group)

```
  0..189  tracking_goal        ref joint pos (29) + ref joint vel (29)
                               + motion_anchor_ori_b (6)
                               + robot body_pos_b (14x3) + body_ori_b (14x6)
190..218  last_action          previous raw policy output, actuator order
219..247  joint_pos_rel        qpos[joints] - default_joint_pos (the init_state keyframe)
248..276  joint_vel            qvel[joints]
277..279  projected_gravity    R_root^T @ (0, 0, -1)
280..282  base_ang_vel         qvel[3:6]   (free joint, already body-frame = the IMU gyro)
```

The order follows `actor_obs_group`, and `spec.obs_terms` is the authority on it.

The critic's `privileged_state` terms are deliberately not exported.

Checkpoints trained before `use_anchor_pos` existed also carry `motion_anchor_pos_b` (3)
ahead of the orientation, for 286 dims. The bundle records which layout it uses. That term
needs a world position, which a real robot does not have, so those policies are sim only.

Two encoding traps:

* `tracking_goal` orientations use rotation-matrix **columns 0 and 1**
  (`mat_to_6d_cols01`); the privileged all-body observations use columns 0 and 2. Don't
  unify them.
* the `tracking_goal` frame is the robot anchor's **full** orientation, not the
  heading-only frame the privileged observations use.

## Bundle format

```
policy.json      spec: orders, action scale/offset, ctrl range, kp/kd, effort limits,
                 tracked bodies, anchor index, termination thresholds, obs term layout,
                 network shape, compiled mj_option
policy.npz       w0/b0..wN/bN, log_std, obs_mean, obs_var, obs_norm_count
motions.lmj.npz  a loco_mujoco Trajectory: info (frequency, 30 joint names, 31 body
                 names, 21 site names, reduced model, metadata) + data (qpos, qvel,
                 xpos, xquat, cvel, subtree_com, site_xpos, site_xmat, split_points)
```

The motions are stored in the **native trajectory format**, not a bespoke per-clip npz
plus a JSON manifest. Three reasons:

* the clips stay loadable by every existing tool -- replay, retargeting,
  `ExtendTrajData` -- so the export is no longer one-way and lossy,
* the per-frame quantities a deploy-only format would drop (`cvel`, `subtree_com`,
  sites, and all 31 bodies rather than the 14 tracked ones) survive the round trip,
* there is no second description of joint and body order that can drift away from
  `policy.json`.

`MotionBundle` slices out the policy's 29 joints and 14 tracked bodies **by name** at
load time, since the trajectory carries all 30 joints (including the free joint) and all
31 bodies in a different order. Clip boundaries come from `split_points`; clip names
from `info.metadata['clip_names']`.

The cost is size: the bundle stores quantities deployment never reads, so 5 clips is
~7 MB where the old format would have been roughly half. Fine at `--max-motions 20`;
do not export 10k clips.

`obs_mean` / `obs_var` are already restricted to the actor slice. Training normalizes the
**full** 734-dim observation with a `RunningMeanStd` and only then indexes
`actor_obs_ind`; normalization is elementwise, so slicing the statistics is equivalent.

`MlpActor` freezes those statistics. Training's `RunningMeanStd` folds the incoming batch
into them before normalizing, but with a count in the billions the difference is below
`1e-8` in the action — measured by the exporter's self-test.

Bundles for the X2 on the robot hold clips that start and end standing; see
[x2/README.md](x2/README.md#robot-bundles-clips-that-start-and-end-standing).

### `--std-floor`

Dimensions that were constant during training have a near-zero normalizer std, so any
kinematic difference on such a dimension reaches the network as a z-score of ~1e4.
Default is `0.0` (bit-exact with training). Raise it to `1e-3` when running a bring-your-own
MJCF. Both the exporter and the rig report how many dims
are affected.

## One thing about the training stack to know before reading results

**`data.xpos` is stale in the first observation of every episode.**
`Mujoco.reset` calls `mj_forward` *before* the initial-state handler writes `qpos`
and never again, so the episode's first observation is built on the previous pose
(~1 cm of body-position error on the G1). Only the first step of each episode is
affected; `mj_step` refreshes it thereafter. `validate_bundle.py` calls `mj_forward`
before its closed-loop phase so that phase measures the bundle rather than this.

## Validation results on this repo's stack

Teacher-forced against the live environment (per-term max absolute difference):

```
joint_pos_rel        0.0        base_ang_vel_{r,p,y}  0.0
joint_vel            0.0        last_action           0.0
projected_gravity    5.7e-10    tracking_goal         7.7e-08
action               9.3e-09    ctrl target           0.0
closed-loop |qpos| divergence after 30 steps: 1.4e-09
```

## Limitations

* **Observation delay and history exist in the env but are OFF by default, and the
  bundle does not carry them.** Delay is `G1RealRandomizer.obs_delay_steps`, history is
  `experiment.len_obs_history` + `obs_history_group`; both default to off, and
  `PolicySpec` has no field for either, so a policy trained WITH them cannot be exported
  faithfully yet. For a policy trained without them the bundle matches the env exactly.
  Either way, a real robot's sensing latency is not modelled anywhere in this chain
  unless you turn the delay on during training.
* **Hardware status is per robot**: see [g1/README.md](g1/README.md#where-it-stands) and
  [x2/README.md](x2/README.md#where-it-stands).
* **Clips can ask for joints beyond their range.** The `squat` reference passes
  `waist_pitch_joint`'s 0.52 rad limit by up to 0.29 rad, on 36% of its frames.
* **`G1RealPositionControl` only, and only observation terms `lmj_deploy.py`
  implements.** The exporter refuses anything else rather than silently producing a wrong
  observation. Of SuperMimic's actors, only the encoder-decoder one (`history_len > 0`).
* **Only the mean action** is exported (`log_std` is stored but unused). Deployment is
  deterministic.
