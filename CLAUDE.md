# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

`loco-mujoco-real` is sim-to-real tooling for motion-tracking policies trained with
[LocoMuJoCo](https://github.com/robfiras/loco-mujoco), for the Unitree G1 and the AgiBot X2
Ultra. The importable package is `lmj_real`. It ships two environments, `G1Real` / `MjxG1Real`
and `X2UltraReal` / `MjxX2UltraReal`, and the components they need.

The X2 has run on the robot, from a laptop over Ethernet and onboard on SoC1
(`deployment/x2/X2_ROBOT_GUIDE.md`, `deployment/x2/soc1/`). Layout is per robot: `training/{g1,x2}/` (each with `train.py`, `conf.yaml`, `env/`; the shared loop is `training/ppo.py`) and `deployment/{g1,x2}/` (controller, bridge, README); `deployment/` itself and the main README hold only what the robots share. `lmj_real/` mirrors LocoMuJoCo's folder structure on purpose, so users can add a robot without touching LocoMuJoCo's core. The G1 runs on
the robot from a laptop over Ethernet (`deploy_g1.py --domain 0`); it has no onboard path yet.

The AgiBot X2 Ultra (`X2UltraReal` / `MjxX2UltraReal`) is the second robot. It
subclasses `G1Real` and overrides only the model path and the body, geom and height
properties; the G1-named control, observation, reward, terminal and randomizer classes read
names from the env and serve both robots. Its configs are in `training/x2/`, its model
`lmj_real/models/x2_ultra_real/`, its retargeting config `X2UltraReal.yaml`. The exporter,
`validate_bundle.py` and `rollout_mujoco.py` take either robot (the model comes from
`lmj_deploy.training_model_xml`, and a `ctrl_range` bound of `null` in `policy.json` means
unclipped); `deploy_g1.py` and the DDS bridge are still G1 only. The X2 has its own
asynchronous pair on AgiBot's AimDK ROS 2 topics, `x2/sim2sim_bridge.py` and
`deploy_x2.py` (library `lmj_aimdk.py`), which need ROS 2 Humble and `aimdk_msgs`. Only
`deploy_x2.py --robot` reaches ROS domain 0, the robot's: it switches the X2 to `Develop_MC`
first, and is driven from the terminal or a USB gamepad, not `/joy`, which the robot does not
publish. It caps OpenBLAS at four threads before numpy loads: uncapped, the spinning pool
delays single policy steps by tens of ms whenever anything else needs a core. Robot clips
must start and end standing (`robot_clips.py`). `x2_joint_test.py` swings named joints on the
hanging robot and is how the link was first proven. Its joint order differs from the G1:
waist and wrist are both yaw,
pitch, roll, and actuator order equals joint order. Its retargeting config sets
`joint_limit_solref` / `joint_limit_solimp`, which LocoMuJoCo applies in `add_mocap_bodies`;
without them the wrists leave their ranges.

## Dependency on loco-mujoco (the backbone)

This repo is a thin extension layer. Everything in `lmj_real/` subclasses, registers with, or
plugs into classes in `loco_mujoco`: `BaseRobotHumanoid`, `ControlFunction`, `Goal`,
`StatefulObservation`, `TerminalStateHandler`, `DefaultRandomizer` and
`ImitationFactory.make(...)`.

When something looks impossible without changing a base class or registration mechanism, the
fix likely belongs in the backbone, not here.

## Commands

See the README for setup and the full pipeline. Run from the repo root:

```bash
python training/g1/verify_env.py walk
python training/g1/bones_to_g1.py <csv or dir> --out <dir>   # BONES-SEED G1 CSVs -> .lmj.npz
python training/g1/train.py          # or training/x2/train.py
python deployment/export_bundle.py --checkpoint <pkl> --output <dir>
python deployment/validate_bundle.py --bundle <dir> --checkpoint <pkl>
python deployment/rollout_mujoco.py --bundle <dir> [--plant servo|torque|<mjcf>]
python deployment/g1/sim2sim_bridge.py --bundle <dir>
python deployment/g1/deploy_g1.py --bundle <dir> --motion <clip> --auto-start
python deployment/x2/sim2sim_bridge.py --bundle <dir>          # X2, ROS 2
python deployment/x2/deploy_x2.py --bundle <dir> --motion <clip> --auto-start
```

`lmj_deploy.py` reimplements the actor observation by hand (283 dims for PPO, 829 for the
SuperMimic encoder-decoder); `validate_bundle.py` keeps that honest. Don't change one without
re-running the other. The exporter reads PPOJax and SuperMimicJax checkpoints and refuses any
not using `G1RealPositionControl`, or whose actor reads a term `ObservationBuilder` lacks.

`deploy_g1.py` and the bridge need the Unitree SDK. The bridge refuses domain 0, which is the
robot; keep that. `deploy_g1.py` reaches it with `--domain 0` and refuses `--auto-start` there. Over DDS
there is no separate observation code: `LowStateAdapter` fills an `MjData` for `ObservationBuilder`.

Two orderings coexist and mixing them fails silently: observations and the reference `command`
are in MuJoCo **joint** order (legs first, `robot.joints_to_consider`), while actions,
`last_action`, `action_scale`, `kp`, `kd` and `ctrl` are in **actuator** order (shoulders
first). Both are in `policy.json`.

## Architecture

`import lmj_real` calls `register_lmj()`, which delegates to each subpackage's `register_*`
function. Every new component must be added to its subpackage's `register_*()` call, or
`ImitationFactory` won't see it. `ImitationFactory.make("<ClassName>")` looks up by class name.

| piece | file |
|---|---|
| MJCF and meshes | `lmj_real/models/{g1_real,x2_ultra_real}/` |
| envs | `lmj_real/environments/{g1_real,x2_ultra_real}{,_mjx}.py` |
| control | `lmj_real/core/control_functions/g1_real_position.py` |
| observations | `lmj_real/core/observations/g1_real_obs.py` |
| anchored goal | `lmj_real/core/observations/goals_anchor.py` |
| reward | `lmj_real/core/reward/g1_real_tracking_reward.py` |
| terminal handler | `lmj_real/core/terminal_state_handler/g1_real_tracking.py` |
| randomizer | `lmj_real/core/domain_randomizer/g1_real_randomizer.py` |
| shared anchor math | `lmj_real/core/utils/tracking.py`, `.../keyframe.py` |
| retargeting config | `lmj_real/smpl_robot_confs/{G1Real,X2UltraReal}.yaml` |
| training config | `training/<robot>/conf.yaml`, `training/<robot>/env/` |

## Conventions to preserve

- **The default pose is a keyframe, never joint `ref`.** Joint `ref` sets `qpos0` *and shifts
  forward kinematics*. `lmj_real/core/utils/keyframe.default_qpos` reads the `init_state`
  keyframe.
- **The free joint must stay named `root`.** `TrajectoryHandler.filter_and_extend` matches
  trajectories to the model by joint name; another name silently drops the 7 root `qpos`
  entries.
- **`G1Real.yaml` must match the env class name.** The lookup strips `Mjx` and appends `.yaml`.
- **The env chooses its dataset source.** `default_dataset_source_env` and
  `lafan1_dataset_source_env` return `UnitreeG1`. Read them from the env; don't hard-code.
- **Default clips load through `default_dataset_conf`.** `ImitationFactory` retargets them to
  `G1Real` on first use and caches them, which needs the SMPL setup in the README. Don't add a
  separate data-preparation step.
- **Two 6D rotation encodings coexist**: tracking *observations* use columns 0 and 1
  (`mat_to_6d_cols01`), the privileged all-body observations use columns 0 and 2
  (`quat_to_tan_norm_from_mat`). Don't unify them.
- **New observation types are `StatefulObservation` subclasses on purpose.**
  `ObservationIndexContainer` builds per-type slots from
  `ObservationType.list_all_non_stateful()`, so a non-stateful type defined outside
  `loco_mujoco` would need `ObservationType` monkeypatched.
- `MjxG1Real._modify_spec_for_mjx` is intentionally a no-op: the collision geoms are already in
  the XML.
- `get_default_xml_file_path` resolves via `__file__`. Don't add a models-path variable.
- **Never branch on the truthiness of an MjSpec limit flag** (`ctrllimited`, `limited`,
  `forcelimited`). On newer MuJoCo they are enums that are truthy for every value. Read the
  compiled model (`model.actuator_ctrllimited`). This once clipped every X2 target to its
  unset ctrlrange of [0, 0], so actions had no effect and nothing trained.
- **Reference positions in the world frame need `env.get_traj_offset_xy`.** The env may start
  episodes at the origin; the reference stays in clip coordinates until that xy offset is
  added. `GoalTrajMimicAnchor`, `G1RealTrackingReward` and `validate_bundle.py` add it.
- Keep site-specific job launchers (for example sbatch files) out of this repo.

## The MjWarp contact buffer

Read this before touching anything that consumes `data.contact`.

MJX batches `data.contact` per environment. MjWarp does not: `contact__*` arrays are sized by
`naconmax`, form **one global buffer** with no batch dimension, and `nacon` is one global count.
Anything per-env read off it without a mask is wrong. `G1RealTrackingReward._self_collision_count`
masks by `worldid == carry.env_id`, and three things hold it together:

1. `slot < nacon` is **mandatory**: MjWarp does not clear unused slots between steps.
2. Plain MJX has no `worldid` and needs no mask; the reward branches on the layout it gets.
3. **Any vmap without `axis_name` silently degrades to `env_id=0`.** `VecEnv` sets it; anything
   hand-rolling `jax.vmap(env.mjx_reset, ...)` must too. With nested vmaps the stride comes from
   `jax.lax.axis_size`, not the env's `n_envs` argument.

`naconmax` bounds contacts across *all* worlds, so the reward asserts at construction that
`n_envs * 32 <= naconmax`. Past that ceiling MuJoCo drops contacts and the count under-reports.
