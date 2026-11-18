"""Verify the G1Real environment behaves the way it is designed to.

Run on one or more default mocap clips (``walk`` if none are given)::

    python training/g1/verify_env.py walk run

Exits non-zero if any check fails. These are behavioural checks on the MDP.

The checks:

1. **Perfect replay saturates the reward.** Drive the robot to the reference state
   exactly; all six tracking terms should hit ~1.0, so the weighted sum reaches ~5.0.
2. **Anchor invariance.** Teleport the robot 10 m downrange and yaw it 90 degrees while
   still replaying the reference. The re-anchored body terms must be *unchanged* and
   only the two global anchor terms may drop. This is the central
   property of the anchored reward.
3. **Terminations are z-only.** A 0.3 m pelvis height error must terminate; a 10 m xy
   error must not.
4. **Pushes fire on schedule**, detected from the randomizer's own countdown rather
   than from velocity heuristics (with zero action the robot is falling, so root
   velocity changes every step regardless).
5. **The anchored goal observation** has the expected layout and collapses to zero offset /
   identity orientation on perfect replay.
"""

from __future__ import annotations

import sys

import mujoco
import numpy as np
from loco_mujoco import ImitationFactory
from scipy.spatial.transform import Rotation as R

import lmj_real  # noqa: F401  (registers the components)

TASKS = sys.argv[1:] or ["walk"]

_failures: list[str] = []


def check(label: str, passed: bool, detail: str = "") -> None:
    print(f'  {"OK  " if passed else "FAIL"} {label:<40}{detail}')
    if not passed:
        _failures.append(label)


def build_env(env_name: str = "G1Real", **overrides):
    """Build the env.

    The CPU env defaults to ``NoGoal``: ``Mujoco._create_observation`` does not forward
    ``traj`` to stateful observations (``mujoco_base.py:741``), so any
    trajectory-backed goal raises there. The training configs only ever run on the
    MJX path, whose ``_mjx_create_observation`` does forward ``traj``. Check 5 exercises the goal through that path instead.
    """
    kwargs = dict(
        default_dataset_conf=dict(task=TASKS),
        control_type="G1RealPositionControl",
        reward_type="G1RealTrackingReward",
        goal_type="NoGoal",
        terminal_state_type="G1RealTrackingTerminalStateHandler",
        init_state_type="TrajInitialStateHandler",
        horizon=500,
    )
    kwargs.update(overrides)
    return ImitationFactory.make(env_name, **kwargs)


def set_state_from_traj(env, traj, step, xy_offset=(0.0, 0.0), yaw=0.0):
    """Put the robot exactly on the reference frame, optionally displaced.

    The reference is in clip coordinates; the env's xy offset moves it to the episode's
    world frame, where the reward and the goal read it.
    """
    ref = traj.data.get(int(env._additional_carry.traj_state.traj_no), step, np)
    env._data.qpos[:] = ref.qpos
    env._data.qvel[:] = ref.qvel
    env._data.qpos[:2] += np.asarray(env.get_traj_offset_xy(env._additional_carry, np, traj))
    if xy_offset != (0.0, 0.0):
        env._data.qpos[0] += xy_offset[0]
        env._data.qpos[1] += xy_offset[1]
    if yaw:
        rot = R.from_euler("z", yaw) * R.from_quat(np.r_[ref.qpos[4:7], ref.qpos[3]])
        q = rot.as_quat()
        env._data.qpos[3:7] = np.r_[q[3], q[:3]]
    mujoco.mj_forward(env._model, env._data)
    return ref


def reward_components(env, traj, action=None):
    """Evaluate the reward in the current state against the current traj step."""
    if action is None:
        action = np.zeros(env.info.action_space.shape[0])
    carry = env._additional_carry
    reward, carry = env._reward_function(
        env._obs, action, env._obs, False, {}, env,
        env._model, env._data, carry, np, traj,
    )
    return float(reward), dict(carry.reward_state.reward_components)


def main() -> int:
    env, traj = build_env()
    env.reset(traj=traj)
    step = int(env._additional_carry.traj_state.subtraj_step_no)
    print(f"env: {type(env).__name__}, obs {env.info.observation_space.shape}, "
          f"action {env.info.action_space.shape}, traj step {step}")

    # ---- 1. perfect replay ----
    print("\n1. perfect replay saturates the tracking terms")
    set_state_from_traj(env, traj, step)
    total, comp = reward_components(env, traj)
    tracking = {k: v for k, v in comp.items() if k.startswith("tracking/")}
    penalties = {k: v for k, v in comp.items() if k.startswith("penalities/")}
    for k, v in tracking.items():
        print(f"       {k:<38} {float(v):+.4f}")
    for k, v in penalties.items():
        if abs(float(v)) > 1e-9:
            print(f"       {k:<38} {float(v):+.4f}")
    track_sum = sum(float(v) for v in tracking.values())
    check("tracking sum ~= 5.0", abs(track_sum - 5.0) < 0.05, f"{track_sum:.4f}")
    # Some reference frames are themselves past a joint limit, so that penalty is excluded.
    limit_penalty = float(penalties.get("penalities/joint_position_limit_reward", 0.0))
    check("total reward ~= 5.0", abs(total - limit_penalty - 5.0) < 0.15,
          f"{total - limit_penalty:.4f}")

    absorbing, _ = env._terminal_state_handler.is_absorbing(
        env, env._obs, {}, env._data, env._additional_carry, traj
    )
    check("no termination on perfect replay", not bool(absorbing))

    # ---- 2a. invariance to pure xy translation ----
    # Only the global anchor *position* term may react; everything else, including the
    # world-frame velocity terms, is untouched by a rigid translation.
    print("\n2a. +10 m in x: only root_pos may drop")
    set_state_from_traj(env, traj, step, xy_offset=(10.0, 0.0))
    _, comp_t = reward_components(env, traj)
    for k in tracking:
        a, b = float(tracking[k]), float(comp_t[k])
        label = k.split("/")[1]
        if k.endswith("root_pos_reward"):
            check(f"{label} drops", b < a - 1e-6, f"{a:.4f} -> {b:.4f}")
        else:
            check(f"{label} unchanged", abs(a - b) < 1e-6, f"{a:.4f} -> {b:.4f}")

    # ---- 2b. invariance of the re-anchored terms to heading ----
    # body_pos / body_ori are measured against the re-anchored reference and so must be
    # heading-invariant. The velocity terms are deliberately NOT: they are compared in
    # the world frame, so rotating the robot genuinely changes them.
    print("\n2b. +90 deg yaw: re-anchored terms invariant, global terms react")
    set_state_from_traj(env, traj, step, yaw=np.pi / 2)
    _, comp_r = reward_components(env, traj)
    for k in tracking:
        a, b = float(tracking[k]), float(comp_r[k])
        label = k.split("/")[1]
        if k.endswith("root_pos_reward"):
            # Yawing about the pelvis does not translate the pelvis.
            check(f"{label} unchanged", abs(a - b) < 1e-6, f"{a:.4f} -> {b:.4f}")
        elif k.endswith("body_pos_reward") or k.endswith("body_ori_reward"):
            check(f"{label} heading-invariant", abs(a - b) < 1e-6, f"{a:.4f} -> {b:.4f}")
        else:
            check(f"{label} reacts (global frame)", b < a - 1e-6, f"{a:.4f} -> {b:.4f}")

    # ---- 3. z-only terminations ----
    print("\n3. terminations are z-only")
    set_state_from_traj(env, traj, step, xy_offset=(10.0, 0.0))
    absorbing, _ = env._terminal_state_handler.is_absorbing(
        env, env._obs, {}, env._data, env._additional_carry, traj
    )
    check("10 m xy error does NOT terminate", not bool(absorbing))

    set_state_from_traj(env, traj, step)
    env._data.qpos[2] += 0.30
    mujoco.mj_forward(env._model, env._data)
    absorbing, _ = env._terminal_state_handler.is_absorbing(
        env, env._obs, {}, env._data, env._additional_carry, traj
    )
    check("0.30 m z error DOES terminate", bool(absorbing))

    set_state_from_traj(env, traj, step)
    env._data.qpos[2] += 0.10
    mujoco.mj_forward(env._model, env._data)
    absorbing, _ = env._terminal_state_handler.is_absorbing(
        env, env._obs, {}, env._data, env._additional_carry, traj
    )
    check("0.10 m z error does NOT terminate", not bool(absorbing))

    # ---- 4. pushes ----
    print("\n4. push randomizer")
    env_p, traj_p = build_env(
        domain_randomization_type="G1RealRandomizer",
        domain_randomization_params=dict(
            enable_push=True,
            push_interval_range_s=[1.0, 3.0],
            enable_encoder_bias=False,
            enable_foot_friction=False,
            enable_base_com=False,
        ),
    )
    env_p.reset(traj=traj_p)
    action = np.zeros(env_p.info.action_space.shape[0])
    n_steps = 1500
    fires = []
    prev_countdown = int(env_p._additional_carry.domain_randomizer_state.push_countdown)
    for i in range(n_steps):
        env_p.step(action, traj_p)
        cur = int(env_p._additional_carry.domain_randomizer_state.push_countdown)
        # The countdown decrements each step and is resampled upward when a push fires.
        if cur > prev_countdown:
            fires.append(i)
        prev_countdown = cur
    gaps = np.diff(fires) * env_p.dt if len(fires) > 1 else np.array([])
    lo, hi = 1.0, 3.0
    print(f"       {len(fires)} pushes in {n_steps * env_p.dt:.1f} s of sim "
          f"(dt={env_p.dt}, expect one every {lo}-{hi} s)")
    check("at least two pushes fired", len(fires) >= 2, f"at steps {fires[:6]}")
    if gaps.size:
        check(f"push gaps within [{lo}, {hi}] s",
              bool(np.all((gaps >= lo - env_p.dt) & (gaps <= hi + env_p.dt))),
              f"gaps={np.round(gaps, 2).tolist()[:6]} s")

    # ---- 5. anchored goal observation ----
    print("\n5. anchored goal observation")
    goal = env.obs_container.get("GoalTrajMimicAnchor")
    if goal is None:
        from lmj_real.core.observations.goals_anchor import GoalTrajMimicAnchor

        goal = GoalTrajMimicAnchor(env._get_all_info_properties())
        goal._init_from_mj(env, env._model, env._data, 0)
    n_bodies = len(goal._body_ids)
    expected = 29 + 29 + 3 + 6 + n_bodies * 3 + n_bodies * 6
    check("goal dim matches tracking_goal", goal.dim == expected,
          f"{goal.dim} (= 58 + 3 + 6 + {n_bodies}*3 + {n_bodies}*6)")

    set_state_from_traj(env, traj, step)
    g_ref, _ = goal.get_obs_and_update_state(
        env, env._model, env._data, env._additional_carry, np, traj
    )
    # On perfect replay the reference anchor coincides with the robot anchor, so the
    # anchor offset is zero and the anchor orientation is the identity's first two
    # columns flattened ROW-major: [R00,R01,R10,R11,R20,R21] = [1,0, 0,1, 0,0].
    anchor_pos_b = np.asarray(g_ref)[58:61]
    anchor_ori_b = np.asarray(g_ref)[61:67]
    check("anchor_pos_b ~= 0 on perfect replay", np.abs(anchor_pos_b).max() < 1e-6,
          f"max {np.abs(anchor_pos_b).max():.2e}")
    check("anchor_ori_b ~= identity on perfect replay",
          np.allclose(anchor_ori_b, [1, 0, 0, 1, 0, 0], atol=1e-6),
          f"{np.round(anchor_ori_b, 4).tolist()}")

    # Displacing the robot must show up in the anchor offset (that is its whole job).
    set_state_from_traj(env, traj, step, xy_offset=(10.0, 0.0))
    g_off, _ = goal.get_obs_and_update_state(
        env, env._model, env._data, env._additional_carry, np, traj
    )
    off = np.asarray(g_off)[58:61]
    check("anchor_pos_b sees a 10 m offset", abs(np.linalg.norm(off) - 10.0) < 1e-3,
          f"|offset| = {np.linalg.norm(off):.4f} m")

    print()
    if _failures:
        print(f"{len(_failures)} CHECK(S) FAILED: {', '.join(_failures)}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
