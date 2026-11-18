"""Reward, goal and terminations on a synthetic clip that starts away from the origin."""

import numpy as np

from verify_env import reward_components, set_state_from_traj


def _tracking(components):
    return {k.split("/")[1]: float(v) for k, v in components.items() if k.startswith("tracking/")}


def test_reset_puts_robot_on_reference(tracking_env):
    env, traj = tracking_env
    clip_x = []
    for _ in range(8):
        env.reset(traj=traj)
        carry = env._additional_carry
        state = carry.traj_state
        ref = traj.data.get(int(state.traj_no), int(state.subtraj_step_no), np)
        offset = np.asarray(env.get_traj_offset_xy(carry, np, traj))
        clip_x.append(abs(float(ref.qpos[0])))
        np.testing.assert_allclose(env._data.qpos[:2], np.asarray(ref.qpos[:2]) + offset, atol=1e-6)
    # the check only means something if episodes start away from the clip's origin
    assert max(clip_x) > 0.5


def test_perfect_replay_saturates_tracking(tracking_env):
    env, traj = tracking_env
    env.reset(traj=traj)
    step = int(env._additional_carry.traj_state.subtraj_step_no)
    set_state_from_traj(env, traj, step)
    _, components = reward_components(env, traj)
    assert abs(sum(_tracking(components).values()) - 5.0) < 0.05
    absorbing, _ = env._terminal_state_handler.is_absorbing(
        env, env._obs, {}, env._data, env._additional_carry, traj)
    assert not bool(absorbing)


def test_xy_drift_only_costs_the_root_position_term(tracking_env):
    env, traj = tracking_env
    env.reset(traj=traj)
    step = int(env._additional_carry.traj_state.subtraj_step_no)
    set_state_from_traj(env, traj, step)
    on_ref = _tracking(reward_components(env, traj)[1])
    set_state_from_traj(env, traj, step, xy_offset=(10.0, 0.0))
    drifted = _tracking(reward_components(env, traj)[1])
    assert on_ref["root_pos_reward"] > 0.45 and drifted["root_pos_reward"] < 1e-6
    for name in on_ref:
        if name != "root_pos_reward":
            assert abs(on_ref[name] - drifted[name]) < 1e-4, name
    absorbing, _ = env._terminal_state_handler.is_absorbing(
        env, env._obs, {}, env._data, env._additional_carry, traj)
    assert not bool(absorbing), "xy drift alone must not end the episode"


def test_anchor_goal_observation(tracking_env):
    from lmj_real.core.observations.goals_anchor import GoalTrajMimicAnchor

    env, traj = tracking_env
    env.reset(traj=traj)
    goal = GoalTrajMimicAnchor(env._get_all_info_properties())
    goal._init_from_mj(env, env._model, env._data, 0)
    step = int(env._additional_carry.traj_state.subtraj_step_no)

    set_state_from_traj(env, traj, step)
    obs, _ = goal.get_obs_and_update_state(env, env._model, env._data, env._additional_carry, np, traj)
    np.testing.assert_allclose(np.asarray(obs)[58:61], 0.0, atol=1e-6)            # motion_anchor_pos_b
    np.testing.assert_allclose(np.asarray(obs)[61:67], [1, 0, 0, 1, 0, 0], atol=1e-6)  # motion_anchor_ori_b

    set_state_from_traj(env, traj, step, xy_offset=(10.0, 0.0))
    obs, _ = goal.get_obs_and_update_state(env, env._model, env._data, env._additional_carry, np, traj)
    assert abs(np.linalg.norm(np.asarray(obs)[58:61]) - 10.0) < 1e-3


def test_steps_stay_finite(tracking_env):
    env, traj = tracking_env
    obs = env.reset(traj=traj)
    for _ in range(20):
        obs, reward, absorbing, done, info = env.step(np.zeros(env.info.action_space.shape))
        assert np.all(np.isfinite(obs)) and np.isfinite(reward)
