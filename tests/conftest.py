"""Shared fixtures. Nothing here needs a GPU, the SMPL model or a downloaded dataset."""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "deployment"), str(REPO / "training" / "g1")]

ROBOTS = ("G1Real", "X2UltraReal")


def make_slide_clip(env_name: str, n_frames: int = 300, speed: float = 1.0):
    """A synthetic clip: the robot's default pose sliding along x at constant speed.

    A random start therefore lands away from the clip's origin, so anything that mixes
    clip and world coordinates shows up.
    """
    import jax.numpy as jnp
    import mujoco

    import lmj_real  # noqa: F401
    from lmj_real.core.utils.keyframe import default_qpos
    from loco_mujoco.core.trajectory import Trajectory, TrajectoryData, TrajectoryInfo, TrajectoryModel
    from loco_mujoco.datasets.humanoids.LAFAN1.load import extend_motion
    from loco_mujoco.environments import LocoEnv
    from loco_mujoco.smpl.retargeting import load_robot_conf_file

    env = LocoEnv.registered_envs[env_name](
        th_params=dict(random_start=False, fixed_start_conf=(0, 0)), print_obs_summary=False)
    model, freq = env._model, 1.0 / env.dt
    qpos = np.tile(default_qpos(model), (n_frames, 1))
    qpos[:, 0] = speed * np.arange(n_frames) / freq
    qvel = np.zeros((n_frames, model.nv))
    qvel[:, 0] = speed
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(model.njnt)]
    info = TrajectoryInfo(joint_names=names, frequency=freq,
                          model=TrajectoryModel(njnt=model.njnt, jnt_type=jnp.array(model.jnt_type)))
    data = TrajectoryData(qpos=jnp.array(qpos), qvel=jnp.array(qvel),
                          split_points=jnp.array([0, n_frames]))
    return extend_motion(env_name, load_robot_conf_file(env_name), Trajectory(info=info, data=data),
                         replace_qvel_with_finite_diff=False)


@pytest.fixture(scope="session", params=ROBOTS)
def tracking_env(request):
    """``(env, traj)`` for each robot on the synthetic clip, with the tracking reward."""
    from loco_mujoco.task_factories import ImitationFactory
    from loco_mujoco.task_factories.dataset_confs import CustomDatasetConf

    made = ImitationFactory.make(
        request.param,
        custom_dataset_conf=CustomDatasetConf(traj=make_slide_clip(request.param)),
        control_type="G1RealPositionControl",
        reward_type="G1RealTrackingReward",
        goal_type="NoGoal",
        terminal_state_type="G1RealTrackingTerminalStateHandler",
        init_state_type="TrajInitialStateHandler",
        horizon=100,
    )
    env = made[0] if isinstance(made, tuple) else made
    return env, env._traj
