"""Every training config composes and names an environment and components that exist."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

import lmj_real  # noqa: F401

TRAINING = Path(__file__).resolve().parents[1] / "training"


@pytest.mark.parametrize("robot,env_name", [("g1", "MjxG1Real"), ("x2", "MjxX2UltraReal")])
@pytest.mark.parametrize("config", ["conf", "supermimic"])
def test_config_composes(robot, env_name, config):
    from loco_mujoco.core.control_functions import ControlFunction
    from loco_mujoco.core.reward import Reward
    from loco_mujoco.environments import LocoEnv

    with initialize_config_dir(config_dir=str(TRAINING / robot), version_base=None):
        params = compose(config_name=config).experiment.env_params
    assert params.env_name == env_name and env_name in LocoEnv.registered_envs
    assert params.control_type in ControlFunction.registered
    assert params.reward_type in Reward.registered
