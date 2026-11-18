"""`import lmj_real` registers every component, or ImitationFactory cannot find it."""

import lmj_real  # noqa: F401


def test_environments_registered():
    from loco_mujoco.environments import LocoEnv

    for name in ("G1Real", "MjxG1Real", "X2UltraReal", "MjxX2UltraReal"):
        assert name in LocoEnv.registered_envs


def test_components_registered():
    from loco_mujoco.core.control_functions import ControlFunction
    from loco_mujoco.core.domain_randomizer import DomainRandomizer
    from loco_mujoco.core.observations import Observation
    from loco_mujoco.core.reward import Reward
    from loco_mujoco.core.terminal_state_handler import TerminalStateHandler

    assert "G1RealPositionControl" in ControlFunction.registered
    assert "G1RealTrackingReward" in Reward.registered
    assert "G1RealTrackingTerminalStateHandler" in TerminalStateHandler.registered
    assert "G1RealRandomizer" in DomainRandomizer.registered
    for name in ("GoalTrajMimicAnchor", "G1RealJointPosRel", "G1RealBodyPosRel", "G1RealBodyOriRel",
                 "G1RealBodyLinVelRel", "G1RealBodyAngVelRel"):
        assert name in Observation.registered


def test_retargeting_configs_match_env_names():
    from pathlib import Path

    confs = Path(lmj_real.__file__).parent / "smpl_robot_confs"
    assert {p.name for p in confs.glob("*.yaml")} == {"G1Real.yaml", "X2UltraReal.yaml"}
