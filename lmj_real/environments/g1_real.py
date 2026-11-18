"""Unitree G1 (29 DoF) environment for training policies meant for the real robot (CPU).

What makes it suitable for real-world deployment:

* actuators are MuJoCo **position servos** (affine bias), so the PD loop runs inside the
  physics rather than in Python, as it does on the robot -- see
  :class:`~lmj_real.core.control_functions.g1_real_position.G1RealPositionControl`;
* ``armature`` is set per motor type (0.02510 on ``hip_roll``/``knee``, 0.00722 on
  ``waist_roll``/``waist_pitch``);
* ``dof_damping`` and ``dof_frictionloss`` are 0;
* full self-collision is enabled through the ``*_collision`` geom set;
* the standing pose is the ``init_state`` keyframe, which is what the position controller
  and the relative joint observations use as their zero point.
"""

from __future__ import annotations

import os
from typing import List, Tuple, Union

import mujoco
from loco_mujoco.core import Observation, ObservationType
from loco_mujoco.core.utils import info_property
from loco_mujoco.environments.humanoids.base_robot_humanoid import BaseRobotHumanoid
from mujoco import MjSpec

# Joints in the default observation. The configs in training declare their own
# observation_spec; this is what a bare G1Real() falls back to.
_OBSERVED_JOINTS = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
)


class G1Real(BaseRobotHumanoid):
    """Unitree G1 with position servos and full self-collision."""

    mjx_enabled = False

    def __init__(
        self,
        spec: Union[str, MjSpec] = None,
        observation_spec: List[Observation] = None,
        actuation_spec: List[str] = None,
        **kwargs,
    ) -> None:
        """
        Args:
            spec: Path to an XML file or an MjSpec. Defaults to the bundled model.
            observation_spec: Observation space. Defaults to joint positions and
                velocities of the joints in ``_OBSERVED_JOINTS``.
            actuation_spec: Actuator names. Defaults to every actuator in the model.
            **kwargs: Additional parameters for the environment.
        """
        if spec is None:
            spec = self.get_default_xml_file_path()
        spec = mujoco.MjSpec.from_file(spec) if not isinstance(spec, MjSpec) else spec

        if observation_spec is None:
            observation_spec = self._get_observation_specification(spec)
        else:
            observation_spec = self.parse_observation_spec(observation_spec)
        if actuation_spec is None:
            actuation_spec = self._get_action_specification(spec)

        if self.mjx_enabled:
            spec = self._modify_spec_for_mjx(spec)

        super().__init__(
            spec=spec, actuation_spec=actuation_spec, observation_spec=observation_spec, **kwargs
        )

    @staticmethod
    def _get_observation_specification(spec: MjSpec) -> List[Observation]:
        """Returns the default observation specification of the environment."""
        return [
            ObservationType.FreeJointPosNoXY("q_root", xml_name="root"),
            *(ObservationType.JointPos(f"q_{j}", xml_name=j) for j in _OBSERVED_JOINTS),
            ObservationType.FreeJointVel("dq_root", xml_name="root"),
            *(ObservationType.JointVel(f"dq_{j}", xml_name=j) for j in _OBSERVED_JOINTS),
        ]

    @staticmethod
    def _get_action_specification(spec: MjSpec) -> List[str]:
        """Returns the action space specification: every actuator in the model."""
        return [actuator.name for actuator in spec.actuators]

    @classmethod
    def get_default_xml_file_path(cls) -> str:
        models_root = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"
        )
        return os.path.join(models_root, "g1_real", "g1_real.xml")

    @info_property
    def default_dataset_source_env(self) -> str:
        return "UnitreeG1"    # use G1 instead of the default SkeletonTorque

    @info_property
    def lafan1_dataset_source_env(self) -> str:
        return "UnitreeG1"    # use G1 instead of the default UnitreeH1v2

    @info_property
    def upper_body_xml_name(self) -> str:
        """Returns the name of the upper body in the Mujoco XML file."""
        return "torso_link"

    @info_property
    def root_height_healthy_range(self) -> Tuple[float, float]:
        """Returns the healthy range of the root height."""
        return (0.5, 1.0)

    @info_property
    def sites_for_mimic(self) -> List[str]:
        """Returns the default sites that are used for mimic."""
        return [
            "upper_body_mimic",
            "head_mimic",
            "pelvis_mimic",
            "left_shoulder_mimic",
            "left_elbow_mimic",
            "left_hand_mimic",
            "left_hip_mimic",
            "left_knee_mimic",
            "left_foot_mimic",
            "right_shoulder_mimic",
            "right_elbow_mimic",
            "right_hand_mimic",
            "right_hip_mimic",
            "right_knee_mimic",
            "right_foot_mimic",
        ]

    @info_property
    def foot_geom_names(self) -> List[str]:
        """Seven capsules per foot, named ``*_footN_collision`` in the MJCF."""
        return [f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)]

    @info_property
    def self_collision_geom_names(self) -> List[str]:
        """Non-foot collision geoms, used by the self-collision reward penalty.

        These have ``condim=1``; the feet have ``condim=3`` and are excluded here so that
        ground contact is not counted as a self-collision.
        """
        feet = set(self.foot_geom_names)
        return [
            name
            for name in (self._model.geom(i).name for i in range(self._model.ngeom))
            if name.endswith("_collision") and name not in feet
        ]

    @info_property
    def tracked_body_names(self) -> List[str]:
        """The 14 tracked bodies.

        The first entry is the anchor body and must stay first -- the goal, reward and
        terminal handler all index it as element 0.
        """
        return [
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ]

    @info_property
    def anchor_body_name(self) -> str:
        """Body the reference motion is anchored to."""
        return "pelvis"

    @info_property
    def end_effector_body_names(self) -> List[str]:
        """Bodies checked by the z-only end-effector termination."""
        return [
            "left_ankle_roll_link",
            "right_ankle_roll_link",
            "left_wrist_yaw_link",
            "right_wrist_yaw_link",
        ]
