"""AgiBot X2 Ultra (new version, 29 actuated DoF) environment for training policies meant
for the real robot (CPU).

It follows :class:`~lmj_real.environments.g1_real.G1Real` and only replaces what depends on
the robot:

* the model is AgiBot's ``X2-Ultra.xml`` for the unit whose nameplate reads "X2 Ultra -N",
  with the free joint renamed to ``root`` and the two head joints fixed;
* actuators are MuJoCo **position servos**, so
  :class:`~lmj_real.core.control_functions.g1_real_position.G1RealPositionControl` drives it
  unchanged;
* each foot is twelve contact spheres rather than seven capsules;
* the hand link is ``*_wrist_roll_link``, the last link of the arm.

The servo gains, the standing pose and the armature are vendor example values that have not
been identified on the robot. See the header of ``x2_ultra_real.xml``.
"""

from __future__ import annotations

import os
from typing import List, Tuple

from loco_mujoco.core import Observation, ObservationType
from loco_mujoco.core.utils import info_property
from mujoco import MjSpec

from lmj_real.environments.g1_real import G1Real


class X2UltraReal(G1Real):
    """AgiBot X2 Ultra with position servos and self-collision."""

    @staticmethod
    def _get_observation_specification(spec: MjSpec) -> List[Observation]:
        """Returns the default observation specification: the root and all 29 joints."""
        joints = [j.name for j in spec.joints if j.name != "root"]
        return [
            ObservationType.FreeJointPosNoXY("q_root", xml_name="root"),
            *(ObservationType.JointPos(f"q_{j}", xml_name=j) for j in joints),
            ObservationType.FreeJointVel("dq_root", xml_name="root"),
            *(ObservationType.JointVel(f"dq_{j}", xml_name=j) for j in joints),
        ]

    @classmethod
    def get_default_xml_file_path(cls) -> str:
        models_root = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"
        )
        return os.path.join(models_root, "x2_ultra_real", "x2_ultra_real.xml")

    @info_property
    def root_height_healthy_range(self) -> Tuple[float, float]:
        """Returns the healthy range of the root height. The pelvis stands at 0.64 m."""
        return (0.4, 0.9)

    @info_property
    def foot_geom_names(self) -> List[str]:
        """Twelve spheres per foot, named ``*_footN_collision`` in the MJCF."""
        return [f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 13)]

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
            "left_wrist_roll_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_roll_link",
        ]

    @info_property
    def end_effector_body_names(self) -> List[str]:
        """Bodies checked by the z-only end-effector termination."""
        return [
            "left_ankle_roll_link",
            "right_ankle_roll_link",
            "left_wrist_roll_link",
            "right_wrist_roll_link",
        ]
