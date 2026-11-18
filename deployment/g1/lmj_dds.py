"""The Unitree SDK side of deployment: G1 motors, ``LowState`` in, ``LowCmd`` out.

There is deliberately no DDS observation builder here. :class:`LowStateAdapter` writes
what the robot reports into an ``MjData`` and runs forward kinematics, after which
:class:`lmj_deploy.ObservationBuilder` works unchanged. A rollout in plain MuJoCo, one over
DDS and one on the robot therefore share a single observation path, and
``validate_bundle.py`` covers all three.

Three orderings meet here, and everything is resolved by name: joint order and actuator
order from the bundle, and the SDK's motor order, :data:`G1_MOTOR_JOINTS`.
"""

from __future__ import annotations

import json
import time

import mujoco
import numpy as np
from lmj_deploy import ObservationBuilder, PolicySpec
from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC

LOWCMD_TOPIC = "rt/lowcmd"
LOWSTATE_TOPIC = "rt/lowstate"
#: Not a Unitree topic: which reference frame the policy is tracking, for viewers.
REFERENCE_TOPIC = "rt/lmj/reference"

#: The robot is on domain 0, so simulation stays on another one.
ROBOT_DOMAIN = 0
SIM_DOMAIN = 1

#: ``motor_cmd`` / ``motor_state`` index -> joint name, for the 29-DoF G1.
G1_MOTOR_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
N_MOTORS = len(G1_MOTOR_JOINTS)

#: ``motor_cmd[i].mode`` that enables a motor. The firmware ignores the others.
MOTOR_ENABLED = 1

#: Bits of ``LowState.wireless_remote[2]``.
REMOTE_START = 0x04
REMOTE_SELECT = 0x08

#: Radians a measured joint may pass the model's range, which contact can briefly push it.
JOINT_LIMIT_MARGIN = 0.1
#: rad/s, about three times the fastest joint in the default clips.
MAX_JOINT_VEL = 25.0
#: Projected gravity z above this means the pelvis has tilted past about 60 degrees.
MAX_GRAVITY_Z = -0.5


def init_dds(domain: int, interface: str) -> None:
    ChannelFactoryInitialize(domain, interface)


class MotorMap:
    """Bundle joint and actuator orders, as motor indices."""

    def __init__(self, spec: PolicySpec):
        index = {n: i for i, n in enumerate(G1_MOTOR_JOINTS)}
        unknown = [n for n in (*spec.joint_names, *spec.actuator_joint_names) if n not in index]
        if unknown:
            raise ValueError(f"bundle joints with no G1 motor: {sorted(set(unknown))}")
        self.joint_motor = np.array([index[n] for n in spec.joint_names], dtype=np.int32)
        self.actuator_motor = np.array([index[n] for n in spec.actuator_joint_names],
                                       dtype=np.int32)

    def from_actuators(self, values: np.ndarray) -> np.ndarray:
        out = np.zeros(N_MOTORS)
        out[self.actuator_motor] = values
        return out

    def from_joints(self, values: np.ndarray) -> np.ndarray:
        out = np.zeros(N_MOTORS)
        out[self.joint_motor] = values
        return out


def remote_pressed(msg: LowState_, button: int) -> bool:
    return bool(msg.wireless_remote[2] & button)


class SafetyMonitor:
    """Checks a raw ``LowState``, before anything downstream trusts it.

    This looks at what the robot measures, never at the reference: retargeted clips can ask
    for joints beyond their range, and that alone is not a fault.
    """

    def __init__(self, model: mujoco.MjModel):
        jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in G1_MOTOR_JOINTS]
        self.lower, self.upper = model.jnt_range[jids].T

    def check(self, msg: LowState_) -> str | None:
        """Why this state is unsafe, or ``None`` if it is fine."""
        motors = msg.motor_state[:N_MOTORS]
        q = np.array([m.q for m in motors])
        dq = np.array([m.dq for m in motors])
        quat = np.asarray(msg.imu_state.quaternion, dtype=np.float64)
        gyro = np.asarray(msg.imu_state.gyroscope, dtype=np.float64)
        if not all(np.all(np.isfinite(v)) for v in (q, dq, quat, gyro)):
            return "non-finite sensor reading"
        if abs(np.linalg.norm(quat) - 1.0) > 0.1:
            return f"bad IMU quaternion {quat.round(3).tolist()}"

        past = np.maximum(self.lower - q, q - self.upper)
        if past.max() > JOINT_LIMIT_MARGIN:
            i = int(past.argmax())
            return f"{G1_MOTOR_JOINTS[i]} is {past[i]:.2f} rad past its range"
        if np.abs(dq).max() > MAX_JOINT_VEL:
            i = int(np.abs(dq).argmax())
            return f"{G1_MOTOR_JOINTS[i]} is moving at {dq[i]:+.1f} rad/s"
        gravity_z = 2.0 * (quat[1] ** 2 + quat[2] ** 2) - 1.0
        if gravity_z > MAX_GRAVITY_Z:
            return f"tilted too far, gravity z {gravity_z:+.2f}"
        return None


class LowStateAdapter:
    """``LowState`` -> ``MjData``, so :class:`ObservationBuilder` can read it.

    The root position is left alone: the robot does not report one, and nothing a policy
    with ``use_anchor_pos=False`` observes depends on it. Tracked bodies are expressed in
    the pelvis frame, and the gyro and quaternion are the pelvis IMU's.
    """

    def __init__(self, builder: ObservationBuilder, motors: MotorMap):
        self.model = builder.model
        self.motors = motors.joint_motor.tolist()
        self.qpos_adr = builder.qpos_adr
        self.dof_adr = builder.dof_adr
        self.quat_adr = builder.root_qpos_adr + 3
        self.gyro_adr = builder.root_dof_adr + 3

    def write(self, msg: LowState_, data: mujoco.MjData) -> None:
        motor = msg.motor_state
        data.qpos[self.qpos_adr] = [motor[i].q for i in self.motors]
        data.qvel[self.dof_adr] = [motor[i].dq for i in self.motors]
        quat = np.asarray(msg.imu_state.quaternion, dtype=np.float64)  # w, x, y, z
        data.qpos[self.quat_adr:self.quat_adr + 4] = quat / np.linalg.norm(quat)
        data.qvel[self.gyro_adr:self.gyro_adr + 3] = msg.imu_state.gyroscope
        mujoco.mj_kinematics(self.model, data)


class LatestLowState:
    """Takes the newest ``LowState`` when asked, once per control tick.

    Deliberately not a callback. With a handler the SDK deserializes every message in
    Python on a DDS thread, which at the robot's 500 Hz takes the GIL from the control
    loop for no benefit: the policy reads 50 of them a second. Without one, the reader
    keeps only the latest sample and nothing runs until :meth:`poll`.
    """

    def __init__(self):
        self.msg: LowState_ | None = None
        self.received_at = 0.0
        self.count = 0
        self._sub = ChannelSubscriber(LOWSTATE_TOPIC, LowState_)
        self._sub.Init()

    def poll(self, timeout: float = 0.01) -> bool:
        """Take a new sample if there is one. ``msg`` keeps the last one if not."""
        msg = self._sub.Read(timeout)
        if msg is None:
            return False
        self.msg = msg
        self.received_at = time.perf_counter()
        self.count += 1
        return True

    @property
    def age(self) -> float:
        return time.perf_counter() - self.received_at


class LowCmdWriter:
    """Publishes joint targets. All arrays are in motor order."""

    def __init__(self):
        self.cmd = unitree_hg_msg_dds__LowCmd_()
        self.cmd.mode_pr = 0
        for i in range(N_MOTORS):
            self.cmd.motor_cmd[i].mode = MOTOR_ENABLED
        self._crc = CRC()
        self._pub = ChannelPublisher(LOWCMD_TOPIC, LowCmd_)
        self._pub.Init()

    def send(self, msg: LowState_, q: np.ndarray, kp: np.ndarray, kd: np.ndarray) -> None:
        # The firmware drops commands whose mode_machine does not match its own.
        self.cmd.mode_machine = msg.mode_machine
        for i, m in enumerate(self.cmd.motor_cmd[:N_MOTORS]):
            m.q, m.dq, m.tau, m.kp, m.kd = float(q[i]), 0.0, 0.0, float(kp[i]), float(kd[i])
        self.cmd.crc = self._crc.Crc(self.cmd)
        self._pub.Write(self.cmd)

    def zero_torque(self, msg: LowState_) -> None:
        zeros = np.zeros(N_MOTORS)
        self.send(msg, zeros, zeros, zeros)

    def damp(self, msg: LowState_, kd: np.ndarray) -> None:
        zeros = np.zeros(N_MOTORS)
        self.send(msg, zeros, zeros, kd)


class ReferencePublisher:
    """Says which reference frame the policy is tracking, as JSON in a ``String_``.

    Nothing on the robot reads it; it is there so a viewer can draw the reference.
    """

    def __init__(self):
        self._pub = ChannelPublisher(REFERENCE_TOPIC, String_)
        self._pub.Init()

    def send(self, motion: str, frame: int, yaw: float) -> None:
        """``yaw`` is the heading rotation the controller applied to the reference."""
        self._pub.Write(String_(json.dumps(
            {"motion": motion, "frame": int(frame), "yaw": float(yaw)})))


class ReferenceListener:
    """The latest message from :class:`ReferencePublisher`."""

    def __init__(self):
        self.latest: dict | None = None
        self.received_at = 0.0
        self._sub = ChannelSubscriber(REFERENCE_TOPIC, String_)
        self._sub.Init(self._on_message, 1)

    def _on_message(self, msg: String_) -> None:
        self.latest = json.loads(msg.data)
        self.received_at = time.perf_counter()

    @property
    def age(self) -> float:
        return time.perf_counter() - self.received_at


def new_low_state() -> LowState_:
    return unitree_hg_msg_dds__LowState_()
