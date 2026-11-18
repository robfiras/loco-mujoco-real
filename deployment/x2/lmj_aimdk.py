"""The AimDK side of deployment: the X2's ROS 2 topics, joint state in, joint commands out.

The counterpart of :mod:`lmj_dds` for the AgiBot X2. As there, nothing here builds an
observation: :class:`StateAdapter` writes what the robot reports into an ``MjData``, after
which :class:`lmj_deploy.ObservationBuilder` works unchanged.

The interface is the robot's hardware layer, as AgiBot's ``x2_rl_deploy`` example uses it:

==================================  ================================  =====================
topic                               type                              direction
==================================  ================================  =====================
``/aima/hal/joint/<group>/state``   ``aimdk_msgs/JointStateArray``    robot -> controller
``/aima/hal/joint/<group>/command`` ``aimdk_msgs/JointCommandArray``  controller -> robot
``/aima/hal/imu/torso/state``       ``sensor_msgs/Imu``               robot -> controller
``/aima/hal/imu/chest/state``       ``sensor_msgs/Imu``               robot -> controller
``/joy``                            ``sensor_msgs/Joy``               gamepad -> controller
==================================  ================================  =====================

``<group>`` is one of ``leg``, ``waist``, ``arm`` and ``head``.

A command is a target angle with a stiffness and a damping per joint, and the robot closes
that loop itself. It has no timeout: the last command stays in force until another arrives.

In AgiBot's simulator the ``torso`` IMU sits on the pelvis and the ``chest`` IMU on
``torso_link``. Which body the real ``torso`` IMU is on has not been checked on the robot.

Needs ROS 2 Humble and the ``aimdk_msgs`` package from the AimDK, built and sourced.
"""

from __future__ import annotations

import atexit
import json
import math
import os
import queue
import struct
import sys
import termios
import threading
import time
import tty

import mujoco
import numpy as np
import rclpy
from aimdk_msgs.msg import JointCommand, JointCommandArray, JointState, JointStateArray
from lmj_deploy import ObservationBuilder, PolicySpec
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import Imu, Joy
from std_msgs.msg import String

_LEG = ["hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll"]
_ARM = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
        "wrist_yaw", "wrist_pitch", "wrist_roll"]

#: Joint group -> joint names, in the order of the group's message.
GROUPS: dict[str, list[str]] = {
    "leg": [f"{side}_{j}_joint" for side in ("left", "right") for j in _LEG],
    "waist": ["waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint"],
    "arm": [f"{side}_{j}_joint" for side in ("left", "right") for j in _ARM],
    "head": ["head_yaw_joint", "head_pitch_joint"],
}
#: The groups a tracking policy drives. The head is only held still.
BODY_GROUPS = ("leg", "waist", "arm")

IMU_TOPIC = "/aima/hal/imu/torso/state"
CHEST_IMU_TOPIC = "/aima/hal/imu/chest/state"
JOY_TOPIC = "/joy"
#: Not an AimDK topic: which reference frame the policy is tracking, for viewers.
REFERENCE_TOPIC = "/lmj/reference"

#: ROS 2's default domain, so the one a robot on the network is assumed to be on.
#: Simulation stays on another.
ROBOT_DOMAIN = 0
SIM_DOMAIN = 1

#: ``Joy.buttons`` indices, as AgiBot's example maps a PS5 pad: circle damps, square runs.
JOY_DAMP = 1
JOY_START = 3

#: What the head joints are held at zero with. AgiBot's example uses these as its defaults.
HEAD_STIFFNESS = 20.0
HEAD_DAMPING = 1.0

#: Radians a measured joint may pass the model's range, which contact can briefly push it.
JOINT_LIMIT_MARGIN = 0.1
#: rad/s, about three times the fastest joint in the default clips.
MAX_JOINT_VEL = 25.0
#: Projected gravity z above this means the pelvis has tilted past about 60 degrees.
MAX_GRAVITY_Z = -0.5

#: What the controller listens and commands with. Best effort, as the robot's topics are,
#: and volatile, which matches both a volatile and a transient-local publisher.
LATEST_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                        durability=DurabilityPolicy.VOLATILE,
                        history=HistoryPolicy.KEEP_LAST, depth=1)
#: What the robot publishes its state with.
STATE_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                       durability=DurabilityPolicy.TRANSIENT_LOCAL,
                       history=HistoryPolicy.KEEP_LAST, depth=1)


def state_topic(group: str) -> str:
    return f"/aima/hal/joint/{group}/state"


def command_topic(group: str) -> str:
    return f"/aima/hal/joint/{group}/command"


def init_ros(domain: int, name: str, network: bool = False) -> Node:
    """Start ROS 2 on ``domain`` and return a node.

    Traffic stays on this machine unless ``network`` is set. A ROS environment exports
    ``ROS_LOCALHOST_ONLY=0`` on activation, so that variable is set here rather than read.
    """
    os.environ["ROS_LOCALHOST_ONLY"] = "0" if network else "1"
    # No ROS signal handlers: Ctrl-C must reach Python, so the controller can damp.
    rclpy.init(domain_id=domain, signal_handler_options=SignalHandlerOptions.NO)
    return Node(name)


def shutdown_ros(node: Node) -> None:
    """Leave ROS 2. Without this the process does not exit."""
    node.destroy_node()
    rclpy.try_shutdown()


class Latest:
    """The newest message on a topic, taken when asked.

    Deliberately not a callback. With one, every message is deserialized in Python on an
    executor thread, which at the robot's rates takes the GIL from the control loop for no
    benefit: the policy reads 50 states a second. Here nothing runs until :meth:`poll`.
    """

    def __init__(self, node: Node, msg_type, topic: str):
        self.msg = None
        self.received_at = 0.0
        self.count = 0
        self._type = msg_type
        self._sub = node.create_subscription(msg_type, topic, lambda _: None, LATEST_QOS)

    def poll(self) -> bool:
        """Take a new sample if there is one. ``msg`` keeps the last one if not."""
        handle = self._sub.handle
        with handle:
            taken = handle.take_message(self._type, False)
        if taken is None:
            return False
        self.msg = taken[0]
        self.received_at = time.perf_counter()
        self.count += 1
        return True

    @property
    def age(self) -> float:
        return time.perf_counter() - self.received_at


def joint_names(group: str, joints) -> list[str]:
    """Names of a message's joints. ``name`` is optional, so fall back to the group's order."""
    return [j.name or GROUPS[group][i] for i, j in enumerate(joints)]


class RobotState:
    """The latest joint state of the body groups and the pelvis IMU, in bundle joint order."""

    def __init__(self, node: Node, spec: PolicySpec):
        self.names = list(spec.joint_names)
        self._index = {n: i for i, n in enumerate(self.names)}
        self._groups = {g: Latest(node, JointStateArray, state_topic(g)) for g in BODY_GROUPS}
        self._imu = Latest(node, Imu, IMU_TOPIC)
        self._joy = Latest(node, Joy, JOY_TOPIC)
        self.q = np.zeros(len(self.names))
        self.dq = np.zeros(len(self.names))
        self.quat = np.array([1.0, 0.0, 0.0, 0.0])  # w, x, y, z
        self.gyro = np.zeros(3)
        self.error: str | None = None

    def poll(self) -> None:
        for group, latest in self._groups.items():
            if not latest.poll():
                continue
            msg = latest.msg
            for name, joint in zip(joint_names(group, msg.joints), msg.joints):
                if name in self._index:
                    self.q[self._index[name]] = joint.position
                    self.dq[self._index[name]] = joint.velocity
            codes = [j.error_code for j in msg.joints]
            if msg.state.value or any(codes):
                self.error = f"{group} joints report state {msg.state.value}, errors {codes}"
        if self._imu.poll():
            o, w = self._imu.msg.orientation, self._imu.msg.angular_velocity
            self.quat = np.array([o.w, o.x, o.y, o.z])
            self.gyro = np.array([w.x, w.y, w.z])
        self._joy.poll()

    @property
    def complete(self) -> bool:
        """Whether every source has been heard from at least once."""
        return all(s.count for s in (*self._groups.values(), self._imu))

    @property
    def age(self) -> float:
        """Seconds since the stalest source last reported."""
        return max(s.age for s in (*self._groups.values(), self._imu))

    @property
    def sequence(self) -> int:
        """The leg message's counter, for telling how many states pass per control step."""
        msg = self._groups["leg"].msg
        return 0 if msg is None else int(msg.header.sequence)

    def pressed(self, button: int) -> bool:
        """Whether a gamepad button is down. A press older than a second is ignored."""
        joy = self._joy
        return bool(joy.msg is not None and joy.age < 1.0
                    and len(joy.msg.buttons) > button and joy.msg.buttons[button])


class SafetyMonitor:
    """Checks a raw state, before anything downstream trusts it.

    This looks at what the robot measures, never at the reference: retargeted clips can ask
    for joints beyond their range, and that alone is not a fault.
    """

    def __init__(self, model: mujoco.MjModel, names: list[str]):
        self.names = names
        jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in names]
        self.lower, self.upper = model.jnt_range[jids].T

    def check(self, state: RobotState) -> str | None:
        """Why this state is unsafe, or ``None`` if it is fine."""
        q, dq, quat, gyro = state.q, state.dq, state.quat, state.gyro
        if state.error:
            return state.error
        if not all(np.all(np.isfinite(v)) for v in (q, dq, quat, gyro)):
            return "non-finite sensor reading"
        if abs(np.linalg.norm(quat) - 1.0) > 0.1:
            return f"bad IMU quaternion {quat.round(3).tolist()}"

        past = np.maximum(self.lower - q, q - self.upper)
        if past.max() > JOINT_LIMIT_MARGIN:
            i = int(past.argmax())
            return f"{self.names[i]} is {past[i]:.2f} rad past its range"
        if np.abs(dq).max() > MAX_JOINT_VEL:
            i = int(np.abs(dq).argmax())
            return f"{self.names[i]} is moving at {dq[i]:+.1f} rad/s"
        gravity_z = 2.0 * (quat[1] ** 2 + quat[2] ** 2) - 1.0
        if gravity_z > MAX_GRAVITY_Z:
            return f"tilted too far, gravity z {gravity_z:+.2f}"
        return None


class StateAdapter:
    """:class:`RobotState` -> ``MjData``, so :class:`ObservationBuilder` can read it.

    The root position is left alone: the robot does not report one, and nothing a policy
    with ``use_anchor_pos=False`` observes depends on it. The IMU's orientation and angular
    velocity are taken as the pelvis's, which holds while its frame is parallel to the
    pelvis frame, as it is in the model.
    """

    def __init__(self, builder: ObservationBuilder):
        self.model = builder.model
        self.qpos_adr = builder.qpos_adr
        self.dof_adr = builder.dof_adr
        self.quat_adr = builder.root_qpos_adr + 3
        self.gyro_adr = builder.root_dof_adr + 3

    def write(self, state: RobotState, data: mujoco.MjData) -> None:
        data.qpos[self.qpos_adr] = state.q
        data.qvel[self.dof_adr] = state.dq
        data.qpos[self.quat_adr:self.quat_adr + 4] = state.quat / np.linalg.norm(state.quat)
        data.qvel[self.gyro_adr:self.gyro_adr + 3] = state.gyro
        mujoco.mj_kinematics(self.model, data)


class CommandWriter:
    """Publishes joint targets. All arrays are in the bundle's actuator order."""

    def __init__(self, node: Node, spec: PolicySpec):
        self._node = node
        index = {n: i for i, n in enumerate(spec.actuator_joint_names)}
        missing = sorted(set(index) - {n for g in BODY_GROUPS for n in GROUPS[g]})
        if missing:
            raise ValueError(f"bundle joints the X2 does not have: {missing}")
        self._slots = {g: [index.get(n) for n in names] for g, names in GROUPS.items()}
        self._msgs = {g: JointCommandArray(joints=[JointCommand(name=n) for n in names])
                      for g, names in GROUPS.items()}
        self._pubs = {g: node.create_publisher(JointCommandArray, command_topic(g), LATEST_QOS)
                      for g in GROUPS}
        self._sequence = 0

    def send(self, q: np.ndarray, kp: np.ndarray, kd: np.ndarray) -> None:
        stamp = self._node.get_clock().now().to_msg()
        self._sequence += 1
        held = bool(np.any(kp))
        for group, msg in self._msgs.items():
            msg.header.stamp, msg.header.sequence = stamp, self._sequence
            for joint, i in zip(msg.joints, self._slots[group]):
                if i is None:  # a joint the policy does not drive: keep it at zero
                    joint.position = 0.0
                    joint.stiffness = HEAD_STIFFNESS if held else 0.0
                    joint.damping = HEAD_DAMPING
                else:
                    joint.position = float(q[i])
                    joint.stiffness, joint.damping = float(kp[i]), float(kd[i])
            self._pubs[group].publish(msg)

    def damp(self, kd: np.ndarray) -> None:
        zeros = np.zeros(len(kd))
        self.send(zeros, zeros, kd)


class ReferencePublisher:
    """Says which reference frame the policy is tracking, as JSON in a ``String``.

    Nothing on the robot reads it; it is there so a viewer can draw the reference.
    """

    def __init__(self, node: Node):
        self._pub = node.create_publisher(String, REFERENCE_TOPIC, LATEST_QOS)

    def send(self, motion: str, frame: int, yaw: float) -> None:
        """``yaw`` is the heading rotation the controller applied to the reference."""
        self._pub.publish(String(data=json.dumps(
            {"motion": motion, "frame": int(frame), "yaw": float(yaw)})))


#: The system mode in which AgiBot's motion control is stopped and the joints are ours.
DEVELOP_MC = "Develop_MC"


class SystemMode:
    """The robot's system state machine, which decides who commands the joints.

    In ``Business`` AgiBot's motion control publishes on the joint command topics; in
    ``Develop_MC`` it is stopped. A migration is asynchronous: the call only says the request
    was accepted, then the state reads ``IN_MOVE`` for tens of seconds (36 s on our robot).

    Needs the SDK's full ``aimdk_msgs``. The RL deploy extras' subset lacks these services,
    so they are imported here, where only the robot path reaches them.
    """

    def __init__(self, node: Node):
        from aimdk_msgs.srv import GetSystemState, MigrateSystemState
        self._node = node
        self._get = node.create_client(GetSystemState, "/aimdk_5Fmsgs/srv/GetSystemState")
        self._migrate = node.create_client(MigrateSystemState,
                                           "/aimdk_5Fmsgs/srv/MigrateSystemState")

    def _call(self, client, request, timeout: float, attempts: int):
        """A reply is lost if the call goes out before discovery has matched the reply
        channel, so a query may be retried. A migration may not."""
        if not client.wait_for_service(timeout_sec=10.0):
            raise RuntimeError(f"{client.srv_name} is not available")
        for _ in range(attempts):
            request.header.header.stamp = self._node.get_clock().now().to_msg()
            future = client.call_async(request)
            rclpy.spin_until_future_complete(self._node, future, timeout_sec=timeout)
            if future.done():
                return future.result()
            client.remove_pending_request(future)
        raise RuntimeError(f"{client.srv_name} did not reply")

    def get(self) -> tuple[str, int]:
        """The current mode and its status (``SystemStatus``; 1 is ready, 2 is migrating)."""
        r = self._call(self._get, self._get.srv_type.Request(), timeout=5.0, attempts=3)
        return r.cur_state, r.curr_status.value

    def enter(self, target: str, timeout: float = 120.0) -> None:
        """Migrate to ``target`` and wait until it is ready."""
        from aimdk_msgs.msg import CommonState, SystemStatus
        state, status = self.get()
        print(f"[aimdk] system state {state!r}, status {status}")
        if state == target and status == SystemStatus.IN_READY:
            return
        request = self._migrate.srv_type.Request()
        request.state = target
        try:
            r = self._call(self._migrate, request, timeout=10.0, attempts=1)
        except RuntimeError as e:
            print(f"[aimdk] {e}; watching the system state instead")
        else:
            if r.header.status.value != CommonState.SUCCESS:
                raise RuntimeError(f"MigrateSystemState({target}) refused: {r.header.message!r}")
            print(f"[aimdk] migrating to {target}, which takes tens of seconds")
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            time.sleep(1.0)
            state, status = self.get()
            print(f"[aimdk]   {state!r}, status {status}")
            if state == target and status == SystemStatus.IN_READY:
                return
        raise RuntimeError(f"{target} not reached within {timeout:.0f} s")


def wait_for_command_topics(node: Node, timeout: float = 15.0) -> None:
    """Wait until nothing publishes joint commands, i.e. native motion control has stopped.

    Call it before creating a :class:`CommandWriter`, whose own publishers would count.
    """
    end = time.monotonic() + timeout
    for group in GROUPS:
        while (n := node.count_publishers(command_topic(group))) > 0:
            if time.monotonic() > end:
                raise RuntimeError(f"{n} publisher(s) still on {command_topic(group)}")
            time.sleep(0.5)


class Keyboard:
    """Keys pressed in the controller's terminal, each acted on as it is pressed.

    The robot publishes no ``/joy``, so on the robot this stands in for the gamepad. A
    terminal is put in cbreak mode, so keys need no Enter and are not echoed, and restored
    at exit; Ctrl-C still interrupts. Keys are read on a daemon thread, so the control loop
    never waits on stdin.
    """

    HELP = ("Enter start the policy, Enter again play the clip | n / p next / previous clip "
            "| s default pose | Ctrl-C damp and quit")

    def __init__(self, path: str | None = None):
        """Read the terminal, or with ``path`` a named pipe, which is created if missing.

        A pipe lets a controller run detached, on the robot, with keys sent in by whoever
        writes to it (``echo -n d > path``); a lost SSH session then cannot reach the loop.
        """
        self.start = threading.Event()
        self.safe = threading.Event()
        self.damp = threading.Event()
        self._steps: queue.SimpleQueue[int] = queue.SimpleQueue()
        self.last_heartbeat = -math.inf  # perf_counter time of the last ``h``
        self._pipe = bool(path)
        if path:
            if not os.path.exists(path):
                os.mkfifo(path)
            # Opened read-write, so that a writer closing it never reads as end of file.
            self._fd = os.open(path, os.O_RDWR)
        else:
            self._fd = sys.stdin.fileno()
            if os.isatty(self._fd):
                saved = termios.tcgetattr(self._fd)
                tty.setcbreak(self._fd)
                atexit.register(termios.tcsetattr, self._fd, termios.TCSADRAIN, saved)
        threading.Thread(target=self._read, daemon=True).start()

    def step(self) -> int:
        """Clips to move by since last asked: +1 per ``n``, -1 per ``p``."""
        total = 0
        while not self._steps.empty():
            total += self._steps.get_nowait()
        return total

    def press_step(self, step: int) -> None:
        self._steps.put(step)

    @property
    def heartbeat_age(self) -> float:
        """Seconds since an operator link last sent ``h`` (see soc1/x2_operator.py)."""
        return time.perf_counter() - self.last_heartbeat

    def _read(self) -> None:
        while chunk := os.read(self._fd, 1):
            key = chunk.decode(errors="ignore").lower()
            if key == "h":
                self.last_heartbeat = time.perf_counter()
            elif key in ("\n", "\r", " "):
                self.start.set()
            elif key == "s":
                self.safe.set()
            elif key == "d" and self._pipe:
                # Only from the pipe, where the operator sends it for Ctrl-C. In a terminal
                # d sits next to s, the default pose: there Ctrl-C damps instead.
                self.damp.set()
            elif key in ("n", "p"):
                self.press_step(1 if key == "n" else -1)


class Gamepad:
    """A USB gamepad read straight from ``/dev/input/js*``, feeding a :class:`Keyboard`.

    Local to this machine, so nothing goes onto the robot's network. Button numbers are the
    Linux xpad driver's, i.e. a Logitech F710 with its switch on X (XInput):

    =========  =========================================
    A          as Enter: start the policy, then play
    B          as ``s``: the policy tracks the default pose
    LB / RB    previous / next clip
    LT + B     damp
    =========  =========================================

    Only presses count, not buttons already held when it opens. Wireless pads drop out
    silently, so this is a convenience, not a dead-man switch.
    """

    HELP = "gamepad: A start / play | B default pose | LB / RB clip | LT + B damp"
    A, B, LB, RB = 0, 1, 4, 5
    LT_AXIS = 2
    #: Of +-32767: how far LT must be pulled to count as held.
    LT_HELD = 16000

    def __init__(self, keys: Keyboard, path: str):
        self.keys, self.path = keys, path
        self._fd = os.open(path, os.O_RDONLY)
        self._lt = -32767
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        while True:
            try:
                event = os.read(self._fd, 8)
            except OSError:
                print(f"[deploy] gamepad {self.path} lost; the keyboard still works")
                return
            _, value, kind, number = struct.unpack("IhBB", event)
            if kind & 0x80:  # the state on opening, not a press
                if kind & 0x02 and number == self.LT_AXIS:
                    self._lt = value
                continue
            if kind == 0x02 and number == self.LT_AXIS:
                self._lt = value
            elif kind == 0x01 and value == 1:
                if number == self.A:
                    self.keys.start.set()
                elif number == self.B:
                    (self.keys.damp if self._lt > self.LT_HELD else self.keys.safe).set()
                elif number in (self.LB, self.RB):
                    self.keys.press_step(1 if number == self.RB else -1)


__all__ = [
    "BODY_GROUPS", "CHEST_IMU_TOPIC", "DEVELOP_MC", "GROUPS", "Gamepad", "IMU_TOPIC", "JOY_DAMP",
    "JOY_START", "JOY_TOPIC", "LATEST_QOS", "REFERENCE_TOPIC", "ROBOT_DOMAIN", "SIM_DOMAIN",
    "STATE_QOS", "CommandWriter", "Imu", "JointCommandArray", "JointState", "JointStateArray",
    "Joy", "Keyboard", "Latest", "ReferencePublisher", "RobotState", "SafetyMonitor",
    "StateAdapter", "String", "SystemMode", "command_topic", "init_ros", "joint_names",
    "shutdown_ros", "state_topic", "wait_for_command_topics",
]
