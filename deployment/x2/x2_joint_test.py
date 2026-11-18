"""Control test on a real X2: switch to Develop_MC and swing a few named joints a little.

::

    source ~/x2_aimdk/env.sh
    python3 deployment/x2/x2_joint_test.py --joints left_elbow_joint right_elbow_joint

Run it with the robot hanging. Each named joint fades its stiffness in at the measured pose,
moves to a centre that keeps the swing ``--margin`` inside the model's range, swings
``+-amplitude`` for ``--cycles`` periods, returns to where it started and is damped.

Every group (leg, waist, arm, head) holding a named joint is commanded in full: the other
joints of that group get damping only. Groups without a named joint are not commanded at
all. Gains are AgiBot's ``x2_rl_deploy`` defaults times ``--gain-scale``.

Damps the commanded groups and stops on Ctrl-C, stale state, a named joint leaving its
range or moving faster than ``MAX_SPEED``. The command topics have no timeout, so the
damping command stays in force after exit. The robot is left in Develop_MC; ``--restore``
migrates it back to ``Ready``, which returns to ``Business``.

Needs only rclpy and ``aimdk_msgs``, so it runs with the system Python of ROS 2 Humble.
"""

from __future__ import annotations

import argparse
import math
import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import rclpy
from aimdk_msgs.msg import CommonState, JointCommand, JointCommandArray, JointStateArray, SystemStatus
from aimdk_msgs.srv import GetSystemState, MigrateSystemState
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.signals import SignalHandlerOptions

QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE,
                 history=HistoryPolicy.KEEP_LAST, depth=1)
GROUPS = ("leg", "waist", "arm", "head")
MODEL_XML = (Path(__file__).resolve().parents[2]
             / "lmj_real/models/x2_ultra_real/x2_ultra_real.xml")
#: The model welds the head. AgiBot's SDK reference range for head yaw; head pitch has none.
EXTRA_RANGES = {"head_yaw_joint": (-0.384, 0.384)}

#: AgiBot's x2_rl_deploy motion_control.yaml; joints it does not list use its 20 / 1 default.
DEFAULT_GAINS = {
    "hip_pitch": (40, 4), "hip_roll": (40, 4), "hip_yaw": (30, 3), "knee": (80, 8),
    "ankle_pitch": (40, 4), "ankle_roll": (20, 2),
    "shoulder_pitch": (30, 1), "elbow": (50, 1), "wrist_yaw": (50, 1),
    "waist_yaw": (150, 3), "waist_pitch": (300, 3), "waist_roll": (300, 3),
}
MAX_STATE_AGE = 0.1
#: rad/s. A swing of 0.3 rad at a 4 s period peaks at 0.47 rad/s.
MAX_SPEED = 3.0
RATE_HZ = 100.0


def joint_ranges() -> dict[str, tuple[float, float]]:
    ranges = dict(EXTRA_RANGES)
    for j in ET.parse(MODEL_XML).iter("joint"):
        if "range" in j.attrib:
            lo, hi = map(float, j.attrib["range"].split())
            ranges[j.attrib["name"]] = (lo, hi)
    return ranges


def default_gains(name: str) -> tuple[float, float]:
    key = name.removeprefix("left_").removeprefix("right_").removesuffix("_joint")
    return DEFAULT_GAINS.get(key, (20, 1))


class Fault(RuntimeError):
    pass


class JointTest:
    def __init__(self, node: Node, args):
        self.node, self.args = node, args
        self.state: dict[str, JointStateArray] = {}
        self.stamp: dict[str, float] = {}
        for g in GROUPS:
            node.create_subscription(JointStateArray, f"/aima/hal/joint/{g}/state",
                                     self._on_state(g), QOS)
        self.get_state = node.create_client(GetSystemState, "/aimdk_5Fmsgs/srv/GetSystemState")
        self.migrate = node.create_client(MigrateSystemState, "/aimdk_5Fmsgs/srv/MigrateSystemState")
        self.pubs: dict[str, object] = {}

    def _on_state(self, group):
        def cb(msg):
            self.state[group] = msg
            self.stamp[group] = time.monotonic()
        return cb

    def spin(self, seconds: float) -> None:
        end = time.monotonic() + seconds
        while (left := end - time.monotonic()) > 0:
            rclpy.spin_once(self.node, timeout_sec=min(left, 0.005))

    def names(self, group: str) -> list[str]:
        return [j.name for j in self.state[group].joints]

    def measured(self, group: str, check_age: bool = True) -> dict[str, tuple[float, float, float]]:
        if check_age and time.monotonic() - self.stamp.get(group, -1e9) > MAX_STATE_AGE:
            raise Fault(f"{group} state is stale")
        return {j.name: (j.position, j.velocity, j.effort) for j in self.state[group].joints}

    def print_body(self, label: str) -> None:
        print(f"[x2] {label}")
        for g in GROUPS:
            joints = self.state[g].joints
            print(f"   {g:5s} q   " + " ".join(f"{j.position:+.2f}" for j in joints))
            print(f"   {g:5s} tau " + " ".join(f"{j.effort:+.2f}" for j in joints))

    def call(self, client, request, timeout: float = 5.0, attempts: int = 1):
        """Call a service. A reply can be lost if the call goes out before discovery has
        matched the reply channel, so a query may be retried; a command may not."""
        if not client.wait_for_service(timeout_sec=10.0):
            raise RuntimeError(f"{client.srv_name} is not available")
        for _ in range(attempts):
            if hasattr(request, "header"):
                request.header.header.stamp = self.node.get_clock().now().to_msg()
            future = client.call_async(request)
            rclpy.spin_until_future_complete(self.node, future, timeout_sec=timeout)
            if future.done():
                return future.result()
            client.remove_pending_request(future)
        raise RuntimeError(f"{client.srv_name} did not reply")

    def system_state(self) -> tuple[str, int]:
        r = self.call(self.get_state, GetSystemState.Request(), attempts=3)
        return r.cur_state, r.curr_status.value

    def migrate_to(self, target: str, done: tuple[str, ...], timeout: float = 120.0) -> None:
        state, status = self.system_state()
        print(f"[x2] system state {state!r}, status {status}")
        if state in done and status == SystemStatus.IN_READY:
            return
        req = MigrateSystemState.Request()
        req.state = target
        try:
            r = self.call(self.migrate, req, timeout=10.0)
        except RuntimeError as e:  # not resent: the request may have arrived anyway
            print(f"[x2] {e}; watching the system state instead")
        else:
            if r.header.status.value != CommonState.SUCCESS:
                raise RuntimeError(f"MigrateSystemState({target}) refused: {r.header.message!r}")
            print(f"[x2] migrating to {target}: {r.header.message!r} (status 2 = in progress)")
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            self.spin(1.0)
            state, status = self.system_state()
            print(f"[x2]   {state!r}, status {status}")
            if state in done and status == SystemStatus.IN_READY:
                return
        raise RuntimeError(f"{target} not reached within {timeout:.0f} s")

    def take_over(self, groups: list[str], timeout: float = 15.0) -> None:
        """Wait until native control has stopped commanding ``groups``, then publish on them."""
        end = time.monotonic() + timeout
        for g in groups:
            topic = f"/aima/hal/joint/{g}/command"
            while (n := self.node.count_publishers(topic)) > 0:
                if time.monotonic() > end:
                    raise RuntimeError(f"{n} publisher(s) still on {topic}; native control is running")
                self.spin(0.5)
            self.pubs[g] = self.node.create_publisher(JointCommandArray, topic, QOS)

    def send(self, group: str, commands: dict[str, tuple[float, float, float]]) -> None:
        """``commands``: joint -> (position, stiffness, damping), for every joint of ``group``."""
        msg = JointCommandArray()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.joints = [JointCommand(name=n, position=commands[n][0], velocity=0.0, effort=0.0,
                                   stiffness=commands[n][1], damping=commands[n][2])
                      for n in self.names(group)]
        self.pubs[group].publish(msg)

    def run(self, joints: list[str], ranges, gains) -> None:
        a = self.args
        group_of = {n: g for g in GROUPS for n in self.names(g)}
        groups = sorted({group_of[n] for n in joints}, key=GROUPS.index)
        q0, centre = {}, {}
        for n in joints:
            q0[n] = self.measured(group_of[n])[n][0]
            lo, hi = ranges[n]
            c_lo, c_hi = lo + a.margin + a.amplitude, hi - a.margin - a.amplitude
            if c_lo > c_hi:
                raise SystemExit(f"{n}: range [{lo:+.3f}, {hi:+.3f}] is too narrow for "
                                 f"+-{a.amplitude} with a {a.margin} margin")
            centre[n] = min(max(q0[n], c_lo), c_hi)
            kp, kd = gains[n]
            print(f"[x2] {n:28s} at {q0[n]:+.3f}, range [{lo:+.3f}, {hi:+.3f}], swing "
                  f"{centre[n] - a.amplitude:+.3f}..{centre[n] + a.amplitude:+.3f}, "
                  f"kp {kp:.1f} kd {kd:.2f}")

        dt = 1.0 / RATE_HZ
        n_fade, n_move = int(1.0 / dt), int(2.0 / dt)
        n_sine = int(a.cycles * a.period / dt)
        log = {n: [] for n in joints}
        t0 = time.monotonic()
        for i in range(n_fade + n_move + n_sine + n_move):
            while time.monotonic() < t0 + i * dt:
                rclpy.spin_once(self.node, timeout_sec=0.001)
            fade = min(1.0, (i + 1) / n_fade)
            for g in groups:
                now = self.measured(g)
                cmd = {name: (q, 0.0, gains[name][1]) for name, (q, _, _) in now.items()}
                for n in joints:
                    if group_of[n] != g:
                        continue
                    q, dq, tau = now[n]
                    lo, hi = ranges[n]
                    if not lo - a.margin <= q <= hi + a.margin:
                        raise Fault(f"{n} at {q:+.3f} is outside [{lo:+.3f}, {hi:+.3f}]")
                    if abs(dq) > MAX_SPEED:
                        raise Fault(f"{n} moving at {dq:+.2f} rad/s")
                    if i < n_fade:
                        target = q0[n]
                    elif i < n_fade + n_move:
                        target = q0[n] + (centre[n] - q0[n]) * (i - n_fade + 1) / n_move
                    elif i < n_fade + n_move + n_sine:
                        s = (i - n_fade - n_move) * dt
                        target = centre[n] + a.amplitude * math.sin(2 * math.pi * s / a.period)
                        log[n].append((target, q, tau))
                    else:
                        k = (i - n_fade - n_move - n_sine + 1) / n_move
                        target = centre[n] + (q0[n] - centre[n]) * k
                    target = min(max(target, ranges[n][0]), ranges[n][1])
                    cmd[n] = (target, fade * gains[n][0], gains[n][1])
                self.send(g, cmd)
        for n in joints:
            qs = [q for _, q, _ in log[n]]
            err = [abs(t - q) for t, q, _ in log[n]]
            tau = max(abs(x) for _, _, x in log[n])
            print(f"[x2] {n:28s} swing {max(qs) - min(qs):.3f} rad (commanded "
                  f"{2 * a.amplitude:.3f}); error mean {sum(err) / len(err):.3f}, "
                  f"max {max(err):.3f} rad; |tau| max {tau:.2f} Nm")

    def damp(self, gains, seconds: float = 1.0) -> None:
        if not self.pubs:
            return
        print(f"[x2] damping {', '.join(self.pubs)}")
        for _ in range(int(seconds * RATE_HZ)):
            for g in self.pubs:
                # The last known pose, stale or not: with zero stiffness the target does not matter.
                now = self.measured(g, check_age=False)
                self.send(g, {n: (q, 0.0, gains[n][1]) for n, (q, _, _) in now.items()})
            self.spin(1.0 / RATE_HZ)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--joints", nargs="+", required=True, help="joint names, e.g. left_elbow_joint")
    ap.add_argument("--amplitude", type=float, default=0.15, help="rad either side of the centre")
    ap.add_argument("--margin", type=float, default=0.1, help="rad kept clear of the range ends")
    ap.add_argument("--period", type=float, default=4.0, help="s per swing")
    ap.add_argument("--cycles", type=float, default=2.0)
    ap.add_argument("--gain-scale", type=float, default=0.5,
                    help="fraction of AgiBot's default kp and kd")
    ap.add_argument("--restore", action="store_true",
                    help="migrate back to Ready (then Business) at the end")
    ap.add_argument("--domain", type=int, default=0, help="ROS domain; the robot is on 0")
    args = ap.parse_args()
    if not 0.0 < args.amplitude <= 0.3:
        raise SystemExit("--amplitude must be in (0, 0.3]")
    if not 0.0 < args.gain_scale <= 1.0:
        raise SystemExit("--gain-scale must be in (0, 1]")
    if args.period < 2.0:
        raise SystemExit("--period must be at least 2 s")
    ranges = joint_ranges()
    unknown = [n for n in args.joints if n not in ranges]
    if unknown:
        raise SystemExit(f"no known range for {unknown}")
    if os.environ.get("RMW_IMPLEMENTATION") != "rmw_fastrtps_cpp":
        print("[x2] warning: RMW_IMPLEMENTATION is not rmw_fastrtps_cpp; source ~/x2_aimdk/env.sh")

    os.environ["ROS_LOCALHOST_ONLY"] = "0"
    rclpy.init(domain_id=args.domain, signal_handler_options=SignalHandlerOptions.NO)
    node = Node("lmj_x2_joint_test")
    test = JointTest(node, args)
    try:
        test.spin(1.0)
        missing = [g for g in GROUPS if g not in test.state]
        if missing:
            raise SystemExit(f"no state from {missing}; is the robot on domain {args.domain}?")
        group_of = {n: g for g in GROUPS for n in test.names(g)}
        absent = [n for n in args.joints if n not in group_of]
        if absent:
            raise SystemExit(f"the robot reports no joint named {absent}")
        gains = {n: tuple(args.gain_scale * x for x in default_gains(n)) for n in group_of}
        test.print_body("before")
        test.migrate_to("Develop_MC", ("Develop_MC",))
        test.take_over(sorted({group_of[n] for n in args.joints}, key=GROUPS.index))
        test.spin(0.5)
        try:
            test.run(args.joints, ranges, gains)
        except Fault as e:
            print(f"[x2] stopping: {e}")
        finally:
            test.damp(gains)
        test.print_body("after")
        if args.restore:
            test.migrate_to("Ready", ("Ready", "Business"))
        else:
            print("[x2] left in Develop_MC; --restore, or a reboot, brings native control back")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
