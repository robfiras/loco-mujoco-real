"""Runs an exported bundle over the Unitree SDK, against ``sim2sim_bridge.py`` or the robot.

::

    python deployment/g1/deploy_g1.py --bundle <dir> --motion <clip>
    python deployment/g1/deploy_g1.py --bundle <dir> --motion <clip> --domain 0 --interface <eth>

States: wait for ``LowState`` -> ramp to the clip's first pose -> hold -> policy -> damp.
The hold lasts until START is pressed on the remote, or ``--auto-start`` in simulation.

Nothing is commanded if the first ``LowState`` already fails the checks in
:class:`lmj_dds.SafetyMonitor`. After that, every state goes straight to damp on SELECT, a
failed check, stale state or a non-finite action, and so does Ctrl-C or an exception.

Only a bundle whose actor needs no world position runs here, since the robot reports
none: PPO with ``use_anchor_pos=False``, or SuperMimic. Yaw is relative: when the
policy starts, the reference is rotated onto the robot's heading, so IMU yaw only has to
hold for the length of the clip.

Domain 0 is the robot, reached from a laptop on its Ethernet port; the default is domain 1,
the bridge's. On the robot the policy starts on START only: ``--auto-start`` is refused
there. Hang the robot first, since damping lets it fall.
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]
from lmj_deploy import (  # noqa: E402
    MotionBundle,
    ObservationBuilder,
    PolicySpec,
    PositionActionMapper,
    load_actor,
)
from lmj_dds import (  # noqa: E402
    REMOTE_SELECT,
    REMOTE_START,
    ROBOT_DOMAIN,
    SIM_DOMAIN,
    LatestLowState,
    LowCmdWriter,
    LowStateAdapter,
    MotorMap,
    ReferencePublisher,
    SafetyMonitor,
    init_dds,
    remote_pressed,
)

_G1_XML = Path(__file__).resolve().parents[2] / "lmj_real" / "models" / "g1_real" / "g1_real.xml"

#: Seconds without a new ``LowState`` before the controller gives up.
MAX_STATE_AGE = 0.1
#: rad/s. The ramp takes longer than ``--ramp-seconds`` if a joint would go faster than this.
MAX_RAMP_SPEED = 0.5


class Controller:
    def __init__(self, args):
        self.spec = PolicySpec.load(args.bundle)
        if self.spec.needs_world_position:
            raise SystemExit("this bundle observes motion_anchor_pos_b, which needs a world "
                             "position. Retrain with goal_params.use_anchor_pos=false.")
        self.actor = load_actor(args.bundle, self.spec, std_floor=args.std_floor)
        self.motions = MotionBundle(args.bundle, spec=self.spec)
        self.motion = args.motion or self.motions.names[0]
        self.start_frame = args.start_frame
        self.ramp_seconds = args.ramp_seconds
        self.hold_seconds = args.hold_seconds
        self.damp_seconds = args.damp_seconds
        self.auto_start = args.auto_start
        self.n_steps = args.n_steps or self.motions.length(self.motion) - self.start_frame

        # Kinematics only: this model is never stepped.
        self.model = mujoco.MjModel.from_xml_path(str(_G1_XML))
        self.data = mujoco.MjData(self.model)
        self.builder = ObservationBuilder(self.spec, self.model)
        self.mapper = PositionActionMapper(self.spec, self.model)
        self.motors = MotorMap(self.spec)
        self.adapter = LowStateAdapter(self.builder, self.motors)
        self.safety = SafetyMonitor(self.model)
        self.kp = self.motors.from_actuators(np.asarray(self.spec.kp))
        self.kd = self.motors.from_actuators(np.asarray(self.spec.kd))
        self.dt = self.spec.control_dt

        self.state = LatestLowState()
        self.writer = LowCmdWriter()
        self.reference = ReferencePublisher()
        self.last_action = np.zeros(self.spec.action_dim)
        self.compute_ms: list[float] = []
        self.jitter_ms: list[float] = []
        self.ticks_per_step: list[int] = []

    def wait_for_state(self, timeout: float = 10.0) -> None:
        print("[deploy] waiting for LowState")
        t0 = time.perf_counter()
        while not self.state.poll(1.0):
            if time.perf_counter() - t0 > timeout:
                raise SystemExit("no LowState received; is the bridge running on this domain?")
        if fault := self.safety.check(self.state.msg):
            raise SystemExit(f"refusing to start: {fault}")

    def _ticks(self, n: int | None = None):
        """Yield on the control clock, ``n`` times or forever, recording how late each is."""
        t0 = time.perf_counter()
        for i in itertools.count() if n is None else range(n):
            target = t0 + i * self.dt
            while (wait := target - time.perf_counter()) > 0:
                time.sleep(min(wait, 0.001))
            self.jitter_ms.append((time.perf_counter() - target) * 1e3)
            yield i

    def _fault(self) -> str | None:
        """Refresh the state, and say why the robot must stop, if it must."""
        self.state.poll()
        if self.state.age > MAX_STATE_AGE:
            return f"LowState is {self.state.age * 1e3:.0f} ms old"
        if remote_pressed(self.state.msg, REMOTE_SELECT):
            return "SELECT pressed"
        return self.safety.check(self.state.msg)

    def ramp_and_hold(self) -> bool:
        """Move from the measured pose to the clip's first pose, and stay until told to go."""
        ref = self.motions.frame(self.motion, self.start_frame)
        # Clips can pass a joint's range, and the ramp must not push it into the stop.
        goal = np.clip(self.motors.from_joints(ref["joint_pos"]),
                       self.safety.lower, self.safety.upper)
        start = np.array([m.q for m in self.state.msg.motor_state[:len(goal)]])
        ramp_seconds = max(self.ramp_seconds, np.abs(goal - start).max() / MAX_RAMP_SPEED)
        print(f"[deploy] ramp {ramp_seconds:.1f} s, hold {self.hold_seconds:.1f} s")
        n_ramp = int(ramp_seconds / self.dt)
        n_settled = n_ramp + int(self.hold_seconds / self.dt)
        for i in self._ticks():
            if fault := self._fault():
                return self.damp(fault)
            alpha = min(1.0, (i + 1) / n_ramp)
            self.writer.send(self.state.msg, start + alpha * (goal - start), self.kp, self.kd)
            if i == n_settled and not self.auto_start:
                print("[deploy] holding: START runs the policy, SELECT damps")
            if i >= n_settled and (self.auto_start
                                   or remote_pressed(self.state.msg, REMOTE_START)):
                return True

    def run_policy(self) -> bool:
        self.adapter.write(self.state.msg, self.data)
        ref = self.motions.frame(self.motion, self.start_frame)
        yaw = self.builder.align_heading(self.data, ref)
        print(f"[deploy] policy on '{self.motion}', {self.n_steps} steps at "
              f"{1 / self.dt:.0f} Hz; reference rotated by {np.degrees(yaw):+.1f} deg")
        self.builder.reset_history()
        self.actor.reset()
        last_frame = self.motions.length(self.motion) - 1
        last_tick = self.state.msg.tick
        for i in self._ticks(self.n_steps):
            t = time.perf_counter()
            if fault := self._fault():
                return self.damp(fault)
            msg = self.state.msg
            self.ticks_per_step.append(msg.tick - last_tick)
            last_tick = msg.tick
            self.adapter.write(msg, self.data)
            frame = min(self.start_frame + i, last_frame)
            ref = self.motions.frame(self.motion, frame)
            self.reference.send(self.motion, frame, yaw)
            obs = self.builder.build(self.data, ref, self.last_action)
            action = self.actor.act(obs)
            if not np.all(np.isfinite(action)):
                return self.damp("non-finite action")
            self.last_action = action.astype(np.float64)
            target = self.motors.from_actuators(self.mapper.target(action))
            self.writer.send(msg, target, self.kp, self.kd)
            self.compute_ms.append((time.perf_counter() - t) * 1e3)
        return True

    def damp(self, reason: str) -> bool:
        print(f"[deploy] damping: {reason}")
        for _ in self._ticks(int(self.damp_seconds / self.dt)):
            self.writer.damp(self.state.msg, self.kd)
        return False

    def report(self) -> None:
        for name, values in (("compute", self.compute_ms), ("tick lateness", self.jitter_ms)):
            if values:
                v = np.asarray(values)
                print(f"[deploy] {name:13s} p50 {np.median(v):6.2f} ms   "
                      f"p99 {np.percentile(v, 99):6.2f} ms   max {v.max():6.2f} ms")
        if self.ticks_per_step:
            print(f"[deploy] LowState ticks per control step: median "
                  f"{np.median(self.ticks_per_step):.0f}, max {max(self.ticks_per_step)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", required=True, help="bundle dir from export_bundle.py")
    ap.add_argument("--motion", default=None, help="clip name (default: the first one)")
    ap.add_argument("--domain", type=int, default=SIM_DOMAIN)
    ap.add_argument("--interface", default="lo0", help="network interface (lo on Linux)")
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--n-steps", type=int, default=0, help="policy steps (0 = to end of clip)")
    ap.add_argument("--ramp-seconds", type=float, default=2.0)
    ap.add_argument("--hold-seconds", type=float, default=2.0)
    ap.add_argument("--damp-seconds", type=float, default=3.0)
    ap.add_argument("--auto-start", action="store_true",
                    help="run the policy after the hold, without waiting for START")
    ap.add_argument("--std-floor", type=float, default=0.0, help="see MlpActor")
    args = ap.parse_args()
    if args.domain == ROBOT_DOMAIN:
        if args.auto_start:
            raise SystemExit("domain 0 is the robot: --auto-start is for simulation, press START")
        print(f"[deploy] domain 0 on {args.interface}: this is the ROBOT, not the bridge")

    init_dds(args.domain, args.interface)
    controller = Controller(args)
    controller.wait_for_state()
    try:
        if controller.ramp_and_hold() and controller.run_policy():
            controller.damp("end of clip")
    except BaseException as e:  # Ctrl-C or a bug must not leave the last targets in force.
        controller.damp(repr(e))
        raise
    finally:
        controller.report()


if __name__ == "__main__":
    main()
