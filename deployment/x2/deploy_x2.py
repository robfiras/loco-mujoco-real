"""Runs an exported X2 bundle over the AimDK ROS 2 topics, against ``sim2sim_bridge.py``
or, with ``--robot``, on the real X2.

::

    python deployment/x2/deploy_x2.py --bundle <dir> --motion <clip>
    python deployment/x2/deploy_x2.py --bundle <dir> --motion <clip> --robot [--hold-only]

States: wait for state -> ramp to the clip's first pose -> hold -> policy on the first frame
-> policy playing the clip -> damp. Keys in this terminal act as soon as they are pressed:

=======  ====================================================================
Enter    end the hold: the policy balances on the clip's first frame, held
         still, so the crane can go slack (with ``--hold-only``: damp)
Enter    again: play the clip, from the heading the robot has then
n / p    switch the reference to the next / previous clip window
s        the intermediate safety layer: the policy keeps running but tracks the
         model's ``init_state`` pose, standing still, until Enter replays the
         clip from its start or n / p switch
Ctrl-C   damp, which ends the run, as does the gamepad's circle (no letter key damps:
         one next to s would be hit by accident; d only through ``--keys-from``)
=======  ====================================================================

A USB gamepad (:class:`lmj_aimdk.Gamepad`) gives the same: A is Enter, B is s, LB / RB are
p / n, and LT + B damps.

``--play-on-start`` skips the first-frame stage.

``--stand-first`` is for a simulator that has nothing to hang the robot from, such as
AgiBot's own: the ramp goes to the model's ``init_state`` pose instead of the clip's first
one, and the policy starts as after ``s``, standing still, so that it balances from the
start. Enter then plays the clip, or ``--stand-first SECONDS`` plays it after that long.

In simulation the gamepad's square (key 6 in the bridge's viewer) and ``--auto-start`` also
end the hold. The windows are ``--motion`` and the bundle's ``calm_windows.txt``; a switch
starts the new one at its first frame, from the robot's current heading. After a window the
policy keeps tracking its last frame until stopped (``--end hold``), plays the next clip in
the list (``--end next``, holding after the last), replays it (``--end loop``), or damps
(``--end damp``). Clips should start and end standing, both feet down and still, so that
switching between them, or holding the last frame, never asks for a pose in mid-stride. A status line each second gives the achieved
control rate, compute time, how late ticks ran, the age of the state and how many states
arrived per step.

``--robot`` is the only way onto ROS domain 0. It lets traffic onto the network, checks the
first state, switches the robot to ``Develop_MC`` (AgiBot's motion control off, tens of
seconds), waits until nothing else publishes joint commands, and only then creates its own
publishers. The robot has no ``/joy``, so only the keys above start it, and ``--auto-start``
is refused. The robot is left in Develop_MC, damped; a reboot, or migrating to ``Ready``,
brings AgiBot's control back.

Nothing is commanded if the first state already fails the checks in
:class:`lmj_aimdk.SafetyMonitor`. After that, every state goes straight to damp on circle, a
failed check, stale state or a non-finite action, and so does Ctrl-C or an exception. The
command topics have no timeout, so the damping command is the last thing sent and stays in
force after this exits.

Only a bundle whose actor needs no world position runs here, since the robot reports none:
PPO with ``use_anchor_pos=False``, or SuperMimic. Yaw is relative: when the policy starts, the
reference is rotated onto the robot's heading, so IMU yaw only has to hold for the length of
the clip.

``--robot`` has been run against the fake hardware layer only, not yet on the X2. Hang the
robot first: damping lets it fall.
"""

from __future__ import annotations

import argparse
import itertools
import os
import signal
import sys
import time
from pathlib import Path

# Before numpy loads: OpenBLAS otherwise starts one spinning thread per core, and any other
# program that needs a core then holds up the policy's matrix multiplies by tens of ms
# (worst step 26 ms on a 32-core machine). Four threads are also faster for this network
# (1.7 ms mean against 2.3); one is far slower (9.2).
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]
from lmj_aimdk import (  # noqa: E402
    DEVELOP_MC,
    JOY_DAMP,
    JOY_START,
    ROBOT_DOMAIN,
    SIM_DOMAIN,
    CommandWriter,
    Gamepad,
    Keyboard,
    ReferencePublisher,
    RobotState,
    SafetyMonitor,
    StateAdapter,
    SystemMode,
    init_ros,
    shutdown_ros,
    wait_for_command_topics,
)
from lmj_deploy import (  # noqa: E402
    MotionBundle,
    ObservationBuilder,
    PolicySpec,
    PositionActionMapper,
    load_actor,
    training_model_xml,
)

#: Seconds without a new state before the controller gives up.
MAX_STATE_AGE = 0.1
#: rad/s. The ramp takes longer than ``--ramp-seconds`` if a joint would go faster than this.
MAX_RAMP_SPEED = 0.5
#: Seconds of heartbeat before a lost operator link counts as back (see _operator_link).
LINK_SETTLE = 1.0


def playlist(args, motions: MotionBundle) -> list[tuple[str, int, int]]:
    """The (clip, start frame, steps) windows that ``n`` and ``p`` step through.

    The bundle's ``calm_windows.txt`` (one ``clip:start:steps`` per line), started at the
    window given by ``--motion``, or with it in front if the file lacks it. Steps 0 means to
    the end of the clip.
    """
    windows = []
    path = Path(args.bundle) / "calm_windows.txt"
    if path.exists():
        for line in path.read_text().split():
            clip, start, steps = line.split(":")
            windows.append((clip, int(start), int(steps)))
    if args.motion or not windows:
        first = (args.motion or motions.names[0], args.start_frame, args.n_steps)
        if first in windows:  # start the list there, keeping its order for n and --end next
            i = windows.index(first)
            windows = windows[i:] + windows[:i]
        else:
            windows.insert(0, first)
    unknown = sorted({c for c, _, _ in windows} - set(motions.names))
    if unknown:
        raise SystemExit(f"clips not in the bundle: {unknown}")
    return windows


def default_frame(model: mujoco.MjModel, builder: ObservationBuilder, n_future: int) -> dict:
    """The ``init_state`` keyframe as a reference frame, standing still: the target B gives.

    Shaped like :meth:`MotionBundle.frame`; the trajectory's qpos layout is the model's.
    """
    data = mujoco.MjData(model)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "init_state")
    if key < 0:
        raise SystemExit("the model has no init_state keyframe to stand in")
    mujoco.mj_resetDataKeyframe(model, data, key)
    mujoco.mj_kinematics(model, data)
    qpos, qvel = data.qpos.copy(), np.zeros(model.nv)
    return {"joint_pos": qpos[builder.qpos_adr], "joint_vel": np.zeros(len(builder.qpos_adr)),
            "body_pos_w": data.xpos[builder.body_ids].copy(),
            "body_quat_w": data.xquat[builder.body_ids].copy(), "qpos": qpos, "qvel": qvel,
            "qpos_future": np.tile(qpos, (n_future + 1, 1)),
            "qvel_future": np.zeros((n_future + 1, model.nv))}


class Controller:
    def __init__(self, args, node):
        self.spec = PolicySpec.load(args.bundle)
        if self.spec.needs_world_position:
            raise SystemExit("this bundle observes motion_anchor_pos_b, which needs a world "
                             "position. Retrain with goal_params.use_anchor_pos=false.")
        self.actor = load_actor(args.bundle, self.spec, std_floor=args.std_floor)
        self.motions = MotionBundle(args.bundle, spec=self.spec)
        self.windows = playlist(args, self.motions)
        self._select(0)
        self.ramp_seconds = args.ramp_seconds
        self.hold_seconds = args.hold_seconds
        self.damp_seconds = args.damp_seconds
        self.auto_start = args.auto_start
        self.hold_only = args.hold_only
        self.pause_at_start = not args.play_on_start
        self.stand_first = args.stand_first  # None: off; seconds before the clip plays, 0 = Enter
        self.paused = False
        self.end = args.end
        self.robot = args.robot
        self.status_every = args.status_every
        self.keyboard = Keyboard(args.keys_from)
        # The operator's link (--heartbeat): until it is up, no key but damp counts.
        self.heartbeat = args.heartbeat
        self.link_ok = not self.heartbeat
        self._link_back_since: float | None = None
        if self.heartbeat:
            print(f"[deploy] waiting for the operator's heartbeat (lost after "
                  f"{self.heartbeat:.1f} s without one)")

        # Kinematics only: this model is never stepped.
        self.model = mujoco.MjModel.from_xml_path(str(training_model_xml(self.spec)))
        self.data = mujoco.MjData(self.model)
        self.builder = ObservationBuilder(self.spec, self.model)
        self.mapper = PositionActionMapper(self.spec, self.model)
        self.adapter = StateAdapter(self.builder)
        self.default_ref = default_frame(self.model, self.builder, self.spec.n_future)
        self.safe = False
        self.safety = SafetyMonitor(self.model, list(self.spec.joint_names))
        self.kp = np.asarray(self.spec.kp, dtype=np.float64)
        self.kd = np.asarray(self.spec.kd, dtype=np.float64)
        self.dt = self.spec.control_dt
        # Observations are in joint order and commands in actuator order.
        joint = {n: i for i, n in enumerate(self.spec.joint_names)}
        self.joint_of_actuator = np.array([joint[n] for n in self.spec.actuator_joint_names])

        self.node = node
        self.state = RobotState(node, self.spec)
        self.writer: CommandWriter | None = None  # see take_command
        self.reference = ReferencePublisher(node)
        self.last_action = np.zeros(self.spec.action_dim)
        self.compute_ms: list[float] = []
        self.jitter_ms: list[float] = []
        self.states_per_step: list[int] = []

        self.viewer = None
        self._status_text = ""
        if args.viser:
            from live_viser import LiveViewer
            self.viewer = LiveViewer(self.model, self.motions, self.builder.root_qpos_adr,
                                     args.viser)
            print("[deploy] live view at " + "  or  ".join(self.viewer.urls()))
        print(f"[deploy] keys in this terminal: {Keyboard.HELP}"
              + ("" if self.robot else " | viewer: 6 is square, which also starts"))
        self.gamepad = None
        if args.gamepad != "none" and os.path.exists(args.gamepad):
            try:
                self.gamepad = Gamepad(self.keyboard, args.gamepad)
                print(f"[deploy] {Gamepad.HELP} ({args.gamepad})")
            except OSError as e:
                print(f"[deploy] no gamepad: {e}")
        print("[deploy] clips: " + "  ".join(
            f"{i + 1}) {c} {s}+{n or 'end'}" for i, (c, s, n) in enumerate(self.windows)))

    def _select(self, index: int) -> None:
        """Make window ``index`` of the playlist the reference, from its first frame."""
        self.w = index % len(self.windows)
        self.motion, self.start_frame, n = self.windows[self.w]
        length = self.motions.length(self.motion)
        self.n_steps = n or length - self.start_frame
        self.last_frame = min(self.start_frame + self.n_steps, length) - 1

    def wait_for_state(self, timeout: float = 10.0) -> None:
        print("[deploy] waiting for joint state and IMU")
        t0 = time.perf_counter()
        while not self.state.complete:
            self.state.poll()
            if time.perf_counter() - t0 > timeout:
                raise SystemExit("no state received; is the bridge running on this domain?")
            time.sleep(0.01)
        if fault := self.safety.check(self.state):
            raise SystemExit(f"refusing to start: {fault}")

    def take_command(self) -> None:
        """Create the command publishers. On the robot, only once nothing else publishes."""
        self.writer = CommandWriter(self.node, self.spec)

    def _operator_link(self) -> None:
        """With ``--heartbeat``, follow the operator's link and drop what it sends while down.

        While it is down every key but damp is dropped, and ``run_policy`` falls back to the
        default pose. It counts as back only after ``LINK_SETTLE`` seconds of heartbeat, so
        keys that an SSH connection delivers late, in a burst after an outage, are dropped
        too rather than acted on.
        """
        if not self.heartbeat:
            return
        now = time.perf_counter()
        alive = self.keyboard.heartbeat_age < self.heartbeat
        if not alive:
            self._link_back_since = None
            if self.link_ok:
                self.link_ok = False
                print("[deploy] operator link LOST: default pose; keys other than damp are "
                      "ignored until it is back")
        elif not self.link_ok:
            self._link_back_since = self._link_back_since or now
            if now - self._link_back_since >= LINK_SETTLE:
                self.link_ok = True
                print("[deploy] operator link up")
        if not self.link_ok:
            self.keyboard.start.clear()
            self.keyboard.safe.clear()
            self.keyboard.step()

    def _start_requested(self) -> bool:
        if self.keyboard.start.is_set():
            return True
        return not self.robot and (self.auto_start or self.state.pressed(JOY_START))

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
            return f"state is {self.state.age * 1e3:.0f} ms old"
        if self.state.pressed(JOY_DAMP):
            return "circle pressed"
        if self.keyboard.damp.is_set():
            return "damp pressed"
        return self.safety.check(self.state)

    def ramp_and_hold(self) -> bool:
        """Move from the measured pose to the clip's first pose, and stay until told to go.

        With ``--stand-first`` the pose is the default standing one.
        """
        ref = (self.default_ref if self.stand_first is not None
               else self.motions.frame(self.motion, self.start_frame))
        # Clips can pass a joint's range, and the ramp must not push it into the stop.
        goal = np.clip(ref["joint_pos"], self.safety.lower, self.safety.upper)
        goal = goal[self.joint_of_actuator]
        start = self.state.q[self.joint_of_actuator].copy()
        ramp_seconds = max(self.ramp_seconds, np.abs(goal - start).max() / MAX_RAMP_SPEED)
        print(f"[deploy] ramp {ramp_seconds:.1f} s, hold {self.hold_seconds:.1f} s")
        n_ramp = int(ramp_seconds / self.dt)
        n_settled = n_ramp + int(self.hold_seconds / self.dt)
        for i in self._ticks():
            if fault := self._fault():
                return self.damp(fault)
            self._operator_link()
            alpha = min(1.0, (i + 1) / n_ramp)
            self.writer.send(start + alpha * (goal - start), self.kp, self.kd)
            self._show(None, "ramping" if i < n_ramp else "holding: Enter starts")
            if i == n_settled:
                self.keyboard.start.clear()  # an Enter pressed during the ramp does not count
                if not self.auto_start:
                    print("[deploy] holding: Enter " + ("ends the hold" if self.hold_only
                                                         else "runs the policy")
                          + ("" if self.robot else " (or square)") + ", Ctrl-C damps")
            if i >= n_settled and self._start_requested():
                if self.hold_only:
                    return self.damp("hold finished")
                return True

    def _begin(self, label: str) -> float:
        """Start the current window from the robot's present heading. Returns that yaw."""
        yaw = self.builder.align_heading(self.data,
                                         self.motions.frame(self.motion, self.start_frame))
        print(f"[deploy] {label} {self.w + 1}) '{self.motion}' frames {self.start_frame}.."
              f"{self.last_frame}; reference rotated by {np.degrees(yaw):+.1f} deg")
        return yaw

    def run_policy(self) -> bool:
        self.adapter.write(self.state, self.data)
        print(f"[deploy] policy at {1 / self.dt:.0f} Hz; after a clip: {self.end}")
        # Unless told to play at once, the policy first balances on the clip's first frame,
        # held still, until Enter: time to let the crane go slack.
        self.keyboard.start.clear()
        self.keyboard.safe.clear()
        self.paused = self.pause_at_start and not self.auto_start
        yaw, k0 = self._begin("balancing on the first frame of" if self.paused
                              else "policy on"), 0
        if self.stand_first is not None:  # start as after s: balance standing still first
            self.safe, self.paused = True, False
            yaw = self.builder.align_heading(self.data, self.default_ref)
            print("[deploy] standing first: the policy tracks the default pose; "
                  + (f"the clip plays in {self.stand_first:.1f} s" if self.stand_first
                     else "Enter plays the clip"))
        elif self.paused:
            print("[deploy] Enter plays the clip; n / p pick another, Ctrl-C damps")
        self.builder.reset_history()
        self.actor.reset()
        last_sequence = self.state.sequence
        self._status_at, self._status_i = time.perf_counter() + self.status_every, 0
        for i in self._ticks():
            t = time.perf_counter()
            if fault := self._fault():
                return self.damp(fault)
            self.states_per_step.append(self.state.sequence - last_sequence)
            last_sequence = self.state.sequence
            self.adapter.write(self.state, self.data)
            self._operator_link()
            if self.keyboard.safe.is_set() or not self.link_ok:
                self.keyboard.safe.clear()
                if not self.safe:  # the intermediate safety layer: stand in the default pose
                    self.safe, self.paused = True, False
                    self.keyboard.start.clear()  # only a press after this one plays again
                    yaw = self.builder.align_heading(self.data, self.default_ref)
                    print("[deploy] default pose: the policy now tracks standing still; A / Enter "
                          "plays the clip from its start, LB / RB or n / p switch, LT + B or Ctrl-C damps")
            if step := self.keyboard.step():
                self.safe = False
                self._select(self.w + step)
                yaw, k0 = self._begin("switched to"), i
            if self.safe and self.stand_first and i == int(self.stand_first / self.dt):
                self.keyboard.start.set()  # --stand-first SECONDS: play without a key
            if self.safe:
                if self.keyboard.start.is_set():
                    self.keyboard.start.clear()
                    self.safe = False
                    yaw = self._begin("playing")
                k0 = i
            elif self.paused:
                if self.keyboard.start.is_set():  # played from the heading the robot has now
                    self.keyboard.start.clear()
                    self.paused = False
                    yaw = self._begin("playing")
                k0 = i  # the clip's clock starts when it plays
            else:
                # Playing already, so start means nothing: drop it rather than let it linger
                # and replay the clip the moment the default pose is left.
                self.keyboard.start.clear()
            k, window = i - k0, self.last_frame - self.start_frame + 1
            if k >= window:
                if self.end == "damp":
                    return True
                if self.end == "next" and self.w + 1 < len(self.windows):
                    self._select(self.w + 1)
                    yaw, k0, k = self._begin("next:"), i, 0
                    window = self.last_frame - self.start_frame + 1
                elif self.end == "loop":
                    yaw, k0, k = self._begin("looping"), i, 0
                else:
                    if k == window:
                        print("[deploy] end of the clip: holding its last frame; "
                              "n / p switch, Ctrl-C damps")
                    k = window - 1
            frame = self.start_frame + k
            if self.safe:
                ref = self.default_ref
            else:
                # While paused the future frames repeat the first one, as at a clip's end.
                ref = self.motions.frame(self.motion, frame,
                                         end=frame if self.paused else self.last_frame)
                self.reference.send(self.motion, frame, yaw)
            obs = self.builder.build(self.data, ref, self.last_action)
            action = self.actor.act(obs)
            if not np.all(np.isfinite(action)):
                return self.damp("non-finite action")
            self.last_action = action.astype(np.float64)
            self.writer.send(self.mapper.target(action), self.kp, self.kd)
            self.compute_ms.append((time.perf_counter() - t) * 1e3)
            self._status(i, frame)
            self._show(None if self.safe else {"motion": self.motion, "frame": frame, "yaw": yaw},
                       self._status_text)

    def _show(self, reference: dict | None, text: str) -> None:
        """Hand the measured pose to the live view, if there is one."""
        if self.viewer is not None:
            self.adapter.write(self.state, self.data)
            self.viewer.update(self.data.qpos, reference, text)

    def _status(self, i: int, frame: int) -> None:
        """Every ``--status-every`` seconds: the achieved rate and where the time went."""
        now = time.perf_counter()
        if not self.status_every or now < self._status_at:
            return
        n = i + 1 - self._status_i
        hz = n / (now - self._status_at + self.status_every)
        compute = np.asarray(self.compute_ms[-n:])
        late = np.asarray(self.jitter_ms[-n:])
        states = np.asarray(self.states_per_step[-n:])
        self._status_text = (
            f"{hz:5.1f} Hz | compute {np.median(compute):4.1f} ms, max {compute.max():4.1f} | "
            f"late max {late.max():5.1f} ms | state age {self.state.age * 1e3:4.1f} ms, "
            f"{states.mean():.1f} states/step | "
            + ("default pose (A / Enter plays)" if self.safe else f"{self.motion} {frame}"
               + (" (first frame, Enter plays)" if self.paused else ""))
            + ("" if self.link_ok else " | OPERATOR LINK LOST"))
        print(f"[deploy] {self._status_text}")
        self._status_at, self._status_i = now + self.status_every, i + 1

    def damp(self, reason: str) -> bool:
        if self.writer is None:
            print(f"[deploy] stopping before any command was sent: {reason}")
            return False
        print(f"[deploy] damping: {reason}")
        # A second Ctrl-C must not cut the damping short: ignore it until the sequence is sent.
        previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            for _ in self._ticks(int(self.damp_seconds / self.dt)):
                self.writer.damp(self.kd)
                self.state.poll()
                self._show(None, f"damped: {reason}")
        finally:
            signal.signal(signal.SIGINT, previous)
        return False

    def report(self) -> None:
        for name, values in (("compute", self.compute_ms), ("tick lateness", self.jitter_ms)):
            if values:
                v = np.asarray(values)
                print(f"[deploy] {name:13s} p50 {np.median(v):6.2f} ms   "
                      f"p99 {np.percentile(v, 99):6.2f} ms   max {v.max():6.2f} ms")
        if self.states_per_step:
            print(f"[deploy] states per control step: median "
                  f"{np.median(self.states_per_step):.0f}, max {max(self.states_per_step)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", required=True, help="bundle dir from export_bundle.py")
    ap.add_argument("--motion", default=None, help="clip name (default: the first one)")
    ap.add_argument("--domain", type=int, default=None,
                    help=f"ROS domain id (default {SIM_DOMAIN}, or {ROBOT_DOMAIN} with --robot; "
                         "pass another with --robot to rehearse against a fake robot)")
    ap.add_argument("--network", action="store_true",
                    help="let ROS traffic leave this machine (default: localhost only)")
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--n-steps", type=int, default=0, help="policy steps (0 = to end of clip)")
    ap.add_argument("--ramp-seconds", type=float, default=2.0)
    ap.add_argument("--hold-seconds", type=float, default=2.0)
    ap.add_argument("--damp-seconds", type=float, default=3.0)
    ap.add_argument("--auto-start", action="store_true",
                    help="run the policy after the hold, without waiting for square")
    ap.add_argument("--viser", type=int, nargs="?", const=8080, default=None, metavar="PORT",
                    help="serve a live view of the robot and the reference (port 8080)")
    ap.add_argument("--status-every", type=float, default=1.0,
                    help="seconds between status lines while the policy runs (0 = none)")
    ap.add_argument("--end", choices=("hold", "next", "loop", "damp"), default="hold",
                    help="after a clip: keep the policy on its last frame until stopped, play "
                         "the next clip in the list (holding after the last), replay it, or damp")
    ap.add_argument("--heartbeat", type=float, default=0.0, metavar="SECONDS",
                    help="expect the operator's heartbeat (h on the key input) and fall back to "
                         "the default pose after this long without one (0 = off)")
    ap.add_argument("--keys-from", default=None, metavar="FIFO",
                    help="read the keys from this named pipe instead of the terminal, so the "
                         "controller can run detached on the robot (see soc1/start.sh)")
    ap.add_argument("--gamepad", default="/dev/input/js0",
                    help="joystick device read for A / LB / RB / LT + B, if present ('none' = off)")
    ap.add_argument("--play-on-start", action="store_true",
                    help="play the clip as soon as the policy starts, instead of balancing on "
                         "its first frame until a second Enter")
    ap.add_argument("--stand-first", type=float, nargs="?", const=0.0, default=None,
                    metavar="SECONDS",
                    help="for a simulator with no band: ramp to the default standing pose and "
                         "start the policy standing still; play the clip on Enter, or after "
                         "SECONDS")
    ap.add_argument("--wait-seconds", type=float, default=10.0,
                    help="how long to wait for the first state, e.g. to start before a simulator")
    ap.add_argument("--hold-only", action="store_true",
                    help="ramp and hold the first pose; never run the policy")
    ap.add_argument("--robot", action="store_true",
                    help="the real X2: domain 0, network, Develop_MC, keyboard start")
    ap.add_argument("--std-floor", type=float, default=0.0, help="see MlpActor")
    args = ap.parse_args()
    if args.robot:
        if args.auto_start:
            raise SystemExit("--auto-start is refused on the robot: start with Enter")
        args.network = True
        if args.domain is None:
            args.domain = ROBOT_DOMAIN
    elif args.domain is None:
        args.domain = SIM_DOMAIN
    elif args.domain == ROBOT_DOMAIN:
        raise SystemExit("domain 0 is where the robot is: pass --robot to mean it")

    node = init_ros(args.domain, "lmj_x2_controller", args.network)
    try:
        controller = Controller(args, node)
        controller.wait_for_state(args.wait_seconds)
        if args.robot:
            SystemMode(node).enter(DEVELOP_MC)
            wait_for_command_topics(node)
            controller.wait_for_state()  # still safe after the switch
        controller.take_command()
        try:
            if controller.ramp_and_hold() and controller.run_policy():
                controller.damp("end of clip")
        except KeyboardInterrupt:  # how a hold or a loop is normally ended
            controller.damp("Ctrl-C")
        except BaseException as e:  # a bug must not leave the last targets in force
            controller.damp(repr(e))
            controller.damp(repr(e))
            raise
        finally:
            controller.report()
    finally:
        shutdown_ros(node)


if __name__ == "__main__":
    main()
