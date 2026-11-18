"""MuJoCo standing in for the X2 on its AimDK ROS 2 topics.

Runs the physics in real time in its own process, publishes the joint-state and IMU topics
and obeys the joint-command topics (see :mod:`lmj_aimdk`), so ``deploy_x2.py`` talks to it
through the interface it would use on the robot. Unlike ``rollout_mujoco.py``, the controller
here is asynchronous: it sees state that is already old, and a late command simply leaves the
previous targets in force.

The bridge behaves like the joint drives. Every physics step it applies, per joint,
``effort + stiffness * (position - q) + damping * (velocity - dq)`` from the latest command,
clipped to the effort limit, and it keeps that command until another arrives. That is what
AgiBot's own simulator, ``x2_rl_deploy_mujoco``, does with the same topics; this bridge
exists because that one is a Linux binary and loads their model, not the one trained on.

State goes out once per physics step, 200 Hz with the training timestep, where the robot
sends 500 to 1000 Hz. The head joints are fixed in the model: their state reads zero and
their commands are ignored.

The robot starts hanging from an invisible elastic band, ``--lift`` above the ``init_state``
keyframe. In the viewer:

====  =================================================
8     shorten the band, lifting the robot
9     lengthen it, lowering the robot
7     release the band, or attach it again
6     press the gamepad's square button, which starts the policy
====  =================================================

To push the robot, double-click a body and drag with Ctrl and the right mouse button.

While ``deploy_x2.py`` runs a policy, the viewer also draws the reference frame being tracked
as a translucent ghost, started on the robot and rotated by the controller's heading
alignment::

    mjpython deployment/x2/sim2sim_bridge.py --bundle <dir> --viewer  # macOS GUI
    python deployment/x2/sim2sim_bridge.py --bundle <dir> --release-seconds 5
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import mujoco
import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]
from lmj_aimdk import (  # noqa: E402
    CHEST_IMU_TOPIC,
    GROUPS,
    IMU_TOPIC,
    JOY_START,
    JOY_TOPIC,
    LATEST_QOS,
    MAX_JOINT_VEL,
    REFERENCE_TOPIC,
    ROBOT_DOMAIN,
    SIM_DOMAIN,
    STATE_QOS,
    Imu,
    JointCommandArray,
    JointState,
    JointStateArray,
    Joy,
    Latest,
    String,
    command_topic,
    init_ros,
    joint_names,
    shutdown_ros,
    state_topic,
)
from lmj_deploy import MotionBundle, PolicySpec, PositionActionMapper, quat_to_mat  # noqa: E402
from rollout_mujoco import build_model  # noqa: E402

#: N/m and N s/m. A few centimetres of stretch under the robot's weight, and close to
#: critically damped, so it hangs still rather than bouncing.
BAND_STIFFNESS = 10000.0
BAND_DAMPING = 1000.0
#: Metres per key press.
BAND_STEP = 0.05
#: Hung from the torso, above the robot's centre of mass, so it hangs upright.
BAND_BODY = "torso_link"
#: How long a key press holds the gamepad button down.
JOY_PRESS_SECONDS = 0.3

#: Sites the two IMU topics report, as in AgiBot's simulator.
IMU_SITES = {IMU_TOPIC: "imu_in_pelvis", CHEST_IMU_TOPIC: "imu_in_torso"}

KEY_START, KEY_BAND, KEY_SHORTER, KEY_LONGER = (ord(k) for k in "6789")

VIEWER_FPS = 30.0
GHOST_RGBA = (0.2, 0.8, 0.3, 0.35)
#: A reference stream older than this means no policy is running.
GHOST_TIMEOUT = 0.5

#: Radians short of its range at which a joint counts as at its limit. MuJoCo's limits are
#: soft, so a joint pressed against one sits a few milliradians either side of it.
AT_LIMIT = 0.01
#: Seconds a contact or a limit must have been clear before it is reported again.
REPORT_GAP = 1.0


class Drives:
    """The latest command of every joint, as arrays in the bundle's actuator order."""

    def __init__(self, node, names: list[str]):
        self._index = {n: i for i, n in enumerate(names)}
        n = len(names)
        self.q, self.dq, self.tau = np.zeros(n), np.zeros(n), np.zeros(n)
        self.kp, self.kd = np.zeros(n), np.zeros(n)
        self.first_gains_at: float | None = None
        self.accepted = 0
        self._subs = {g: Latest(node, JointCommandArray, command_topic(g)) for g in GROUPS}

    def poll(self) -> None:
        for group, latest in self._subs.items():
            if not latest.poll():
                continue
            joints = latest.msg.joints
            for name, c in zip(joint_names(group, joints), joints):
                i = self._index.get(name)
                if i is None:
                    continue
                self.q[i], self.dq[i], self.tau[i] = c.position, c.velocity, c.effort
                self.kp[i], self.kd[i] = c.stiffness, c.damping
            self.accepted += 1
        if self.first_gains_at is None and self.kp.any():
            self.first_gains_at = time.perf_counter()


class ElasticBand:
    """A rope from a fixed point to one body. It pulls when stretched and never pushes."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, body: str,
                 length: float = 1.0):
        self.model = model
        self.body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
        if self.body < 0:
            raise ValueError(f"model has no body {body!r} to hang from")
        self.length = length
        self.attached = False
        self.attach(data)

    def attach(self, data: mujoco.MjData) -> None:
        """Hang from a point ``length`` straight above the body, so it starts slack."""
        self.anchor = data.xpos[self.body] + np.array([0.0, 0.0, self.length])
        self.attached = True

    def release(self, data: mujoco.MjData) -> None:
        self.attached = False
        data.xfrc_applied[self.body, :3] = 0.0

    def apply(self, data: mujoco.MjData) -> None:
        if not self.attached:
            return
        offset = self.anchor - data.xpos[self.body]
        dist = float(np.linalg.norm(offset))
        force = np.zeros(3)
        if dist > self.length:
            force += BAND_STIFFNESS * (dist - self.length) * offset / dist
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, data, mujoco.mjtObj.mjOBJ_BODY, self.body, vel, 0)
        force -= BAND_DAMPING * vel[3:]
        data.xfrc_applied[self.body, :3] += force


class Watch:
    """Self-collision and joint limits, while ``deploy_x2.py`` streams a reference.

    Only then: the ramp, the hanging and the collapse after damping don't count. Each contact
    between two robot bodies, and each joint at or past its range, is reported as it starts,
    with the clip frame being tracked, and :meth:`summary` sums them up.
    """

    def __init__(self, plant: Plant):
        m = plant.model
        self.plant = plant
        self.reference = Latest(plant.node, String, REFERENCE_TOPIC)
        self.where = ""
        self.names = list(plant.spec.actuator_joint_names)
        jids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in self.names]
        self.limited = m.jnt_limited[jids].astype(bool)
        self.lower, self.upper = m.jnt_range[jids].T
        self.steps = 0
        #: (body, body) -> [steps, deepest m, first frame]; joint -> [steps, most past, first]
        self.contacts: dict[tuple[str, str], list] = {}
        self.at_limit: dict[int, list] = {}
        self.target_past = np.zeros(len(self.names))
        self.peak_speed = np.zeros(len(self.names))
        self.fastest_at = ""
        self.saturated = np.zeros(len(self.names), dtype=int)
        self._seen: dict = {}

    def step(self, saturated: np.ndarray) -> None:
        if self.reference.poll():
            ref = json.loads(self.reference.msg.data)
            self.where = f"{ref['motion']} frame {ref['frame']}"
        if self.reference.msg is None or self.reference.age > GHOST_TIMEOUT:
            return
        m, d, now = self.plant.model, self.plant.data, time.perf_counter()
        self.steps += 1

        body = m.geom_bodyid[d.contact.geom[:d.ncon]]
        dist = d.contact.dist[:d.ncon]
        own = (body != 0).all(1) & (dist < 0)  # both on the robot, and penetrating
        for (b1, b2), depth in zip(body[own], -dist[own]):
            key = tuple(sorted((m.body(b1).name, m.body(b2).name)))
            self._count(self.contacts, key, depth, now,
                        f"SELF-COLLISION {key[0]} / {key[1]}, {depth * 1e3:.0f} mm")

        q = d.qpos[self.plant.qpos_adr]
        past = np.where(self.limited, np.maximum(self.lower - q, q - self.upper), -np.inf)
        for i in np.flatnonzero(past > -AT_LIMIT):
            side = "upper" if q[i] - self.upper[i] >= self.lower[i] - q[i] else "lower"
            self._count(self.at_limit, i, max(0.0, past[i]), now,
                        f"JOINT LIMIT {self.names[i]} at its {side} end, "
                        f"{q[i]:+.2f} of [{self.lower[i]:+.2f}, {self.upper[i]:+.2f}] rad")

        target = self.plant.drives.q
        t_past = np.where(self.limited, np.maximum(self.lower - target, target - self.upper), 0)
        np.maximum(self.target_past, t_past, out=self.target_past)
        speed = np.abs(d.qvel[self.plant.dof_adr])
        if speed.max() > self.peak_speed.max():
            self.fastest_at = self.where
        np.maximum(self.peak_speed, speed, out=self.peak_speed)
        self.saturated += saturated

    def _count(self, table: dict, key, amount: float, now: float, text: str) -> None:
        entry = table.setdefault(key, [0, 0.0, self.where])
        entry[0] += 1
        entry[1] = max(entry[1], amount)
        if now - self._seen.get((id(table), key), -np.inf) > REPORT_GAP:
            print(f"[bridge] {text} ({self.where})")
        self._seen[(id(table), key)] = now

    def summary(self) -> list[str]:
        dt = self.plant.model.opt.timestep
        if not self.steps:
            return ["no policy ran: nothing watched"]
        out = [f"while a policy ran ({self.steps * dt:.1f} s):"]
        out.append("  self-collision: " + ("none" if not self.contacts else ""))
        for (a, b), (n, deep, first) in sorted(self.contacts.items(), key=lambda e: -e[1][0]):
            out.append(f"    {a} / {b}: {n * dt:.2f} s, deepest {deep * 1e3:.0f} mm, "
                       f"first at {first}")
        out.append(f"  joints within {AT_LIMIT} rad of their range: "
                   + ("none" if not self.at_limit else ""))
        for i, (n, most, first) in sorted(self.at_limit.items(), key=lambda e: -e[1][0]):
            out.append(f"    {self.names[i]}: {n * dt:.2f} s, at most {most:.3f} rad past, "
                       f"first at {first}")
        over = np.flatnonzero(self.target_past > AT_LIMIT)
        out.append(f"  targets more than {AT_LIMIT} rad past the range: " + (", ".join(
            f"{self.names[i]} {self.target_past[i]:.2f} rad" for i in over) or "none"))
        fast = int(self.peak_speed.argmax())
        out.append(f"  fastest joint {self.names[fast]} {self.peak_speed[fast]:.1f} rad/s at "
                   f"{self.fastest_at} (the controller damps above {MAX_JOINT_VEL:.0f})")
        sat = np.flatnonzero(self.saturated)
        out.append("  at the effort limit: " + (", ".join(
            f"{self.names[i]} {100 * self.saturated[i] / self.steps:.1f}%" for i in sat)
            or "never"))
        return out


class Plant:
    def __init__(self, args, node):
        self.node = node
        self.spec = PolicySpec.load(args.bundle)
        self.model = build_model(args.plant, self.spec, match_solver=True)
        self.data = mujoco.MjData(self.model)
        mapper = PositionActionMapper(self.spec, self.model)
        if mapper.position_actuators:
            raise SystemExit("--plant must have torque motors: the bridge closes the PD "
                             "loop here, as the joint drives do on the robot")
        self.qpos_adr, self.dof_adr = mapper.qpos_adr, mapper.dof_adr
        self.ctrl_index, self.effort = mapper.ctrl_index, mapper.effort_limit

        root = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, self.spec.root_joint_name)
        self.root_q = int(self.model.jnt_qposadr[root])

        key = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "init_state")
        mujoco.mj_resetDataKeyframe(self.model, self.data, key)
        half = 0.5 * args.start_yaw
        self.data.qpos[self.root_q + 3:self.root_q + 7] = [np.cos(half), 0, 0, np.sin(half)]
        self.data.qpos[self.root_q + 2] += args.lift
        mujoco.mj_forward(self.model, self.data)
        self.band = ElasticBand(self.model, self.data, BAND_BODY)
        # The viewer's mouse push, per body. The viewer thread swaps in a new array.
        self.push = np.zeros((self.model.nbody, 6))
        self.release_seconds = args.release_seconds
        self.start_until = 0.0
        self.min_up = 1.0

        names = list(self.spec.actuator_joint_names)
        self.drives = Drives(node, names)
        slot = {n: i for i, n in enumerate(names)}
        # Per group: the message, reused, and where each of its joints is in the arrays above.
        self._states = {
            g: (JointStateArray(joints=[JointState(name=n) for n in group]),
                [slot.get(n) for n in group],
                node.create_publisher(JointStateArray, state_topic(g), STATE_QOS))
            for g, group in GROUPS.items()}
        self._imus = []
        for topic, site in IMU_SITES.items():
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site)
            if sid < 0:
                raise ValueError(f"model has no site {site!r} for {topic}")
            msg = Imu()
            msg.header.frame_id = "base_link"
            self._imus.append((msg, sid, node.create_publisher(Imu, topic, STATE_QOS)))
        self._joy = node.create_publisher(Joy, JOY_TOPIC, LATEST_QOS)
        self._sequence = 0
        self.watch = Watch(self)

    def step(self) -> None:
        d, drives = self.data, self.drives
        drives.poll()
        q, dq = d.qpos[self.qpos_adr], d.qvel[self.dof_adr]
        tau = drives.tau + drives.kp * (drives.q - q) + drives.kd * (drives.dq - dq)
        d.ctrl[self.ctrl_index] = np.clip(tau, -self.effort, self.effort)
        saturated = np.abs(tau) >= self.effort
        started = drives.first_gains_at
        if (self.band.attached and self.release_seconds and started is not None
                and time.perf_counter() - started > self.release_seconds):
            self.release_band()
        d.xfrc_applied[:] = self.push
        self.band.apply(d)
        mujoco.mj_step(self.model, d)
        self.watch.step(saturated)
        up = quat_to_mat(d.qpos[self.root_q + 3:self.root_q + 7])[2, 2]
        if not self.band.attached:
            self.min_up = min(self.min_up, float(up))

    def release_band(self) -> None:
        self.band.release(self.data)
        print("[bridge] band released")

    def key_callback(self, key: int) -> None:
        """Viewer keys. Runs on the viewer's thread, so it only sets state."""
        band = self.band
        if key == KEY_START:
            self.start_until = time.perf_counter() + JOY_PRESS_SECONDS
            print("[bridge] square pressed")
        elif key == KEY_BAND:
            if band.attached:
                self.release_band()
            else:
                band.attach(self.data)
                print(f"[bridge] band attached, length {band.length:.2f} m")
        elif key in (KEY_SHORTER, KEY_LONGER):
            step = -BAND_STEP if key == KEY_SHORTER else BAND_STEP
            band.length = max(0.0, band.length + step)
            print(f"[bridge] band length {band.length:.2f} m")

    def publish(self) -> None:
        m, d = self.model, self.data
        stamp = self.node.get_clock().now().to_msg()
        self._sequence += 1
        for msg, slots, pub in self._states.values():
            msg.header.stamp, msg.header.sequence = stamp, self._sequence
            for joint, i in zip(msg.joints, slots):
                if i is not None:
                    joint.position = float(d.qpos[self.qpos_adr[i]])
                    joint.velocity = float(d.qvel[self.dof_adr[i]])
                    joint.effort = float(d.ctrl[self.ctrl_index[i]])
            pub.publish(msg)

        mujoco.mj_rnePostConstraint(m, d)  # body accelerations, for the accelerometers
        quat, vel, acc = np.empty(4), np.empty(6), np.empty(6)
        for msg, sid, pub in self._imus:
            mujoco.mju_mat2Quat(quat, d.site_xmat[sid])
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_SITE, sid, vel, 1)
            mujoco.mj_objectAcceleration(m, d, mujoco.mjtObj.mjOBJ_SITE, sid, acc, 1)
            msg.header.stamp = stamp
            o, w, a = msg.orientation, msg.angular_velocity, msg.linear_acceleration
            o.w, o.x, o.y, o.z = quat.tolist()
            w.x, w.y, w.z = vel[:3].tolist()
            a.x, a.y, a.z = acc[3:].tolist()
            pub.publish(msg)

        if time.perf_counter() < self.start_until:
            buttons = [0] * (JOY_START + 1)
            buttons[JOY_START] = 1
            self._joy.publish(Joy(buttons=buttons))


class Viewer:
    """The passive viewer, drawing a copy of the plant from its own thread.

    ``viewer.sync()`` can block for over 100 ms while a frame renders, which would stall the
    state long enough for the controller to fault. Here the physics loop never touches the
    viewer: this thread copies the state into its own ``MjData``, draws the reference ghost
    and syncs, at ``VIEWER_FPS``.
    """

    def __init__(self, plant: Plant, bundle: str):
        import mujoco.viewer

        self.plant = plant
        self.model = plant.model
        self.display = mujoco.MjData(self.model)
        self.ghost = mujoco.MjData(self.model)
        self.motions = MotionBundle(bundle, spec=plant.spec)
        self.reference = Latest(plant.node, String, REFERENCE_TOPIC)
        self.opt, self.pert = mujoco.MjvOption(), mujoco.MjvPerturb()
        self._ghost_anchor: tuple | None = None
        self.handle = mujoco.viewer.launch_passive(self.model, self.display,
                                                   key_callback=plant.key_callback)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def is_running(self) -> bool:
        return self.handle.is_running()

    def close(self) -> None:
        self.handle.close()

    def _loop(self) -> None:
        src, dst = self.plant.data, self.display
        while self.handle.is_running():
            t = time.perf_counter()
            dst.qpos[:], dst.qvel[:], dst.time = src.qpos, src.qvel, src.time
            mujoco.mj_kinematics(self.model, dst)
            with self.handle.lock():
                self._draw_ghost()
                self._forward_push()
            self.handle.sync()
            time.sleep(max(0.0, 1.0 / VIEWER_FPS - (time.perf_counter() - t)))

    def _forward_push(self) -> None:
        """Hand a mouse push (Ctrl + right-drag on a selected body) to the physics.

        The viewer applies it to its own copy of the state, which never steps, so the force
        is computed here, on that copy's pose, and handed to the plant.
        """
        pert = self.handle.perturb
        push = np.zeros((self.model.nbody, 6))
        if pert.select > 0 and pert.active & mujoco.mjtPertBit.mjPERT_TRANSLATE:
            self.display.xfrc_applied[:] = 0.0
            mujoco.mjv_applyPerturbForce(self.model, self.display, pert)
            push[:] = self.display.xfrc_applied
        self.plant.push = push

    def _draw_ghost(self) -> None:
        scn = self.handle.user_scn
        scn.ngeom = 0
        self.reference.poll()
        if self.reference.msg is None or self.reference.age > GHOST_TIMEOUT:
            self._ghost_anchor = None
            return
        ref = json.loads(self.reference.msg.data)
        qpos = np.array(self.motions.frame(ref["motion"], ref["frame"])["qpos"])
        r = self.plant.root_q
        yaw = ref["yaw"]
        c, s = np.cos(yaw), np.sin(yaw)
        pos = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ qpos[r:r + 3]
        # Start the ghost on the robot, once per policy run, so drift shows as separation.
        run = (ref["motion"], yaw)
        if self._ghost_anchor is None or self._ghost_anchor[0] != run:
            self._ghost_anchor = (run, self.display.qpos[r:r + 2] - pos[:2])
        pos[:2] += self._ghost_anchor[1]
        quat = np.empty(4)
        mujoco.mju_mulQuat(quat, np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]),
                           qpos[r + 3:r + 7])
        qpos[r:r + 3], qpos[r + 3:r + 7] = pos, quat
        self.ghost.qpos[:] = qpos
        mujoco.mj_kinematics(self.model, self.ghost)
        mujoco.mjv_addGeoms(self.model, self.ghost, self.opt, self.pert,
                            mujoco.mjtCatBit.mjCAT_DYNAMIC, scn)
        for i in range(scn.ngeom):
            scn.geoms[i].rgba[:] = GHOST_RGBA


def run(plant: Plant, args, viewer: Viewer | None = None) -> None:
    dt = plant.model.opt.timestep
    t0 = time.perf_counter()
    n, late = 0, 0
    last_pub, max_gap = t0, 0.0
    try:
        while viewer is None or viewer.is_running():
            plant.step()
            plant.publish()
            now = time.perf_counter()
            max_gap, last_pub = max(max_gap, now - last_pub), now
            n += 1
            wait = t0 + n * dt - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
            else:
                late += 1
            if args.duration and n * dt >= args.duration:
                break
    except KeyboardInterrupt:  # Ctrl-C ends the run as closing the viewer does
        pass

    wall = time.perf_counter() - t0
    print(f"[bridge] {n * dt:.1f} s simulated in {wall:.1f} s "
          f"(real-time factor {n * dt / wall:.2f}, {late}/{n} steps late), "
          f"state {1 / dt:.0f} Hz")
    print(f"[bridge] longest gap between states {max_gap * 1e3:.0f} ms")
    print(f"[bridge] commands: {plant.drives.accepted} group messages; "
          f"pelvis z {plant.data.qpos[plant.root_q + 2]:.3f} m, "
          f"lowest uprightness after release {plant.min_up:.2f}"
          + ("" if not plant.band.attached else " (band still attached)"))
    for line in plant.watch.summary():
        print(f"[bridge] {line}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", required=True, help="bundle dir from export_bundle.py")
    ap.add_argument("--plant", default="torque", help="torque | <path to an MJCF with motors>")
    ap.add_argument("--domain", type=int, default=SIM_DOMAIN, help="ROS domain id")
    ap.add_argument("--network", action="store_true",
                    help="let ROS traffic leave this machine (default: localhost only)")
    ap.add_argument("--lift", type=float, default=0.1,
                    help="metres above the init_state keyframe the robot starts at")
    ap.add_argument("--release-seconds", type=float, default=0.0,
                    help="release the band this long after the first gains "
                         "(0 = only with key 7)")
    ap.add_argument("--start-yaw", type=float, default=0.0,
                    help="initial heading in rad, to exercise heading alignment")
    ap.add_argument("--duration", type=float, default=0.0, help="seconds to run (0 = forever)")
    ap.add_argument("--viewer", action="store_true", help="interactive viewer")
    args = ap.parse_args()
    if args.domain == ROBOT_DOMAIN:
        raise SystemExit("domain 0 is where a robot would be: the bridge must not publish on it")

    node = init_ros(args.domain, "lmj_aimdk_bridge", args.network)
    try:
        plant = Plant(args, node)
        print(f"[bridge] ROS domain {args.domain}, plant {args.plant}")
        if args.viewer:
            print("[bridge] keys: 8 shorter band, 9 longer, 7 release/attach, 6 square (start)")
            viewer = Viewer(plant, args.bundle)
            try:
                run(plant, args, viewer)
            finally:
                viewer.close()
        else:
            run(plant, args)
    finally:
        shutdown_ros(node)


if __name__ == "__main__":
    main()
