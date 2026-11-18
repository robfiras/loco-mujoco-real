"""MuJoCo standing in for the G1 on DDS.

Runs the physics in real time in its own process, publishes ``rt/lowstate`` and obeys
``rt/lowcmd``, so ``deploy_g1.py`` cannot tell it from the robot except by domain. Unlike
``rollout_mujoco.py``, the controller here is asynchronous: it sees state that is already
old, and a late command simply leaves the previous targets in force.

The bridge behaves like the motor firmware. It closes ``kp``/``kd`` at the physics rate,
holds the last command, ignores disabled motors and rejects bad CRCs.

The robot starts hanging from an invisible elastic band, as the real one hangs from a
crane, ``--lift`` above the ``init_state`` keyframe. In the viewer:

====  =================================================
8     shorten the band, lifting the robot
9     lengthen it, lowering the robot
7     release the band, or attach it again
6     press START on the simulated remote
====  =================================================

The viewer draws from its own thread, so rendering never holds up the physics. While
``deploy_g1.py`` runs a policy, it also draws the reference frame being tracked as a
translucent ghost, started on the robot and rotated by the controller's heading alignment.

So a start goes as on the robot: run ``deploy_g1.py`` without ``--auto-start``, let it ramp
and hold, lower the robot onto its feet, press START, then release the band. Headless,
``--release-seconds`` releases it on a timer and ``deploy_g1.py --auto-start`` starts::

    mjpython deployment/g1/sim2sim_bridge.py --bundle <dir> --viewer  # macOS GUI
    python deployment/g1/sim2sim_bridge.py --bundle <dir> --release-seconds 5
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import mujoco
import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]
from lmj_deploy import (  # noqa: E402
    MotionBundle,
    PolicySpec,
    PositionActionMapper,
    quat_to_mat,
)
from lmj_dds import (  # noqa: E402
    G1_MOTOR_JOINTS,
    LOWCMD_TOPIC,
    LOWSTATE_TOPIC,
    MOTOR_ENABLED,
    N_MOTORS,
    REMOTE_START,
    ROBOT_DOMAIN,
    SIM_DOMAIN,
    ChannelPublisher,
    ChannelSubscriber,
    CRC,
    LowCmd_,
    LowState_,
    MotorMap,
    ReferenceListener,
    init_dds,
    new_low_state,
)
from rollout_mujoco import build_model  # noqa: E402

#: ``mode_machine`` the bridge reports, and expects back. Any fixed value works in sim.
MODE_MACHINE = 5

#: N/m and N s/m. About 3.5 cm of stretch under the G1's weight, and close to critically
#: damped, so the robot hangs still rather than bouncing.
BAND_STIFFNESS = 10000.0
BAND_DAMPING = 1000.0
#: Metres per key press.
BAND_STEP = 0.05
#: Hung from the torso, above the robot's centre of mass, so it hangs upright.
BAND_BODY = "torso_link"
#: How long a START key press holds the button down.
REMOTE_PRESS_SECONDS = 0.3

KEY_START, KEY_BAND, KEY_SHORTER, KEY_LONGER = (ord(k) for k in "6789")

VIEWER_FPS = 30.0
GHOST_RGBA = (0.2, 0.8, 0.3, 0.35)
#: A reference stream older than this means no policy is running.
GHOST_TIMEOUT = 0.5


class Firmware:
    """The latest valid ``LowCmd``, as motor-order arrays."""

    def __init__(self):
        self.q = np.zeros(N_MOTORS)
        self.dq = np.zeros(N_MOTORS)
        self.tau = np.zeros(N_MOTORS)
        self.kp = np.zeros(N_MOTORS)
        self.kd = np.zeros(N_MOTORS)
        self.first_gains_at: float | None = None
        self.accepted = 0
        self.bad_crc = 0
        self.bad_mode_machine = 0
        self._crc = CRC()
        self._sub = ChannelSubscriber(LOWCMD_TOPIC, LowCmd_)
        self._sub.Init(self._on_message, 1)

    def _on_message(self, cmd: LowCmd_) -> None:
        if self._crc.Crc(cmd) != cmd.crc:
            self.bad_crc += 1
            return
        if cmd.mode_machine != MODE_MACHINE:
            self.bad_mode_machine += 1
            return
        motors = cmd.motor_cmd[:N_MOTORS]
        on = np.array([m.mode == MOTOR_ENABLED for m in motors], dtype=np.float64)
        self.q = np.array([m.q for m in motors])
        self.dq = np.array([m.dq for m in motors])
        self.tau = on * np.array([m.tau for m in motors])
        self.kp = on * np.array([m.kp for m in motors])
        self.kd = on * np.array([m.kd for m in motors])
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
        data.xfrc_applied[self.body, :3] = force


class Plant:
    def __init__(self, args):
        self.spec = PolicySpec.load(args.bundle)
        self.model = build_model(args.plant, self.spec, match_solver=True)
        self.data = mujoco.MjData(self.model)
        mapper = PositionActionMapper(self.spec, self.model)
        if mapper.position_actuators:
            raise SystemExit("--plant must have torque motors: the firmware closes the PD "
                             "loop here, as it does on the robot")

        motors = MotorMap(self.spec)
        jids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
                for n in G1_MOTOR_JOINTS]
        self.qpos_adr = self.model.jnt_qposadr[jids]
        self.dof_adr = self.model.jnt_dofadr[jids]
        self.ctrl_index = motors.from_actuators(mapper.ctrl_index).astype(np.int32)
        self.effort = motors.from_actuators(mapper.effort_limit)

        root = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, self.spec.root_joint_name)
        self.root_q = int(self.model.jnt_qposadr[root])
        self.root_v = int(self.model.jnt_dofadr[root])

        key = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "init_state")
        mujoco.mj_resetDataKeyframe(self.model, self.data, key)
        half = 0.5 * args.start_yaw
        self.data.qpos[self.root_q + 3:self.root_q + 7] = [np.cos(half), 0, 0, np.sin(half)]
        self.data.qpos[self.root_q + 2] += args.lift
        mujoco.mj_forward(self.model, self.data)
        self.band = ElasticBand(self.model, self.data, BAND_BODY)
        self.release_seconds = args.release_seconds
        self.start_until = 0.0

        self.firmware = Firmware()
        self.state = new_low_state()
        self.state.mode_machine = MODE_MACHINE
        self.pub = ChannelPublisher(LOWSTATE_TOPIC, LowState_)
        self.pub.Init()
        self.min_up = 1.0

    def step(self) -> None:
        d, fw = self.data, self.firmware
        q, dq = d.qpos[self.qpos_adr], d.qvel[self.dof_adr]
        tau = fw.tau + fw.kp * (fw.q - q) + fw.kd * (fw.dq - dq)
        d.ctrl[self.ctrl_index] = np.clip(tau, -self.effort, self.effort)
        started = fw.first_gains_at
        if (self.band.attached and self.release_seconds and started is not None
                and time.perf_counter() - started > self.release_seconds):
            self.release_band()
        self.band.apply(d)
        mujoco.mj_step(self.model, d)
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
            self.start_until = time.perf_counter() + REMOTE_PRESS_SECONDS
            print("[bridge] START pressed")
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
        d, s = self.data, self.state
        s.tick += 1
        for i in range(N_MOTORS):
            m = s.motor_state[i]
            m.q = float(d.qpos[self.qpos_adr[i]])
            m.dq = float(d.qvel[self.dof_adr[i]])
            m.tau_est = float(d.ctrl[self.ctrl_index[i]])
        s.imu_state.quaternion = d.qpos[self.root_q + 3:self.root_q + 7].astype(float).tolist()
        s.imu_state.gyroscope = d.qvel[self.root_v + 3:self.root_v + 6].astype(float).tolist()
        s.wireless_remote[2] = REMOTE_START if time.perf_counter() < self.start_until else 0
        self.pub.Write(s)


class Viewer:
    """The passive viewer, drawing a copy of the plant from its own thread.

    ``viewer.sync()`` can block for over 100 ms while a frame renders, which stalled
    ``LowState`` long enough for the controller to fault. Here the physics loop never
    touches the viewer: this thread copies the state into its own ``MjData``, draws the
    reference ghost and syncs, at ``VIEWER_FPS``.
    """

    def __init__(self, plant: Plant, bundle: str):
        import mujoco.viewer

        self.plant = plant
        self.model = plant.model
        self.display = mujoco.MjData(self.model)
        self.ghost = mujoco.MjData(self.model)
        self.motions = MotionBundle(bundle, spec=plant.spec)
        self.reference = ReferenceListener()
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
            self.handle.sync()
            time.sleep(max(0.0, 1.0 / VIEWER_FPS - (time.perf_counter() - t)))

    def _draw_ghost(self) -> None:
        scn = self.handle.user_scn
        scn.ngeom = 0
        ref = self.reference.latest
        if ref is None or self.reference.age > GHOST_TIMEOUT:
            self._ghost_anchor = None
            return
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
    every = max(1, round(1.0 / (args.publish_hz * dt)))
    t0 = time.perf_counter()
    n, late = 0, 0
    last_pub, max_gap = t0, 0.0
    while viewer is None or viewer.is_running():
        plant.step()
        if n % every == 0:
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

    wall = time.perf_counter() - t0
    fw = plant.firmware
    print(f"[bridge] {n * dt:.1f} s simulated in {wall:.1f} s "
          f"(real-time factor {n * dt / wall:.2f}, {late}/{n} steps late), "
          f"state {1 / (every * dt):.0f} Hz")
    print(f"[bridge] longest gap between LowState messages {max_gap * 1e3:.0f} ms")
    print(f"[bridge] commands: {fw.accepted} accepted, {fw.bad_crc} bad crc, "
          f"{fw.bad_mode_machine} bad mode_machine; "
          f"pelvis z {plant.data.qpos[plant.root_q + 2]:.3f} m, "
          f"lowest uprightness after release {plant.min_up:.2f}"
          + ("" if not plant.band.attached else " (band still attached)"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", required=True, help="bundle dir from export_bundle.py")
    ap.add_argument("--plant", default="torque", help="torque | <path to an MJCF with motors>")
    ap.add_argument("--domain", type=int, default=SIM_DOMAIN)
    ap.add_argument("--interface", default="lo0", help="network interface (lo on Linux)")
    ap.add_argument("--publish-hz", type=float, default=200.0,
                    help="LowState rate. The robot does 500 Hz, which Python may not sustain")
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
        raise SystemExit("domain 0 is the robot's: the bridge must not publish on it")

    init_dds(args.domain, args.interface)
    plant = Plant(args)
    print(f"[bridge] domain {args.domain} on {args.interface}, plant {args.plant}")
    if args.viewer:
        print("[bridge] keys: 8 shorter band, 9 longer, 7 release/attach, 6 START")
        viewer = Viewer(plant, args.bundle)
        try:
            run(plant, args, viewer)
        finally:
            viewer.close()
    else:
        run(plant, args)


if __name__ == "__main__":
    main()
