"""A live browser view of the X2 as its joint encoders and IMU report it, with the reference.

Standalone it only listens, so it can run beside anything, the robot included::

    python deployment/x2/live_viser.py --bundle <dir>             # the robot, domain 0
    python deployment/x2/live_viser.py --bundle <dir> --domain 1  # the bridge

and ``deploy_x2.py --viser`` runs the same view fed directly by the controller. Either way
the view is served by a child process, so drawing and viser never hold the controller's
GIL. Both print the address to open.

The robot reports no position, so it stands at the origin: at the reference's pelvis height
while a policy runs, at the model's otherwise. The translucent blue robot is the reference
the controller is tracking (``/lmj/reference``), started on the robot and turned by the
controller's heading alignment, as in the bridge's viewer. A walking reference therefore
walks away from the robot; a gap in posture or orientation is tracking error.

It looks like LocoMuJoCo's ``ViserViewer``: the same floor, sky, lighting and panel, and its
camera modes, of which ``follow`` (the default) keeps the pelvis centred by moving the world.
The styling is copied, not imported, so it needs only ``viser`` (which brings ``trimesh``)
next to mujoco and numpy.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import select
import socket
import struct
import subprocess
import sys
import threading
import time
import warnings
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
from lmj_deploy import MotionBundle  # noqa: E402

FPS = 30.0
#: Seconds after the last reference before the ghost is hidden.
GHOST_TIMEOUT = 0.5

# The look of LocoMuJoCo's ViserViewer and of the mjviser scene under it.
GHOST_RGBA = (0.25, 0.55, 1.0, 0.35)  # rollout_viser.py's reference robot
FLOOR_RGB = (28, 30, 34)
ENVIRONMENT_INTENSITY = 0.8
FOV = 60
#: Fetched by the browser, so it needs the viewing machine online; "Hide logo" removes it.
LOGO = ("https://github.com/robfiras/loco-mujoco/assets/69359729/"
        "bd2a219e-ddfd-4355-8024-d9af921fb92a")
LOGO_WIDTH = 170
#: MujocoViewer's camera modes, all looking at the origin.
CAMERAS = {"follow": dict(distance=3.5, elevation=0.0, azimuth=90.0),
           "static": dict(distance=15.0, elevation=-45.0, azimuth=90.0),
           "top_static": dict(distance=5.0, elevation=-90.0, azimuth=90.0)}


class LiveViewer:
    """The live view in a child process, which draws the poses handed to :meth:`update`.

    Posing 2 x 32 bodies 30 times a second, and viser serialising it for a connected browser,
    is Python: on threads of the controller it took the GIL from the control loop and made
    ticks late. :meth:`update` writes one small message to a non-blocking pipe and returns.
    A pose the view has no room for is dropped, and if the view dies the controller carries
    on without it. The view ends when this process does.
    """

    def __init__(self, model: mujoco.MjModel, motions: MotionBundle, root_qpos_adr: int,
                 port: int = 8080):
        ready_r, ready_w = os.pipe()
        poses_r, self._fd = os.pipe()
        # Its own session, so Ctrl-C in this terminal reaches the controller only and the view
        # keeps showing the damping that follows.
        self._child = subprocess.Popen(
            [sys.executable, "-c", f"import sys; sys.path.insert(0, {str(HERE)!r}); "
             "import live_viser; live_viser._serve()"],
            stdin=poses_r, pass_fds=(ready_w,), start_new_session=True)
        os.close(poses_r)
        os.close(ready_w)
        try:
            _write_all(self._fd, _frame((model, str(motions.dir), root_qpos_adr, port, ready_w)))
        except BrokenPipeError:  # it died first; its exit code is reported below
            pass
        with os.fdopen(ready_r, "rb") as ready:
            reply = ready.read()  # the port, once the scene is up; nothing if it failed
        if not reply:
            raise RuntimeError(f"the live view did not start (exit code {self._child.wait()})")
        self._port = int(reply)
        os.set_blocking(self._fd, False)

    def urls(self) -> list[str]:
        hosts = ["localhost", socket.gethostname()]
        try:  # the address other machines on the default route reach this one by
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                hosts.append(s.getsockname()[0])
        except OSError:
            pass
        return [f"http://{h}:{self._port}" for h in dict.fromkeys(hosts)]

    def update(self, qpos: np.ndarray, reference: dict | None = None, text: str | None = None):
        """``qpos`` of the model (joints and root orientation; the position is ignored),
        ``reference`` as published on ``/lmj/reference``, and a line for the status panel."""
        if self._fd is None:
            return
        frame = _frame((np.asarray(qpos, dtype=np.float64), reference, text))
        if len(frame) > select.PIPE_BUF:  # only writes this small are all or nothing
            frame = _frame((np.asarray(qpos, dtype=np.float64), reference, None))
        try:
            os.write(self._fd, frame)
        except BlockingIOError:  # the view is behind: it only wants the newest pose anyway
            pass
        except OSError as e:
            print(f"[viser] the live view stopped ({e}); the controller carries on")
            os.close(self._fd)
            self._fd = None


def _frame(obj) -> bytes:
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    return struct.pack("<I", len(data)) + data


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        view = view[os.write(fd, view):]


def _read_frame(stream):
    head = stream.read(4)
    if len(head) < 4:
        return None
    return pickle.loads(stream.read(struct.unpack("<I", head)[0]))


def _serve() -> None:
    """The child of :class:`LiveViewer`: sets up the scene, then draws what arrives on stdin
    until the parent closes it."""
    os.nice(10)  # never ahead of the controller for a core
    stream = sys.stdin.buffer
    model, bundle, root_qpos_adr, port, ready = _read_frame(stream)
    scene = Scene(model, MotionBundle(bundle), root_qpos_adr, port)
    os.write(ready, str(scene.server.get_port()).encode())
    os.close(ready)
    while (pose := _read_frame(stream)) is not None:
        scene.update(*pose)
    scene.server.stop()


class Scene:
    """Draws poses handed to :meth:`update` from a daemon thread, ``FPS`` times a second.

    :meth:`update` only swaps in a reference to new arrays, so it never waits for drawing,
    and viser serves the browser from threads of its own.
    """

    def __init__(self, model: mujoco.MjModel, motions: MotionBundle, root_qpos_adr: int,
                 port: int = 8080):
        import viser

        self.model, self.motions, self.r = model, motions, root_qpos_adr
        self.robot, self.ghost = mujoco.MjData(model), mujoco.MjData(model)
        self.server = viser.ViserServer(host="0.0.0.0", port=port)
        self.server.scene.set_up_direction("+z")
        self.server.scene.configure_environment_map(environment_intensity=ENVIRONMENT_INTENSITY)
        self._fixed = (model.body_weldid == 0) & (model.body_mocapid[model.body_rootid] < 0)
        self._tracked = int(np.flatnonzero(~self._fixed)[0])  # the pelvis
        self._tracking = True
        self._world = self.server.scene.add_frame("/fixed_bodies", show_axes=False)
        self._floor_rgb = self._sky_rgb = None
        self._set_floor(FLOOR_RGB)
        self._set_sky(None)
        meshes = self._body_meshes()
        self._bodies = self._add_bodies("/bodies", meshes, ghost=False)
        self._ghost_bodies = self._add_bodies("/reference", meshes, ghost=True)
        az, el = np.radians(120.0), np.radians(20.0)  # mjviser's framing for a new browser
        self._home = 3.0 * model.stat.extent * np.array(
            [-np.cos(el) * np.cos(az), -np.cos(el) * np.sin(az), np.sin(el)])
        self._build_gui()
        self.server.on_client_connect(self._on_connect)
        self._latest: tuple | None = None
        self._anchor: tuple | None = None
        threading.Thread(target=self._loop, daemon=True).start()

    def update(self, qpos: np.ndarray, reference: dict | None = None, text: str | None = None):
        """As :meth:`LiveViewer.update`."""
        self._latest = (np.array(qpos), reference, text, time.perf_counter())

    def _set_floor(self, rgb: tuple) -> None:
        """Draws the model's ground planes as ViserViewer does: an infinite grid in ``rgb``,
        its lines pushed towards white on a dark floor and towards black on a light one."""
        base = np.clip(np.asarray(rgb, dtype=np.float64), 0, 255)
        target = 0.0 if base @ np.array([0.299, 0.587, 0.114]) > 127.0 else 255.0
        plane, cell, section = (tuple(int(c) for c in base + (target - base) * k)
                                for k in (0.0, 0.18, 0.32))
        m = self.model
        for g in np.flatnonzero(m.geom_type == mujoco.mjtGeom.mjGEOM_PLANE):
            if not self._fixed[m.geom_bodyid[g]]:
                continue
            self._floor_rgb = plane
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom_{g}"
            self.server.scene.add_grid(  # re-adding under the same name replaces it
                f"/fixed_bodies/{name}", infinite_grid=True, fade_distance=50.0,
                shadow_opacity=0.2, plane_color=plane, cell_color=cell, section_color=section,
                plane_opacity=1.0, position=m.geom_pos[g], wxyz=m.geom_quat[g])

    def _set_sky(self, sky: tuple | None) -> None:
        """A vertical gradient behind the scene: ``(top, horizon)`` RGB, or None for the
        model's skybox, which reproduces the OpenGL viewer's sky as ViserViewer does."""
        if sky is None:
            column = self._skybox_column()
            if column is None:
                return
        else:
            top, bottom = (np.asarray(c, dtype=np.float64) for c in sky)
            ramp = np.linspace(0.0, 1.0, 256)[:, None]
            column = np.clip(top * (1.0 - ramp) + bottom * ramp, 0, 255).astype(np.uint8)
        self._sky_rgb = (tuple(int(c) for c in column[0]), tuple(int(c) for c in column[-1]))
        # the background is stretched across the viewport, so a narrow strip is enough
        self.server.scene.set_background_image(np.repeat(column[:, None, :], 8, axis=1))

    def _skybox_column(self) -> np.ndarray | None:
        """Zenith-to-horizon colours of the model's skybox. MuJoCo stores it as a strip of six
        square faces (right, left, up, down, front, back); a column of a side face is enough."""
        m = self.model
        for i in range(m.ntex):
            w, h, n = int(m.tex_width[i]), int(m.tex_height[i]), int(m.tex_nchannel[i])
            if m.tex_type[i] != mujoco.mjtTexture.mjTEXTURE_SKYBOX or n < 3 or h < 5 * w:
                continue
            adr = int(m.tex_adr[i])
            strip = np.asarray(m.tex_data[adr:adr + w * h * n], dtype=np.uint8).reshape(h, w, n)
            return strip[4 * w:5 * w, w // 2, :3]
        return None

    def _body_meshes(self) -> dict:
        """Each moving body's mesh geoms in groups 0 to 2, merged into one mesh in the body's
        frame and coloured per vertex by material or geom, as mjviser draws them."""
        with warnings.catch_warnings():  # ROS's system SciPy objects to numpy<2, harmlessly
            warnings.filterwarnings("ignore", "A NumPy version", UserWarning)
            import trimesh

        m = self.model
        parts: dict[int, list] = {}
        for g in range(m.ngeom):
            b = int(m.geom_bodyid[g])
            if (self._fixed[b] or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH
                    or m.geom_group[g] > 2 or m.geom_rgba[g, 3] == 0):
                continue
            k = m.geom_dataid[g]
            v0, nv = m.mesh_vertadr[k], m.mesh_vertnum[k]
            f0, nf = m.mesh_faceadr[k], m.mesh_facenum[k]
            part = trimesh.Trimesh(m.mesh_vert[v0:v0 + nv], m.mesh_face[f0:f0 + nf],
                                   process=False)
            mat = m.geom_matid[g]
            rgba = np.clip(m.mat_rgba[mat] if mat >= 0 else m.geom_rgba[g], 0, 1)
            part.visual = trimesh.visual.ColorVisuals(
                vertex_colors=np.tile((rgba * 255).astype(np.uint8), (nv, 1)))
            rot, pose = np.empty(9), np.eye(4)
            mujoco.mju_quat2Mat(rot, m.geom_quat[g])
            pose[:3, :3], pose[:3, 3] = rot.reshape(3, 3), m.geom_pos[g]
            part.apply_transform(pose)
            parts.setdefault(b, []).append(part)
        meshes = {}
        for b, ps in parts.items():
            mesh = ps[0] if len(ps) == 1 else trimesh.util.concatenate(ps)
            if len(np.unique(mesh.visual.vertex_colors, axis=0)) == 1:
                mesh.merge_vertices()  # smooth shading, unless it would blend two colours
            meshes[b] = mesh
        return meshes

    def _add_bodies(self, root: str, meshes: dict, ghost: bool) -> dict[int, object]:
        """A hidden scene frame per body holding its mesh: the robot in its own colours and
        casting shadows, the reference in translucent blue without. Returns body id -> frame."""
        color = tuple(int(c * 255) for c in GHOST_RGBA[:3])
        frames = {}
        for b, mesh in meshes.items():
            name = f"{root}/{mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b) or b}"
            frames[b] = self.server.scene.add_frame(name, show_axes=False, visible=False)
            if ghost:
                self.server.scene.add_mesh_simple(
                    f"{name}/mesh", mesh.vertices, mesh.faces, color=color,
                    opacity=GHOST_RGBA[3], cast_shadow=False, receive_shadow=False)
            else:
                self.server.scene.add_mesh_trimesh(f"{name}/mesh", mesh)
        return frames

    def _build_gui(self) -> None:
        """ViserViewer's panel (the logo, then Scene, Visualization and LocoMuJoCo tabs), and
        below it the robot's status, where ``rollout_viser.py`` puts its clip controls."""
        import viser

        gui = self.server.gui
        logo = gui.add_html(  # pinned to the viewport's corner, click-through
            f'<div style="position:fixed; left:16px; top:14px; width:{LOGO_WIDTH}px;'
            f' z-index:100; pointer-events:none; opacity:0.85;">'
            f'<img src="{LOGO}" style="width:100%; height:auto; display:block;"'
            f' alt="LocoMuJoCo"/></div>')
        tabs = gui.add_tab_group()
        with tabs.add_tab("Scene", icon=viser.Icon.VIDEO), gui.add_folder("Camera"):
            track = gui.add_checkbox("Track camera", initial_value=self._tracking,
                                     hint="Keep the pelvis centred.")
            self._fov = gui.add_slider("FOV (°)", min=20, max=150, step=1,
                                       initial_value=FOV, hint="Vertical FOV in degrees.")
        with tabs.add_tab("Visualization", icon=viser.Icon.EYE):
            if self._floor_rgb is not None:
                with gui.add_folder("Floor"):
                    floor = gui.add_rgb("Color", initial_value=self._floor_rgb)
                    floor_reset = gui.add_button("Reset")

                @floor.on_update
                def _(_):
                    self._set_floor(tuple(floor.value))

                @floor_reset.on_click
                def _(_):
                    self._set_floor(FLOOR_RGB)
                    floor.value = self._floor_rgb

            if self._sky_rgb is not None:
                with gui.add_folder("Sky"):
                    top = gui.add_rgb("Top", initial_value=self._sky_rgb[0])
                    horizon = gui.add_rgb("Horizon", initial_value=self._sky_rgb[1])
                    sky_reset = gui.add_button("Reset")

                def _sky(_):
                    self._set_sky((top.value, horizon.value))

                top.on_update(_sky)
                horizon.on_update(_sky)

                @sky_reset.on_click
                def _(_):
                    self._set_sky(None)
                    top.value, horizon.value = self._sky_rgb

            with gui.add_folder("Logo"):
                hide_logo = gui.add_checkbox("Hide logo", initial_value=False)

            @hide_logo.on_update
            def _(_):
                logo.visible = not hide_logo.value

        with tabs.add_tab("LocoMuJoCo", icon=viser.Icon.RUN):
            self._pause = gui.add_checkbox("Pause", initial_value=False,
                                           hint="Freeze the view; the robot is not affected.")
            camera = gui.add_dropdown("Camera mode", tuple(CAMERAS), initial_value="follow")
        with gui.add_folder("Robot"):
            self._show_ghost = gui.add_checkbox("Show reference", initial_value=True)
            self.status = gui.add_markdown("waiting for the robot")

        @track.on_update
        def _(_):
            if track.value == self._tracking:
                return
            self._tracking = track.value
            if self._tracking:
                for client in self.server.get_clients().values():
                    client.camera.position, client.camera.look_at = self._home, np.zeros(3)

        @self._fov.on_update
        def _(_):
            for client in self.server.get_clients().values():
                client.camera.fov = np.radians(self._fov.value)

        @camera.on_update
        def _(_):
            self._set_camera(str(camera.value))
            track.value = self._tracking

    def _on_connect(self, client) -> None:
        client.camera.fov = np.radians(self._fov.value)
        client.camera.position = self._home
        client.camera.look_at = np.zeros(3) if self._tracking else self.model.stat.center

    def _set_camera(self, mode: str) -> None:
        """Puts every browser's camera where MujocoViewer's ``mode`` would; only ``follow``
        tracks the pelvis."""
        p = CAMERAS[mode]
        self._tracking = mode == "follow"
        az, el = np.radians(p["azimuth"]), np.radians(p["elevation"])
        # MuJoCo's camera looks along `forward`, so the eye sits `distance` behind the origin
        forward = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        for client in self.server.get_clients().values():
            client.camera.up_direction = np.array([0.0, 0.0, 1.0])
            client.camera.position = -p["distance"] * forward
            client.camera.look_at = np.zeros(3)

    def _loop(self) -> None:
        while True:
            t = time.perf_counter()
            latest = self._latest
            if latest is not None:
                qpos, reference, text, stamp = latest
                ghost = None
                if reference is not None and t - stamp < GHOST_TIMEOUT:
                    ghost = self._ghost_qpos(reference, qpos)  # also while hidden or paused
                if not self._pause.value:
                    self._draw(qpos, ghost, self._show_ghost.value)
                if text is not None:
                    self.status.content = text
            time.sleep(max(0.0, 1.0 / FPS - (time.perf_counter() - t)))

    def _draw(self, qpos: np.ndarray, ghost: np.ndarray | None, show_ghost: bool) -> None:
        r = self.r
        self.robot.qpos[:] = qpos
        self.robot.qpos[r:r + 2] = 0.0
        self.robot.qpos[r + 2] = ghost[r + 2] if ghost is not None else self.model.qpos0[r + 2]
        mujoco.mj_kinematics(self.model, self.robot)
        # tracking moves the world, not the camera, so the pelvis stays at the origin
        offset = -self.robot.xpos[self._tracked] if self._tracking else np.zeros(3)
        show_ghost = show_ghost and ghost is not None
        if show_ghost:
            self.ghost.qpos[:] = ghost
            mujoco.mj_kinematics(self.model, self.ghost)
        with self.server.atomic():
            self._world.position = tuple(offset)
            self._pose(self._bodies, self.robot, offset)
            if show_ghost:
                self._pose(self._ghost_bodies, self.ghost, offset)
            else:
                for frame in self._ghost_bodies.values():
                    frame.visible = False

    @staticmethod
    def _pose(frames: dict, data: mujoco.MjData, offset: np.ndarray) -> None:
        for b, frame in frames.items():
            frame.position = tuple(data.xpos[b] + offset)
            frame.wxyz = tuple(data.xquat[b])
            frame.visible = True

    def _ghost_qpos(self, reference: dict, robot_qpos: np.ndarray) -> np.ndarray:
        """The reference frame, turned by the controller's yaw and started on the robot."""
        r = self.r
        qpos = np.array(self.motions.frame(reference["motion"], reference["frame"])["qpos"])
        yaw = float(reference["yaw"])
        c, s = np.cos(yaw), np.sin(yaw)
        pos = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ qpos[r:r + 3]
        run = (reference["motion"], yaw)
        if self._anchor is None or self._anchor[0] != run:
            self._anchor = (run, -pos[:2])  # the robot stands at the origin
        pos[:2] += self._anchor[1]
        quat = np.empty(4)
        mujoco.mju_mulQuat(quat, np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]),
                           qpos[r + 3:r + 7])
        qpos[r:r + 3], qpos[r + 3:r + 7] = pos, quat
        return qpos


def pelvis_gravity(quat: np.ndarray) -> np.ndarray:
    """Gravity's direction in the frame of an IMU with orientation ``quat`` (w, x, y, z)."""
    w, x, y, z = quat / np.linalg.norm(quat)
    return -np.array([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)])


def main() -> None:
    from lmj_aimdk import (REFERENCE_TOPIC, ROBOT_DOMAIN, Latest, RobotState, StateAdapter,
                           String, init_ros, shutdown_ros)
    from lmj_deploy import ObservationBuilder, PolicySpec, training_model_xml

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", required=True, help="bundle dir, for the model and the clips")
    ap.add_argument("--domain", type=int, default=ROBOT_DOMAIN,
                    help=f"ROS domain (default {ROBOT_DOMAIN}, the robot's)")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    spec = PolicySpec.load(args.bundle)
    model = mujoco.MjModel.from_xml_path(str(training_model_xml(spec)))
    data = mujoco.MjData(model)
    builder = ObservationBuilder(spec, model)
    adapter = StateAdapter(builder)
    viewer = LiveViewer(model, MotionBundle(args.bundle, spec=spec), builder.root_qpos_adr,
                        args.port)
    node = init_ros(args.domain, "lmj_x2_live_viser", network=True)
    state = RobotState(node, spec)
    reference = Latest(node, String, REFERENCE_TOPIC)
    print("[viser] listening only, commanding nothing; open " + "  or  ".join(viewer.urls()))
    try:
        while True:
            state.poll()
            reference.poll()
            if state.complete:
                adapter.write(state, data)
                ref = json.loads(reference.msg.data) \
                    if reference.msg is not None and reference.age < GHOST_TIMEOUT else None
                gravity = pelvis_gravity(state.quat)
                viewer.update(data.qpos, ref,
                              f"pelvis tilt {np.degrees(np.arccos(-gravity[2])):.0f} deg, "
                              f"gravity {np.round(gravity, 2).tolist()} | state age "
                              f"{state.age * 1e3:.0f} ms"
                              + (f" | {ref['motion']} frame {ref['frame']}" if ref else ""))
            time.sleep(1.0 / FPS)
    except KeyboardInterrupt:
        pass
    finally:
        shutdown_ros(node)


if __name__ == "__main__":
    main()
