"""Pick a reference clip in the browser and watch an exported policy track it live.

The rollout is the one in :mod:`rollout_mujoco` (plain MuJoCo, the bundle's numpy actor,
observations rebuilt by hand), so it plays PPO and SuperMimic bundles for either robot.
Only the picture comes from LocoMuJoCo, through its ``ViserViewer``::

    python deployment/rollout_viser.py --bundle deploy_bundles/x2_rel_full

Then open http://localhost:8080. The "Motion" panel filters the bundle's clips by name and
plays the chosen one from its first frame. The clip restarts when it ends and, with
"Restart on fall", when a training termination condition fires. The translucent blue robot
is the reference, drawn where the policy is asked to be.

Requires the viser extra: ``pip install loco-mujoco[viser]``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rollout_mujoco import Runner  # noqa: E402

_MAX_OPTIONS = 300  # viser dropdowns get sluggish with very long option lists
_GHOST_RGBA = np.array([0.25, 0.55, 1.0, 0.35])


class LiveRunner(Runner):
    """``Runner`` with the reference robot's pose and without the per-reset printing."""

    def __init__(self, args):
        super().__init__(args)
        self._ghost_data = mujoco.MjData(self.model)
        m = self.model
        mesh = m.geom_type == mujoco.mjtGeom.mjGEOM_MESH
        visual = mesh & (m.geom_contype == 0) & (m.geom_conaffinity == 0)
        self._ghost_geoms = np.flatnonzero(visual if visual.any() else mesh)

    def _debug_obs(self, obs, action) -> None:
        pass

    def ghost_scene(self, show: bool) -> SimpleNamespace:
        """The reference robot's visual meshes, in the layout ``ViserViewer`` reads markers from."""
        ids = self._ghost_geoms if show else self._ghost_geoms[:0]
        d = self._ghost_data
        if show:
            ref = self.motions.frame(self.motion_name, self.frame)
            n = min(self.model.nq, ref["qpos"].shape[0])
            d.qpos[:n] = ref["qpos"][:n]
            mujoco.mj_kinematics(self.model, d)
        geoms = SimpleNamespace(
            type=np.full(len(ids), int(mujoco.mjtGeom.mjGEOM_MESH)),
            size=np.ones((len(ids), 3)),
            pos=d.geom_xpos[ids].copy(),
            mat=d.geom_xmat[ids].copy(),
            rgba=np.tile(_GHOST_RGBA, (len(ids), 1)),
            dataid=self.model.geom_dataid[ids].copy(),
        )
        return SimpleNamespace(geoms=geoms)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", required=True, help="bundle dir from export_bundle.py")
    ap.add_argument("--plant", default="servo", help="servo | torque | <path to an MJCF>")
    ap.add_argument("--motion", default=None,
                    help="clip to start with (default: first clip whose name contains 'walk')")
    ap.add_argument("--port", type=int, default=8080, help="viser port")
    ap.add_argument("--std-floor", type=float, default=0.0, help="see rollout_mujoco.py")
    ap.add_argument("--camera", choices=("static", "follow", "top_static"), default="follow")
    args = ap.parse_args()
    args.decimation, args.start_frame = 0, 0
    args.loop = args.stop_on_terminate = args.no_match_solver = False

    from loco_mujoco.core.visuals.viser_viewer import ViserViewer

    runner = LiveRunner(args)
    names = list(runner.motions.names)
    viewer = ViserViewer(runner.model, runner.dt_control, port=args.port,
                         default_camera_mode=args.camera)
    gui = viewer.server.gui

    def matches(text: str) -> list[str]:
        text = text.strip().lower()
        hits = [n for n in names if text in n.lower()]
        return hits[:_MAX_OPTIONS] or [runner.motion_name]

    request: dict[str, str | None] = {"motion": None}
    counts = {"done": 0, "falls": 0}

    with gui.add_folder("Motion"):
        search = gui.add_text("Filter", initial_value="",
                              hint=f"{len(names)} clips; the list shows at most {_MAX_OPTIONS}.")
        clip = gui.add_dropdown("Clip", options=matches(""), initial_value=runner.motion_name
                                if runner.motion_name in matches("") else matches("")[0])
        restart = gui.add_button("Restart clip")
        on_fall = gui.add_checkbox("Restart on fall", initial_value=True)
        ghost = gui.add_checkbox("Show reference", initial_value=True)
        status = gui.add_markdown("")

    @search.on_update
    def _(_):
        clip.options = matches(search.value)

    @clip.on_update
    def _(_):
        request["motion"] = str(clip.value)

    @restart.on_click
    def _(_):
        request["motion"] = runner.motion_name

    runner.reset_to(0)
    print(f"[viser] {len(names)} clips, open http://localhost:{viewer.server.get_port()} "
          f"(Ctrl-C to stop)", flush=True)

    try:
        while True:
            wanted, request["motion"] = request["motion"], None
            if wanted is not None and wanted in names:
                if wanted != runner.motion_name:
                    counts["done"] = counts["falls"] = 0
                runner.motion_name = wanted
                runner.reset_to(0)

            runner.step_once()
            length = runner.motions.length(runner.motion_name)
            fell = runner.terminated_at is not None
            status.content = (f"**{runner.motion_name}**  \nframe {runner.frame + 1} / {length}"
                              f"{'  (terminated)' if fell else ''}  \n"
                              f"completed {counts['done']}, falls {counts['falls']}")
            viewer.render(runner.data, SimpleNamespace(user_scene=runner.ghost_scene(ghost.value)),
                          record=False)

            if fell and on_fall.value:
                counts["falls"] += 1
                runner.reset_to(0)
            elif runner.frame >= length - 1:
                counts["done"] += not fell
                runner.reset_to(0)
    except KeyboardInterrupt:
        viewer.stop()


if __name__ == "__main__":
    main()
