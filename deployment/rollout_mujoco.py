"""Rolls out an exported bundle in plain MuJoCo, for G1Real and X2UltraReal tracking policies.

One process, plain MuJoCo, the exported bundle, and nothing from the training stack.
Observations are rebuilt by hand from ``MjData`` (:mod:`lmj_deploy`), the actor runs in
numpy, and the reference clip is read from the bundle -- so a run here exercises the
same code path a robot would, minus DDS and the IMU.

Two built-in plants, ``--plant``, or a path to any MJCF:

``servo`` (default)
    The training MJCF, unchanged: MuJoCo position servos closing the PD loop inside the
    physics. Same plant, same actuation, different observation code -- so a failure here
    is a pipeline bug, not a sim2sim gap. Run this first.

``torque``
    The same MJCF with the position servos replaced by torque motors, and the PD loop
    moved into Python at the simulation rate using the bundle's ``kp`` / ``kd``. This is
    what the real G1 does (the joint MCU closes the loop on a target angle at ~500 Hz),
    so this isolates the actuation gap from every other difference.

Point ``--plant`` at the stock Unitree MJCF to measure the full plant gap: joint damping
and frictionloss, the vendor armature values, and mesh collision geometry. Read those
numbers as a robustness probe, not a regression test.

Runs headless by default and prints tracking error; pass ``--viewer`` for the
interactive window. On macOS the passive viewer needs ``mjpython``::

    python deployment/rollout_mujoco.py --bundle <dir>            # headless
    mjpython deployment/rollout_mujoco.py --bundle <dir> --viewer # macOS GUI

Viewer keys: ``r`` reset, ``n`` next motion, ``space`` pause, ``q`` quit.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lmj_deploy import (  # noqa: E402
    MotionBundle,
    ObservationBuilder,
    PolicySpec,
    PositionActionMapper,
    apply_model_contact,
    apply_model_options,
    check_bundle_against_model,
    load_actor,
    training_model_xml,
)


def build_model(plant: str, spec: PolicySpec, match_solver: bool) -> mujoco.MjModel:
    """Compile the requested plant and put it on the training solver settings."""
    if plant == "servo":
        model = mujoco.MjModel.from_xml_path(str(training_model_xml(spec)))
    elif plant == "torque":
        model = _torque_variant(training_model_xml(spec))
    else:
        model = mujoco.MjModel.from_xml_path(plant)  # treat as a path

    if plant in ("servo", "torque"):
        apply_model_contact(model, spec)  # the training model, so also its contact setup
    if match_solver:
        apply_model_options(model, spec)
    else:
        model.opt.timestep = spec.sim_timestep
    return model


def _torque_variant(xml: Path) -> mujoco.MjModel:
    """The MJCF with its position servos turned into plain torque motors.

    Only the actuators change: same bodies, same joints, same armature, same collision
    set, same keyframe. ``PositionActionMapper`` notices the non-affine bias and closes
    the PD loop itself.
    """
    spec = mujoco.MjSpec.from_file(str(xml))
    for a in spec.actuators:
        force = np.asarray(a.forcerange, dtype=float).copy()
        a.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        a.gainprm = [1.0] + [0.0] * (len(a.gainprm) - 1)
        a.biastype = mujoco.mjtBias.mjBIAS_NONE
        a.biasprm = [0.0] * len(a.biasprm)
        a.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
        a.ctrlrange = force
    return spec.compile()


class Runner:
    """Owns the plant, the actor, and the motion cursor."""

    def __init__(self, args):
        self.spec = PolicySpec.load(args.bundle)
        self.actor = load_actor(args.bundle, self.spec, std_floor=args.std_floor)
        self.motions = MotionBundle(args.bundle)
        self.model = build_model(args.plant, self.spec, not args.no_match_solver)
        self.data = mujoco.MjData(self.model)

        for w in check_bundle_against_model(self.spec, self.motions, self.model):
            print(f"[rollout] WARNING: {w}")

        self.obs_builder = ObservationBuilder(self.spec, self.model)
        self.action_mapper = PositionActionMapper(self.spec, self.model)

        self.decimation = args.decimation or self.spec.n_substeps
        self.dt_control = self.model.opt.timestep * self.decimation
        self.loop = bool(args.loop)
        self.stop_on_terminate = bool(args.stop_on_terminate)
        self.start_frame = int(args.start_frame)

        mode = "position servos (in-physics PD)" if self.action_mapper.position_actuators \
            else "torque motors (Python PD at sim rate)"
        print(f"[rollout] plant={args.plant}  {mode}")
        print(f"[rollout] control {1 / self.dt_control:.1f} Hz "
              f"({self.model.opt.timestep * 1e3:.1f} ms x {self.decimation}); "
              f"motion dt {self.motions.dt * 1e3:.1f} ms")
        if abs(self.dt_control - self.motions.dt) > 1e-9:
            print(f"[rollout] WARNING: control step {self.dt_control:.4f}s != motion dt "
                  f"{self.motions.dt:.4f}s -- the reference will play at the wrong rate")
        if self.actor.n_low_std and args.std_floor == 0.0:
            print(f"[rollout] NOTE: {self.actor.n_low_std} observation dims have a "
                  f"normalizer std < 1e-3. On a plant other than 'servo' consider "
                  f"--std-floor 1e-3; see MlpActor.")

        self.motion_name = args.motion or self._pick_motion()
        if self.motion_name not in self.motions.names:
            raise SystemExit(f"motion {self.motion_name!r} not in bundle. Available: "
                             f"{self.motions.names[:10]}...")
        self.paused = False
        self.frame = self.start_frame
        self.steps = 0
        self.terminated_at: int | None = None
        self._err_sum = 0.0
        self.last_action = np.zeros(self.spec.action_dim, dtype=np.float64)

    def _pick_motion(self) -> str:
        walks = [n for n in self.motions.names if "walk" in n.lower()]
        return walks[0] if walks else self.motions.names[0]

    # -- episode -----------------------------------------------------------

    def reset_to(self, frame: int) -> None:
        """Teleport to the reference frame, as ``TrajInitialStateHandler`` does.

        The bundle carries the full ``qpos`` / ``qvel`` of the clip, so this is the same
        state training reset to -- not a reconstruction from pelvis pose + joint angles.

        The robot is placed on the reference. Training may start the episode at the
        origin instead, but it moves the reference by the same xy shift, so the two are
        the same up to a translation of the world.
        """
        ref = self.motions.frame(self.motion_name, frame)
        mujoco.mj_resetData(self.model, self.data)
        n = min(self.model.nq, ref["qpos"].shape[0])
        self.data.qpos[:n] = ref["qpos"][:n]
        nv = min(self.model.nv, ref["qvel"].shape[0])
        self.data.qvel[:nv] = ref["qvel"][:nv]
        mujoco.mj_forward(self.model, self.data)

        self.frame = frame
        self.last_action[:] = 0.0
        self.actor.reset()
        self.obs_builder.reset_history()
        self.steps = 0
        self.terminated_at: int | None = None
        self._err_sum = 0.0

        feet = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n_)
                for n_ in ("left_ankle_roll_link", "right_ankle_roll_link")]
        foot_z = [float(self.data.xpos[i, 2]) for i in feet if i >= 0]
        anchor_z = float(self.data.xpos[self.obs_builder.anchor_body_id, 2])
        print(f"[rollout] reset '{self.motion_name}'[{frame}] "
              f"(T={self.motions.length(self.motion_name)})  "
              f"anchor_z={anchor_z:+.3f}  foot_z={[round(z, 3) for z in foot_z]}")
        if any(z < -0.005 for z in foot_z):
            print("[rollout]   WARNING: feet below ground at reset")

    def step_once(self) -> None:
        ref = self.motions.frame(self.motion_name, self.frame)
        obs = self.obs_builder.build(self.data, ref, self.last_action)
        action = self.actor.act(obs)
        self.last_action = action.astype(np.float64)

        if self.steps < 3:
            self._debug_obs(obs, action)

        target = self.action_mapper.target(action)
        for _ in range(self.decimation):
            self.action_mapper.apply(self.data, target)
            mujoco.mj_step(self.model, self.data)
            if not np.all(np.isfinite(self.data.qvel)):
                print(f"[rollout] BLOWUP: non-finite qvel at step {self.steps}")
                self.terminated_at = self.steps
                return

        metrics = self.obs_builder.termination_metrics(self.data, ref)
        self._err_sum += metrics["ee_pos_z"]
        hits = self.obs_builder.would_terminate(metrics)
        if hits and self.terminated_at is None:
            self.terminated_at = self.steps
            print(f"[rollout] step {self.steps}: termination condition met "
                  f"({', '.join(hits)})"
                  + ("" if self.stop_on_terminate else "  [continuing]"))

        self.steps += 1
        T = self.motions.length(self.motion_name)
        self.frame = (self.frame + 1) % T if self.loop else min(self.frame + 1, T - 1)

    def _debug_obs(self, obs: np.ndarray, action: np.ndarray) -> None:
        """Per-term z-score summary: the fastest way to spot an obs the actor never saw."""
        z = self.actor.z_scores(obs)
        print(f"[rollout] step {self.steps}: action [{action.min():+.2f}, "
              f"{action.max():+.2f}] |a|.mean={np.abs(action).mean():.2f}")
        for name, sl in self.obs_builder.term_slices.items():
            zs = np.abs(z[sl])
            i = int(np.argmax(zs))
            print(f"[rollout]     {name:20s} |z| mean {zs.mean():5.2f} "
                  f"max {zs.max():7.2f} @ local index {i}")

    def summary(self) -> dict[str, float]:
        return {
            "steps": self.steps,
            "terminated_at": -1 if self.terminated_at is None else self.terminated_at,
            "mean_ee_pos_z_err": self._err_sum / max(self.steps, 1),
        }

    # -- ui ----------------------------------------------------------------

    def next_motion(self) -> None:
        i = (self.motions.names.index(self.motion_name) + 1) % len(self.motions.names)
        self.motion_name = self.motions.names[i]
        self.reset_to(self.start_frame)

    def key_callback(self, key: int) -> None:
        if key in (ord("R"), ord("r")):
            self.reset_to(self.start_frame)
        elif key in (ord("N"), ord("n")):
            self.next_motion()
        elif key == ord(" "):
            self.paused = not self.paused
            print("[rollout] " + ("paused" if self.paused else "resumed"))
        elif key in (ord("Q"), ord("q")):
            os._exit(0)

    def draw_reference(self, scn, raw: bool = False) -> None:
        """Green spheres at the re-anchored reference bodies (what the reward scores).

        With ``raw``, red spheres at the unanchored world-frame reference too. The
        re-anchored markers are pinned to the robot's xy and heading by construction, so
        only the raw ones can show global drift.
        """
        ref = self.motions.frame(self.motion_name, self.frame)
        targets = [(self.obs_builder.reanchored_reference(self.data, ref),
                    (0.1, 0.9, 0.2, 0.65))]
        if raw:
            targets.append((np.asarray(ref["body_pos_w"], dtype=np.float64),
                            (1.0, 0.25, 0.2, 0.35)))
        eye = np.eye(3).flatten()
        size = np.array([0.04, 0.0, 0.0])
        i = 0
        for pts, color in targets:
            for p in pts:
                if i >= scn.maxgeom:
                    break
                mujoco.mjv_initGeom(
                    scn.geoms[i], type=mujoco.mjtGeom.mjGEOM_SPHERE, size=size,
                    pos=np.asarray(p, dtype=np.float64), mat=eye,
                    rgba=np.array(color, dtype=np.float32),
                )
                i += 1
        scn.ngeom = i


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", required=True, help="bundle dir from export_bundle.py")
    ap.add_argument("--plant", default="servo",
                    help="servo | torque | <path to an MJCF>")
    ap.add_argument("--motion", default=None,
                    help="clip name (default: first clip whose name contains 'walk')")
    ap.add_argument("--n-steps", type=int, default=500,
                    help="control steps per episode (0 = the clip's length)")
    ap.add_argument("--episodes", type=int, default=1,
                    help="headless only: run this many clips, starting at --motion")
    ap.add_argument("--loop", action="store_true",
                    help="wrap to frame 0 at end-of-clip. Off by default: wrapping "
                         "teleports the reference anchor across the whole clip in one "
                         "control tick and the policy blows up. Default holds the last "
                         "frame.")
    ap.add_argument("--start-frame", type=int, default=0,
                    help="reference frame to start each episode from (training uses "
                         "random state initialization, so mid-clip starts are normal)")
    ap.add_argument("--decimation", type=int, default=0,
                    help="physics steps per control step (0 = the bundle's n_substeps)")
    ap.add_argument("--std-floor", type=float, default=0.0,
                    help="floor on the observation-normalizer std. 0.0 reproduces "
                         "training exactly; 1e-3 tames dims that were constant during "
                         "training and blow up on a different plant.")
    ap.add_argument("--no-match-solver", action="store_true",
                    help="do not copy the training model's mj_option onto the plant")
    ap.add_argument("--stop-on-terminate", action="store_true",
                    help="end the episode when a training termination condition fires")
    ap.add_argument("--viewer", action="store_true", help="interactive viewer")
    ap.add_argument("--draw-raw-reference", action="store_true",
                    help="with --viewer, also draw the unanchored reference in red")
    args = ap.parse_args()

    runner = Runner(args)

    if args.viewer:
        import mujoco.viewer

        runner.reset_to(runner.start_frame)
        print("\nKeys: r reset   n next motion   space pause/resume   q quit\n")
        with mujoco.viewer.launch_passive(
            runner.model, runner.data, key_callback=runner.key_callback
        ) as viewer:
            next_t = time.perf_counter() + runner.dt_control
            while viewer.is_running():
                if not runner.paused:
                    runner.step_once()
                    runner.draw_reference(viewer.user_scn, args.draw_raw_reference)
                    if args.stop_on_terminate and runner.terminated_at is not None:
                        runner.reset_to(runner.start_frame)
                viewer.sync()
                sleep = next_t - time.perf_counter()
                if sleep > 0:
                    time.sleep(sleep)
                next_t += runner.dt_control
                if next_t < time.perf_counter():
                    next_t = time.perf_counter() + runner.dt_control
        return

    start = runner.motions.names.index(runner.motion_name)
    results = []
    for k in range(args.episodes):
        runner.motion_name = runner.motions.names[(start + k) % len(runner.motions.names)]
        runner.reset_to(runner.start_frame)
        limit = args.n_steps or runner.motions.length(runner.motion_name)
        for _ in range(limit):
            runner.step_once()
            if args.stop_on_terminate and runner.terminated_at is not None:
                break
        s = runner.summary()
        results.append(s)
        print(f"[rollout] {runner.motion_name}: {s['steps']} steps, "
              f"terminated_at={s['terminated_at']}, "
              f"mean ee z err {s['mean_ee_pos_z_err']:.4f} m")

    if len(results) > 1:
        alive = [r for r in results if r["terminated_at"] < 0]
        print(f"\n[rollout] {len(alive)}/{len(results)} episodes never hit a "
              f"termination condition; mean ee z err "
              f"{np.mean([r['mean_ee_pos_z_err'] for r in results]):.4f} m")


if __name__ == "__main__":
    main()
