"""Convert BONES-SEED Unitree G1 motions (CSV) into trajectories for ``G1Real``.

BONES-SEED ships every motion already retargeted to the G1, so no retargeting is needed:
this only changes units and layout, then replays the motion once in ``G1Real`` to add the
body and site quantities the tracking reward and observations read.

::

    python training/g1/bones_to_g1.py bones-seed/g1/csv/220728/walk_forward_start_001__A042.csv
    python training/g1/bones_to_g1.py bones-seed/g1/csv/220728 --out data/bones --match walk_forward

Each CSV becomes ``<name>.lmj.npz``, next to it or in ``--out``. Train on them with
``custom_dataset_conf.paths`` in ``training/g1/conf.yaml``, and look at one with
``training/replay_trajectory.py <name>.lmj.npz``.

The CSV format, one row per frame at 120 fps::

    Frame,
    root_translate{X,Y,Z}     centimetres, Z up
    root_rotate{X,Y,Z}        degrees, extrinsic xyz Euler (intrinsic ZYX)
    <29 joint angles>         degrees, named like the model's joints plus ``_dof``

Joints are matched to the model by name. Velocities are central differences of the
positions.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import jax.numpy as jnp
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as sRot

import lmj_real  # noqa: F401  (registers G1Real)
from loco_mujoco import LocoEnv
from loco_mujoco.core.trajectory import Trajectory, TrajectoryData, TrajectoryInfo, TrajectoryModel
from loco_mujoco.datasets.humanoids.LAFAN1.load import extend_motion
from loco_mujoco.smpl.retargeting import load_robot_conf_file

ENV_NAME = "G1Real"
BONES_FPS = 120.0
CM_TO_M = 0.01
N_ROOT_COLUMNS = 7  # Frame + 3 translation + 3 rotation


def parse_bones_csv(path: Path) -> dict:
    """Root translation, root rotation and joint angles of one CSV, in its own units."""
    with open(path) as f:
        header = f.readline().strip().split(",")
    arr = np.atleast_2d(np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64))
    if arr.shape[1] != len(header) or arr.shape[1] <= N_ROOT_COLUMNS:
        raise ValueError(f"{path}: {arr.shape[1]} columns for a header of {len(header)}")
    return dict(
        root_xyz_cm=arr[:, 1:4],
        root_euler_deg=arr[:, 4:7],
        joint_angles_deg=arr[:, N_ROOT_COLUMNS:],
        joint_names=[c.removesuffix("_dof") for c in header[N_ROOT_COLUMNS:]],
    )


def central_difference(x: np.ndarray, dt: float) -> np.ndarray:
    v = np.zeros_like(x)
    v[1:-1] = (x[2:] - x[:-2]) / (2 * dt)
    if len(x) > 2:
        v[0], v[-1] = v[1], v[-2]
    return v


def csv_to_trajectory(path: Path, model: mujoco.MjModel, euler: str, z_lift: float) -> Trajectory:
    """A qpos / qvel trajectory in ``model``'s joint layout, at the CSV's 120 fps."""
    parsed = parse_bones_csv(path)

    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
    types = np.array(model.jnt_type)
    if types[0] != mujoco.mjtJoint.mjJNT_FREE or np.any(types[1:] != mujoco.mjtJoint.mjJNT_HINGE):
        raise ValueError("expected one free joint followed by hinges")
    hinges = names[1:]
    missing = [n for n in hinges if n not in parsed["joint_names"]]
    if missing:
        raise ValueError(f"{path}: no column for {missing}")
    columns = [parsed["joint_names"].index(n) for n in hinges]

    n = len(parsed["root_xyz_cm"])
    rot = sRot.from_euler(euler, np.deg2rad(parsed["root_euler_deg"]))
    quat_xyzw = rot.as_quat()

    qpos = np.zeros((n, model.nq))
    qpos[:, 0:3] = parsed["root_xyz_cm"] * CM_TO_M
    qpos[:, 2] += z_lift
    qpos[:, 3] = quat_xyzw[:, 3]
    qpos[:, 4:7] = quat_xyzw[:, 0:3]
    qpos[:, 7:] = np.deg2rad(parsed["joint_angles_deg"][:, columns])

    dt = 1.0 / BONES_FPS
    qvel = np.zeros((n, model.nv))
    qvel[:, 0:3] = central_difference(qpos[:, 0:3], dt)
    if n > 2:  # world-frame angular velocity from the rotation between the two neighbours
        qvel[1:-1, 3:6] = (rot[2:] * rot[:-2].inv()).as_rotvec() / (2 * dt)
        qvel[0, 3:6], qvel[-1, 3:6] = qvel[1, 3:6], qvel[-2, 3:6]
    qvel[:, 6:] = central_difference(qpos[:, 7:], dt)

    info = TrajectoryInfo(joint_names=names, frequency=BONES_FPS,
                          model=TrajectoryModel(njnt=model.njnt, jnt_type=jnp.array(types)))
    data = TrajectoryData(qpos=jnp.array(qpos), qvel=jnp.array(qvel), split_points=jnp.array([0, n]))
    return Trajectory(info=info, data=data)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", nargs="+", type=Path,
                    help="BONES-SEED G1 CSV files, or directories of them (searched recursively)")
    ap.add_argument("--out", type=Path, default=None,
                    help="output directory (default: next to each CSV)")
    ap.add_argument("--match", default=None, help="only motions whose name matches this regex")
    ap.add_argument("--euler", default="xyz",
                    help="scipy Euler convention of root_rotate{X,Y,Z} (default: extrinsic xyz)")
    ap.add_argument("--z-lift", type=float, default=0.0,
                    help="metres added to the root height, if the feet sink into the floor")
    args = ap.parse_args()

    csvs = sorted(p for s in args.source for p in ([s] if s.is_file() else s.rglob("*.csv")))
    if args.match:
        csvs = [p for p in csvs if re.search(args.match, p.stem)]
    if not csvs:
        raise SystemExit("no CSV files found")
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)

    robot_conf = load_robot_conf_file(ENV_NAME)
    model = LocoEnv.registered_envs[ENV_NAME](
        th_params=dict(random_start=False, fixed_start_conf=(0, 0)), print_obs_summary=False)._model

    for i, path in enumerate(csvs, 1):
        traj = csv_to_trajectory(path, model, args.euler, args.z_lift)
        traj = extend_motion(ENV_NAME, robot_conf, traj, replace_qvel_with_finite_diff=False)
        out = (args.out or path.parent) / f"{path.stem}.lmj.npz"
        traj.save(str(out))
        n = int(traj.data.qpos.shape[0])
        print(f"[{i}/{len(csvs)}] {out}  {n} frames at {traj.info.frequency:.0f} Hz "
              f"({n / traj.info.frequency:.1f} s)")


if __name__ == "__main__":
    main()
