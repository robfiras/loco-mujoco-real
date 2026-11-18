"""
Replay a default mocap clip, or a saved loco-mujoco Trajectory (.npz), in a robot's
environment.

Usage:
    python training/replay_trajectory.py [--env ENV] [--task TASK] [--n-episodes N]
    python training/replay_trajectory.py PATH.npz [--env ENV] [--n-episodes N]
"""
from __future__ import annotations

import argparse

import lmj_real  # registers the environments
from loco_mujoco.core.trajectory import Trajectory
from loco_mujoco.task_factories import ImitationFactory, CustomDatasetConf


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz", type=str, nargs="?", default=None,
                    help="Path to a saved Trajectory .npz file. Overrides --task.")
    ap.add_argument("--env", type=str, default="G1Real", help="Environment: G1Real or X2UltraReal.")
    ap.add_argument("--task", type=str, default="walk", help="Default mocap clip to replay.")
    ap.add_argument("--n-episodes", type=int, default=1,
                    help="Number of times to replay the trajectory.")
    args = ap.parse_args()

    if args.npz:
        dataset = dict(custom_dataset_conf=CustomDatasetConf(traj=Trajectory.load(args.npz)))
    else:
        dataset = dict(default_dataset_conf=dict(task=args.task))

    env, traj = ImitationFactory.make(args.env, **dataset)
    T = int(traj.data.trim().qpos.shape[0])
    print(f"Loaded trajectory: {T} frames @ {traj.info.frequency} Hz "
          f"({T / float(traj.info.frequency):.2f} s)")

    env.play_trajectory(
        n_episodes=args.n_episodes,
        n_steps_per_episode=T,
        render=True,
    )


if __name__ == "__main__":
    main()
