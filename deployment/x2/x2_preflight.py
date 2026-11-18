"""Read-only check of the real X2 before a policy runs on it: commands nothing.

::

    source ~/x2_aimdk/env.sh
    python3 deployment/x2/x2_preflight.py --bundle <dir>

Twice a second it prints what ``deploy_x2.py`` would refuse on (the joint groups' status and
error codes, the joint furthest past its range) and the two inputs a balance policy leans on
most, computed by the controller's own code: the projected gravity and the pelvis angular
velocity. Tilt the hanging robot by hand to check their signs against the model's frame
(x forward, y left, z up). Gravity points to the side that is lowered: tipping it forward
makes x positive, tipping it onto its left side makes y positive. Turning it to its left, seen
from above, makes the yaw rate (the last one) positive while it turns.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]
from lmj_aimdk import (  # noqa: E402
    BODY_GROUPS,
    ROBOT_DOMAIN,
    RobotState,
    SafetyMonitor,
    StateAdapter,
    init_ros,
    shutdown_ros,
)
from lmj_deploy import ObservationBuilder, PolicySpec, training_model_xml  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", required=True, help="bundle dir, for the joint list and model")
    ap.add_argument("--domain", type=int, default=ROBOT_DOMAIN)
    ap.add_argument("--seconds", type=float, default=0.0, help="stop after this long (0 = Ctrl-C)")
    args = ap.parse_args()

    spec = PolicySpec.load(args.bundle)
    model = mujoco.MjModel.from_xml_path(str(training_model_xml(spec)))
    data = mujoco.MjData(model)
    builder = ObservationBuilder(spec, model)
    adapter = StateAdapter(builder)
    safety = SafetyMonitor(model, list(spec.joint_names))
    node = init_ros(args.domain, "lmj_x2_preflight", network=True)
    state = RobotState(node, spec)
    try:
        t0 = time.perf_counter()
        while not state.complete:
            state.poll()
            if time.perf_counter() - t0 > 10.0:
                raise SystemExit("no joint state or IMU on this domain")
            time.sleep(0.01)
        groups = state._groups  # noqa: SLF001 -- the raw messages, for their status fields
        for g in BODY_GROUPS:
            msg = groups[g].msg
            print(f"[preflight] {g:5s} group state {msg.state.value}, joint error codes "
                  f"{[j.error_code for j in msg.joints]}")
        print("[preflight] gravity in the pelvis frame (x fwd, y left, z up), pelvis rate, "
              "and the deploy check; Ctrl-C ends")
        end = time.perf_counter() + args.seconds if args.seconds else float("inf")
        while time.perf_counter() < end:
            time.sleep(0.5)
            state.poll()
            adapter.write(state, data)
            gravity, ang_vel = _gravity(data, builder)
            past = np.maximum(safety.lower - state.q, state.q - safety.upper)
            worst = int(past.argmax())
            fault = safety.check(state)
            print(f"[preflight] gravity {np.round(gravity, 2)}  rate {np.round(ang_vel, 2)} rad/s  "
                  f"| {spec.joint_names[worst]} {past[worst]:+.2f} rad past range | "
                  f"state age {state.age * 1e3:.0f} ms | deploy would "
                  + (f"REFUSE: {fault}" if fault else "start"))
    except KeyboardInterrupt:
        pass
    finally:
        shutdown_ros(node)


def _gravity(data: mujoco.MjData, builder: ObservationBuilder):
    """Projected gravity and root angular velocity, as ``ObservationBuilder.state`` makes them."""
    qpos, qvel = np.asarray(data.qpos), np.asarray(data.qvel)
    quat = qpos[builder.root_qpos_adr + 3: builder.root_qpos_adr + 7]
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, quat / np.linalg.norm(quat))
    gravity = rot.reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])
    return gravity, qvel[builder.root_dof_adr + 3: builder.root_dof_adr + 6]


if __name__ == "__main__":
    main()
