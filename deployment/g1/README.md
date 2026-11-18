# Unitree G1

Train, export and validate a policy as in the [main README](../../README.md); this page
takes the bundle to the robot. [../README.md](../README.md) has the steps both robots share, and
[../REFERENCE.md](../REFERENCE.md) the bundle format and the observation layout.

`G1Real` is the 29-DoF G1 with zero joint damping and friction.

```bash
python training/g1/verify_env.py walk
```

```bash
python training/g1/train.py
```

## Files

| file | what it does |
|---|---|
| `deploy_g1.py` | the controller: runs the policy against the simulated robot, or the real one with `--domain 0` |
| `sim2sim_bridge.py` | the simulated robot: MuJoCo answering over the Unitree SDK as the G1 would |
| `lmj_dds.py` | the library both share: Unitree SDK messages in and out |

## Setup: the Unitree SDK

Both scripts below need it. The SDK needs Python 3.10 or older for a prebuilt `cyclonedds==0.10.2`:

```bash
pip install cyclonedds==0.10.2 mujoco
git clone https://github.com/unitreerobotics/unitree_sdk2_python
pip install -e unitree_sdk2_python
```

Install it editable. On Linux, `CRC` loads a shared library from `utils/lib`, which a plain
`pip install` of the repository leaves out. macOS has no such library and falls back to
pure Python, at about 3 ms per message.

## Check the bundle

Run the bundle in plain MuJoCo first, policy and physics in one process. The model is the
training one, so a failure here is a bug in the export, not a sim-to-real gap.

```bash
python deployment/rollout_mujoco.py --bundle deploy_bundles/<name> --viewer
```

## Sim2sim

The policy and the physics run as separate
processes that talk over the SDK, so the controller sees state that is already old and a late
command leaves the previous targets in force, as on the robot. The bridge simulates the
robot, hanging from an elastic band, and the controller is started in a second terminal:

```bash
python deployment/g1/sim2sim_bridge.py --bundle deploy_bundles/<name> --viewer
```

```bash
python deployment/g1/deploy_g1.py --bundle deploy_bundles/<name> --motion <clip>
```

A start then goes as on the robot. The robot hangs `--lift` (0.1 m) above its `init_state`
keyframe.

1. The controller ramps to the clip's first pose and holds it.
2. Lower the robot onto its feet with **9** (each press lengthens the band by 5 cm; **8**
   shortens it).
3. Press **6**, START on the simulated remote, to run the policy.
4. Press **7** to release the band. **7** again attaches it above where the robot is.

While the policy runs, the viewer draws the reference frame it is tracking as a green ghost.
The ghost starts on the robot, turned by the controller's heading alignment, so any gap
that grows between them is tracking drift. The controller sends the frame on
`rt/lmj/reference`. Nothing on the robot reads that topic. The viewer renders from its own
thread, so a slow frame never delays `LowState`.

Keys 0 to 5 toggle MuJoCo's geom groups, so the band uses 6 to 9. On macOS, start
the bridge's viewer with `mjpython`. Headless, release the band on a timer and start without the remote:

```bash
python deployment/g1/sim2sim_bridge.py --bundle deploy_bundles/<name> --release-seconds 5
python deployment/g1/deploy_g1.py --bundle deploy_bundles/<name> --motion <clip> --auto-start
```

Both default to domain 1 on `lo0`; pass `--interface lo` on Linux. `--start-yaw` starts the
robot on an arbitrary heading.

## On the robot

> [!WARNING]
> Running a learned policy on a real robot is hazardous. A policy can fail without notice,
> and the robot can fall, hit people or objects, or damage itself. Everything in this
> repository is research code, provided as is and without warranty (see [LICENSE](../../LICENSE)):
> you deploy at your own risk and are responsible for a safe setup. Hang the robot from a
> crane or gantry, keep people out of its reach, keep the emergency stop in hand, and run
> every policy in sim2sim first.

The same controller drives the robot from a laptop on its Ethernet port. The policy starts when START is pressed on the remote, and SELECT damps. Hang the robot
first: damping lets it fall.

```bash
python deployment/g1/deploy_g1.py --bundle deploy_bundles/<name> --motion <clip> --domain 0 --interface <ethernet interface>
```

There is no example of running the controller on the G1's own computer yet.

## Safety checks

`deploy_g1.py` commands nothing if the first `LowState` fails its checks. After that it damps
on any of these, in every state:

| trigger | limit |
|---|---|
| SELECT on the remote | |
| stale `LowState` | 100 ms |
| measured joint past its range | 0.1 rad beyond the model's range |
| joint speed | 25 rad/s |
| pelvis tilt | about 60 degrees |
| bad IMU quaternion, or non-finite sensor reading or action | |
| Ctrl-C or an exception | |

The ramp to the first pose clamps it to the joint ranges and runs at 0.5 rad/s at most, so
it can take longer than `--ramp-seconds`. Damping lets the robot fall, so hang it first.

## Design notes

Three decisions worth knowing:

* **There is no DDS observation builder.** `LowStateAdapter` writes `LowState` into an
  `MjData` and runs `mj_kinematics`, and `ObservationBuilder` carries on unchanged. All
  rollouts share one observation path.
* **`LowState` is polled, not subscribed to with a handler.** A handler deserializes every
  message in Python and takes the GIL from the control loop.
* **Yaw is relative.** `ObservationBuilder.align_heading` rotates the reference onto the
  robot's heading when the policy starts, so IMU yaw only has to hold for one clip.
  Only bundles with `use_anchor_pos=False` run, since the robot reports no world position.

Python may not sustain the robot's 500 Hz of `LowState`, so the bridge defaults to
`--publish-hz 200`, once per physics step. The bridge's accelerometer output is zero.

## Where it stands

* **The G1 has had little time on hardware.** `deploy_g1.py --domain 0` has been tried on
  the robot from a laptop over Ethernet, far less than the X2's path, and there is no onboard
  example. Its safety layer was developed against the bridge, with faults injected. A lost
  remote cannot be detected, since it reads as no button pressed. Hang the robot first.
