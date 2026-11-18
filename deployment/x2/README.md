# AgiBot X2 Ultra

Train, export and validate a policy as in the [main README](../../README.md); this page
takes the bundle to the robot. [../README.md](../README.md) has the steps both robots share, and
[../REFERENCE.md](../REFERENCE.md) the bundle format and the observation layout.

`X2UltraReal` is the AgiBot X2 Ultra whose nameplate reads "X2 Ultra -N".

```bash
python training/x2/train.py
```

Export, validation and `rollout_mujoco.py` work as in the main README's quickstart. The X2 is driven over
the ROS 2 topics of AgiBot's AimDK, so everything below needs ROS 2 Humble and `aimdk_msgs`.

## Files

| file | what it does |
|---|---|
| `deploy_x2.py` | the controller: runs the policy against the simulated robot, or the real one with `--robot` |
| `sim2sim_bridge.py` | the simulated robot: MuJoCo answering on the X2's ROS 2 topics as the robot would. AgiBot's own simulator can take its place |
| `x2_preflight.py` | read-only check of the robot before a run: status, gravity, joint ranges |
| `x2_joint_test.py` | swings a few joints of the hanging robot, to prove the connection |
| `live_viser.py` | live browser view of the robot and the reference motion |
| `robot_clips.py` | builds bundles whose clips start and end standing |
| `lmj_aimdk.py` | the library they share: ROS 2 messages in and out, mode switch, keyboard and gamepad |
| `soc1/` | running the controller on the robot's own computer |
| `X2_ROBOT_GUIDE.md` | the robot itself: network, modes, safety procedure, measurements |

## Check the bundle

Run the bundle in plain MuJoCo first, policy and physics in one process, in the browser with
a clip picker. The model is the training one, so a failure here is a bug in the export, not
a sim-to-real gap. This needs no ROS.

```bash
python deployment/rollout_viser.py --bundle deploy_bundles/<name>
```

## Sim2sim

`sim2sim_bridge.py` is MuJoCo standing
in for the robot: it publishes joint states and both IMUs on the X2's own topics and applies
the joint commands it receives. `deploy_x2.py` is the controller that later runs on the
robot, unchanged. They are separate processes, so the controller sees state that is already
old and a late command leaves the previous targets in force.

```bash
python deployment/x2/sim2sim_bridge.py --bundle deploy_bundles/<name> --viewer
```

```bash
python deployment/x2/deploy_x2.py --bundle deploy_bundles/<name> --motion <clip> --viser
```

The robot starts hanging from an elastic band, as on a crane, and a start goes as on the
robot: the controller ramps to the clip's first pose and holds it, you lower the robot onto
its feet, start the policy, and release the band.

| key, in the bridge's viewer | |
|---|---|
| **9** / **8** | lower / raise the robot on its band |
| **6** | square on the simulated gamepad: start the policy |
| **7** | release the band; again: attach it above the robot |
| double-click a body, then Ctrl + right-drag | push it |

The viewer draws the reference frame being tracked as a green ghost, so a growing gap
between ghost and robot is tracking drift. `--viser` adds a live browser view from the
controller's side, the one you also get on the robot. Both processes stay on ROS domain 1
and on this machine, so they cannot reach a robot on the same network. Headless, use
`--release-seconds` on the bridge and `--auto-start` on the controller.

**A rehearsal of the robot start-up.** `deploy_x2.py --robot --domain <n>` runs the full
robot sequence, mode switch included, against the bridge.

### Against AgiBot's own simulator

AgiBot ships a simulator of the X2, `x2_rl_deploy_mujoco`, in the "X2 RL Deploy extras" on the
[AimDK site](https://x2-aimdk.agibot.com/en/latest/). It stands in for the robot on the same
topics, with AgiBot's model instead of ours, so it tests what the bridge above cannot: whether
the policy survives a model it was not trained on.

| | AgiBot's simulator | our training model |
|---|---|---|
| joint friction | 0.3 Nm on every joint | none |
| collision | meshes | capsules and spheres |
| physics step | 1 ms | 5 ms |
| total mass | 43.5 kg | 44.8 kg |
| joint ranges | differ from ours by up to 0.26 rad | |
| state rate | about 950 Hz, IMU 485 Hz | 200 Hz from our bridge |

It needs Ubuntu 22.04 with ROS 2 Humble, as on the laptop used for the robot, plus
`ros-humble-grid-map-msgs`, `ros-humble-grid-map-ros`, `ros-humble-rmw-cyclonedds-cpp`,
`libgflags-dev` and `libyaml-cpp-dev`. Build `aimdk_msgs` from the extras with `colcon build
--packages-select aimdk_msgs` and source it. For the controller, install `mujoco` and
`numpy==1.26.4`: the 1.21 that comes with Humble cannot read a bundle's motion file.

The simulator has no band, and the robot stays up for less than a second on its own. So the
controller is started first and waits, and `--stand-first` makes it ramp to the default
standing pose and start the policy standing still, as after `s`:

```bash
export ROS_DOMAIN_ID=1 ROS_LOCALHOST_ONLY=1 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
python3 deployment/x2/deploy_x2.py --bundle deploy_bundles/<name> --motion <clip> --stand-first --wait-seconds 120 --auto-start --ramp-seconds 0.3 --hold-seconds 0
```

Then, in a second terminal, the simulator. Choose robot 0 when it asks:

```bash
export ROS_DOMAIN_ID=1 ROS_LOCALHOST_ONLY=1
cd x2_rl_deploy/x2_rl_deploy_mujoco/bin && ./start_sim.sh -s
```

Enter in the controller's terminal plays the clip; `--stand-first 4` plays it after four
seconds without a key. Set the two variables in both terminals: the simulator's default is
ROS domain 0 on the network, which is where a real robot is. To reset the robot, restart the
simulator.

What we have seen: a SuperMimic policy trained on BONES-SEED stood, then tracked a 43 s
routine of dance moves to its end without falling and held the last frame, at 50 Hz with
2.5 ms of compute and state about 3 ms old. A PPO policy trained for only 80M steps on the
single clip `stepinplace1` also tracked its 10 s clip. Those are two policies and two clips,
not a benchmark. The simulator leaves the message counter at zero, so the controller's
"states per control step" reads 0.

### Topics and requirements

| topic | type | |
|---|---|---|
| `/aima/hal/joint/{leg,waist,arm,head}/state` | `aimdk_msgs/JointStateArray` | bridge publishes, every physics step (200 Hz) |
| `/aima/hal/joint/{leg,waist,arm,head}/command` | `aimdk_msgs/JointCommandArray` | controller publishes, 50 Hz |
| `/aima/hal/imu/torso/state` | `sensor_msgs/Imu` | the pelvis IMU, which the controller reads |
| `/aima/hal/imu/chest/state` | `sensor_msgs/Imu` | the IMU on `torso_link` |

Both need ROS 2 Humble and `aimdk_msgs`, built from the "X2 RL Deploy extras" of the AimDK
and sourced. On macOS, RoboStack provides Humble as conda packages; it needs `empy` 3.3.4 to
generate messages. Both default to ROS domain 1 and keep traffic on this machine; `--network`
lets it out.

The robot sends state at 500 Hz (head 334 Hz) where the bridge sends 200.

## On the robot

> [!WARNING]
> Running a learned policy on a real robot is hazardous. A policy can fail without notice,
> and the robot can fall, hit people or objects, or damage itself. Everything in this
> repository is research code, provided as is and without warranty (see [LICENSE](../../LICENSE)):
> you deploy at your own risk and are responsible for a safe setup. Hang the robot from a
> crane or gantry, keep people out of its reach, keep the emergency stop in hand, and run
> every policy in sim2sim first.

`deploy_x2.py --robot` runs the policy from a laptop over the robot's Ethernet port, or on
the robot's own computer (SoC1) with the laptop only operating it over Wi-Fi. Start with the
[X2 robot guide](X2_ROBOT_GUIDE.md), which covers the network, the
robot's modes, the safety procedure and what we measured, then
[soc1/README.md](soc1/README.md) for the cable-free setup.

The robot's computers are at 10.0.1.40 to 10.0.1.42 on its rear Ethernet port, on ROS domain 0
with Fast DDS, so this machine needs an address on 10.0.1.0/24 and
`RMW_IMPLEMENTATION=rmw_fastrtps_cpp`. Build `aimdk_msgs` from the SDK, whose version must
match the firmware (we use v1.1.0, the latest; every type we use matched the robot); the RL
deploy extras' subset lacks the mode services. ROS Humble's Python
bindings need numpy 1.x, so install mujoco next to them with `numpy==1.26.4`; older 1.x
releases cannot read a bundle's motion file.

AgiBot's motion control owns the joint command topics until the robot is migrated to
`Develop_MC` (`MigrateSystemState`; about 36 s, status 2 meanwhile). `deploy_x2.py --robot`
does that after checking the first state, waits until nothing else publishes joint commands,
and only then creates its own publishers. The robot is left in `Develop_MC`, damped; a
reboot or a migration to `Ready` brings AgiBot's control back. In the order we first ran it,
hanging on the gantry:

```bash
python deployment/x2/x2_preflight.py --bundle <dir>    # read-only: status fields, gravity, ranges
python deployment/x2/live_viser.py --bundle <dir>      # read-only: move joints, tilt it, compare
python deployment/x2/x2_joint_test.py --joints left_elbow_joint right_elbow_joint
python deployment/x2/deploy_x2.py --robot --bundle <dir> --hold-only   # PD to the first pose, no policy
python deployment/x2/deploy_x2.py --robot --bundle <dir> --motion <clip> --viser
```

A run goes: state checks, `Develop_MC`, a ramp to the clip's first pose (PD, at most 0.5
rad/s), a hold, then the policy balancing on that first frame held still, so the crane can go
slack, then the clip. Keys act as they are pressed, in the controller's terminal or on a USB
gamepad (a Logitech F710 with its switch on X; the robot itself publishes no `/joy`):

| keyboard | gamepad | |
|---|---|---|
| Enter | A | end the hold: the policy starts on the first frame; again: play the clip |
| s | B | the safety layer: the policy keeps running but tracks the `init_state` pose, standing still; Enter / A replays the clip |
| n / p | RB / LB | next / previous clip in the bundle's `calm_windows.txt` |
| Ctrl-C | LT + B | damp, which ends the run and lets the robot fall; no letter key damps |

After a clip, `--end hold` (the default) keeps the policy on its last frame, `next` plays the
next clip, `loop` replays it and `damp` damps. Every clip must start and end standing, both
feet down and still, so that none of these ever asks for a pose in mid-stride; see
[Robot bundles](#robot-bundles-clips-that-start-and-end-standing). A status line each second
gives the control rate, compute time, how late ticks ran, the state's age and how many
states arrived per step (about 10 from the robot at 500 Hz).

`deploy_x2.py` caps numpy's OpenBLAS at four threads before numpy loads. Uncapped, it spins
one thread per core, and any other load, a browser included, held up single steps by 25 to
70 ms on a 32-core machine; capped, the worst step is about 3 ms. `--viser` draws in a child
process for the same reason.

`--robot --domain <n>` rehearses the same sequence against the bridge on another domain, with
something serving the two mode services. Measured on the hanging robot with
`x2_joint_test.py`, a 0.15 rad sine at a 4 s period: the elbows track 85% of it at half
AgiBot's gains, the knees 77% at kp 80, and the knee needs about 3.5 Nm almost regardless of
gain, which reads as friction. The training model has no `frictionloss`.

Without the cable, `deploy_x2.py` runs on the robot's SoC1 (Jetson Orin NX) in a rootless
podman container, detached, at real-time priority, reading keys from a pipe
(`--keys-from`) and expecting the operator's heartbeat (`--heartbeat`): if it stops, the
policy tracks the default pose and only damp gets through. The laptop joins the robot's
hotspot and starts, operates and stops it over SSH, keyboard and gamepad included; Wi-Fi is
never in the control loop. See [soc1/README.md](soc1/README.md).

`deploy_x2.py` commands nothing if the first state fails its checks, and afterwards damps by
itself on a stale state (100 ms), a joint past its range or too fast, a pelvis tilt past
about 60 degrees, and a non-finite reading or action. The
[guide](X2_ROBOT_GUIDE.md#72-on-the-laptop-in-order) lists them.

## Robot bundles: clips that start and end standing

On the robot, `deploy_x2.py` switches clips (n / p, LB / RB), plays them in turn
(`--end next`) and holds the last frame of each. None of that may ask for a pose in
mid-stride, so every clip it steps through starts and ends standing. LAFAN never stands still
in the middle of a walk, so walks from standing to standing come from BONES-SEED, whose
chunks (`chunk_*.npz`, with a `chunk_*.csv` of `name,first_frame,frames`) are Trajectories
in the same format. `robot_clips.py` finds such windows, checks them and builds the bundle,
with numpy only:

```bash
# windows by kind (stand, fwd, back), calmest first, as <clip>:<start>:<steps>
python deployment/x2/robot_clips.py windows <bones_chunks> --match "walk_(forward|backward)_(start|stop)"
python deployment/x2/robot_clips.py windows deploy_bundles/<name> --kind stand
# do they start and end standing (every clip, without windows), and if not, what fails
python deployment/x2/robot_clips.py check deploy_bundles/<name> [<clip>:<start>:<steps> ...]
```

Standing means the root, the joints and both feet still, the feet level, the pelvis high and
upright, and the arms down, for 0.2 s either side; the script's docstring has the
thresholds. The arms rule out LAFAN's opening T-pose, which passes everything else.

`build` crops `<name>=<source>:<clip>` from the end of the clip's first standing run to the
start of its last, keeping 10 frames of standing (`--margin`) at each end, and takes
`<name>=<source>:<clip>:<start>:<steps>` as given, which is how `windows` prints it. It writes
`motions.lmj.npz` and `calm_windows.txt`, the playlist `deploy_x2.py` steps through, and copies
`policy.json` and `policy.npz` from `--policy-from`. A clip that does not start and end
standing stays in the npz, reachable with `--motion` only, and is reported. This rebuilds
the X2 walk bundles, array for array; `stand_a` is a T-pose and stays out of the playlist:

```bash
B=deploy_bundles/<name>_robot   # the policy, and the clips stand_a and stand_b come from
C=<bones_chunks>
python deployment/x2/robot_clips.py build --out deploy_bundles/<name>_walk --policy-from $B \
    stand_a=$B:dance2_subject4:41:286 \
    fwd_A042=$C:220728/walk_forward_start_001__A042 \
    back_A039=$C:220727/walk_backward_start_001__A039 \
    fwd_A041=$C:220728/walk_forward_stop_003__A041 \
    back_A041=$C:220728/walk_backward_stop_003__A041 \
    stand_b=$B:sprint1_subject4:2829:251
```

A chunk is several GB, and loading one of its arrays whole can exhaust a machine's
memory. Chunks are stored uncompressed, so the script memory-maps them and reads only the
frames of the clips it uses: the build above peaks at 80 MB, and `windows` over BONES-SEED's
1319 forward and backward walks takes 18 s and peaks under 1 GB. Scanning many chunks still reads a lot from
disk.

## Where it stands

- **Retargeting works.** The 19 default clips and the 40 LAFAN1 clips retarget from the
  `UnitreeG1` source. `X2UltraReal.yaml` stiffens the joint limits of the retargeting model,
  without which the wrists end up as much as 2.8 rad past their ranges.
- **Training works in simulation.** SuperMimic with LocoMuJoCo's stock-robot observations and
  tracking reward completes 32 to 36 of the 40 LAFAN1 clips after 36M steps.
- **The actuator parameters are vendor example values.** The servo gains and the standing pose
  come from AgiBot's `x2_rl_deploy` example and the armature from its MJCF. None has been
  identified on the robot.
- **The servos have no speed limit.** The real joints stop at 12 to 16 rad/s. In simulation a
  saturated hip-yaw or wrist-yaw servo passes 100 rad/s within two control steps, and a few
  such worlds diverge numerically during training.
- **It has run on one robot.** `deploy_x2.py` has driven our X2 Ultra over the cable and from
  SoC1, with AimDK SDK v1.1.0; the [guide](X2_ROBOT_GUIDE.md) has what was measured. Other
  firmware versions are untested.
