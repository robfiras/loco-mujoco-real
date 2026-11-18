# Running a policy on the AgiBot X2 Ultra

How we connect a laptop to the lab's X2 Ultra, take over its joints and run a learned
whole-body policy on it. This covers deployment only, not training. Everything here was done
and measured on our robot in November 2026, with AimDK SDK v1.1.0.

If you bring your own policy, read sections 1 to 6, then 8. Section 7 is our own tool chain,
which you can reuse if your policy fits it.

---

## 1. The robot in one paragraph

The X2 Ultra has 31 actuated joints: 29 body joints (12 leg, 3 waist, 14 arm) and 2 head
joints. AgiBot's own motion control (MC) normally drives them. For our own controller we
switch the robot into a developer mode (`Develop_MC`), in which MC stops. From then on, our
laptop sends joint targets with stiffness and damping over ROS 2, and the robot's joint
drives close the PD loop themselves. **Up to section 8, the policy runs on the laptop, not
on the robot:** the robot streams joint states and IMU readings to the laptop at 500 Hz, and
the laptop sends commands back at 50 Hz, over an Ethernet cable. Section 9 moves the policy
onto the robot and drops the cable.

| computer | board | address | role |
|---|---|---|---|
| PC1 | RK3588 | 10.0.1.40 | motion control, joint drives (HAL). **Never build or run your own code here.** |
| PC2 | Jetson Orin NX 16 GB (8× A78AE, Ampere GPU) | 10.0.1.41 | the developer computer: cameras, LiDAR, SLAM, perception. SSH login only from AgiBot support. |
| PC3 | RK3588s | 10.0.1.42 | interaction: voice, TTS |

---

## 2. Hardware and network

**What you need:** the robot on a gantry or crane, the emergency stop (kill switch), the
Agibot Go app on a phone, a Linux laptop (we use Ubuntu 22.04 with ROS 2 Humble), an Ethernet
cable, and optionally a USB gamepad (Logitech F710).

**Cable:** plug into the robot's rear SDK Ethernet port (1000BASE-T). The robot's computers
are on `10.0.1.0/24`, so the laptop needs an address there. AgiBot's docs suggest
`10.0.1.2`; we use `10.0.1.200`. Avoid .40, .41 and .42.

```bash
sudo ip addr add 10.0.1.200/24 dev <your ethernet interface>
```

That address is lost whenever the link goes down: a robot reboot, replugging, a laptop
reboot. **This happened to us several times.** If the laptop uses a NetworkManager profile
for that port, make the address permanent instead:

```bash
nmcli connection modify <profile> +ipv4.addresses 10.0.1.200/24 && nmcli connection up <profile>
```

Check: `ping 10.0.1.40` must answer. If ROS sees nothing, check the address first.

**Wi-Fi:** the robot can open its own hotspot (switch it on in the Agibot Go app). Don't use
it for control. AgiBot recommends the cable for development and Wi-Fi only for debugging. Our
controller damps when the robot's state is more than 100 ms old, so a Wi-Fi hiccup drops the
robot. Wi-Fi is fine for read-only monitoring. Going cable-free properly means running the
controller on PC2 (SSH needed); see section 9.

---

## 3. Software on the laptop

### 3.1 ROS 2 and the AimDK messages

- **ROS 2 Humble**, with Fast DDS (`rmw_fastrtps_cpp`), on **domain 0**. The robot uses
  domain 0, and the robot's DDS vendor is eProsima Fast DDS.
- **`aimdk_msgs`**, built from AgiBot's SDK. **The SDK version must match the robot's
  firmware**; our robot works with **v1.1.0** (`aimdk-sdk-v1_1_0-...zip`, 283 MB, from the
  download toolbar at the bottom right of <https://x2-aimdk.agibot.com/en/latest/>). Build
  only the message package:

  ```bash
  unzip aimdk-sdk-v1_1_0-aimdk-aarch64-faf4c945-artifacts.zip -d ~/x2_aimdk/sdk
  cd ~/x2_aimdk/sdk/aimdk-aarch64-*/
  source /opt/ros/humble/setup.bash
  colcon build --packages-select aimdk_msgs     # about 40 s; builds for x86 despite the name
  ```

  The smaller `aimdk_msgs` inside AgiBot's "X2 RL Deploy extras" lacks the mode services in
  section 4, so use the full SDK.

### 3.2 Python packages

ROS Humble's Python bindings only work with **numpy 1.x**, while recent MuJoCo and JAX wheels
pull in numpy 2. We install MuJoCo (the version the policy was trained with, 3.16 for us),
`numpy==1.26.4` and `viser` into a private folder, and put it on `PYTHONPATH`. Ubuntu's system
Python has no `venv` module without sudo, hence `--target`:

```bash
pip3 install --no-cache-dir --target ~/x2_aimdk/pydeps "mujoco==3.16.0" "numpy==1.26.4" viser
```

Use numpy 1.26, not an older 1.x: a bundle's motion file is written by numpy 2, and 1.26 is
the first 1.x release that can read it.

### 3.3 One environment script

`~/x2_aimdk/env.sh`:

```bash
source /opt/ros/humble/setup.bash
source ~/x2_aimdk/sdk/aimdk-aarch64-faf4c945-artifacts/install/local_setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0
export PYTHONPATH=$HOME/x2_aimdk/pydeps${PYTHONPATH:+:$PYTHONPATH}
```

`source ~/x2_aimdk/env.sh` in every terminal that talks to the robot. A quick check:
`ros2 topic list` should show about 270 topics, `/aima/hal/joint/leg/state` among them.

### 3.4 CPU threads

If your controller uses numpy, **cap OpenBLAS before numpy is imported**:

```python
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import numpy as np
```

Uncapped, OpenBLAS starts one spinning thread per core (32 on our laptop). Anything else that
needs a core, even a browser, then delays single policy steps by 25 to 70 ms. With 4
threads our 2048-wide network takes 1.7 ms per step and the worst step stays around 3 ms.
One thread is far slower (9 ms).

---

## 4. Robot modes and the app

### 4.1 System mode: who owns the joints

The robot has a system state machine. After power-on it goes to `Business`, normal operation,
in which AgiBot's MC publishes the joint commands. To command the joints yourself, migrate to
`Develop_MC`:

| service | request | response |
|---|---|---|
| `/aimdk_5Fmsgs/srv/GetSystemState` | `header` (`CommonRequest`) | `cur_state` (string), `curr_status.value` (1 = ready, 2 = migrating) |
| `/aimdk_5Fmsgs/srv/MigrateSystemState` | `header`, `state` = `"Develop_MC"` | `header.status.value` (1 = accepted) |

- **The migration is asynchronous.** The call only says it was accepted. Poll
  `GetSystemState` until `cur_state == "Develop_MC"` and the status reads 1. **On our robot
  that took about 36 s**, with the status reading 2 meanwhile.
- **Then wait until no other publisher is left** on `/aima/hal/joint/*/command` (count the
  publishers with `node.count_publishers(topic)`). Only then create yours.
- **The way back** is a reboot, or migrating to `Ready`, which continues to `Business`.
- **ROS 2 Humble and Fast DDS can lose a service reply** if you call right after creating the
  client. Create the clients early. Retry queries if needed, but **never resend a
  migration**: poll the state instead.

### 4.2 Motion mode (MC action)

Separately, MC has motion modes, readable with `/aimdk_5Fmsgs/srv/GetMcAction` and set from
the Agibot Go app or the remote: `PASSIVE_DEFAULT` (limp), `DAMPING_DEFAULT` (resists
motion), `JOINT_DEFAULT` (position-controlled standing) and so on. After the emergency stop
is released, the robot comes back in `DAMPING_DEFAULT`.

### 4.3 Things the app does

- **Language:** we switched the robot's voice from Chinese to English in the Agibot Go app.
  The robot also exposes an undocumented `SetRobotLanguage` service.
- **Motion mode changes,** the **hotspot**, and robot information.

### 4.4 Emergency stop

We tested it: it kills the joints, and after a reset the robot recovers to `Business` /
`DAMPING_DEFAULT` with all data streams back. Keep it in hand for every motion test.

---

## 5. The ROS 2 interface (hardware layer)

All of these stream in every mode. Commands only take effect, without MC overriding them, in
`Develop_MC`.

| topic | type | direction | rate on our robot |
|---|---|---|---|
| `/aima/hal/joint/{leg,waist,arm,head}/state` | `aimdk_msgs/JointStateArray` | robot → you | 500 Hz (head 334 Hz) |
| `/aima/hal/joint/{leg,waist,arm,head}/command` | `aimdk_msgs/JointCommandArray` | you → robot | we send 50 Hz; AgiBot's example 500 Hz |
| `/aima/hal/imu/torso/state` | `sensor_msgs/Imu` | robot → you | 500 Hz. On the **pelvis** (frame_id `base_link`) |
| `/aima/hal/imu/chest/state` | `sensor_msgs/Imu` | robot → you | 500 Hz. On `torso_link`, above the waist |

**QoS:** the robot publishes best-effort, transient-local. Subscribe and publish
**best-effort, volatile, keep-last depth 1**.

**Message fields:**
- `JointCommand`: `name`, `position`, `velocity`, `effort`, `stiffness`, `damping`
- `JointCommandArray`: `header`, `joints[]`
- `JointState`: `name`, `position`, `velocity`, `effort`, `error_code`
- `JointStateArray`: `header`, `state` (`DomainErrorState`), `joints[]`

In normal operation `state.value` and every `error_code` read 0; treat anything else as a
fault. Velocities are rad/s and positions rad. `effort` in the state reads as torque in Nm
and appears quantized to about 0.09 Nm.

**What a command does:** the drive applies `effort + stiffness·(position − q) + damping·(velocity − dq)`
and **keeps the last command forever**: there is no timeout on the robot side. If your
program dies, the last targets stay in force. So the last thing you send should always be a
damping command: stiffness 0, damping > 0. Send each group as a full array in the order below,
with every joint named.

**Joint order per group** (this is also the order of our policy's 29 actions; head excluded):

| group | joints |
|---|---|
| leg (12) | `left_{hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll}_joint`, then `right_…` the same |
| waist (3) | `waist_yaw_joint`, `waist_pitch_joint`, `waist_roll_joint` |
| arm (14) | `left_{shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_yaw, wrist_pitch, wrist_roll}_joint`, then `right_…` |
| head (2) | `head_yaw_joint`, `head_pitch_joint` (we hold them at 0 with kp 20, kd 1) |

**IMU conventions, checked by tilting the hanging robot:** the orientation quaternion (ROS
`x, y, z, w`) gives gravity in the pelvis frame as roughly `[0, 0, −1]` upright. Tipping the
robot forward makes x positive, tipping it onto its left side makes y positive, and the
accelerometer agrees with the orientation. The pelvis IMU and our MuJoCo model's pelvis frame
agree, so the IMU orientation can go straight into a floating-base model. There is **no
world position**: a policy must not need the root's world position. Yaw drifts slowly, so use
it relative.

**Joint ranges:** AgiBot's "Joint Motion Range (Warranted)" docs page, or the robot's URDF.
Hanging limp, our robot's waist pitch reads +0.35 rad, past the URDF's ±0.314 rad, and the
left shoulder roll sits about 0.06 rad past its range. Allow a margin.

---

## 6. The safety procedure we follow

Always in this order, the first time with a new policy or motion:

1. **The robot hangs** from the gantry, feet clear of the floor or just touching. Emergency
   stop in hand. Nothing near the head or arms.
2. **Read-only checks:**
   - all status fields 0
   - gravity about `[0.16, 0, −0.99]` when hanging (it hangs about 9° forward)
   - a live 3D view that moves with the robot when you move joints by hand or tilt it
3. **Prove the link:** swing a couple of joints a little with gentle gains (section 7.2).
4. **Ramp and hold, no policy:** ramp all joints to the policy's first pose (at most 0.5 rad/s)
   and hold with the policy's PD gains. Any joint moving the wrong way shows up here, slowly.
5. **Policy, hanging.**
6. **Policy, feet on the floor, crane slack:** start the policy balancing on a still standing
   pose, then slacken the crane, then play the motion.
7. **Motions that start and end standing.** Every motion clip must begin and end with both
   feet down and still. Then holding the last frame, switching clips or playing the next
   never asks for a mid-stride pose.

**When something looks wrong:** first go to the **safety layer**, where the policy keeps
balancing but tracks a fixed standing pose. Only **damp** (zero stiffness, the robot goes
limp) if it is falling, because damping drops it into the crane.

---

## 7. Our tool chain (`deployment/`)

Our policies are SuperMimic motion trackers trained in LocoMuJoCo. They're exported to a
**bundle**, a folder with `policy.json` (joint names, gains, action scale, observation
layout), `policy.npz` (weights) and `motions.lmj.npz` (reference clips). The robot side needs
**numpy + MuJoCo (kinematics only) + ROS 2**: no JAX, no ONNX, no LocoMuJoCo.

### 7.1 Make and check a bundle (on the training machine)

```bash
python deployment/export_bundle.py --checkpoint <ckpt.pkl> --output <bundle> --motion-names <clip,clip>
python deployment/validate_bundle.py --bundle <bundle> --checkpoint <ckpt.pkl>
python deployment/x2/robot_clips.py windows <bundle or motions> --kind ...    # find windows that start and end standing
python deployment/x2/robot_clips.py check <bundle> <clip>:<start>:<steps> ...  # verify given windows
python deployment/x2/robot_clips.py build --out <dir> --policy-from <bundle> <name>=<source>:<clip>[:<start>:<steps>] ...
```

`build` writes a robot bundle (the policy plus cropped clips) and its `calm_windows.txt`, the
clip list the controller steps through.

`validate_bundle.py` must print PASS: the hand-written observation, action and control
target must reproduce the training environment term by term. Only bundles with
`use_anchor_pos=False` can run on the robot.

### 7.2 On the laptop, in order

```bash
source ~/x2_aimdk/env.sh && cd loco-mujoco-real

# read-only: status fields, gravity, joints past their range, "deploy would start / REFUSE"
python3 deployment/x2/x2_preflight.py --bundle <bundle> --seconds 5

# read-only live 3D view in the browser (http://localhost:8080): move joints, tilt, compare
python3 deployment/x2/live_viser.py --bundle <bundle>

# prove the link: switches to Develop_MC, swings the named joints ±0.15 rad, then damps them
python3 deployment/x2/x2_joint_test.py --joints left_elbow_joint right_elbow_joint

# ramp and hold the first pose with the policy's gains, no policy
python3 deployment/x2/deploy_x2.py --robot --bundle <bundle> --hold-only

# the policy
python3 deployment/x2/deploy_x2.py --robot --bundle <bundle> --motion <clip> --viser
```

**What `deploy_x2.py --robot` does:**
1. Refuses to start unless the first state passes its checks.
2. Migrates to `Develop_MC` and waits until nothing else commands the joints.
3. Ramps to the clip's first pose and holds it (PD).
4. **Enter or A:** the policy starts, balancing on the clip's first frame held still. Now
   slacken the crane.
5. **Enter or A again:** the clip plays, turned to the heading the robot has at that moment.
6. After the clip: holds its last frame (`--end hold`), plays the next clip
   (`--end next`), loops (`--end loop`) or damps (`--end damp`).

The robot is left in `Develop_MC`, damped.

**Keys,** acting as soon as they are pressed, in the controller's terminal or on the F710
gamepad:

| keyboard | gamepad | effect |
|---|---|---|
| Enter | A | start the policy / play the clip / replay from the safety layer |
| s | B | **safety layer:** the policy keeps running, the target becomes the default standing pose |
| n / p | RB / LB | next / previous clip in the bundle's `calm_windows.txt` |
| Ctrl-C | LT + B | **damp and quit** (the robot goes limp); no letter key damps |

**Gamepad:** a Logitech F710 with its back switch on **X** (XInput) and the **MODE** LED
**off**; with MODE on, the left stick and the D-pad swap. It's read straight from
`/dev/input/js0`, nothing goes over the network. It's wireless, so it's a convenience, not
a dead-man switch.

**Automatic damping,** in every state:
- stale state (> 100 ms)
- a joint more than 0.1 rad past its model range
- a joint faster than 25 rad/s
- pelvis tilt past about 60°
- a non-finite reading or action
- a bad IMU quaternion
- any exception

**Status line, once a second:** control rate, compute time, worst tick lateness, state age,
states per step, clip and frame. Healthy on our robot: 50 Hz, about 2 ms compute, state about
2 ms old, about 10 states per step.

### 7.3 Rehearse in simulation first

The same controller runs against a MuJoCo stand-in that speaks the robot's topics, on another
ROS domain, so it can't reach the robot:

```bash
python3 deployment/x2/sim2sim_bridge.py --bundle <bundle> --viewer             # terminal 1
python3 deployment/x2/deploy_x2.py --bundle <bundle> --motion <clip> --viser         # terminal 2
```

**Viewer keys** (bridge window):
- **9** / **8:** lower / raise the robot on its band
- **7:** release / re-attach the band
- **6:** gamepad square (start)
- **double-click a body, then Ctrl + right-drag:** push it

`--robot --domain <n>` rehearses the robot start-up against the bridge, but you need a small
fake node for the two mode services.

---

## 8. Deploying your own policy

What your controller has to do on this robot, whatever framework trained it:

1. **Inputs:** read `/aima/hal/joint/{leg,waist,arm}/state` and
   `/aima/hal/imu/torso/state` (section 5).
   - **Poll for the newest message** (`subscription.handle.take_message`) rather than using
     callbacks. At 500 Hz × 5 topics, Python callbacks steal the GIL from the control loop.
   - Map joints **by name**, never by position.
   - Projected gravity and base angular velocity come from the pelvis IMU. There's no base
     linear velocity and no world position: train without them, or estimate them.
2. **Outputs:** joint targets plus stiffness and damping per joint, published as the four
   `JointCommandArray` groups. Your training gains are the place to start. Ours (training
   ran the same values) are listed below. AgiBot's own RL example uses softer gains, for
   example knee 80 / 8.
3. **Rate:** we run 50 Hz. The drives hold the last command between your messages.
4. **Start-up:** check the first state, migrate to `Develop_MC`, wait for the command topics
   to be free, ramp to your first pose (we use at most 0.5 rad/s), hold, then start.
5. **Shutdown:** always end with stiffness 0 and damping > 0. Commands never time out.
6. **Heading:** the IMU's yaw is arbitrary, so rotate your reference or command to the robot's
   heading when the policy starts.
7. **Model gap:** our MJCF (`lmj_real/models/x2_ultra_real/`, from AgiBot's URDF v1.4.0, X2
   Ultra new version) matches the robot's kinematics. Measured differences are in section 10.

**Gains we use** (kp / kd, both sides the same):

| joint | kp / kd | joint | kp / kd |
|---|---|---|---|
| hip pitch | 120 / 5 | waist yaw | 40.2 / 2.56 |
| hip roll, hip yaw | 100 / 4 | waist pitch, waist roll | 200 / 2 |
| knee | 150 / 5 | shoulder pitch / roll / yaw, elbow | 50 / 3 |
| ankle pitch, ankle roll | 40 / 2 | wrist yaw / pitch / roll | 20 / 2 |

**Default standing pose** (the model's `init_state` keyframe, pelvis at 0.641 m):
- hip pitch −0.312, knee 0.669, ankle pitch −0.363
- shoulder pitch 0.2, shoulder roll ±0.2 (left +), elbow −0.3
- everything else 0

**Fastest path if your policy is a different network but needs the same inputs:** reuse
`lmj_aimdk.py` (state reader, command writer, mode switch, keyboard and gamepad, safety
monitor) and replace the observation builder and actor in a copy of `deploy_x2.py`.
`x2_joint_test.py` is a short, self-contained example of the raw sequence: mode switch,
take-over, commands, damping. It needs only rclpy and `aimdk_msgs`. AgiBot's
`x2_rl_deploy` (C++, ONNX runtime, in the "X2 RL Deploy extras") is an alternative
reference, but its README says it was only validated in simulation and its gains need
adapting for the real robot.

**Test it in the bridge (section 7.3) before the robot.** The bridge needs a bundle only for
the model and the motions; any of ours will do.

---

## 9. Without the cable

Cable-free, the controller runs on PC2 (Jetson Orin NX), so the 50 Hz loop never crosses
Wi-Fi. Wi-Fi only starts it, sends the operator's keys and shows its log. Setup and scripts:
[soc1/README.md](soc1/README.md).
- **Container:** `deploy_x2.py` runs in a rootless podman container on PC2, as AgiBot
  recommends (SDK v1.2.0 requires it). It is built offline from ROS Humble's arm64 image plus
  MuJoCo and numpy wheels, and started like AgiBot's own: host network, the robot's DDS
  profile. Nothing is installed on the robot.
- **Network:** the laptop joins the robot's hotspot with a second Wi-Fi adapter
  (`ipv4.never-default yes`), so its own Wi-Fi keeps the internet. PC1 serves the hotspot at
  192.168.88.88 and relays SSH to PC2: `ssh -J agi@192.168.88.88 agi@10.0.1.41`. The PC2
  login comes from AgiBot support.
- **A run:** `soc1/x2ctl.sh log` in one terminal, `soc1/x2ctl.sh start --robot --bundle
  bundles/<bundle> --motion <clip>` in another. The controller runs detached on PC2, so a
  dropped SSH session can't stop it. The operator sends the laptop's keys and its USB gamepad
  over SSH, with the same mapping as section 7.2. `x2ctl.sh stop` damps without the operator.
- **Heartbeat:** the operator sends one 10 times a second. After 0.5 s without one (Wi-Fi
  lost, operator closed), the policy tracks the default pose, as with B. Every key except damp
  is ignored until the link has been back for a second. There is no key to leave: a session
  ends with damp.
- **Real-time priority:** the controller runs at SCHED_FIFO 40 on cores 6-7. Without it,
  AgiBot's processes delayed single steps to 26 ms. Against the simulator, compute is about
  7 ms, the worst step 9.3 ms, and no tick more than 0.5 ms late. On the robot, compute is
  5.7 ms. PC2's kernel is not PREEMPT_RT, so this is soft real-time.
- **ROS doesn't cross the relay:** PC1 doesn't route between the hotspot and the internal
  network, and the robot's DDS profile only allows 10.0.1.41. A sim2sim test of PC2's
  controller against the laptop's bridge (section 7.3) therefore still needs the cable. The
  real robot doesn't.

---

## 10. Measured behaviour and pitfalls

**Measured:**
- **Rates and latency over the cable:** joint state 500 Hz (head 334 Hz), both IMUs 500 Hz.
  State about 2 ms old when read. Policy step about 2 ms on the laptop.
- **Joint tracking, hanging, 0.15 rad sine at 4 s:** the elbows reached 85% of the amplitude
  at kp 25, the knees 60% at kp 40 and 77% at kp 80. The knee needs about **3.5 Nm almost
  regardless of gain**, which reads as friction; the elbows need about 1 Nm. Our training
  model has no joint friction. Our policies still worked, and a simulated plant with that
  friction didn't change their tracking much.
- **Develop_MC:** the migration took about 36 s.

**Pitfalls:**
- **The robot's address on the laptop disappears** after a reboot or replug (section 2).
- **numpy 2 breaks ROS Humble's Python bindings.**
- **OpenBLAS threads:** uncapped, the spinning pool delays policy steps (section 3.4).
- **Service replies can get lost** right after creating a client; never resend a migration.
- **Background jobs in non-interactive shells ignore Ctrl-C.** For test scripts, send SIGINT
  through Python's `subprocess` instead of a shell `&`.
- **LAFAN clips start in a T-pose,** which looks perfectly "still". Check the arms when
  picking standing frames.
- **F710 MODE LED on** swaps the left stick with the D-pad.
- **Joint ranges:** a limp robot already sits slightly past some URDF ranges (section 5).

**Links:**
- AimDK docs: <https://x2-aimdk.agibot.com/en/latest/> (system modes in 5.6, sensors in
  1.6, joint ranges in 1.7)
- User manual: <https://agibot.com/public/uploads/file/User_Manual_X2.pdf>
- Our controller and tools: `deployment/` in this repository (`x2/README.md` there has
  the details)
