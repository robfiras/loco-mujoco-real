# The X2 controller on the robot (SoC1), driven over Wi-Fi

`deploy_x2.py` runs on the robot's development computer (SoC1 / PC2, the Jetson Orin NX at
10.0.1.41) inside a rootless podman container, as AgiBot's docs recommend (and require from
SDK v1.2.0). The laptop only starts it, sends keys and the gamepad, and shows its log, over
the robot's Wi-Fi hotspot. Wi-Fi is never in the control loop.

| file | where it runs | what it does |
|---|---|---|
| `fetch_image.py` | laptop | fetches `ros:humble-ros-base-jammy` for arm64 from Docker Hub as a docker-archive, without docker (the robot has no internet) |
| `Containerfile` | SoC1 | that image plus MuJoCo and numpy 1.x, unpacked from wheels: builds offline |
| `run.sh` | SoC1 | starts the container `lmj-x2` as AgiBot's `run.sh` does: host network, the robot's ROS and DDS profile, `AGIBOT_*`, `/dev`, the GPU, and real-time priority allowed |
| `start.sh` | SoC1 | starts `deploy_x2.py` in the container, detached, at real-time priority on cores 6-7, keys from `run/keys`, log to `run/deploy.log`, heartbeat required |
| `key.sh` | SoC1 | writes keys into `run/keys` |
| `x2ctl.sh` | laptop | `start …`, `log`, `stop`, or (no argument) the operator |
| `x2_operator.py` | laptop | keyboard and USB gamepad to `run/keys` over one SSH connection through the relay, with a heartbeat |

## Setup, once

On the laptop (internet), then copied to SoC1 into `~/lmj_deploy` (code under `repo/` in the
repository's layout, a bundle under `bundles/`, the wheels under `wheels/`, the image tarball,
and these scripts):

```bash
python3 fetch_image.py library/ros humble-ros-base-jammy arm64 ros_humble_ros_base_arm64.tar
pip3 download --only-binary=:all: --platform manylinux_2_28_aarch64 --platform manylinux2014_aarch64 \
    --platform manylinux_2_17_aarch64 --platform linux_aarch64 --platform any --python-version 3.10 \
    --implementation cp --abi cp310 --abi abi3 --abi none "mujoco==3.16.0" "numpy==1.26.4" -d wheels
```

On SoC1:

```bash
podman load -i image/ros_humble_ros_base_arm64.tar
podman build -t lmj/x2-deploy:dev ~/lmj_deploy
~/lmj_deploy/run.sh
cp -r ~/aimdk/src/aimdk_msgs ~/lmj_deploy/ws/src/
podman exec lmj-x2 bash -c 'source /opt/ros/humble/setup.bash && cd ~/lmj_deploy/ws && colcon build --packages-select aimdk_msgs'
```

The container has no restart policy: after a robot reboot, `~/lmj_deploy/run.sh` again.

## Network

The laptop joins the robot's hotspot (`X2…`, switched on in the Agibot Go app) with a second
Wi-Fi adapter, so its own stays on the internet; give that connection
`ipv4.never-default yes`. The hotspot is served by PC1 (192.168.88.88), which relays SSH to
SoC1 (`ssh -J agi@192.168.88.88 agi@10.0.1.41`); both need the laptop's key
(`ssh-copy-id`). ROS does not cross the relay, and the robot's DDS profile pins its ROS
traffic to the wired internal network anyway: only SSH goes over Wi-Fi.

## A run

```bash
deployment/x2/soc1/x2ctl.sh log                                   # terminal 1
deployment/x2/soc1/x2ctl.sh start --robot --bundle bundles/<bundle> --motion <clip>   # terminal 2
```

Gamepad: A start/play, B default pose, LB/RB clip, LT + B damp and stop. Keyboard, in the
operator's terminal: Enter, `s`, `p` / `n`, and Ctrl-C to damp and stop; no letter key damps. If the heartbeat stops for 0.5 s (Wi-Fi, the operator closed), the controller
switches to the default pose by itself and ignores every key but damp until the link has
been back for a second. Reconnect with `x2ctl.sh`; stop without the operator with
`x2ctl.sh stop`.

Measured on SoC1 against the laptop's simulator, controller at priority 40 on cores 6-7:
compute 7.2 ms median, 9.3 ms worst, ticks at most 0.5 ms late (without real-time priority:
26 ms and 6 ms). On the real robot: 5.7 ms median, 6.8 ms worst.
