#!/bin/bash
# Start deploy_x2.py in the lmj-x2 container on SoC1, detached from this shell: it keeps
# running if the SSH session or the Wi-Fi drops. Keys go in through run/keys (key.sh),
# output goes to run/deploy.log. The operator's heartbeat is expected (--heartbeat 0.5;
# x2_operator.py sends it), so without an operator the policy never starts, and losing
# it falls back to the default pose. Arguments are passed to deploy_x2.py, e.g.
#   ~/lmj_deploy/start.sh --robot --bundle bundles/<name>_walk --motion fwd_A042
set -euo pipefail
HERE=/home/agi/lmj_deploy
mkdir -p "$HERE/run"
if pgrep -f "^python3 -u repo/deployment/x2/deploy_x2.py" >/dev/null; then
  echo "a controller is already running; stop it first (key.sh d)" >&2; exit 1
fi
# A fresh pipe for every controller. An operator connection that outlived its controller
# waits on the old pipe for a reader, holding whatever was pressed meanwhile, and would hand
# it to this one: once, a damp that stopped the next controller right after its hold.
pkill -u "$(id -u)" -f "^bash -c p=~/lmj_deploy/run/keys" || true
rm -f "$HERE/run/keys"
mkfifo "$HERE/run/keys"
ARGS=$(printf ' %q' "$@")
# Real-time priority (SCHED_FIFO) on the quieter cores: AgiBot's own processes are
# ordinary ones and busiest on cores 0-3, and without this they delayed single control
# steps to 26 ms. 40 preempts every ordinary process and stays below AgiBot's camera
# daemon (75). SoC1's kernel is not PREEMPT_RT, so this is soft real-time.
RT="chrt -f ${RT_PRIO:-40} taskset -c ${RT_CPUS:-6,7}"
podman exec -d lmj-x2 bash -c "source /opt/ros/humble/setup.bash && \
  source ~/lmj_deploy/ws/install/local_setup.bash && cd ~/lmj_deploy && \
  exec $RT python3 -u repo/deployment/x2/deploy_x2.py --keys-from run/keys --gamepad none --heartbeat 0.5$ARGS \
  > run/deploy.log 2>&1"
sleep 2
PID=$(pgrep -f "^python3 -u repo/deployment/x2/deploy_x2.py" || true)
if [ -n "$PID" ]; then
  echo "$(chrt -p "$PID" | tail -1); $(taskset -cp "$PID")"
fi
echo "started; follow it with: tail -f $HERE/run/deploy.log"
