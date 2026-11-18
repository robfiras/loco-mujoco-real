#!/bin/bash
# On the laptop, over Wi-Fi through the robot's relay:
#   x2ctl.sh start --robot --bundle bundles/<bundle> --motion <clip> [--end next]
#                         start the controller on SoC1, then operate it (x2_operator.py)
#   x2ctl.sh              operate a controller that is already running
#   x2ctl.sh log          follow the controller's output (Ctrl-C only stops the viewing)
#   x2ctl.sh stop         damp and quit, without the operator program
RELAY=${X2_RELAY:-agi@192.168.88.88}
SOC1=${X2_SOC1:-agi@10.0.1.41}
HERE="$(cd "$(dirname "$0")" && pwd)"
SSH=(ssh -o ConnectTimeout=5 -J "$RELAY" "$SOC1")
[ -f "$HOME/x2_aimdk/env.sh" ] && source "$HOME/x2_aimdk/env.sh"
case "${1:-}" in
  log) exec "${SSH[@]}" "tail -n 40 -f ~/lmj_deploy/run/deploy.log" ;;
  stop) exec "${SSH[@]}" "~/lmj_deploy/key.sh d" ;;
  start) shift; "${SSH[@]}" "~/lmj_deploy/start.sh $(printf ' %q' "$@")" || exit 1 ;;
esac
exec python3 "$HERE/x2_operator.py"
