#!/bin/bash
# Send keys to the controller started by start.sh:  key.sh enter | s | n | p | d ...
#   enter  start the policy / play the clip      s  default pose (safety layer)
#   n, p   next / previous clip                  d  damp and quit
set -euo pipefail
FIFO=/home/agi/lmj_deploy/run/keys
pgrep -f "^python3 -u repo/deployment/x2/deploy_x2.py" >/dev/null || { echo "no controller is running" >&2; exit 1; }
for k in "$@"; do
  case "$k" in
    enter) printf '\n' ;;
    s|n|p|d) printf '%s' "$k" ;;
    *) echo "unknown key '$k' (enter s n p d)" >&2; exit 1 ;;
  esac
done | timeout 2 tee "$FIFO" >/dev/null
