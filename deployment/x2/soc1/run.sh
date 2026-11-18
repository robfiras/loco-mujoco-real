#!/bin/bash
# Start the lmj tracker container on SoC1, as AgiBot's container guide starts theirs: host
# network, IPC and PID namespaces, the robot's own ROS and DDS profile, the AGIBOT_* variables,
# /dev (gamepad, later), and the GPU (for TensorRT, later). Nothing is installed on the host.
# --ulimit rtprio lets start.sh run the controller with real-time priority: the agi user may
# use up to 98 without sudo (ulimit -r), and the container otherwise gets 0.
set -euo pipefail
IMAGE="${IMAGE:-lmj/x2-deploy:dev}"
NAME="${NAME:-lmj-x2}"
HERE="$(cd "$(dirname "$0")" && pwd)"

# Host groups owning device nodes, translated to the container's GIDs (rootless mapping).
GROUP_ADDS=()
MAP_RAW=$(podman info --format '{{.Host.IDMappings.GIDMap}}')
for HG in 20 29 44 106 109 995 1000; do   # dialout audio video input render debug run
  CG=$(echo "$MAP_RAW" | grep -oE '\{[0-9]+ '"$HG"' [0-9]+\}' | head -1 | grep -oE '^\{[0-9]+' | tr -d '{')
  [ -n "$CG" ] && GROUP_ADDS+=("--group-add" "$CG")
done

ENV_FILE=$(mktemp /tmp/lmj-env.XXXXXX); trap 'rm -f "$ENV_FILE"' EXIT
sed -nE "s/^export (AGIBOT_[A-Z_]+)='(.*)'\$/\1=\2/p" /home/agi/.aima/env/bashrc > "$ENV_FILE" 2>/dev/null || true

podman rm -f "$NAME" >/dev/null 2>&1 || true
podman run -d --name "$NAME" \
  --ipc=host --network=host --pid=host \
  --ulimit rtprio=98:98 \
  --device nvidia.com/gpu=all \
  "${GROUP_ADDS[@]}" \
  -e HOME=/home/agi \
  -e FASTRTPS_DEFAULT_PROFILES_FILE=/etc/ros_dds_configuration.xml \
  --env-file "$ENV_FILE" \
  -v /etc/machine-id:/etc/machine-id:ro \
  -v /opt/ros:/opt/ros:ro \
  -v /home/agi/.aima/env/ros_dds_configuration.xml:/etc/ros_dds_configuration.xml:ro \
  -v /dev:/dev:rslave \
  -v "$HERE":/home/agi/lmj_deploy:rw \
  "$IMAGE"
echo "started $NAME; enter with: podman exec -it $NAME bash"
