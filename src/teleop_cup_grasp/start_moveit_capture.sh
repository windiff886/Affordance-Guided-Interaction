#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"
WORKSPACE_SETUP="${WORKSPACE_SETUP:-${PROJECT_ROOT}/install/setup.bash}"
PYTHON_BIN="${PYTHON_BIN:-python}"
USE_RVIZ="${USE_RVIZ:-true}"
LAUNCH_LOG="${LAUNCH_LOG:-/tmp/cup_grasp_moveit_servo.launch.log}"

if [[ ! -f "${ROS_SETUP}" ]]; then
  echo "Missing ROS setup: ${ROS_SETUP}" >&2
  exit 1
fi

if [[ ! -f "${WORKSPACE_SETUP}" ]]; then
  echo "Missing workspace setup: ${WORKSPACE_SETUP}" >&2
  echo "Build the package first with: colcon build --packages-select cup_grasp_moveit_bridge" >&2
  exit 1
fi

source "${ROS_SETUP}"
source "${WORKSPACE_SETUP}"

LAUNCH_PID=""

cleanup() {
  if [[ -n "${LAUNCH_PID}" ]] && kill -0 "${LAUNCH_PID}" 2>/dev/null; then
    kill "${LAUNCH_PID}" 2>/dev/null || true
    wait "${LAUNCH_PID}" 2>/dev/null || true
  fi
}

trap cleanup EXIT

echo "Starting MoveIt Servo sidecar..."
ros2 launch cup_grasp_moveit_bridge servo_teleop.launch.py use_rviz:="${USE_RVIZ}" >"${LAUNCH_LOG}" 2>&1 &
LAUNCH_PID=$!

echo "Waiting for /servo_node/start_servo ..."
for _ in $(seq 1 60); do
  if ! kill -0 "${LAUNCH_PID}" 2>/dev/null; then
    echo "MoveIt launch exited early. Check ${LAUNCH_LOG}" >&2
    exit 1
  fi
  if ros2 service list 2>/dev/null | grep -q "/servo_node/start_servo"; then
    break
  fi
  sleep 0.5
done

if ! ros2 service list 2>/dev/null | grep -q "/servo_node/start_servo"; then
  echo "Timed out waiting for /servo_node/start_servo. Check ${LAUNCH_LOG}" >&2
  exit 1
fi

echo "Starting Isaac capture..."
cd "${PROJECT_ROOT}"
exec "${PYTHON_BIN}" src/teleop_cup_grasp/isaac_servo_bridge.py
