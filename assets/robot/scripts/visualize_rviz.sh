#!/bin/bash
#
# 在 RViz2 中可视化 Uni-Dingo 双臂机器人
#
# 用法:
#   bash assets/robot/scripts/visualize_rviz.sh
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOT_DIR="$(dirname "$SCRIPT_DIR")"
URDF_FILE="$ROBOT_DIR/urdf/uni_dingo_dual_arm_absolute.urdf"
RVIZ_CONFIG="$ROBOT_DIR/rviz/visualize.rviz"

# 检查文件
if [ ! -f "$URDF_FILE" ]; then
    echo "❌ URDF 文件不存在: $URDF_FILE"
    exit 1
fi

# Source ROS 2
source /opt/ros/humble/setup.bash

echo "================================================"
echo " Uni-Dingo 双臂机器人 RViz2 可视化"
echo "================================================"
echo ""
echo "URDF:  $URDF_FILE"
echo "RViz:  $RVIZ_CONFIG"
echo ""
echo "启动中... (Ctrl+C 退出)"
echo ""

# 读取 URDF 内容
ROBOT_DESC=$(cat "$URDF_FILE")

# 启动 robot_state_publisher (后台)
ros2 run robot_state_publisher robot_state_publisher \
    --ros-args -p robot_description:="$ROBOT_DESC" &
RSP_PID=$!

# 启动 joint_state_publisher_gui (后台, 有滑块可以操控关节)
ros2 run joint_state_publisher_gui joint_state_publisher_gui &
JSP_PID=$!

# 等一秒让 publisher 启动
sleep 1

# 启动 rviz2
if [ -f "$RVIZ_CONFIG" ]; then
    ros2 run rviz2 rviz2 -d "$RVIZ_CONFIG" &
else
    ros2 run rviz2 rviz2 &
fi
RVIZ_PID=$!

# 清理函数
cleanup() {
    echo ""
    echo "正在关闭..."
    kill $RSP_PID $JSP_PID $RVIZ_PID 2>/dev/null
    wait $RSP_PID $JSP_PID $RVIZ_PID 2>/dev/null
    echo "已退出"
}
trap cleanup EXIT INT TERM

# 等待任意进程退出
wait -n $RSP_PID $JSP_PID $RVIZ_PID 2>/dev/null
