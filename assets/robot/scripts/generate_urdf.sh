#!/bin/bash
#
# 将 Xacro 文件展开为纯 URDF，供 Isaac Sim 导入使用
#
# 用法:
#   cd <项目根目录>
#   bash asserts/robot/scripts/generate_urdf.sh
#
# 前提条件:
#   - 安装了 ROS 2 (humble) 和 xacro
#   - Z1-Dingo-Robot 工作空间需要先 colcon build
#     或者使用本脚本中的 fallback 方案

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOT_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(cd "$ROBOT_DIR/../.." && pwd)"
Z1_DINGO_ROOT="$PROJECT_ROOT/Z1-Dingo-Robot"
OUTPUT_DIR="$ROBOT_DIR/urdf"
OUTPUT_FILE="$OUTPUT_DIR/uni_dingo_dual_arm.urdf"

echo "================================================"
echo " Uni-Dingo 双臂机器人 URDF 生成脚本"
echo "================================================"
echo ""
echo "项目根目录:   $PROJECT_ROOT"
echo "Z1-Dingo 源:  $Z1_DINGO_ROOT"
echo "输出文件:     $OUTPUT_FILE"
echo ""

# 检查 xacro 是否可用
if ! command -v xacro &> /dev/null; then
    echo "❌ 错误: xacro 未安装。请运行:"
    echo "   pip install xacro"
    echo "   或 sudo apt install ros-humble-xacro"
    exit 1
fi

# 检查 source 是否已加载
if [ -z "$ROS_DISTRO" ]; then
    echo "⚠️  ROS 环境未加载，正在自动 source..."
    source /opt/ros/humble/setup.bash
fi

# 方案1: 尝试使用已构建的工作空间
if [ -f "$Z1_DINGO_ROOT/install/setup.bash" ]; then
    echo "✅ 发现已构建的工作空间，使用包路径解析..."
    source "$Z1_DINGO_ROOT/install/setup.bash"
    
    xacro "$Z1_DINGO_ROOT/src/unidingo_description/urdf/uni_dingo_dual_arm.urdf.xacro" \
        UnitreeGripper:=true \
        PanTilt:=true \
        -o "$OUTPUT_FILE"
    
    echo ""
    echo "✅ URDF 已生成: $OUTPUT_FILE"
    exit 0
fi

# 方案2: 手动设置 ROS_PACKAGE_PATH 来模拟包查找
echo "⚠️  工作空间未构建，使用手动路径解析..."
echo "   (建议先运行: cd $Z1_DINGO_ROOT && colcon build)"
echo ""

# 创建临时的包查找目录结构
TEMP_DIR=$(mktemp -d -p "$PROJECT_ROOT" .tmp_pkg_XXXXXX)
trap "rm -rf $TEMP_DIR" EXIT

# 创建符号链接来模拟 ROS 包结构
ln -sf "$Z1_DINGO_ROOT/src/unidingo_description" "$TEMP_DIR/uni_dingo"
ln -sf "$Z1_DINGO_ROOT/src/z1_ros/unitree_ros/robots/z1_description" "$TEMP_DIR/z1_description"
ln -sf "$Z1_DINGO_ROOT/src/dingo/dingo_description" "$TEMP_DIR/dingo_description"
ln -sf "$Z1_DINGO_ROOT/src/pan_tilt_ros/pan_tilt_description" "$TEMP_DIR/pan_tilt_description"
ln -sf "$Z1_DINGO_ROOT/src/uni_dingo_api" "$TEMP_DIR/uni_dingo_api"

# 为 realsense2_description 创建空包（如果不存在）
if ! ros2 pkg prefix realsense2_description 2>/dev/null; then
    echo "⚠️  realsense2_description 包未安装，D455 相机描述将不可用"
    echo "   安装: sudo apt install ros-humble-realsense2-description"
fi

# 设置环境变量  
export CMAKE_PREFIX_PATH="$TEMP_DIR:${CMAKE_PREFIX_PATH:-}"
export AMENT_PREFIX_PATH="$TEMP_DIR:${AMENT_PREFIX_PATH:-}"

# 为每个符号链接创建 package.xml（如果不存在）让 xacro 的 $(find) 能找到
for pkg_dir in "$TEMP_DIR"/*/; do
    pkg_name=$(basename "$pkg_dir")
    if [ ! -f "$pkg_dir/package.xml" ]; then
        cat > "$pkg_dir/package.xml" << EOF_PKG
<?xml version="1.0"?>
<package format="3">
  <name>$pkg_name</name>
  <version>0.0.0</version>
  <description>Temp package for xacro resolution</description>
  <maintainer email="temp@temp.com">temp</maintainer>
  <license>MIT</license>
</package>
EOF_PKG
    fi
done

# 设置 ROS_PACKAGE_PATH (ROS 1 风格, xacro 也支持)
export ROS_PACKAGE_PATH="$TEMP_DIR:${ROS_PACKAGE_PATH:-}"

# 设置 Dingo 环境变量
export DINGO_OMNI=1
export DINGO_PACS_ENABLED=0

echo "正在生成 URDF..."
xacro "$Z1_DINGO_ROOT/src/unidingo_description/urdf/uni_dingo_dual_arm.urdf.xacro" \
    UnitreeGripper:=true \
    PanTilt:=true \
    -o "$OUTPUT_FILE" 2>&1 || {
    echo ""
    echo "❌ xacro 展开失败。"
    echo ""
    echo "最常见原因:"
    echo "  1. 需要先构建 Z1-Dingo-Robot ROS 2 工作空间:"
    echo "     cd $Z1_DINGO_ROOT && colcon build"
    echo "  2. 缺少 realsense2_description 包:"
    echo "     sudo apt install ros-humble-realsense2-description"
    echo ""
    exit 1
}

echo ""
echo "✅ URDF 已生成: $OUTPUT_FILE"
echo ""
echo "文件大小: $(du -h "$OUTPUT_FILE" | cut -f1)"
echo "Link 数量: $(grep -c '<link' "$OUTPUT_FILE")"
echo "Joint 数量: $(grep -c '<joint' "$OUTPUT_FILE")"
echo ""
echo "下一步: 在 Isaac Sim 中导入此 URDF 文件"
