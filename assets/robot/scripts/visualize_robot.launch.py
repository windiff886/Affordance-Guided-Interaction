"""
ROS 2 Launch file for visualizing the Uni-Dingo dual-arm robot in RViz2.

Usage:
    source /opt/ros/humble/setup.bash
    ros2 launch assets/robot/scripts/visualize_robot.launch.py
"""

import os
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    robot_dir = os.path.dirname(script_dir)
    urdf_file = os.path.join(robot_dir, 'urdf', 'uni_dingo_dual_arm_absolute.urdf')
    rviz_config = os.path.join(robot_dir, 'rviz', 'visualize.rviz')

    # Read URDF
    with open(urdf_file, 'r') as f:
        robot_description = f.read()

    return LaunchDescription([
        # Robot State Publisher
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': robot_description}],
        ),

        # Joint State Publisher GUI (sliders to control joints)
        Node(
            package='joint_state_publisher_gui',
            executable='joint_state_publisher_gui',
            name='joint_state_publisher_gui',
            output='screen',
        ),

        # RViz2
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', rviz_config] if os.path.exists(rviz_config) else [],
        ),
    ])
