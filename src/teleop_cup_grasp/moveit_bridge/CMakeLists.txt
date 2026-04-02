cmake_minimum_required(VERSION 3.16)
project(cup_grasp_moveit_bridge)

find_package(ament_cmake REQUIRED)

install(DIRECTORY config launch rviz
  DESTINATION share/${PROJECT_NAME}
)

install(
  FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../assets/robot/urdf/uni_dingo_dual_arm_absolute.urdf
  DESTINATION share/${PROJECT_NAME}/urdf
)

ament_package()
