cmake_minimum_required(VERSION 3.10)
project(sensors_tools_ros)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  geometry_msgs
  nav_msgs
  tf2_ros
  std_srvs
  cv_bridge
  tf_conversions
)

catkin_package(
  CATKIN_DEPENDS
    rospy
    sensor_msgs
    geometry_msgs
    nav_msgs
    tf2_ros
    std_srvs
    cv_bridge
    tf_conversions
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)
