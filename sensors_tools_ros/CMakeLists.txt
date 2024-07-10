cmake_minimum_required(VERSION 3.10)
project(sensors_tools_ros)

find_package(catkin REQUIRED COMPONENTS
  rospy
  roscpp
  message_generation
  sensor_msgs
  geometry_msgs
  nav_msgs
  tf2_ros
  std_srvs
  cv_bridge
  tf_conversions
)

add_service_files(
  FILES
  MoveSensor.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
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

