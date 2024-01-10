cmake_minimum_required(VERSION 3.10)
project(sensors_tools_ros)


## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
# catkin_simple
cv_bridge
rospy
sensor_msgs
geometry_msgs
tf_conversions
)

catkin_package()

###########
## Build ##
###########

## Nothing to build, this is a Python only package