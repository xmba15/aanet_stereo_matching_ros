cmake_minimum_required(VERSION 3.10)
project(aanet_stereo_matching_ros)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  message_filters
)

catkin_package(
 CATKIN_DEPENDS
   rospy
   sensor_msgs
   message_filters
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)
