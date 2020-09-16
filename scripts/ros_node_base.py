#!/usr/bin/env python
import enum
import sys


__all__ = ["RosNodeBase"]


class SynchronizerType(enum.IntEnum):
    TIME_SYNCHRONIZER = 0
    APPROX_TIME_SYNCHRONIZER = 1
    MAX = APPROX_TIME_SYNCHRONIZER


class RosNodeBase(object):
    def __init__(self, internal_rospy):
        self._rospy = internal_rospy

    def to_synchronizer(self, synchronizer_type: SynchronizerType, fs, queue_size=10, slop=0.1):
        if synchronizer_type > SynchronizerType.MAX:
            self._rospy.logfatal("Not supported synchronizer type")
            sys.exit()

        import message_filters

        if synchronizer_type == 0:
            return message_filters.TimeSynchronizer(fs=fs, queue_size=queue_size)
        if synchronizer_type == 1:
            return message_filters.ApproximateTimeSynchronizer(fs=fs, queue_size=queue_size, slop=slop)

    def get_new_header(self, old_header):
        import std_msgs.msg

        frame_id = "map"
        if old_header.frame_id != "":
            frame_id = old_header.frame_id

        time_stamp = self._rospy.Time.now()
        cur_header = std_msgs.msg.Header()
        cur_header.frame_id = frame_id
        cur_header.stamp = time_stamp

        return cur_header
