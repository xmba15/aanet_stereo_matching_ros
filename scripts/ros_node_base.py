#!/usr/bin/env python
import enum


__all__ = ["RosNodeBase"]


class SynchronizerType(enum.IntEnum):
    TIME_SYNCHRONIZER = 0
    APPROX_TIME_SYNCHRONIZER = 1
    MAX = APPROX_TIME_SYNCHRONIZER


class RosNodeBase(object):
    def __init__(self, rospy):
        self._rospy = rospy

    @staticmethod
    def to_cv_image(image_msg):
        import numpy as np
        import cv2

        if image_msg is None:
            return None

        width = image_msg.width
        height = image_msg.height
        channels = int(len(image_msg.data) / (width * height))

        encoding = None
        if image_msg.encoding.lower() in ["rgb8", "bgr8"]:
            encoding = np.uint8
        elif image_msg.encoding.lower() == "mono8":
            encoding = np.uint8
        elif image_msg.encoding.lower() == "32fc1":
            encoding = np.float32
            channels = 1

        cv_img = np.ndarray(shape=(image_msg.height, image_msg.width, channels), dtype=encoding, buffer=image_msg.data)

        if image_msg.encoding.lower() == "mono8":
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
        elif image_msg.encoding.lower() == "rgb8":
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)

        return cv_img

    @staticmethod
    def to_img_msg(disp_img, encoding):
        from sensor_msgs.msg import Image

        img_msg = Image()
        height, width = disp_img.shape[:2]
        img_msg.height = height
        img_msg.width = width

        if disp_img.dtype.byteorder == ">":
            img_msg.is_bigendian = True

        img_msg.encoding = encoding
        img_msg.data = disp_img.tostring()
        img_msg.step = len(img_msg.data) // img_msg.height

        return img_msg

    @staticmethod
    def uint16_to_uint8(img_uint16):
        import cv2

        return cv2.convertScaleAbs(img_uint16, alpha=(255.0 / 65535.0))

    def to_synchronizer(self, synchronizer_type: SynchronizerType, fs, queue_size=10, slop=0.1):
        if synchronizer_type > SynchronizerType.MAX:
            raise Exception("Not supported synchronizer type\n")

        import message_filters

        if synchronizer_type == 0:
            return message_filters.TimeSynchronizer(fs=fs, queue_size=queue_size)
        if synchronizer_type == 1:
            return message_filters.ApproximateTimeSynchronizer(fs=fs, queue_size=queue_size, slop=slop)
