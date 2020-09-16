#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
import enum
import os
import sys
import numpy as np
import torch
import cv2

from aanet_stereo_matching import AANetStereoMatcher, AANetStereoMatcherConfig
from utility import Utility as utils
from ros_node_base import RosNodeBase
import rospy
import message_filters
from sensor_msgs.msg import Image, CameraInfo, PointCloud2


class MatcherType(enum.IntEnum):
    AANET = 0
    MAX = AANET


_MODEL_FACTORY = {MatcherType.AANET: [AANetStereoMatcher, AANetStereoMatcherConfig]}


class StereoMatcherNode(RosNodeBase):
    def __init__(self, internal_rospy):
        RosNodeBase.__init__(self, internal_rospy)
        self._init_parameter()

        if self._matcher_type > MatcherType.MAX:
            self._rospy.logfatal("Not supported stereo matcher")
            sys.exit()

        self._MODEL_CLASS, self._MODEL_CONFIG_CLASS = _MODEL_FACTORY[self._matcher_type]
        self._model_config = self._MODEL_CONFIG_CLASS(rospy)

        if (self._model_config.model_path == "") or (not os.path.isfile(self._model_config.model_path)):
            self._rospy.logfatal("Invalid model path {}".format(self._model_config.model_path))
            sys.exit()

        if self._gpu_idx < 0:
            self._device = torch.device("cpu")
        else:
            if not torch.cuda.is_available():
                self._rospy.logfatal("GPU environment not available")
                sys.exit()
            else:
                self._device = torch.device("cuda:{}".format(self._gpu_idx))

        self._model = self._MODEL_CLASS(self._model_config, self._device)

        self._disp_pub = self._rospy.Publisher("~disparity", Image, queue_size=1)
        if self._debug:
            self._disp_debug_pub = self._rospy.Publisher("~disp_debug", Image, queue_size=1)

        if self._publish_point_cloud:
            self._pointcloud_pub = self._rospy.Publisher("~pointcloud", PointCloud2, queue_size=1)

        self._subscribe_once()
        self._subscribe()

    def _init_parameter(self):
        self._matcher_type = self._rospy.get_param("~matcher_type", 0)
        self._gpu_idx = self._rospy.get_param("~gpu_idx", -1)
        self._debug = self._rospy.get_param("~debug", True)
        self._img_scale = self._rospy.get_param("~img_scale", 1.0)
        self._disparity_multiplier = self._rospy.get_param("~disparity_multiplier", 256.0)
        self._max_depth = self._rospy.get_param("~max_depth", 30.0)
        self._publish_point_cloud = self._rospy.get_param("~publish_point_cloud", False)
        self._use_raw_img = self._rospy.get_param("~use_raw_img", False)

    def _subscribe(self):

        self._synchronizer_type = self._rospy.get_param("~synchronizer_type", 0)
        self._left_rect_img_sub = message_filters.Subscriber("~left_rect_img", Image)
        self._right_rect_img_sub = message_filters.Subscriber("~right_rect_img", Image)

        self._ts = self.to_synchronizer(
            self._synchronizer_type,
            fs=[
                self._left_rect_img_sub,
                self._right_rect_img_sub,
            ],
            queue_size=10,
            slop=0.1,
        )
        self._ts.registerCallback(self.callback)

    def _subscribe_once(self):
        self._left_camera_info = self._rospy.wait_for_message("~left_camera_info", CameraInfo)
        self._right_camera_info = self._rospy.wait_for_message("~right_camera_info", CameraInfo)
        self._q_matrix = utils.get_q_matrix(self._left_camera_info, self._right_camera_info)
        if not np.isclose(self._img_scale, 1.0):
            scaled_left_camera_info = utils.scale_camera_info(self._left_camera_info, self._img_scale, self._img_scale)
            scaled_right_camera_info = utils.scale_camera_info(
                self._right_camera_info, self._img_scale, self._img_scale
            )
            self._q_matrix = utils.get_q_matrix(scaled_left_camera_info, scaled_right_camera_info)

    def callback(self, left_img_msg: Image, right_img_msg: Image):
        left_img = utils.to_cv_image(left_img_msg)
        if left_img is None:
            self._rospy.logwarn("Left image empty")
            return

        right_img = utils.to_cv_image(right_img_msg)
        if right_img is None:
            self._rospy.logwarn("Right image empty")
            return

        if self._use_raw_img:
            left_img = utils.remap(left_img, self._left_camera_info)
            right_img = utils.remap(right_img, self._right_camera_info)

        if not np.isclose(self._img_scale, 1.0):
            width = int(left_img.shape[1] * self._img_scale)
            height = int(left_img.shape[0] * self._img_scale)
            resize_dim = (width, height)
            left_img = cv2.resize(left_img, resize_dim, interpolation=cv2.INTER_CUBIC)
            right_img = cv2.resize(right_img, resize_dim, interpolation=cv2.INTER_CUBIC)

        disp_img = self._model.run(left_img, right_img)

        cur_header = self.get_new_header(left_img_msg.header)

        disp_img_msg = utils.to_img_msg(disp_img, "32FC1", cur_header)
        self._disp_pub.publish(disp_img_msg)

        if self._debug:
            disp_img_scaled = disp_img * self._disparity_multiplier
            disp_img_scaled = disp_img_scaled.astype(np.uint16)

            disp_debug_img = cv2.applyColorMap(utils.uint16_to_uint8(disp_img_scaled), cv2.COLORMAP_JET)
            disp_debug_img_msg = utils.to_img_msg(disp_debug_img, "bgr8", cur_header)
            self._disp_debug_pub.publish(disp_debug_img_msg)

        if self._publish_point_cloud:
            projected_points = cv2.reprojectImageTo3D(disp_img, self._q_matrix)
            pointcloud_msg = utils.xyzrgb_array_to_pointcloud2(projected_points, left_img, self._max_depth, cur_header)
            self._pointcloud_pub.publish(pointcloud_msg)


def main(argv):
    try:
        rospy.init_node("stereo_matcher", anonymous=False)
        stereo_matcher = StereoMatcherNode(rospy)
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.logfatal("error with stereo_matcher setup")
        sys.exit()


if __name__ == "__main__":
    main(sys.argv)
