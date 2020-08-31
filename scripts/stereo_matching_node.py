#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
import enum
import os
import sys

import rospy
from aanet_stereo_matching import AANetStereoMatcher, AANetStereoMatcherConfig
from ros_node_base import RosNodeBase

from sensor_msgs.msg import Image


import torch


class MatcherType(enum.IntEnum):
    AANET = 0
    MAX = AANET


_MODEL_FACTORY = {MatcherType.AANET: [AANetStereoMatcher, AANetStereoMatcherConfig]}


class StereoMatcherNode(RosNodeBase):
    def __init__(self, rospy):
        RosNodeBase.__init__(self, rospy)
        self._matcher_type = self._rospy.get_param("~matcher_type", 0)

        if self._matcher_type > MatcherType.MAX:
            self._rospy.logfatal("Not supported stereo matcher\n")
            sys.exit()

        self._MODEL_CLASS, self._MODEL_CONFIG_CLASS = _MODEL_FACTORY[self._matcher_type]
        self._model_config = self._MODEL_CONFIG_CLASS(rospy)

        if (self._model_config.model_path == "") or (not os.path.isfile(self._model_config.model_path)):
            self._rospy.logfatal("Invalid model path {}\n".format(self._model_config.model_path))
            sys.exit()

        self._gpu_idx = self._rospy.get_param("~gpu_idx", -1)
        if self._gpu_idx < 0:
            self._device = torch.device("cpu")
        else:
            if not torch.cuda.is_available():
                self._rospy.logfatal("GPU environment not available\n")
                sys.exit()
            else:
                self._device = torch.device("cuda:{}".format(self._gpu_idx))

        self._model = self._MODEL_CLASS(self._model_config, self._device)

        self._debug = rospy.get_param("~debug", True)

        self._disp_pub = self._rospy.Publisher("~disparity", Image, queue_size=1)
        if self._debug:
            self._disp_debug_pub = self._rospy.Publisher("~disp_debug", Image, queue_size=1)

        self._disparity_multiplier = self._rospy.get_param("~disparity_multiplier", 256.0)

        self._subscribe()

    def _subscribe(self):
        import message_filters

        self._synchronizer_type = self._rospy.get_param("~synchronizer_type", 0)

        self._left_rect_img_sub = message_filters.Subscriber("~left_rect_img", Image)
        self._right_rect_img_sub = message_filters.Subscriber("~right_rect_img", Image)

        self._ts = self.to_synchronizer(
            self._synchronizer_type, fs=[self._left_rect_img_sub, self._right_rect_img_sub], queue_size=10, slop=0.1
        )
        self._ts.registerCallback(self.callback)

    def callback(self, left_rect_img_msg, right_rect_img_msg):
        import cv2
        import numpy as np

        left_img = RosNodeBase.to_cv_image(left_rect_img_msg)
        right_img = RosNodeBase.to_cv_image(right_rect_img_msg)

        if left_img is None:
            rospy.logwarn("Left image empty\n")
            pass
        if right_img is None:
            rospy.logwarn("Right image empty\n")
            pass

        disp_img = self._model.run(left_img, right_img)
        disp_img = disp_img * self._disparity_multiplier
        disp_img = disp_img.astype(np.uint16)

        self._disp_pub.publish(RosNodeBase.to_img_msg(disp_img, "16UC1"))

        if self._debug:
            disp_debug_img = cv2.applyColorMap(RosNodeBase.uint16_to_uint8(disp_img), cv2.COLORMAP_JET)
            self._disp_debug_pub.publish(RosNodeBase.to_img_msg(disp_debug_img, "8UC3"))


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
