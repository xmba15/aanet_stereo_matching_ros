#!/usr/bin/env python
import rospy
from aanet_stereo_matching import AANetStereoMatcher, AANetStereoMatcherConfig
import enum


class MatcherType(enum.Enum):
    AANET = 0
    MAX = AANET


_MODEL_FACTORY = {MatcherType.AANET: [AANetStereoMatcher, AANetStereoMatcherConfig]}


def StereoMatcherNode(object):
    def __init__(self):
        self._matcher_type = rospy.get_param("~matcher_type", 0)
        if self._matcher_type > MatcherType.MAX:
            raise Exception("Not supported stereo matcher\n")

        self._MODEL_CLASS, self._MODEL_CLASS_CONFIG = _MODEL_FACTORY[self._matcher_type]
        self._model_config = self._MODEL_CLASS_CONFIG()

