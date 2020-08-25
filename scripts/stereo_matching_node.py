#!/usr/bin/env python
import sys
import rospy
from aanet_stereo_matching import AANetStereoMatcher, AANetStereoMatcherConfig
import enum


class MatcherType(enum.Enum):
    AANET = 0
    MAX = AANET


_MODEL_FACTORY = {MatcherType.AANET: [AANetStereoMatcher, AANetStereoMatcherConfig]}


def StereoMatcherNode(object):
    def __init__(self, rospy):
        self._matcher_type = rospy.get_param("~matcher_type", 0)
        if self._matcher_type > MatcherType.MAX:
            raise Exception("Not supported stereo matcher\n")

        self._MODEL_CLASS, self._MODEL_CLASS_CONFIG = _MODEL_FACTORY[self._matcher_type]
        self._model_config = self._MODEL_CLASS_CONFIG(rospy)


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
