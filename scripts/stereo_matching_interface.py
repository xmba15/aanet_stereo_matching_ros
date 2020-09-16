#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc
import os


class StereoMatcherBase(abc.ABC):
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(self, config, device, transform):
        self._config = config
        self._device = device
        self._transform = transform

        if not os.path.exists(config.model_path):
            raise Exception("{} does not exist".format(config.model_path))

    @abc.abstractmethod
    def run(self, left_rectified_image, right_rectified_image):
        return NotImplemented
