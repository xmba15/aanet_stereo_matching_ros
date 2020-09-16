#!/usr/bin/env python
import math
import os
import sys
import torch
import torch.nn.functional as F
from stereo_matching_interface import StereoMatcherBase


ROOT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "aanet")
sys.path.append(ROOT_DIR)
try:
    from utils import utils
    import nets
    from dataloader import transforms
except Exception as e:
    print(e)
    sys.exit()


__all__ = ["AANetStereoMatcher", "AANetStereoMatcherConfig"]


class AANetStereoMatcherConfig(object):
    __name__ = "aanet"

    max_disp = 192  # Max disparity

    feature_type = "aanet"  # Type of feature extractor

    no_feature_mdconv = True  # Whether to use mdconv for feature extraction

    feature_pyramid = True  # Use pyramid feature

    feature_pyramid_network = True  # Use FPN

    feature_similarity = "correlation"  # Similarity measure for matching cost

    num_downsample = 2  # Number of downsample layer for feature extraction

    aggregation_type = "adaptive"  # Type of cost aggregation

    num_scales = 3  # Number of stages when using parallel aggregation

    num_fusions = 6  # Number of multi-scale fusions when using parallel

    num_stage_blocks = 1  # Number of deform blocks for ISA

    num_deform_blocks = 3  # Number of DeformBlocks for aggregation

    no_intermediate_supervision = True  # Whether to add intermediate supervision

    deformable_groups = 2  # Number of deformable groups

    mdconv_dilation = 2  # Dilation rate for deformable conv

    refinement_type = "stereodrnet"  # Type of refinement module

    model_path = ""  # Pretrained network

    def __init__(self, internal_rospy=None):
        if internal_rospy is None:
            return

        def _rospy_set(attr):
            object.__setattr__(
                self,
                attr,
                internal_rospy.get_param(
                    "~" + self.__name__ + "/" + attr,
                    object.__getattribute__(self, attr),
                ),
            )

        [
            _rospy_set(attr)
            for attr in [
                "max_disp",
                "feature_type",
                "no_feature_mdconv",
                "feature_pyramid",
                "feature_pyramid_network",
                "feature_similarity",
                "num_downsample",
                "aggregation_type",
                "num_scales",
                "num_fusions",
                "num_stage_blocks",
                "num_deform_blocks",
                "no_intermediate_supervision",
                "deformable_groups",
                "mdconv_dilation",
                "refinement_type",
                "model_path",
            ]
        ]


class AANetStereoMatcher(StereoMatcherBase):
    def __init__(
        self,
        config,
        device,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=StereoMatcherBase.IMAGENET_MEAN,
                    std=StereoMatcherBase.IMAGENET_STD,
                ),
            ]
        ),
    ):

        StereoMatcherBase.__init__(self, config=config, device=device, transform=transform)

        self._model = nets.AANet(
            config.max_disp,
            num_downsample=config.num_downsample,
            feature_type=config.feature_type,
            no_feature_mdconv=config.no_feature_mdconv,
            feature_pyramid=config.feature_pyramid,
            feature_pyramid_network=config.feature_pyramid_network,
            feature_similarity=config.feature_similarity,
            aggregation_type=config.aggregation_type,
            num_scales=config.num_scales,
            num_fusions=config.num_fusions,
            num_stage_blocks=config.num_stage_blocks,
            num_deform_blocks=config.num_deform_blocks,
            no_intermediate_supervision=config.no_intermediate_supervision,
            refinement_type=config.refinement_type,
            mdconv_dilation=config.mdconv_dilation,
            deformable_groups=config.deformable_groups,
        )

        self._model = self._model.to(device)
        utils.load_pretrained_net(self._model, config.model_path, no_strict=True)

        self._model.eval()

    def run(self, left_rectified_image, right_rectified_image):
        import numpy as np
        import cv2

        left_rectified_image = cv2.cvtColor(left_rectified_image, cv2.COLOR_BGR2RGB)
        right_rectified_image = cv2.cvtColor(right_rectified_image, cv2.COLOR_BGR2RGB)
        left_rectified_image = left_rectified_image.astype(np.float32)
        right_rectified_image = right_rectified_image.astype(np.float32)

        sample = {"left": left_rectified_image, "right": right_rectified_image}
        sample = self._transform(sample)
        left = sample["left"].to(self._device)  # [3, H, W]
        left = left.unsqueeze(0)  # [1, 3, H, W]
        right = sample["right"].to(self._device)
        right = right.unsqueeze(0)

        ori_height, ori_width = left.size()[2:]
        factor = 48 if self._config.refinement_type != "hourglass" else 96

        img_height = math.ceil(ori_height / factor) * factor
        img_width = math.ceil(ori_width / factor) * factor

        if ori_height < img_height or ori_width < img_width:
            top_pad = img_height - ori_height
            right_pad = img_width - ori_width

            # Pad size: (left_pad, right_pad, top_pad, bottom_pad)
            left = F.pad(left, (0, right_pad, top_pad, 0))
            right = F.pad(right, (0, right_pad, top_pad, 0))

        with torch.no_grad():
            pred_disp = self._model(left, right)[-1]  # [B, H, W]

        if pred_disp.size(-1) < left.size(-1):
            pred_disp = pred_disp.unsqueeze(1)  # [B, 1, H, W]
            pred_disp = F.interpolate(pred_disp, (left.size(-2), left.size(-1)), mode="bilinear") * (
                left.size(-1) / pred_disp.size(-1)
            )
            pred_disp = pred_disp.squeeze(1)  # [B, H, W]

        # Crop
        if ori_height < img_height or ori_width < img_width:
            if right_pad != 0:
                pred_disp = pred_disp[:, top_pad:, :-right_pad]
            else:
                pred_disp = pred_disp[:, top_pad:]

        disp = pred_disp[0].detach().cpu().numpy()  # [H, W]

        return disp.astype(np.float32)
