import logging
import numpy as np
import fvcore.nn.weight_init as weight_init
import torch
import torch.nn.functional as F
from torch import nn

from detectron2.layers import Conv2d, ShapeSpec, get_norm

from fpn import FPN

class FTT(FPN):
    """
        in_features should be p2, p3
    """
    def __init__(
        self, fpn, in_features, out_channels, norm = "BN"
    ):
        assert isinstance(fpn, FPN)
        assert in_features == ['p2', 'p3']
        input_shapes = fpn.output_shape()
        # in_channels_per_feature = [input_shapes[f].channels for f in in_features]

        # Apply before content extractor to scale up channels from C to 4C
        self.channel_scaler = Conv2d(
            input_shapes['p2'].channels,
            input_shapes['p3'].channels,
            kernel_size=1,
            bias=False,
            norm=''
        )
        self.content_extractor = #
        self.texture_extractor = #
        self.sub_pixel_conv = #

    def forward(self, x):
        

# Content and Texture Extractor
class Extractor():
    # in and out channels are the same, so for context extractor we'll conv2D
    # before to up it from C to 4C channels
    def __init__(self, num_channels, iterations, norm):
        self.iterations = iterations
        self.conv1 = Conv2d(
            num_channels,
            num_channels,
            kernel_size=1,
            bias=False,
            norm=get_norm(norm, num_channels),
        )
        self.conv2 = Conv2d(
            num_channels,
            num_channels,
            kernel_size=1,
            bias=False,
            norm=get_norm(norm, num_channels),
        )
    
    def _forward_one(self, x):
        out = self.conv1(x)
        out = F.relu_(out)
        out = self.conv2(out)
        return out

    def forward(self, x):
        out = x
        for i in range(self.iterations):
            out = self._forward_one(out)
        return out