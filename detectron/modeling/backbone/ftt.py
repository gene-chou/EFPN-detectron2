import logging
import numpy as np
import fvcore.nn.weight_init as weight_init
import torch
import torch.nn.functional as F
from torch import nn

from detectron2.layers import Conv2d, ShapeSpec, get_norm

import math
import fvcore.nn.weight_init as weight_init
import torch.nn.functional as F
from torch import nn

from detectron2.layers import Conv2d, ShapeSpec, get_norm

from .backbone import Backbone
from .build import BACKBONE_REGISTRY
from .resnet import build_resnet_backbone


# p2, p3 in the paper is p3, p4 for us 
# format of p2, p3 is both [bs, channels, height, width]
def FTT_get_p3pr(p2, p3, out_channels, norm):
    channel_scaler = Conv2d(
        out_channels,
        out_channels * 4,
        kernel_size=1,
        bias=False
        #norm=''
    )

    # tuple of (conv2d, conv2d, iter)
    def create_convs(num_channels, iter=3):
        conv1 = Conv2d(
        num_channels,
        num_channels,
        kernel_size=1,
        bias=False,
        #norm=get_norm(norm, num_channels),
        )

        conv2 = Conv2d(
        num_channels,
        num_channels,
        kernel_size=1,
        bias=False,
        #norm=get_norm(norm, num_channels),
        )
        return (conv1, conv2, iter)

    content_extractor = create_convs(out_channels * 4)
    texture_extractor = create_convs(out_channels * 2)

    # subpixelconv is replaced by nn.PixelShuffle 
    # def sub_pixel_conv(x, r=2):
    #     C, H, W = x.shape[1:]
    #     print ("shape: ",x.shape)
    #     # assert C == self.in_channels
    #     output = torch.zeros((x.shape[0], int(C/r**2), r*H, r*W))

    #     for bs in range(x.shape[0]):
    #         for c in range(int(C/r**2)):
    #             values = x[bs][c*r*r:(c+1)*r*r] #shape would be [4, height, width]
    #             #output[bs][c][i:i + r][j:j + r] = torch.reshape(values, (r,r))
    #             output[bs][c] = torch.reshape(values, (r,r))
    #     return output


    def extractor_helper(extractor, x): # extractor is tuple of (conv2D, cov2D, int)
        def each_iter(x):
            out = extractor[0](x)
            out = F.relu_(out)
            out = extractor[1](out)
            return out

        out = x
        for i in range(extractor[2]):
            out = each_iter(out)
        return out


    bottom = p3
    bottom = channel_scaler(bottom)
    bottom = extractor_helper(content_extractor, bottom)
    sub_pixel_conv = nn.PixelShuffle(2)
    bottom = sub_pixel_conv(bottom)
    print("\np3 shape: ",bottom.shape,"\n")

    # We interpreted "wrap" as concatenating bottom and top
    # so the total channels is doubled after (basically place one on top
    # of the other)
    top = p2
    top = torch.cat((bottom, top), axis=1)
    top = extractor_helper(texture_extractor, top)
    print("\np2 shape: ",top.shape,"\n")

    # Since top has double the original # of channels, we "cast" bottom
    # to the same shape, then add to get p3'
    bottom = torch.cat((bottom, bottom), axis=1)
    result = bottom + top
    print("\nftt final result shape: ",result.shape,"\n")

    return result
