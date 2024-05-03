import math

import numpy as np
from .normalize import Normalize
from .residual_block import ResidualBlock
from .same_pad_conv_3d import SamePadConv3D
from torch import nn


class Encoder(nn.Module):
    def __init__(
        self,
        number_of_hiddens,
        downsample,
        image_channel,
        norm_type="group",
        padding_type="replicate",
        num_groups=32,
    ):
        super().__init__()

        number_of_downsamples = np.array([int(math.log2(d)) for d in downsample])

        self.conv = nn.Sequential(
            SamePadConv3D(
                image_channel,
                number_of_hiddens,
                kernel_size=3,
                padding_type=padding_type,
            )
        )

        out_channels = 0
        for i in range(number_of_downsamples.max()):
            in_channels = number_of_hiddens * 2**i
            out_channels = number_of_hiddens * 2 ** (i + 1)
            stride = tuple([2 if d > 0 else 1 for d in number_of_downsamples])

            self.conv.append(
                SamePadConv3D(
                    in_channels,
                    out_channels,
                    4,
                    stride=stride,  # type: ignore
                    padding_type=padding_type,
                )
            )
            self.conv.append(
                ResidualBlock(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    norm_type=norm_type,
                    num_groups=num_groups,
                )
            )

            number_of_downsamples -= 1
        
        self.out_channels = out_channels

        self.conv.append(Normalize(out_channels, norm_type, num_groups=num_groups))  # type: ignore
        self.conv.append(nn.SiLU())

    def forward(self, x):
        return self.conv(x)
