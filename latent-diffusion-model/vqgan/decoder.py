import math

import numpy as np
from .normalize import Normalize
from .residual_block import ResidualBlock
from .same_pad_conv_3d import SamePadConv3D, SamePadConvTranspose3D
from torch import nn


class Decoder(nn.Module):
    def __init__(
        self,
        number_of_hiddens,
        upsample,
        image_channel,
        norm_type="group",
        num_groups=32,
    ):
        super().__init__()

        number_of_upsamples = np.array([int(math.log2(u)) for u in upsample])
        max_upsample = number_of_upsamples.max()

        in_channels = number_of_hiddens * 2**max_upsample

        self.conv = nn.Sequential(
            Normalize(in_channels, norm_type, num_groups=num_groups), nn.SiLU()
        )

        for i in range(max_upsample):
            in_channels = (
                in_channels
                if i == 0
                else number_of_hiddens * 2 ** (max_upsample - i + 1)
            )
            out_channels = number_of_hiddens * 2 ** (max_upsample - i)
            stride = tuple([2 if u > 0 else 1 for u in number_of_upsamples])

            self.conv.append(
                SamePadConvTranspose3D(in_channels, out_channels, 4, stride=stride)  # type: ignore
            )
            self.conv.append(
                ResidualBlock(
                    out_channels,
                    out_channels,
                    norm_type=norm_type,
                    num_groups=num_groups,
                )
            )
            self.conv.append(
                ResidualBlock(
                    out_channels,
                    out_channels,
                    norm_type=norm_type,
                    num_groups=num_groups,
                )
            )

            number_of_upsamples -= 1

        self.conv.append(SamePadConv3D(out_channels, image_channel, kernel_size=3))  # type: ignore

    def forward(self, x):
        return self.conv(x)
