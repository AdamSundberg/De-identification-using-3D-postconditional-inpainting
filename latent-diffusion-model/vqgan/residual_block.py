from .normalize import Normalize
from .same_pad_conv_3d import SamePadConv3D
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=None,
        kernel_size=3,
        norm_type="group",
        padding_type="replicate",
        num_groups=32,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels is not None else in_channels

        self.conv = nn.Sequential(
            Normalize(self.in_channels, norm_type, num_groups=num_groups),
            nn.SiLU(inplace=True),
            SamePadConv3D(
                self.in_channels,
                self.out_channels,
                kernel_size=kernel_size,
                padding_type=padding_type,
            ),
            Normalize(self.out_channels, norm_type, num_groups=num_groups),
            nn.SiLU(inplace=True),
            SamePadConv3D(
                self.out_channels,
                self.out_channels,
                kernel_size=kernel_size,
                padding_type=padding_type,
            ),
        )

        if self.in_channels != self.out_channels:
            self.conv_shortcut = SamePadConv3D(
                self.in_channels,
                self.out_channels,
                kernel_size=kernel_size,
                padding_type=padding_type,
            )

    def forward(self, x):
        h = self.conv(x)

        if self.in_channels != self.out_channels:
            x = self.conv_shortcut(x)

        return x + h
