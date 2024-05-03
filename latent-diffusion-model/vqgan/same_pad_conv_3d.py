import torch.nn.functional as F
from torch import nn


class SamePadConv3D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        bias=True,
        padding_type="replicate",
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3

        if isinstance(stride, int):
            stride = (stride,) * 3

        total_padding = tuple([k - s for k, s in zip(kernel_size, stride)])
        self.padding_input = []
        for padding in total_padding[::-1]:  # Reverse since F.pad starts from last dim
            self.padding_input.append((padding // 2 + padding % 2, padding // 2))

        self.padding_input = sum(self.padding_input, tuple())
        self.padding_type = padding_type

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=0, bias=bias)  # type: ignore

    def forward(self, x):
        return self.conv(F.pad(x, self.padding_input, mode=self.padding_type))


class SamePadConvTranspose3D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        bias=True,
        padding_type="replicate",
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3

        if isinstance(stride, int):
            stride = (stride,) * 3

        total_padding = tuple([k - s for k, s in zip(kernel_size, stride)])
        self.padding_input = []
        for padding in total_padding[::-1]:  # Reverse since F.pad starts from last dim
            self.padding_input.append((padding // 2 + padding % 2, padding // 2))

        self.padding_input = sum(self.padding_input, tuple())
        self.padding_type = padding_type

        self.convt = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,  # type: ignore
            stride=stride,  # type: ignore
            bias=bias,
            padding=tuple([k - 1 for k in kernel_size]),  # type: ignore
        )

    def forward(self, x):
        return self.convt(F.pad(x, self.padding_input, mode=self.padding_type))
