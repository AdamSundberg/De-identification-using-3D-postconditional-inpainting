import numpy as np
from torch import nn
from torch.nn.modules import BatchNorm2d


class NLayerDiscriminator(nn.Module):
    def __init__(
        self,
        input_channels,
        conv_layer: type[nn.Conv2d] | type[nn.Conv3d],
        discriminator_channels=64,
        number_of_layers=3,
        norm_layer: type[nn.BatchNorm2d] | type[nn.BatchNorm3d] = nn.BatchNorm2d,
    ):
        super().__init__()

        self.number_of_layers = number_of_layers

        kernel_size = 4
        padding = int(np.ceil((kernel_size - 1.0) / 2))

        self.sequence = nn.Sequential()

        # Block 1
        self.sequence.append(
            nn.Sequential(
                conv_layer(
                    input_channels,
                    discriminator_channels,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=padding,
                ),
                nn.LeakyReLU(0.2, True),
            )
        )

        # Block 2
        nf = discriminator_channels
        for n in range(0, self.number_of_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)

            # stride = 1 if last loop else 2
            stride = 1 if n == self.number_of_layers - 1 else 2
            self.sequence.append(
                nn.Sequential(
                    conv_layer(
                        nf_prev,
                        nf,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                    ),
                    norm_layer(nf),
                    nn.LeakyReLU(0.2, True),
                )
            )

        # Block 3
        self.sequence.append(
            conv_layer(nf, 1, kernel_size=kernel_size, stride=1, padding=padding)
        )

    def forward(self, input):
        result = [input]

        for model in self.sequence:
            result.append(model(result[-1]))

        return result[-1], result[1:]


class NLayerDiscriminator2D(NLayerDiscriminator):
    def __init__(
        self,
        input_channels,
        discriminator_channels=64,
        number_of_layers=3,
        norm_layer: type[nn.BatchNorm2d] | type[nn.BatchNorm3d] = nn.BatchNorm2d,
    ) -> None:
        super().__init__(
            input_channels,
            nn.Conv2d,
            discriminator_channels=discriminator_channels,
            number_of_layers=number_of_layers,
            norm_layer=norm_layer,
        )


class NLayerDiscriminator3D(NLayerDiscriminator):
    def __init__(
        self,
        input_channels,
        discriminator_channels=64,
        number_of_layers=3,
        norm_layer: type[nn.BatchNorm2d] | type[nn.BatchNorm3d] = nn.BatchNorm2d,
    ) -> None:
        super().__init__(
            input_channels,
            nn.Conv3d,
            discriminator_channels=discriminator_channels,
            number_of_layers=number_of_layers,
            norm_layer=norm_layer,
        )
