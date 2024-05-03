from torch import nn
from einops import rearrange


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, groups=8):
        super().__init__()

        self.conv = nn.Conv3d(in_channels, out_channels, (1, 3, 3), padding=(0, 1, 1))
        self.norm = nn.GroupNorm(groups, out_channels)
        self.activation = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.conv(x)
        x = self.norm(x)

        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        return self.activation(x)


class ResidualBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, time_emb_dim=None, groups=8
    ):
        super().__init__()

        self.mlp = (
            nn.Sequential(
                nn.SiLU(), nn.Linear(time_emb_dim, out_channels * 2)
            )
            if time_emb_dim is not None
            else None
        )

        self.block1 = Block(in_channels, out_channels, groups=groups)
        self.block2 = Block(out_channels, out_channels, groups=groups)
        self.residual_conv = (
            nn.Conv3d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, time_embedding=None):
        scale_shift = None
        if self.mlp:
            assert time_embedding is not None, "Time embeddint must be passed in."

            time_embedding = self.mlp(time_embedding)
            time_embedding = rearrange(time_embedding, "b c -> b c 1 1 1")
            scale_shift = time_embedding.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)

        return h + self.residual_conv(x)






