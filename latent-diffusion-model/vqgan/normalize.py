from torch import nn


def Normalize(in_channels, norm_type="group", num_groups=32):
    assert norm_type in ["group", "batch"]

    if norm_type == "batch":
        return nn.SyncBatchNorm(in_channels)

    return nn.GroupNorm(
        num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True
    )
