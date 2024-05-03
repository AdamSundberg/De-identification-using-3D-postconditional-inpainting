import math
from functools import partial

import torch
from einops import rearrange
from einops_exts import rearrange_many
from .residual_block import ResidualBlock
from rotary_embedding_torch import RotaryEmbedding
from torch import einsum, nn
from .utils import default, prob_mask_like


class EinopsToAndFrom(nn.Module):
    def __init__(self, from_einops, to_einops, fn):
        super().__init__()
        self.from_einops = from_einops
        self.to_einops = to_einops
        self.fn = fn

    def forward(self, x, **kwargs):
        shape = x.shape
        reconstitute_kwargs = dict(tuple(zip(self.from_einops.split(" "), shape)))
        x = rearrange(x, f"{self.from_einops} -> {self.to_einops}")
        x = self.fn(x, **kwargs)
        x = rearrange(
            x, f"{self.to_einops} -> {self.from_einops}", **reconstitute_kwargs
        )
        return x


class Attention(nn.Module):
    def __init__(self, dimension, heads=4, head_dimension=32, rotary_embedding=None):
        super().__init__()

        self.scale = head_dimension**-0.5
        self.heads = heads
        hidden_dimension = head_dimension * heads

        self.rotary_embedding = rotary_embedding
        self.to_qkv = nn.Linear(dimension, hidden_dimension * 3, bias=False)
        self.to_out = nn.Linear(hidden_dimension, dimension, bias=False)

    def forward(self, x, pos_bias=None, focus_present_mask=None):
        n, device = x.shape[-2], x.device

        qkv = self.to_qkv(x).chunk(3, dim=-1)

        if focus_present_mask is not None and focus_present_mask.all():
            # if all batch samples are focusing on present
            # it would be equivalent to passing that token's values through to the output
            return self.to_out(qkv[-1])

        # Split out heads
        q, k, v = rearrange_many(qkv, "... n (h d) -> ... h n d", h=self.heads)

        # Scale
        q = q * self.scale

        # Rotate positions into queries and keys for time attention
        if self.rotary_embedding is not None:
            q = self.rotary_embedding.rotate_queries_or_keys(q)
            k = self.rotary_embedding.rotate_queries_or_keys(k)

        # Similarity
        similarity = einsum("... h i d, ... h j d -> ... h i j", q, k)

        # Relative positional bias
        if pos_bias is not None:
            similarity = similarity + pos_bias

        if focus_present_mask is not None and not (~focus_present_mask).all():
            attend_all_mask = torch.ones((n, n), device=device, dtype=torch.bool)
            attend_self_mask = torch.eye(n, device=device, dtype=torch.bool)

            mask = torch.where(
                rearrange(focus_present_mask, "b -> b 1 1 1 1"),
                rearrange(attend_self_mask, "i j -> 1 1 1 i j"),
                rearrange(attend_all_mask, "i j -> 1 1 1 i j"),
            )

            similarity = similarity.masked_fill(
                ~mask, -torch.finfo(similarity.dtype).max
            )

        # Numerical stability
        similarity = similarity - similarity.amax(dim=-1, keepdim=True).detach()
        attn = similarity.softmax(dim=-1)

        # Aggregate values
        out = einsum("... h i j, ... h j d -> ... h i d", attn, v)
        out = rearrange(out, "... h n d -> ... n (h d)")
        return self.to_out(out)


class SpatialLinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, f, h, w = x.shape
        x = rearrange(x, "b c f h w -> (b f) c h w")

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = rearrange_many(qkv, "b (h c) x y -> b h c (x y)", h=self.heads)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = einsum("b h d n, b h e n -> b h d e", k, v)

        out = einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        out = self.to_out(out)
        return rearrange(out, "(b f) c h w -> b c f h w", b=b)


class RelativePositionBias(nn.Module):
    def __init__(self, heads=8, num_buckets=32, max_distance=128):
        super().__init__()

        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    def relative_position_buckets(self, relative_position):
        result = 0
        n = -relative_position

        num_buckets = self.num_buckets // 2
        result += (n < 0).long() * num_buckets
        n = torch.abs(n)

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = (
            max_exact
            + (
                torch.log(n.float() / max_exact)
                / math.log(self.max_distance / max_exact)
                * (num_buckets - max_exact)
            ).long()
        )
        val_if_large = torch.min(
            val_if_large, torch.full_like(val_if_large, num_buckets - 1)
        )

        result += torch.where(is_small, n, val_if_large)
        return result

    def forward(self, n, device):
        q_pos = torch.arange(n, dtype=torch.long, device=device)
        k_pos = torch.arange(n, dtype=torch.long, device=device)

        rel_pos = rearrange(k_pos, "j -> 1 j") - rearrange(q_pos, "i -> i 1")
        rp_bucket = self.relative_position_buckets(rel_pos)

        values = self.relative_attention_bias(rp_bucket)
        return rearrange(values, "i j h -> h i j")


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.gamma


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Unet3D(nn.Module):
    def __init__(
        self,
        dim,
        cond_dim=None,
        out_dim=None,
        dimension_mults=(1, 2, 4, 8),
        channels=3,
        attn_heads=8,
        attn_dim_head=32,
        use_bert_text_cond=False,
        init_dim=None,
        init_kernel_size=7,
        use_sparse_linear_attn=True,
        resnet_groups=8,
    ):
        super().__init__()

        self.channels = channels

        rotary_embedding = RotaryEmbedding(min(32, attn_dim_head))

        def temporal_attn(dim):
            return EinopsToAndFrom(
                "b c f h w",
                "b (h w) f c",
                Attention(
                    dim,
                    heads=attn_heads,
                    head_dimension=attn_dim_head,
                    rotary_embedding=rotary_embedding,
                ),
            )

        self.time_rel_pos_bias = RelativePositionBias(heads=attn_heads, max_distance=32)

        # Initial conv
        init_dim = default(init_dim, dim)
        assert (init_kernel_size % 2) == 1

        init_padding = init_kernel_size // 2
        self.init_conv = nn.Conv3d(
            channels,
            init_dim,
            (1, init_kernel_size, init_kernel_size),
            padding=(0, init_padding, init_padding),
        )

        self.init_temporal_attn = Residual(PreNorm(init_dim, temporal_attn(init_dim)))

        # Dimensions
        dims = [init_dim, *map(lambda m: dim * m, dimension_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # Time conditioning
        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        self.has_cond = cond_dim is not None or use_bert_text_cond

        cond_dim = time_dim + int(cond_dim or 0)

        self.null_cond_emb = nn.Parameter(
            torch.randn(1, cond_dim)) if self.has_cond else None

        # Layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        # Block type
        block_klass = partial(ResidualBlock, groups=resnet_groups)
        block_klass_cond = partial(block_klass, time_emb_dim=cond_dim)

        # Modules for all layers
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass_cond(dim_in, dim_out),
                        block_klass_cond(dim_out, dim_out),
                        (
                            Residual(
                                PreNorm(
                                    dim_out,
                                    SpatialLinearAttention(dim_out, heads=attn_heads),
                                )
                            )
                            if use_sparse_linear_attn
                            else nn.Identity()
                        ),
                        Residual(PreNorm(dim_out, temporal_attn(dim_out))),
                        (
                            nn.Conv3d(dim_out, dim_out, (1, 4, 4), (1, 2, 2), (0, 1, 1))
                            if not is_last
                            else nn.Identity()
                        ),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass_cond(mid_dim, mid_dim)

        spatial_attn = EinopsToAndFrom(
            "b c f h w", "b f (h w) c", Attention(mid_dim, heads=attn_heads)
        )

        self.mid_spatial_attn = Residual(PreNorm(mid_dim, spatial_attn))
        self.mid_temporal_attn = Residual(PreNorm(mid_dim, temporal_attn(mid_dim)))

        self.mid_block2 = block_klass_cond(mid_dim, mid_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass_cond(dim_out * 2, dim_in),
                        block_klass_cond(dim_in, dim_in),
                        (
                            Residual(
                                PreNorm(
                                    dim_in,
                                    SpatialLinearAttention(dim_in, heads=attn_heads),
                                )
                            )
                            if use_sparse_linear_attn
                            else nn.Identity()
                        ),
                        Residual(PreNorm(dim_in, temporal_attn(dim_in))),
                        (
                            nn.ConvTranspose3d(
                                dim_in, dim_in, (1, 4, 4), (1, 2, 2), (0, 1, 1)
                            )
                            if not is_last
                            else nn.Identity()
                        ),
                    ]
                )
            )

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            block_klass(dim * 2, dim), nn.Conv3d(dim, out_dim, 1)
        )

    def forward_with_cond_scale(self, *args, cond_scale=2, **kwargs):
        logits = self.forward(*args, null_cond_prob=0, **kwargs)

        if cond_scale == 1 or not self.has_cond:
            return logits

        null_logits = self.forward(*args, null_cond_prob=1, **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(
        self,
        x,
        time,
        cond=None,
        null_cond_prob=0,
        focus_present_mask=None,
        prob_focus_present=0,
    ):
        assert not (
            self.has_cond and not cond is not None
        ), "Cond must be passed in if cond_dim is specified."

        batch, device = x.shape[0], x.device

        focus_present_mask = default(
            focus_present_mask,
            lambda: prob_mask_like((batch,), prob_focus_present, device=device),
        )

        time_rel_pos_bias = self.time_rel_pos_bias(x.shape[2], device=device)

        x = self.init_conv(x)
        r = x.clone()

        x = self.init_temporal_attn(x, pos_bias=time_rel_pos_bias)
        t = self.time_mlp(time)

        # Classifier free guidance
        if self.has_cond and self.null_cond_emb and cond:
            batch, device = x.shape[0], x.device
            mask = prob_mask_like((batch, ), null_cond_prob, device=device)
            cond = torch.where(rearrange(mask, "b -> b 1"), self.null_cond_emb, cond)
            t = torch.cat((t, cond), dim=-1)

        h = []

        for block1, block2, spatial_attn, temporal_attn, downsample in self.downs:  # type: ignore
            x = block1(x, t)
            x = block2(x, t)
            x = spatial_attn(x)
            x = temporal_attn(x, pos_bias=time_rel_pos_bias, focus_present_mask=focus_present_mask)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_spatial_attn(x)
        x = self.mid_temporal_attn(x, pos_bias=time_rel_pos_bias, focus_present_mask=focus_present_mask)
        x = self.mid_block2(x, t)
        
        for block1, block2, spatial_attn, temporal_attn, upsample in self.ups:  # type: ignore
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = spatial_attn(x)
            x = temporal_attn(x, pos_bias=time_rel_pos_bias, focus_present_mask=focus_present_mask)
            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        return self.final_conv(x)



            







