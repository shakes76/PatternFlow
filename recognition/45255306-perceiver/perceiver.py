import torch
import numpy as np
from torch import nn, einsum
from einops import rearrange, repeat
from math import pi

"""
Positional Encoding Method for data encoding
"""
def fourier_encode(x, max_freq, num_bands=4):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x

    scales = torch.linspace(1., max_freq/2, num_bands, device=device, dtype=dtype)
    scales = torch.reshape(scales, (1, ) * (len(x.shape) - 1) + (len(scales), ))

    x = x * scales * pi
    x = torch.cat([x.sin(), x.cos()], dim = -1)
    x = torch.cat((x, orig_x), dim=-1)
    return x

"""
Feed forward network after the attention layer
"""
class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            # optional, but it is shown to degrade performance in the paper
            # nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)


"""
Attention block, can be used for cross attention and self attention
"""
class Attention(nn.Module):

    # query dimension for cross attention purposes
    def __init__(self, q_dim, context_dim=None, heads=8, dim_head=8, dropout=0.0):
        super().__init__()

        # self attention
        if context_dim is None:
            context_dim = q_dim

        # scaling for attention based on the formula sqrt(dimension)
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.query = nn.Linear(q_dim, dim_head * heads, bias=False)
        # taking account for crossattention, input_dim == byte array dim
        self.key = nn.Linear(context_dim, dim_head * heads, bias=False)
        self.val = nn.Linear(context_dim, dim_head * heads, bias=False)

        self.out = nn.Sequential(
            # for cross attention results to latent_dim/query_dim
            nn.Linear(dim_head * heads, q_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None):
        # self attention
        if context is None:
            context = x

        # forward pass q(x), (k, v)(context)
        # for cross attention purposes, context is input data, x is latent
        q, k, v = self.query(x), self.key(context), self.val(context)
        q, k, v = map(lambda x: rearrange(x, 'b n (h d) -> (b h) n d', h=self.heads), (q, k, v))

        attention = einsum("b q d, b k d -> b q k", q, k) * self.scale
        # the same dimension is value and key, therefore perform matrix 
        # multiplication on k, v dimensions
        # softmax(qk/sqrt(dimension)) * v
        attention = einsum("b q k, b k d -> b q d",
                           torch.softmax(attention, dim=-1), v)
        
        # rearrange to match the input
        out = rearrange(attention, "(b h) n d -> b n (h d)", h=self.heads)
        return self.out(out)

"""
Self attention latent block
"""
class LatentBlock(nn.Module):

    def __init__(self, latent_dim, heads, dim_head, dropout):
        super().__init__()
        self.latent_attn = Attention(latent_dim, heads=heads, 
                                     dim_head=dim_head, dropout=dropout)
        
        # pre layer norm input data
        self.latent_norm = nn.LayerNorm(latent_dim)

    def forward(self, x):
        x = self.latent_norm(x)
        return self.latent_attn(x)


"""
Cross attention block
"""
class CrossAttentionBlock(nn.Module):

    def __init__(self, latent_dim, input_dim, heads, dim_head, dropout):
        super().__init__()
        self.cross_attn = Attention(latent_dim, input_dim, heads=heads, 
                               dim_head=dim_head, dropout=dropout)
        
        # pre layer norm input data
        self.latent_norm = nn.LayerNorm(latent_dim)
        self.norm_context = nn.LayerNorm(input_dim)

    def forward(self, x, context):
        x = self.latent_norm(x)
        context = self.norm_context(context)
        return self.cross_attn(x, context)


"""
Perceiver Model
"""
class Perceiver(nn.Module):

    def __init__(self, num_freq_bands, max_freq, depth, input_channels=3, input_axis=2,
                 num_latents=512, latent_dim=512, cross_heads=1, latent_heads=8,
                 cross_dim_head=64, latent_dim_head=64, num_classes=2, 
                 attention_dropout=0.):
        super().__init__()

        self.max_freq = max_freq
        self.num_freq_bands = num_freq_bands

        # assumptions is that all data uses fourier encoding
        input_dim = (input_axis * ((num_freq_bands * 2) + 1)) + input_channels
        
        # the paper mention using truncated normal distribution with 
        # mean 0, sd 0.02, with truncated bounds [-2, 2]
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))

        # attentions blocks
        cross_attn = CrossAttentionBlock(latent_dim, input_dim, cross_heads, 
                               cross_dim_head, attention_dropout)
        latent_attn = LatentBlock(latent_dim, latent_heads, 
                                latent_dim_head, attention_dropout)
        
        # feed forward layers
        cross_ff = FeedForward(latent_dim)
        latent_ff = FeedForward(latent_dim)

        self.layers = nn.ModuleList([])
        for i in range(depth):
            self.layers.append(nn.ModuleList([
                cross_attn,
                cross_ff,
                latent_attn,
                latent_ff
            ]))

        self.head = nn.Sequential(
            # before normalizing, get the mean latent value, this is 
            # based on the paper
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, num_classes)
        )

    def forward(self, data):
        batch_size, _, *axis, device = *data.shape, data.device

        # positional encodings
        # using fourier encodings
        axis_pos = list(map(lambda size: torch.linspace(-1., 1., steps=size, device=device), axis))
        grid = torch.stack(torch.meshgrid(*axis_pos), dim=-1)
        enc_pos = fourier_encode(grid, self.max_freq, self.num_freq_bands)
        enc_pos = rearrange(enc_pos, '... n d -> (n d) ...')
        enc_pos = repeat(enc_pos, '... -> b ...', b=batch_size)
        data = torch.cat((data, enc_pos), dim=1)
        data = rearrange(data, 'b d ... -> b (...) d')
        # context / latent
        latent = repeat(self.latents, 'n d -> b n d', b=batch_size)

        for cross_attn, cross_ff, latent_attn, latent_ff in self.layers:
            x = cross_attn(latent, context=data)
            x = cross_ff(x) + x
            x = latent_attn(x)
            x = latent_ff(x)
        return self.head(torch.mean(x, dim=1))