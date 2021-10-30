#!/usr/bin/env python3

# pi/log for Fourier encoding
from math import pi, log

import torch
import torch.nn as nn


"""
    Unofficial implementation of the Perceiver Transformer model by Deepmind.

    The perceiver consists of two components:
        - The cross attention layer
        - The latent transformer layer

    The components are alternated one after another.
"""

__author__ = "Chegne Eu Joe"
__email__ = "e.chegne@uqconnect.edu.au"


"""
def fourier_encode(x, max_freq, num_bands=4):
    Allows parameterized fourier feature positional encodings which:

    1. Directly represents the position structure of the input data (To compensate for the lack of explicit grid structures).
    2. Control the number of frequency bands in position encoding independent of the cutoff frequency.
    3. Uniformly sample all frequencies up to a target resolution.

    Args:
        x - input
        max_freq - maximmum frequency
        num_bands - size of constructed tensor
        base - base of logarithm function

    Returns:
        Fourier encoded input x.

    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x

    scales = torch.logspace(
        start=1.0,
        log(max_freq / 2) / log(10),
        device=device,
        dtype=dtype,
    )

    scales = scales[(*((None,) * len(x.shape) - 1)), Ellipsis)]
    x = x * scales * pi
    x = torch.cat([x.sin(), x.cos()], dim=1)
    x= torch.cat((x, orig_x), dim=-1)

    return x

"""


class PreNorm(nn.Module):
    """
    A wrapper used to normalize values before each procedure using LayerNorm.

    Args:
        dim - input dimension.
        fn - Layer to be applied post normalization.
        context_dim - Normalizes the context dimension (of fn) if available. Used for cross attention layers.
    Returns:
        Layer with normalized values
    """

    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        if context_dim is not None:
            self.norm_context = nn.LayerNorm(context_dim)
        else:
            self.norm_context = None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if self.norm_context is not None:
            context = kwargs["x_kv"]
            normed_context = self.norm_context(context)
            kwargs.update(context=normed_context)

        return self.fn(x, **kwargs)


class Attention(nn.Module):
    """
    Allows the model to jointly attend to information from different
    representation subspaces.

    Wrapper for {self/cross}-attention layers which will be further
    elaborated in the respective classes

    See https://arxiv.org/abs/1706.03762 for more details.

    Args:
        q_channels - Total dimension of model.
        kv_channels - Total number of features of keys and values.
        num_heads - Number of parallel attention heads.
        dropout - Dropout probability.
    Returns:
        Attention outputs of shape (N, L, E) where L is the target
        sequence length, N is the batch size, and E is embed_dim
    """

    def __init__(self, q_channels, kv_channels, num_heads=8, dropout=0.0):
        super().__init__()
        self.attention = nn.MultiHeadAttention(
            embed_dim=q_channels,
            num_heads=num_heads,
            kdim=kv_channels,
            v_dim=kv_channels,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, query, key_value, key_padding_mask=None, attn_mask=None):
        attn_output, _ = self.attention(
            query,
            key_value,
            key_value,
            key_padding_mask=key_padding_mask,
            attn_mask=attend_mask,
        )
        return attn_output


class SelfAttention(nn.Module):
    """
    For each input vector v_i, compute the output o_i by taking the
    weighted sum of all non-query input vectors, where the weight is proportional
    exp(v_i * v_j / sqrt(d)), dot product.

    This allows us to measure similarity between two vectors. The higher the weight,
    the more similar the input sequence is compared to the query and vice versa.

    See https://arxiv.org/abs/1706.03762 for more details.
    """

    def __init__(self, num_channels, num_heads=8, dropout=0.0):
        super().__init__()

        self.norm = nn.LayerNorm(num_channels)
        self.attention = PreNorm(
            num_channels, Attention(num_channels, num_channels, num_heads, dropout)
        )

    def forward(self, x, key_padding_mask=None, attn_mask=None):
        return self.attention(
            x, x, key_padding_mask=key_padding_mask, attn_mask=attn_mask
        )


class CrossAttention(nn.Module):
    """
    Similar to self attention. However in cross attention,
    queries are generated from a separate embedding to the keys and values.
    Hence the name cross attention.

    In this implementation, embed_dim is determined by q_channels for simplification.
    A non-simplified CrossAttention layer will require building the attention layer
    from the ground up, which takes too much time, so I'll leave it as a TODO.

    See https://arxiv.org/abs/1706.03762 for more details.
    """

    def __init__(self, q_channels, kv_channels, num_heads=8, dropout=0.0):
        super().__init__()
        self.attention = PreNorm(
            q_channels,
            Attention(num_channels, num_channels, num_heads, dropout),
            context_dim=kv_channels,
        )

    def forward(self, x_q, x_kv, key_padding_mask=None, attn_mask=None):
        return self.attention(
            x, x, key_padding_mask=key_padding_mask, attn_mask=attn_mask
        )


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class MLP(nn.Module):
    def __init__(self, num_channels, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(num_channels),
            nn.Linear(num_channels, num_channels),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(num_channels, num_channels),
        )


class Perceiver(nn.Module):
    """
        A scalable, fully attentional architecture.
        Note that data has to be pre-fourier encoded or it will not work.

        Args:
            depth - The depth of the network. See code for more information.
            num_channels - Number of channels.
            num_latents - Number of latents.
            latent_dim - Latent dimension.
            cross_heads - Number of heads for cross attention.
            latent_heads - Number of heads for self attention.
            num_classes - Output number of classes.
            attn_dropout - Attention dropout probability.
            ff_dropout - MLP dropout probability.
        Returns:
            Perceiver layer
    """

    def __init__(
        self,
        depth,
        num_channels,
        num_latents,
        latent_dim,
        cross_heads=1,
        latent_heads=8,
        num_classes=1000,
        attn_dropout=0.0,
        ff_dropout=0.0,
    ):
        super().__init__()
        self.depth = depth
        # Initial latent vectorss
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))

        cross_attn_layer = lambda: CrossAttention(
            num_latents, input_channels, cross_heads, attn_dropout
        )
        cross_ff_layer = lambda: MLP(num_latents, ff_dropout)
        latent_attn_layer = lambda: SelfAttention(
            num_channels, latent_heads, attn_dropout
        )
        latent_ff_layer = lambda: MLP(num_latents, ff_dropout)

        # Iteratively build the architecture based on params
        # Depth * (Cross Attention + num_self_per_cross * Self Attention)
        self.layers = nn.ModuleList([])
        for i in range(depth):
            self_attns = nn.ModuleList([])

            # Construct self attention block
            for _ in range(num_self_per_cross):
                self_attns.append(
                    nn.ModuleList([latent_attn_layer(), latent_ff_layer()])
                )

            # Construct one perceiver block
            self.layers.append(
                nn.Modulelist([cross_attn_layer(), cross_ff_layer(), self_attns])
            )

        self.to_logits = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, num_classes),
        )

        def forward(self, data, pad_mask=None):
            b, *axis, _, device = *data.shape, data.device

            # TODO Fourier encode, possibly preprocess data prior to sending to perceiver.

            for cross_attn, cross_ff, self_attns in self.layers:
                x = cross_attn(x, context=data, mask=mask) + x
                x = cross_ff(x) + x

                for self_attn, self_ff in self_attns:
                    x = self_attn(x) + x
                    x = self_ff(x) + x

            return self.to_logits(x)
