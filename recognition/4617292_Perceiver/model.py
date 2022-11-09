#!/usr/bin/env python3

from math import pi, log, e as exp

import torch
import torch.nn as nn
import torch.nn.functional as F


"""
    Unofficial implementation of the Perceiver Transformer model by Deepmind.

    The perceiver consists of two components:
        - The cross attention layer
        - The latent transformer layer

    The components are alternated one after another.
"""

__author__ = "Chegne Eu Joe"
__email__ = "e.chegne@uqconnect.edu.au"


# Copypasta, as sanity check for fourier encoding
def fourier_encode(shape, bands, device):
    dims = len(shape)

    pos = torch.stack(
        list(torch.meshgrid(*(torch.linspace(-1.0, 1.0, steps=n) for n in list(shape))))
    )
    pos = pos.unsqueeze(0).expand((bands,) + pos.shape).to(device)

    band_frequencies = (
        (
            torch.logspace(
                log(1.0), log(shape[0] / 2), steps=bands, base=exp, device=device
            )
        )
        .view((bands,) + tuple(1 for _ in pos.shape[1:]))
        .expand(pos.shape)
    )

    result = (band_frequencies * pi * pos).view((dims * bands,) + shape)

    result = torch.cat(
        [
            torch.sin(result),
            torch.cos(result),
        ],
        dim=0,
    )

    return result


class Attention(nn.Module):
    """
    Allows the model to jointly attend to information from different
    representation subspaces.

    Wrapper for {self/cross}-attention layers which will be further
    elaborated in the respective classes

    See https://arxiv.org/abs/1706.03762 for more details.

    Args:
        embed_dim - Total dimension of model.
        num_heads - Number of parallel attention heads.
        dropout - Dropout probability.
    Returns:
        Attention outputs of shape (N, L, E) where L is the target
        sequence length, N is the batch size, and E is embed_dim
    """

    def __init__(self, latent_dim, latent_heads, dropout):
        super().__init__()

        self.layer_norm_x = nn.LayerNorm([latent_dim])
        self.layer_norm_1 = nn.LayerNorm([latent_dim])
        self.attention = nn.MultiheadAttention(
            latent_dim,
            latent_heads,
            dropout=0.0,
            bias=True,
            add_bias_kv=True,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(latent_dim, latent_dim)
        self.layer_norm_2 = nn.LayerNorm([latent_dim])
        self.linear2 = nn.Linear(latent_dim, latent_dim)
        self.linear3 = nn.Linear(latent_dim, latent_dim)

    def forward(self, x, z_input):
        # Pre-norm
        x = self.layer_norm_x(x)
        z = self.layer_norm_1(z_input)

        z, _ = self.attention(z, x, x)
        z = self.dropout(z)
        z = self.linear1(z)

        # MLP
        z = self.layer_norm_2(z)
        z = self.linear2(z)
        z = F.gelu(z)
        z = self.dropout(z)
        z = self.linear3(z)

        return z + z_input


class SelfAttention(nn.Module):
    """
    For each input vector v_i, compute the output o_i by taking the
    weighted sum of all non-query input vectors, where the weight is proportional
    exp(v_i * v_j / sqrt(d)), dot product.

    This allows us to measure similarity between two vectors. The higher the weight,
    the more similar the input sequence is compared to the query and vice versa.

    See https://arxiv.org/abs/1706.03762 for more details.
    """

    def __init__(self, embed_dim, latent_heads=8, dropout=0.0):
        super().__init__()
        self.attention = Attention(embed_dim, latent_heads, dropout)

    def forward(self, x, z):
        # Self-attention
        return self.attention(x, z)


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

    def __init__(self, embed_dim, dropout=0.0):
        super().__init__()
        self.attention = Attention(embed_dim, 1, dropout)

    def forward(self, x, z):
        return self.attention(x, z)


class PerceiverBlock(nn.Module):
    """
    A perceiver.
    Depth * (cross attention layer + (self_per_cross_attn * self attention layer))
    """

    def __init__(self, latent_dim, self_per_cross_attn, dropout, latent_heads):
        super().__init__()

        self.cross_attention = CrossAttention(latent_dim, dropout=dropout)
        self.latent_attentions = nn.ModuleList(
            [
                SelfAttention(latent_dim, latent_heads=latent_heads, dropout=dropout)
                for _ in range(self_per_cross_attn)
            ]
        )

    def forward(self, x, z):
        z = self.cross_attention(x, z)
        for latent_attention in self.latent_attentions:
            z = latent_attention(z, z)
        return z


class Perceiver(nn.Module):
    """
    A scalable, fully attentional architecture.
    Note that data has to be pre-fourier encoded or it will not work.

    Args:
        depth - The depth of the network. See code for more information.
        num_channels - Number of channels of each input.
        input_shape - Size & shape of input.
        fourier_bands - Number of bands for fourier encoding.
        num_latents - Number of latent vectors.
        latent_dim - Latent dimension.
        latent_heads - Number of heads for self attention.
        attn_dropout - Attention dropout probability.
        ff_dropout - MLP dropout probability.
        num_features - How many different classes for output
        self_per_cross_attn - Number of self attention blocks per cross attention.
    Returns:
        Perceiver layer
    """

    def __init__(
        self,
        depth,
        num_channels,
        input_shape,
        fourier_bands,
        num_latents,
        latent_dim,
        latent_heads=8,
        attn_dropout=0.0,
        num_features=2,
        self_per_cross_attn=3,
        device="cpu",
    ):
        super().__init__()

        # Initial latent vectors
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))
        self.depth = depth
        self.fourier_features = fourier_encode(input_shape, fourier_bands, device)
        self.embeddings = nn.Conv1d(
            num_channels + self.fourier_features.shape[0], latent_dim, 1
        )

        # Logits
        self.linear = nn.Linear(latent_dim, latent_dim)
        self.output = nn.Linear(latent_dim, num_features)

        self.block1 = PerceiverBlock(
            latent_dim,
            self_per_cross_attn=self_per_cross_attn,
            latent_heads=latent_heads,
            dropout=attn_dropout,
        )

        self.block2 = nn.ModuleList(
            [
                PerceiverBlock(
                    latent_dim,
                    self_per_cross_attn=self_per_cross_attn,
                    latent_heads=latent_heads,
                    dropout=attn_dropout,
                )
                for _ in range(depth - 1)
            ]
        )

        self.linear1 = nn.Linear(latent_dim, latent_dim)
        self.linear2 = nn.Linear(latent_dim, num_features)

    def forward(self, x):
        batch_size = x.shape[0]
        pos = self.fourier_features.unsqueeze(0).expand(
            (batch_size,) + self.fourier_features.shape
        )

        # Transform x
        x = torch.cat([x, pos], dim=1)
        x = x.view((x.shape[0], x.shape[1], -1))
        x = self.embeddings(x)

        x = x.permute(2, 0, 1)

        # Transform latent
        z = self.latents.unsqueeze(1)
        z = z.expand(-1, x.shape[1], -1)

        # Train
        z = self.block1(x, z)
        for perceiver_block in self.block2:
            z = perceiver_block(x, z)

        # Final classification head
        z = self.linear1(z)
        z = z.mean(dim=0)
        z = self.linear2(z)

        return F.log_softmax(z, dim=-1)
