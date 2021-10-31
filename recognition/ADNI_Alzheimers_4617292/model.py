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
def fourier_encode(shape, bands):
    # This first "shape" refers to the shape of the input data, not the output of this function
    dims = len(shape)

    # Every tensor we make has shape: (bands, dimension, x, y, etc...)

    # Pos is computed for the second tensor dimension
    # (aptly named "dimension"), with respect to all
    # following tensor-dimensions ("x", "y", "z", etc.)
    pos = torch.stack(list(torch.meshgrid(
        *(torch.linspace(-1.0, 1.0, steps=n) for n in list(shape))
    )))
    pos = pos.unsqueeze(0).expand((bands,) + pos.shape)

    # Band frequencies are computed for the first
    # tensor-dimension (aptly named "bands") with
    # respect to the index in that dimension
    band_frequencies = (torch.logspace(
        log(1.0),
        log(shape[0]/2),
        steps=bands,
        base=exp
    )).view((bands,) + tuple(1 for _ in pos.shape[1:])).expand(pos.shape)

    # For every single value in the tensor, let's compute:
    #             freq[band] * pi * pos[d]

    # We can easily do that because every tensor is the
    # same shape, and repeated in the dimensions where
    # it's not relevant (e.g. "bands" dimension for the "pos" tensor)
    result = (band_frequencies * pi * pos).view((dims * bands,) + shape)

    # Use both sin & cos for each band, and then add raw position as well
    # TODO: raw position
    result = torch.cat([
        torch.sin(result),
        torch.cos(result),
    ], dim=0)

    return result

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

    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args):
        x = self.norm(x)
        # Removed extra
        return self.fn(x, *args)


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

    def __init__(self, embed_dim, heads, dropout=0.0):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim, heads, dropout, bias=True, batch_first=True
        )

    def forward(self, x, z):
        attn_output, _ = self.attention(z, x, x)
        return attn_output + z


class SelfAttention(nn.Module):
    """
    For each input vector v_i, compute the output o_i by taking the
    weighted sum of all non-query input vectors, where the weight is proportional
    exp(v_i * v_j / sqrt(d)), dot product.

    This allows us to measure similarity between two vectors. The higher the weight,
    the more similar the input sequence is compared to the query and vice versa.

    See https://arxiv.org/abs/1706.03762 for more details.
    """

    def __init__(self, embed_dim, heads=8, dropout=0.0):
        super().__init__()
        self.attention = PreNorm(embed_dim, Attention(embed_dim, heads, dropout))

    def forward(self, x):
        # Self-attention
        return self.attention(x, x)


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

    def __init__(self, embed_dim, heads=1, dropout=0.0):
        super().__init__()
        self.attention = PreNorm(embed_dim, Attention(embed_dim, heads, dropout))

    def forward(self, x, z):
        return self.attention(x, z)

class MLP(nn.Module):
    def __init__(self, latent_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, x):
        return self.net(x)


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
        ff_dropout=0.0,
        num_features=2,
        self_per_cross_attn=3,
    ):
        super().__init__()

        # Initial latent vectorss
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))
        self.depth = depth
        self.fourier_features = fourier_encode(input_shape, fourier_bands)
        self.embeddings = nn.Conv1d(
            num_channels + self.fourier_features.shape[0], latent_dim, 1
        )

        # Perceiver block
        cross_attn = lambda: CrossAttention(
            latent_dim, heads=1, dropout=attn_dropout
        )
        cross_ff = lambda: MLP(latent_dim, dropout=ff_dropout)
        latent_attn = lambda: SelfAttention(
            latent_dim, heads=8, dropout=attn_dropout
        )
        latent_ff = lambda: MLP(latent_dim, dropout=ff_dropout)

        # Logits
        self.linear = nn.Linear(latent_dim, latent_dim)
        self.output = nn.Linear(latent_dim, num_features)

        # Build architecture based on params
        # Depth * (cross attention layer + (self_per_cross_attn * self attention layer))
        self.layers = nn.ModuleList([])
        for i in range(depth):
            self_attns = nn.ModuleList([])

            # Construct self attention block
            for _ in range(self_per_cross_attn):
                self_attns.append(nn.ModuleList([latent_attn(), latent_ff()]))

            # Construct one perceiver block
            self.layers.append(nn.ModuleList([cross_attn(), cross_ff(), self_attns]))

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
        for cross_attn_lyr, cross_ff_lyr, self_attns_lyr in self.layers:
            z = cross_attn_lyr(x, z)
            z = cross_ff_lyr(z)

            for self_attn_lyr, self_ff_lyr in self_attns_lyr:
                z = self_attn_lyr(z)
                z = self_ff_lyr(z)

        # To logits
        z = self.linear(z)
        z = z.mean(dim=0)
        z = self.output(z)

        return F.log_softmax(z, dim=-1)
