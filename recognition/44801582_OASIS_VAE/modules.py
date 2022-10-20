"""
Source code of the components of your model. Each component must be implemented as a class or a function
"""

import torch
import torch.nn as nn
from functions import vector_quantizer, vector_quantizer_straight_through


def get_encoder(latent_dim=16):
    enc_model = nn.Sequential(
        nn.LazyConv2d(32, 3, stride=2, padding=1),
        nn.ReLU(),
        nn.LazyConv2d(64, 3, stride=2, padding=1),
        nn.ReLU(),
        nn.LazyConv2d(latent_dim, 1, padding=1)
    )

    return enc_model


class VQ(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, x):
        x = torch.permute(x, (0, 2, 3, 1)).contiguous()
        latent = vector_quantizer(x, self.embed.weight)

        return latent

    def straight_through(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        x_quantized_straight_through, ind = vector_quantizer_straight_through(x, self.embed.weight.detach())
        x_quantized_straight_through = x_quantized_straight_through.permute(0, 3, 1, 2).contiguous()

        x_quantized = torch.index_select(self.embedding.weight, dim=0, index=ind)\
            .view_as(x).permute(0, 3, 1, 2).contiguous()

        return x_quantized_straight_through, x_quantized


def get_decoder():
    dec_model = nn.Sequential(
        nn.LazyConvTranspose2d(64, 3, stride=2, padding=1),
        nn.ReLU(),
        nn.LazyConvTranspose2d(32, 3, stride=2, padding=1),
        nn.ReLU(),
        nn.LazyConvTranspose2d(1, 3, stride=2, padding=1)
    )

    return dec_model


class VQ_VAE(nn.Module):
    def __init__(self, latent_dim=16, num_embeddings=64):
        super().__init__()
        self.encoder = get_encoder()
        self.codebook = VQ(latent_dim, num_embeddings)
        self.decoder = get_decoder()

    def encode(self, x):
        return self.codebook(self.encoder(x))

    def decode(self, latents):
        return self.decoder(self.codebook.embedding(latents).permute(0, 3, 1, 2))

    def forward(self, x):
        encoding = self.encoder(x)
        x_quantized_straight_through, x_quantized = self.codebook.straight_through(encoding)
        reconstruct = self.decoder(x_quantized_straight_through)
        return reconstruct, encoding, x_quantized
