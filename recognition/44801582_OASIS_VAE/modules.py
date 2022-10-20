"""
Source code of the components of your model. Each component must be implemented as a class or a function
"""

import torch
import torch.nn as nn
from functions import vector_quantizer, vector_quantizer_straight_through


def get_pad(output, stride):
    return (output * (stride - 1) - stride)/2


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
    def __init__(self, num_embeddings, embedding_dim, beta=0.25):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, x):
        x = torch.permute(x, (0, 2, 3, 1)).contiguous()
        latent = vector_quantizer(x, self.embed.weight)

        return latent

    def straight_through(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        x_q, ind = vector_quantizer_straight_through(x, self.embed.weight.detach())
        x_q = x_q.permute(0, 3, 1, 2).contiguous()

        flat = torch.index_select(self.embedding.weight, dim=0, index=ind).view_as(x).permute(0, 3, 1, 2).contiguous()

        return x_q, flat


def get_decoder(latent_dim=16):
    dec_model = nn.Sequential(
        nn.LazyConvTranspose2d(64, 3, stride=2, padding="same"),
        nn.ReLU(),
        nn.LazyConvTranspose2d(32, 3, stride=2, padding="same"),
        nn.ReLU(),
        nn.LazyConvTranspose2d(1, 3, stride=2, padding="same")
    )

    return dec_model


def get_vqvae(latent_dim=16, num_embeddings=64):
    vqvae_model = nn.Sequential(
        get_encoder(latent_dim),
        VQ(num_embeddings, latent_dim),
        get_decoder(latent_dim)
    )

    return vqvae_model



if __name__ == "__main__":
    model = get_vqvae()
