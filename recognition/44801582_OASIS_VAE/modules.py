"""
Source code of the components of your model. Each component must be implemented as a class or a function
"""

import torch
import torch.nn as nn


class VQVAE(nn.Module):
    def __init__(self, latent, device):
        super(VQVAE, self).__init__()

        self.device = device
        self.encoder = Encoder(latent)
        self.decoder = Decoder(latent)

    def forward(self, x):
        x = x.to(self.device)
        z = self.encoder(x)
        return self.decoder(z)


class Encoder(nn.Module):
    def __init__(self, latent):
        super(Encoder, self).__init__()

        self.encoder = nn.Sequential(
        )

    def forward(self, x):
        z = self.encoder(x)
        # Not a finished implementation, need to implement re-parametrisation

        return z


class Decoder(nn.Module):
    def __init__(self, latent):
        super(Decoder, self).__init__()

        self.decoder = nn.Sequential(
        )

    def forward(self, z):
        x = self.decoder(z)
        x = torch.sigmoid(x)
        return x
