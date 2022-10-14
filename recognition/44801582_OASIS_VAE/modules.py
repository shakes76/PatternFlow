"""
Source code of the components of your model. Each component must be implemented as a class or a function
"""

import torch
import torch.nn as nn


# class VQVAE(nn.Module):
#     def __init__(self, latent, device):
#         super(VQVAE, self).__init__()
#
#         self.device = device
#         self.encoder = Encoder(latent)
#         self.decoder = Decoder(latent)
#
#     def forward(self, x):
#         x = x.to(self.device)
#         z = self.encoder(x)
#         return self.decoder(z)

def get_pad(output, stride):
    return (output * (stride - 1) - stride)/2

# class Encoder(nn.Module):
#     def __init__(self, latent):
#         super(Encoder, self).__init__()
#
#         self.encoder = nn.Sequential(
#         )
#
#     def forward(self, x):
#         z = self.encoder(x)
#         # Not a finished implementation, need to implement re-parametrisation
#
#         return z
def get_encoder(latent_dim=16):
    model = nn.Sequential(
        nn.LazyConv2d(32, 3, stride=2, padding=1),
        nn.ReLU(),
        nn.LazyConv2d(64, 3, stride=2, padding=1),
        nn.ReLU(),
        nn.LazyConv2d(latent_dim, 1, padding=1)
    )

    return model


class VQ(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, beta=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.beta = beta

        self.embeddings = torch.rand(size=(embedding_dim, num_embeddings), dtype=torch.float32)

    def forward(self, x):
        input_shape = torch.Size(x)
        flattened = torch.reshape(x, [-1, self.embedding_dim])

        encoding_indices = self.get_code_indices(flattened)
        encodings = nn.functional.one_hot(encoding_indices, self.num_embeddings)
        quantized = torch.matmul(encodings, torch.transpose(self.embeddings))

        quantized = torch.reshape(quantized, input_shape)

        commitment_loss = torch.mean((quantized.detach() - x) ** 2)
        codebook_loss = torch.mean((quantized.detach() - x.detach()) ** 2)  # quantized may need another detach here
        self.loss += (self.beta * commitment_loss + codebook_loss)

        quantized = x + (quantized - x).detach()
        return quantized

    def get_code_indices(self, flattened_inputs):
        similarity = torch.matmul(flattened_inputs, self.embeddings)
        distances = (
            torch.sum(flattened_inputs ** 2, dim=1, keepdim=True)
            + torch.sum(self.embeddings ** 2, dim=0)
            - 2 * similarity
        )

        # Derive the indices for minimum distances.
        encoding_indices = torch.argmin(distances, dim=1)
        return encoding_indices


# class Decoder(nn.Module):
#     def __init__(self, latent):
#         super(Decoder, self).__init__()
#
#         self.decoder = nn.Sequential(
#         )
#
#     def forward(self, z):
#         x = self.decoder(z)
#         x = torch.sigmoid(x)
#         return x

def get_decoder(latent_dim=16):
    model = nn.Sequential(
        nn.LazyConvTranspose2d(64, 3, stride=2, padding="same"),
        nn.ReLU(),
        nn.LazyConvTranspose2d(32, 3, stride=2, padding="same"),
        nn.ReLU(),
        nn.LazyConvTranspose2d(1, 3, stride=2, padding="same")
    )

    return model


def get_vqvae(latent_dim=16, num_embeddings=64):
    model = nn.Sequential(
        get_encoder(latent_dim),
        VQ(num_embeddings, latent_dim),
        get_decoder(latent_dim)
    )

    return model



if __name__ == "__main__":
    model = get_vqvae()
