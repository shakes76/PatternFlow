import torch.nn as nn
from modules.encoder import Encoder
from modules.decoder import Decoder
from modules.quantizer import Quantizer


class VQVAE(nn.Module):
    def __init__(self, latent_dim, res_h_dim,
                 num_embeddings, embedding_dim, beta):
        super(VQVAE, self).__init__()
        # encode image
        self.encoder = Encoder(input_dim=3,
                               between_latent_dim=res_h_dim,
                               latent_dim=latent_dim
                               )

        self.pre_quantization_conv = nn.Conv2d(in_channels=latent_dim,
                                               out_channels=embedding_dim,
                                               kernel_size=1,
                                               stride=1)

        # pass continuous latent dim to quantizer
        self.vector_quantizer = Quantizer(num_embeddings=num_embeddings,
                                          embedding_dim=embedding_dim,
                                          beta=beta)

        # decode discrete latent repr
        self.decoder = Decoder(input_dim=embedding_dim,
                               between_latent_dim=res_h_dim,
                               latent_dim=latent_dim
                               )

    def forward(self, x, verbose=False):
        z_e = self.encoder(x)

        z_e = self.pre_quantization_conv(z_e)
        embedding_loss, z_q, _, _ = self.vector_quantizer(z_e)
        x_hat = self.decoder(z_q)

        if verbose:
            print(f"Original data shape: {x.shape}")
            print(f"Encoded data shape: {z_e.shape}")
            print(f"Reconstructed data shape: {x_hat.shape}")
            assert False

        return embedding_loss, x_hat
