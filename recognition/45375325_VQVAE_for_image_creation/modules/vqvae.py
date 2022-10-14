import torch.nn as nn
from encoder import Encoder
from decoder import Decoder
from quantizer import Quantizer


class VQVAE(nn.Module):
    def __init__(self, latent_dim, res_h_dim, num_residual_layers,
                 num_embeddings, embedding_dim, beta, save_embedding_map=False):
        super(VQVAE, self).__init__()
        # encode image
        self.encoder = Encoder(3, latent_dim, num_residual_layers, res_h_dim)
        self.pre_quantization_conv = nn.Conv2d(latent_dim, embedding_dim, kernel_size=1, stride=1)
        # pass continuous latent dim to quantizer
        self.vector_quantizer = Quantizer(num_embeddings, embedding_dim, beta)
        # decode discrete latent repr
        self.decoder = Decoder(embedding_dim, latent_dim, num_residual_layers, res_h_dim)

        if save_embedding_map:
            self.embedding_map = {i: [] for i in range(num_embeddings)}
        else:
            self.embedding_map = None

    def forward(self, x, verbose=False):
        z_e = self.encoder(x)

        z_e = self.pre_quantization_conv(z_e)
        embedding_loss, z_q, perplexity, _, _ = self.vector_quantizer(z_e)
        x_hat = self.decoder(z_q)

        if verbose:
            print(f"Original data shape: {x.shape}")
            print(f"Encoded data shape: {z_e.shape}")
            print(f"Reconstructed data shape: {x_hat.shape}")
            assert False

        return embedding_loss, x_hat, perplexity
