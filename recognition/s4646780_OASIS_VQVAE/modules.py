import torch.nn as nn
from models.encoder import Encoder
from models.vectorQuantiser import VectorQuantiser
from models.decoder import Decoder


class VQVAE(nn.Module):
    """
    Final VQ-VAE model.
    params:
        residual_inter: Intermediary residual block channels
        embeddings_num: number of codebook embeddings
        embedding_dim: number of dimensions of each embedding
        commitment_cost: beta parameter for the VQ loss function.
    """

    def __init__(self, embeddings_num, embedding_dim, commitment_cost):
        super(VQVAE, self).__init__()
        self.encoder = Encoder(1, 16)
        # this is to ensure that the output from the decoder matches the input to the VQ module i.e. converts the
        # encoder output to the embedding dimensions.
        self.pre_vq_conv = nn.Conv2d(in_channels=8,
                                     out_channels=embedding_dim, kernel_size=1, stride=1)
        self.vq_vae = VectorQuantiser(embeddings_num, embedding_dim, commitment_cost)
        self.decoder = Decoder(embedding_dim, 8)

    def forward(self, x):
        x = self.encoder(x)
        x = self.pre_vq_conv(x)
        loss, quantized, perplexity, d = self.vq_vae(x)
        x_recon = self.decoder(quantized)

        return loss, x_recon, perplexity
