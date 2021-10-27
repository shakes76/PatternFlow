"""
VQVAE Model
"""
__author__ = "Aryaman Sharma"

import torch
import torch.nn as nn
import torch.nn.functional as F
from .decoder import Decoder
from .encoder import Encoder
from .vector_quantizer import VectorQuantizer
from .residualBlock import Residual_block

class VQVAE(nn.Module):
    """
    VQVAE model
    params:
        num_hiddens: Hidden blocks for encoder convolutions
        residual_inter: Intermediary residual block channels
        num_embeddings: Number of codebook embeddings
        embedding_dim: Dimensions of each embedding
        commitment_cost: loss function beta value
    """
    def __init__(self, num_hiddens, residual_inter, 
                 num_embeddings, embedding_dim, commitment_cost):
        super(VQVAE, self).__init__()
        
        # Create the encoder
        self.encoder = Encoder(
            in_channels=3,
            num_hidden=num_hiddens, 
            residual_inter=residual_inter
        )

        # initial conv Convert input dimensions to embedding dimension
        self.conv1 = nn.Conv2d(
            in_channels=num_hiddens,
            out_channels=embedding_dim,
            kernel_size=1,
            stride=1
        )
       
        # Create vector qunatizer
        self.vq = VectorQuantizer(
            num_embeddings=num_embeddings, 
            embedding_dim=embedding_dim,
            commitment_cost=commitment_cost
        )
        
        # Create decoder
        self.decoder = Decoder(
            in_channels=embedding_dim,
            num_hidden=num_hiddens,
            residual_inter=residual_inter)

    def forward(self, x):
        # encode
        z = self.encoder(x)
        # change channel dim
        z = self.conv1(z)
        loss, quantized, _, _ = self.vq(z)
        # decode
        x_recon = self.decoder(quantized) # reconstructed

        return loss, x_recon