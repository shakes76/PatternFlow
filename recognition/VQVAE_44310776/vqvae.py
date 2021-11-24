import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributed

####################
# Residuals
####################
class Residual(nn.Module):
    """
    Residual connection block.
    """
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(num_hiddens),
            nn.Dropout(0.2)
        )
    
    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
                             for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)

####################

####################
# Responsive Encoder
####################
class ResponsiveEncoder(nn.Module):
    """
    Autoencoder convolutional encoder component. Constructs the right number of layers to downsample from
    given input to output sizes automatically. Includes residual connections.
    """
    def __init__(self, in_size, out_size, in_channels, feature_dim, num_residual_layers, residual_feature_dim):
        super(ResponsiveEncoder, self).__init__()

        self._target_features = feature_dim
        self._downsamples = int(math.log(in_size // out_size, 2))

        self._initial = nn.Sequential(
            nn.Conv2d(in_channels, self.feats(0), kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.feats(0)),
            nn.ReLU(True)
        )

        self._blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.feats(i-1),
                          self.feats(i),
                          kernel_size=4,
                          stride=2, padding=1),
                nn.BatchNorm2d(self.feats(i)),
                nn.ReLU(True)
            ) for i in range(1, self._downsamples + 1)
        ])

        self._final = nn.Sequential(
            nn.Conv2d(self._target_features, self._target_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self._target_features),
        )

        self._residual = ResidualStack(in_channels=self._target_features,
                                       num_hiddens=self._target_features,
                                       num_residual_layers=num_residual_layers,
                                       num_residual_hiddens=residual_feature_dim)

    def forward(self, x):
        y = self._initial(x)
        for l in self._blocks:
            y = l(y)
        y = self._final(y)
        y = self._residual(y)

        return y

    def feats(self, i):
        return int(self._target_features*(2**(i-self._downsamples)))


####################

####################
# Responsive Decoder
####################
class ResponsiveDecoder(nn.Module):
    """
    Autoencoder convolutional decoder component. Constructs the right number of layers to upsample from
    given input to output sizes automatically. Includes residual connections.
    """
    def __init__(self, in_size, out_size, in_channels, out_channels, feature_dim, num_residual_layers, residual_feature_dim):
        super(ResponsiveDecoder, self).__init__()

        self._initial_features = feature_dim
        self._upsamples = int(math.log(out_size // in_size, 2))

        self._initial = nn.Sequential(
            nn.Conv2d(in_channels, self._initial_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self._initial_features),
        )
        
        self._residual = ResidualStack(in_channels=self._initial_features,
                                       num_hiddens=self._initial_features,
                                       num_residual_layers=num_residual_layers,
                                       num_residual_hiddens=residual_feature_dim)

        self._blocks = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(self.feats(i),
                          self.feats(i-1),
                          kernel_size=4,
                          stride=2, padding=1),
                nn.BatchNorm2d(self.feats(i-1)),
                nn.ReLU(True)
            ) for i in range(self._upsamples, 0, -1)
        ])

        self._final = nn.Sequential(
            nn.ConvTranspose2d(self.feats(0), out_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        y = self._initial(x)
        y = self._residual(y)
        for l in self._blocks:
            y = l(y)
        y = self._final(y)

        return y

    def feats(self, i):
        return int(self._initial_features*(2**(i-self._upsamples)))

####################


####################
# VQ Module
####################

class Quantize(nn.Module):
    """
    Source: rosinality on GitHub
    URL: https://github.com/rosinality/vq-vae-2-pytorch/blob/master/vqvae.py

    Comments by me.
    """
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.randn(dim, n_embed) # Codebook vectors.
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, input):
        # Get distances between codebook vectors and input to find best matching quantization.
        flatten = input.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        # Make the codebook representation (indices) by selecting the closest codebook vector.
        _, embed_ind = (-dist).max(1)
        # Get onehot encoding to index embedding layer.
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind) # Get the quantized rep (swap index for actual vector).

        # Update the codebook vectors during training.
        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            # Can't do distributed training for this so reduce to single GPU.
            if distributed.is_available() and distributed.is_initialized() and distributed.get_world_size() != 1:
                distributed.all_reduce(embed_onehot_sum, op=distributed.ReduceOp.SUM)
                distributed.all_reduce(embed_sum, op=distributed.ReduceOp.SUM)
            
            # Update values using exponential moving averages.
            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)
        
        # Get loss for this component.
        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()

        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))

class ResponsiveVQVAE2(nn.Module):
    """
    VQVAE2 implementation allowing for flexible input size (at training time).

    Note: layers are labelled as '1' (bottom) and '2' (top) to allow for future
    implementations to dynamically add more intermediate quantizations.
    """
    def __init__(self, in_size, latent_sizes, in_channels, feat_channels, num_residual_blocks, num_residual_channels, 
                num_embeddings, embedding_dim, commitment_cost, decay=0.99):
        super(ResponsiveVQVAE2, self).__init__()
        
        # Bottom level encoder.
        self.encoder_1 = ResponsiveEncoder(in_size, 
                                          latent_sizes[0],
                                          in_channels,
                                          feat_channels,
                                          num_residual_blocks, 
                                          num_residual_channels)

        # Top level encoder.
        self.encoder_2 = ResponsiveEncoder(latent_sizes[0],
                                           latent_sizes[1],
                                           feat_channels,
                                           feat_channels,
                                           num_residual_blocks,
                                           num_residual_channels)
        
        # Top level quantization.
        self.pre_vq_conv_2 = nn.Conv2d(feat_channels, embedding_dim, 1)
        self.quantize_2 = Quantize(embedding_dim, num_embeddings)
        self.decoder_2 = ResponsiveDecoder(latent_sizes[1], 
                                           latent_sizes[0],
                                           embedding_dim,
                                           embedding_dim,
                                           feat_channels,
                                           num_residual_blocks,
                                           num_residual_channels)
        
        # Bottom level quantization.
        self.pre_vq_conv_1 = nn.Conv2d(embedding_dim + feat_channels, embedding_dim, 1)
        self.quantize_1 = Quantize(embedding_dim, num_embeddings)
        
        # Decoder for both levels.
        self.upsample_1 = nn.ConvTranspose2d(
            embedding_dim, embedding_dim, 4, stride=2, padding=1
        )
        self.decoder_1 = ResponsiveDecoder(latent_sizes[0], 
                                           in_size,
                                           embedding_dim + embedding_dim,
                                           in_channels,
                                           feat_channels,
                                           num_residual_blocks,
                                           num_residual_channels)

    def forward(self, x):
        q2, q1, diff, _, _ = self.encode(x)
        reconstruction = self.decode(q1, q2)   
        return reconstruction, diff # Diff is the VQ loss.

    def encode(self, x):
        e1 = self.encoder_1(x) # Encode the input witht he first encoder.
        e2 = self.encoder_2(e1) # Push the encoded input from the first encoder through to the second encoder.

        # Quantize the deepest representation first.
        q2 = self.pre_vq_conv_2(e2).permute(0, 2, 3, 1)
        q2, diff_2, id_2 = self.quantize_2(q2)
        q2 = q2.permute(0, 3, 1, 2)
        diff_2 = diff_2.unsqueeze(0)

        # Add the decoded deep rep to the encoded shallow rep (as per diagram).
        d2 = self.decoder_2(q2)
        e1 = torch.cat([d2, e1], 1)

        # Quantize the shallow rep.
        q1 = self.pre_vq_conv_1(e1).permute(0, 2, 3, 1)
        q1, diff_1, id_1 = self.quantize_1(q1)
        q1 = q1.permute(0, 3, 1, 2)
        diff_1 = diff_1.unsqueeze(0)

        return q2, q1, diff_2 + diff_1, id_2, id_1

    def decode(self, q1, q2):
        # Decode the fully quantized rep.
        q2 = self.upsample_1(q2)
        q = torch.cat([q1, q2], 1)
        return self.decoder_1(q)

    def decode_codebook(self, id_1, id_2):
        # Decode the codebook representation of the input.
        q2 = self.quantize_2.embed_code(id_2)
        q2 = q2.permute(0, 3, 1, 2)
        q1 = self.quantize_1.embed_code(id_1)
        q1 = q1.permute(0, 3, 1, 2)

        return self.decode(q1, q2)

####################