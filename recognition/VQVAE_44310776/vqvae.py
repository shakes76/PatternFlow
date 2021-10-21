import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import distributed

####################
# Residuals
####################
class Residual(nn.Module):
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
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.randn(dim, n_embed)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, input):
        flatten = input.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            if distributed.is_available() and distributed.is_initialized() and distributed.get_world_size() is not 1:
                distributed.all_reduce(embed_onehot_sum, op=distributed.ReduceOp.SUM)
                distributed.all_reduce(embed_sum, op=distributed.ReduceOp.SUM)

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

        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()

        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))

# class VectorQuantizerEMA(nn.Module):
#     def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
#         super(VectorQuantizerEMA, self).__init__()
        
#         self._embedding_dim = embedding_dim
#         self._num_embeddings = num_embeddings
        
#         self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
#         self._embedding.weight.data.normal_()
#         self._commitment_cost = commitment_cost
        
#         self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
#         self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
#         self._ema_w.data.normal_()
        
#         self._decay = decay
#         self._epsilon = epsilon

#     def forward(self, inputs):
#         # convert inputs from BCHW -> BHWC
#         inputs = inputs.permute(0, 2, 3, 1).contiguous()
#         input_shape = inputs.shape
        
#         # Flatten input
#         flat_input = inputs.view(-1, self._embedding_dim)

#         # Calculate distances
#         distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
#                     + torch.sum(self._embedding.weight**2, dim=1)
#                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

#         # Encoding
#         encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
#         encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
#         encodings.scatter_(1, encoding_indices, 1)
#         encoding_shape = (input_shape[0], 1, input_shape[1], input_shape[2])

#         # Quantize and unflatten
#         quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

#         # Use EMA to update the embedding vectors
#         if self.training:
#             self._ema_cluster_size = self._ema_cluster_size * self._decay + \
#                                      (1 - self._decay) * torch.sum(encodings, 0)
            
#             # Laplace smoothing of the cluster size
#             n = torch.sum(self._ema_cluster_size.data)
#             self._ema_cluster_size = (
#                 (self._ema_cluster_size + self._epsilon)
#                 / (n + self._num_embeddings * self._epsilon) * n)
            
#             dw = torch.matmul(encodings.t(), flat_input)
#             self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)
            
#             self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))
        
#         # Loss
#         e_latent_loss = F.mse_loss(quantized.detach(), inputs)
#         loss = self._commitment_cost * e_latent_loss
        
#         # Straight Through Estimator
#         quantized = inputs + (quantized - inputs).detach()
#         avg_probs = torch.mean(encodings, dim=0)
#         perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

#         # convert quantized from BHWC -> BCHW
#         return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encoding_indices.view(encoding_shape)

####################


####################
# Models
####################
# class VQVAE(nn.Module):
#     def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, 
#                 num_embeddings, embedding_dim, commitment_cost, decay=0.99):
#         super(VQVAE, self).__init__()
        
#         self._encoder = Encoder(in_channels, num_hiddens,
#                                 num_residual_layers, 
#                                 num_residual_hiddens)
#         self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens, 
#                                     out_channels=embedding_dim,
#                                     kernel_size=1, 
#                                     stride=1)
#         self._vq_vae = VectorQuantizerEMA(num_embeddings, embedding_dim, 
#                                             commitment_cost, decay)

#         self._decoder = Decoder(embedding_dim,
#                                 in_channels,
#                                 num_hiddens, 
#                                 num_residual_layers, 
#                                 num_residual_hiddens)
        
#     def forward(self, x):
#         original = x
#         z = self._encoder(x)
#         z = self._pre_vq_conv(z)
#         # print(z.shape)
#         vq_loss, quantized, perplexity, encoding_indices = self._vq_vae(z)
#         reconstruction = self._decoder(quantized)

#         return original, reconstruction, vq_loss, perplexity, encoding_indices

#     def encode(self, x):
#         z = self._encoder(x)
#         z = self._pre_vq_conv(z)
#         vq_loss, quantized, perplexity, encoding_indices = self._vq_vae(z)

#         return encoding_indices

class ResponsiveVQVAE2(nn.Module):
    def __init__(self, in_size, latent_sizes, in_channels, feat_channels, num_residual_blocks, num_residual_channels, 
                num_embeddings, embedding_dim, commitment_cost, decay=0.99):
        super(ResponsiveVQVAE2, self).__init__()
        
        self.encoder_1 = ResponsiveEncoder(in_size, 
                                          latent_sizes[0],
                                          in_channels,
                                          feat_channels,
                                          num_residual_blocks, 
                                          num_residual_channels)

        self.encoder_2 = ResponsiveEncoder(latent_sizes[0],
                                           latent_sizes[1],
                                           feat_channels,
                                           feat_channels,
                                           num_residual_blocks,
                                           num_residual_channels)
        
        self.pre_vq_conv_2 = nn.Conv2d(feat_channels, embedding_dim, 1)
        self.quantize_2 = Quantize(embedding_dim, num_embeddings)
        self.decoder_2 = ResponsiveDecoder(latent_sizes[1], 
                                           latent_sizes[0],
                                           embedding_dim,
                                           embedding_dim,
                                           feat_channels,
                                           num_residual_blocks,
                                           num_residual_channels)
        
        self.pre_vq_conv_1 = nn.Conv2d(embedding_dim + feat_channels, embedding_dim, 1)
        self.quantize_1 = Quantize(embedding_dim, num_embeddings)
        
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
        # Encode everything.
        q2, q1, diff, _, _ = self.encode(x)

        # Decode the fully quantized rep
        q2 = self.upsample_1(q2)
        q = torch.cat([q1, q2], 1)
        reconstruction = self.decoder_1(q)
        
        return reconstruction, diff

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

        # Quantize the shallow rep
        q1 = self.pre_vq_conv_1(e1).permute(0, 2, 3, 1)
        q1, diff_1, id_1 = self.quantize_1(q1)
        q1 = q1.permute(0, 3, 1, 2)
        diff_1 = diff_1.unsqueeze(0)

        return q2, q1, diff_2 + diff_1, id_2, id_1

####################