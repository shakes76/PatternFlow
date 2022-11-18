import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantiser(nn.Module):
    """
    Creates the Vector Quantiser module for VQ-VAE.
    Params:
        num_embeddings: number of embeddings in the embedding space (i.e. codebook).
        embedding_dim: dim of the embedding in embedding space.
        commitment_cost: beta term in the loss function.
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantiser, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)

        # initialises a uniform priors for the codebook
        self._embedding.weight.data.uniform_(-1 / self._num_embeddings, 1 / self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous() # keeps inputs contiguous for view and reshaping later on

        # Flatten input to individually quantise
        input_flatten = inputs.view(-1, self._embedding_dim)

        # Calculate distances to all vectors in the codebook
        distances = (torch.sum(input_flatten ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(input_flatten, self._embedding.weight.t()))

        # Finds the indices of the closest codebook vector
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)

        # creates a one-hot encoded matrix table
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(inputs.shape)

        # Loss terms
        e_latent_loss = F.mse_loss(quantized.detach(), inputs) # stop gradient as specified in paper
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss # loss as described in the paper

        # pass through loss so that the loss passes through to inputs and the codebook vectors are not affected
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, torch.argmin(distances, dim=1)

    def get_quantized(self, x):
        """
        Returns the embedding corresponding to encoding index for singular index
        """
        encoding_indices = x.unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=x.device)
        encodings.scatter_(1, encoding_indices, 1)
        quantized = torch.matmul(encodings, self._embedding.weight).view(1, 64, 64, 64)
        return quantized.permute(0, 3, 1, 2).contiguous()
