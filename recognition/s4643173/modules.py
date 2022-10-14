import torch
import torch.nn as nn
import torch.nn.functional as F

class Embedding(nn.Module):
    def __init__(self, K, D):
        super().__init__()
        self.K = K
        self.embedding = nn.Embedding(K, D)
        self.embedding.weight.data.uniform_(-1./K, 1./K)

    def forward(self, z_e_x):
        # Reshape inputs from BCHW to BHWC
        x = z_e_x.permute(0, 2, 3, 1).contiguous()
        x_shape = x.shape

        flat_x = x.view(-1, self.embedding.weight.size(0))

        # Calculate distances to find the closest codebook vectors
        distances = (torch.sum(flat_x**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_x, self.embedding.weight.t()))
        
        # We get the indices from argmin
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = torch.zeros(encoding_indices.shape[0], self.K, device='cuda')
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize the vector by multiplying the encodings with the embeddings
        quantized = torch.matmul(encodings, self.embedding.weight).view(x_shape)

        # Calculate the loss. Note that the paper has log likelihood as the
        # loss but minimising the mean squared error is equivalent.
        commit_loss = F.mse_loss(quantized.detach(), x)
        vq_loss = F.mse_loss(quantized, x.detach()) 
        
        quantized = x + (quantized - x).detach()

        return vq_loss + 1.0 * commit_loss, quantized.permute(0, 3, 1, 2).contiguous(), encodings, encoding_indices.squeeze()