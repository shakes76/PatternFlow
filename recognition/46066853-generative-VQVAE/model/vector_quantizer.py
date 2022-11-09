"""
Vector Quantizer module
"""

__author__ = "Aryaman Sharma"

import torch 
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    """
    Create a Vector Quantizer module

    Params:
        num_embeddings: Number of embeddings in embeddings codebook
        embedding_dim: dim of each embedding in embeddings codebook
        commitment_cost: commitment term of the loss (beta in loss function)
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings 
        self.embedding_dim = embedding_dim

        # Create embedding table 
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        # the learnable weights of the module of shape (num_embeddings, embedding_dim) initialized from \mathcal{N}(0, 1)N(0,1)
        self.embedding.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)
        self.commitment_cost = commitment_cost
    
    def forward(self, x):
        # convert inputs from BCHW to BHWC
        x = x.permute(0,2,3,1).contiguous() # keep memory contiguous
        x_shape = x.shape
        # Flatten
        # Each flattened layer is individually quantized
        flat_x = x.view(-1, self.embedding_dim)
        # Calculate distances
        # Find closest codebook vectors
        # find distance of encoded vector to all coded vectors
        # shape (#,num encodings)
        distances = (torch.sum(flat_x**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_x, self.embedding.weight.t()))
        
        # Encoding

        #return val for training 
        train_indices_return = torch.argmin(distances, dim=1)
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1) # min d

        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=x.device)
        encodings.scatter_(1, encoding_indices, 1) # place in encodings (eq to keras one-hot)
 
        # Quantize and unflatten
        # Multiply encodings table with embeddings
        quantized = torch.matmul(encodings, self.embedding.weight).view(x_shape)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), x) # stop gradient propogation on quantized
        q_latent_loss = F.mse_loss(quantized, x.detach()) 
        loss = q_latent_loss + self.commitment_cost * e_latent_loss # loss fn (paper)
        
        quantized = x + (quantized - x).detach() # when backprop end up with x (no gradient flow for other term) 
        
        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), encodings, train_indices_return

    """
    Returns embedding corresponding to encoding index
    For one index
    """
    def get_quantized(self, x):
        encoding_indices = x.unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=x.device)
        encodings.scatter_(1, encoding_indices, 1)
        quantized = torch.matmul(encodings, self.embedding.weight).view(1,64,64,64)
        return quantized.permute(0,3,1,2).contiguous()