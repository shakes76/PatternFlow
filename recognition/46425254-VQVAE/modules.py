import torch
import torch.nn as nn
import torch.utils as utils
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#Setting Global Parameters
image_dim = (3,256,256)
learning_rate = 0.0001
latent_space = 256

class Encoder(nn.Module):
    
    def __init__(self):
        #3 convolutional layers for a latent space of 64
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride = 2, padding = 1),
            # 3 * 256 * 256 -> 64 * 128 * 128
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            
            nn.Conv2d(64, 128, kernel_size=4, stride = 2, padding = 1),
            # 64 * 128 * 128 -> 128 * 64 * 64
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            
            nn.Conv2d(128, latent_space, kernel_size=4, stride = 2, padding = 1),
            # 64 * 128 * 128 -> 256 * 64 * 64
            
            nn.Tanh(),)
        
    
    def forward(self, x):
        return self.model(x)
            
            

    
    
class Decoder(nn.Module):
    
    def __init__(self):
        self.model = nn.Sequential(
            nn.Conv2d(latent_space, 128, kernel_size=4, stride = 2, padding = 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            
            nn.Conv2d(128, 64, kernel_size= 4, stride = 2, padding = 1),
            nn.BatchNormal(64),
            nn.LeakyReLU(0.1),
            
            nn.Conv(64, 1, kernel_size = 4, stride = 2, padding = 1)
            )
        
        
    def forward(self, x):
        return self.model(x)
    
"""
Since the dimension of e is not defined, it can grow arbitrarily if the 
encoder outputs are trained faster than the embedded vector. A commitment
loss is therefore needed to regulate the encoder outputs to commit to an
embedding as a Hyperparameter.



"""
class VQ(nn.Module):

    """
    Define a latent embedding space e, which is a real number space with 
    relation K * D, where:
        K is the size of the discrete latent space
        D is the size of the vectors embedded into the space
    
    There are K embedded vectors of dimensionality D
        
    e is the lookup table for the encoded output, and based on the output of
    encoder, chooses the closest embedded vector as input for the decoder.
    
    
    
    Parameters:
        self.num_embeddings -> Parameter K
        self.embedding_dim -> Parameter D
        self.commitment_loss -> the loss value that calculates
    """    
    
    def __init__(self, num_embeddings, embedding_dim, commitment_loss):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        
        self.embedding_table = nn.Embedding(
            self.num_embeddings, self.embedding_dim)
        
        #initialise the weights of the vectors in the embedding table
        #weights are uniform relative to the number of embedded vectors
        self.embedding_table.weight.data.uniform_(-1/self.num_embeddings,
                                                  1/self.num_embeddings)
        
        self.commitment_loss = commitment_loss
    
    
    """
    Quantizes the encoder output.
    
    Parameters:
        encoder_inputs -> the output tensor from encoding
        
    Returns:
        quantized -> the quantized tensor
        loss -> loss of information after quantizing
    """
    def forward(self, encoder_inputs):
        #currently the encoder_inputs are in Batch Channel Height Width Format
        #Channel needs to be moved to so that the input can be quantised
        
        encoder_inputs = encoder_inputs.permute(0, 2, 3, 1)
        #reformat memory layout since it is has been transformed
        encoder_inputs = encoder_inputs.contiguous() 
        shape = encoder_inputs.shape
        
        #flatten the Batch Height and Width into such that the tensor becomes
        #(N = B*H*W, Channel)
        
        flattened_inputs = encoder_inputs.view(-1, self.embedding_dim)
        
        """
        calculate distances between each of the flattened inputs and the
        weights of the of embedded table vectors, creates N*K distances, where 
        N is B*H*W and K the num_embeddings
        """
        
        all_distances = (torch.sum(flattened_inputs**2, dim=1, keepdim = True)
                         + torch.sum(self.embedding.weight**2, dim=1)
                         - 2 * torch.matmul(flattened_inputs, 
                                            self.embedding.weight.t()))
        
        # find the smallest distance from N*K distance combinations
        # get the index tensor of the smallest distance
        indexes = torch.argmin(all_distances, dim=1).unsqueeze(1)
        
        # create a zeros tensor, with the position at indexes being 1
        # This creates a "beacon" so that position can be replaced by a 
        # embedded vector.
        encodings = torch.zeros(indexes.shape[0], self.num_embeddings, device)
        
        encodings.scatter_(1, indexes, 1)
        
        # Quantize and reshape to BHWC format
        
        quantized_bhwc = torch.matmul(encodings, 
                                      self.embedding.weight).view(shape)
        
        """
        the loss function is used from the VQVAE paper provided in the 
        references in readme.The first term of the loss fucntion, the 
        reconstruction loss, is calculated later.
        
        The stop gradients used can be applied using the detach() function, 
        which removes it from the current graph and turns it into a constant.
        
        """
        first_term = nn.functional.mse_loss(quantized_bhwc.detach(), 
                                            encoder_inputs)
        
        second_term = nn.functional.mse_loss(quantized_bhwc, 
                                             encoder_inputs.detach())
        
        beta = self.commitment_loss
        
        
        loss = first_term + beta * second_term
        
        # backpropagate and update gradients back at to the encoder using the 
        # quantized gradients
        
        quantized_bhwc = encoder_inputs + (quantized_bhwc - 
                                           encoder_inputs).detach()
        
        # restructure the VQ output to be Batch Channel Height Width format
        quantized = quantized_bhwc.permute(0, 3, 1, 2)
        # reformat memory, just like when it was transformed to bchw format
        quantized = quantized.continguous()
        
        return quantized, loss
        
        
        
        
        
        
        
        
        
        
        
        
    