import torch
import torch.nn as nn
import torch.utils as utils
import torchvision


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
        
    """
    def forward(self, encoder_inputs):
        #currently the encoder_inputs are in Batch Channel Height Width Format
        #Channel needs to be moved to so that the input can be quantised
        
        encoder_inputs = encoder_inputs.permute(0, 2, 3, 1)
        #reformat memory layout since it is has been transformed
        encoder_inputs = encoder_inputs.contiguous() 
        
        #flatten the Batch Height and Width into such that the tensor becomes
        #(B*H*W, Channel)
        
        flattened_inputs = encoder_inputs.view(-1, self.embedding_dim)
        
        #calculate distances for each one of the inputs
        
        
        
        
        
        
        
    