import torch
import torch.nn as nn



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_dim = (3,128,128)

"""
Encoder for VQVAE
Uses 3 Convolution 2d layers, with leaky ReLU activation functions and batch 
normalisation layers in between. Sigmoid activation function at the end for 
down sampling.

"""
class Encoder(nn.Module):
    """
    Parameters:
        latent_space -> latent dimension of VQVAE
    """
    def __init__(self, latent_space):
        super(Encoder, self).__init__()
        self.latent_space = latent_space
        #3 convolutional layers for a latent space of 64
        self.model = nn.Sequential(
            nn.Conv2d(3, self.latent_space, kernel_size=4, 
                      stride = 2, padding = 1),
            # 3 * 128 * 128 -> latent_space * 64 * 64
            nn.BatchNorm2d(self.latent_space),
            nn.LeakyReLU(0.1),
            
            nn.Conv2d(self.latent_space, self.latent_space*2, 
                      kernel_size=4, stride = 2, padding = 1),
            # latent_space * 64 * 64 -> latent_space*2  * 32 * 32
            nn.BatchNorm2d(self.latent_space*2),
            nn.LeakyReLU(0.1),
            
            nn.Conv2d(self.latent_space*2, self.latent_space, 
                      kernel_size=3, stride = 1, padding = 1),
            # latent_space*2 * 32 * 32 * latent_space * 32 * 32
            
            nn.Sigmoid(),)
        
    
    def forward(self, x):
        return self.model(x)
            
            

"""
Encoder for VQVAE
Uses 3 Convolution transpose 2d layers, with leaky ReLU activation functions 
and batch normalisation layers in between, basically same as encoder. Tanh 
activation function at the end for upsampling

"""    
    
class Decoder(nn.Module):
    """
    Parameters:
        latent_space -> latent dimension of VQVAE
    """
    def __init__(self, latent_space):
        super(Decoder, self).__init__()
        self.latent_space = latent_space
        self.model = nn.Sequential(
            nn.ConvTranspose2d(self.latent_space, self.latent_space*2, 
                               kernel_size=4, stride = 2, padding = 1),
            nn.BatchNorm2d(self.latent_space*2),
            nn.LeakyReLU(0.1),
            
            nn.ConvTranspose2d(self.latent_space*2, self.latent_space, 
                               kernel_size= 4, stride = 2, padding = 1),
            nn.BatchNorm2d(self.latent_space),
            nn.LeakyReLU(0.1),
            
            nn.ConvTranspose2d(self.latent_space, 3, kernel_size = 3, stride = 
                               1, padding = 1),
            nn.Tanh(),
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
        super(VQ, self).__init__()
    
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
        
        encodings, _ = self.argmin_indices(flattened_inputs)
        
        # Quantize and reshape to BHWC format
        quantized_bhwc = torch.matmul(encodings, 
                                      self.embedding_table.weight).view(shape)
        
        
        loss = self.loss(quantized_bhwc, encoder_inputs)
        
        # backpropagate and update gradients back at to the encoder using the 
        # quantized gradients
        
        quantized_bhwc = encoder_inputs + (quantized_bhwc - 
                                           encoder_inputs).detach()
        
        # restructure the VQ output to be Batch Channel Height Width format
        quantized = quantized_bhwc.permute(0, 3, 1, 2)
        # reformat memory, just like when it was transformed to bchw format
        quantized = quantized.contiguous()
        
        return quantized, loss
    
    
    """
    calculate distances between each of the flattened inputs and the
    weights of the of embedded table vectors, creates N*K distances, where 
    N is B*H*W and K the num_embeddings
    """
    
    def argmin_indices(self, flat_inputs):
        # find the smallest distance from N*K distance combinations
        # get the index tensor of the smallest distance
        
        dists =  (torch.sum(flat_inputs**2, dim=1, keepdim = True)
                         + torch.sum(self.embedding_table.weight**2, dim=1)
                         - 2 * torch.matmul(flat_inputs, 
                                            self.embedding_table.weight.t()))
        
        indexes = torch.argmin(dists, dim=1).unsqueeze(1)
        
        standalone_indexes = torch.argmin(dists, dim=1)
        # create a zeros tensor, with the position at indexes being 1
        # This creates a "beacon" so that position can be replaced by a 
        # embedded vector.
        encodings = torch.zeros(indexes.shape[0], 
                                self.num_embeddings).to(device)
        encodings.scatter_(1, indexes, 1)
        return encodings, standalone_indexes
        
    """
    the loss function is used from equation 3 of the VQVAE paper provided 
    in the references in readme.The first term of the loss fucntion, the 
    reconstruction loss, is calculated later.
    
    The stop gradients used can be applied using the detach() function, 
    which removes it from the current graph and turns it into a constant.
    
    """
    def loss(self, quantized_bhwc, encoder_inputs):
        first_term = nn.functional.mse_loss(quantized_bhwc.detach(), 
                                            encoder_inputs)
        
        second_term = nn.functional.mse_loss(quantized_bhwc, 
                                             encoder_inputs.detach())
        
        beta = self.commitment_loss
        
        
        return first_term + beta * second_term
        
        
"""
Model that compiles the encoder, Vector Quantizer and decoder together.
Some extra scaffolding added for tensor dimension compatability
"""
class VQVAE(nn.Module):
    """
    Parameters:
        num_embeddings -> the number of embeddngs for VQVAE
        latent_space -> latent dimension of VQVAE
        commitment_loss -> scalable hyperparam, set to 0.25 as default
    """
    def __init__(self, num_embeddings, latent_space, commitment_loss):
        super(VQVAE, self).__init__()
        self.num_embeddings = num_embeddings
        self.latent_space = latent_space
        self.commitment_loss = commitment_loss
        self.encoder = Encoder(self.latent_space)
        self.VQ = VQ(self.num_embeddings, self.latent_space, 
                     self.commitment_loss)
        self.decoder = Decoder(self.latent_space)
        
    def forward(self, inputs):
        outputs = self.encoder(inputs)
        quantized_outputs, loss = self.VQ(outputs)
        decoder_outputs = self.decoder(quantized_outputs)
        
        return decoder_outputs, loss
        
    #Returns decoder
    def get_decoder(self):
        return self.decoder
    #Returns encoder
    def get_encoder(self):
        return self.encoder
    #Returns Vector Quantizer layer
    def get_VQ(self):
        return self.VQ
        

"""
Basic Masked Convolutional layer for PixelCNN, used standalone and as part of 
residual blocks later.
"""
                
class MaskedConv2d(nn.Conv2d):
    """
    Parameters:
        mask_type -> mask type for PixelCNN layer, A layers do not include 
        centre pixel, while B layers do
        
        MaskedConv2d inherits parameters from regular Conv2d
    """
    def __init__(self, mask_type, *args, **kwargs):
        # Inherits 2d convolutional layer and its parameters
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        #print(self.kernel_size)
        self.mask_type = mask_type
        
        self.register_buffer('mask', self.weight.data.clone())
        
        _, _, height, width = self.weight.size()
        self.mask.fill_(1)
        # setup the mask, use floor operations to cover area above and left of
        # kernel position
        if mask_type == "A":
            self.mask[:, :, height//2, width//2:] = 0
            self.mask[:, :, height // 2 + 1:, :] = 0
        else:
            # include centre pixel
            self.mask[:, :, height // 2, width // 2 + 1:] = 0
            self.mask[:, :, height // 2 + 1:,:] = 0
        
        #register the mask
        
        
    def forward(self, x):
        #apply mask
        self.weight.data = self.weight.data * self.mask
        #use the forward from nn.Conv2d
        return super(MaskedConv2d, self).forward(x)


"""
Resdiual block built from MaskedConv2d layer.
"""
class ResBlock(nn.Module):
    """
    Parameters:
        num_embeddings -> the num of embeddings of VQVAE
        
        
    """
    def __init__(self, num_embeddings):
        super(ResBlock, self).__init__()
        self.num_embeddings = num_embeddings
        
        self.res_block = nn.Sequential(
            
            nn.Conv2d(num_embeddings, num_embeddings, kernel_size = 1),
            nn.ReLU(),
            MaskedConv2d("B", in_channels = num_embeddings, out_channels \
                         = num_embeddings//2, kernel_size = 3, padding = \
                         "same"),
            nn.ReLU(),
            nn.Conv2d(num_embeddings//2, num_embeddings, kernel_size = 1),
            nn.ReLU())
    
    def forward(self, x):
        output = self.res_block(x)
        return output + x
    
"""
PixelCNN model, using MaskedConv2d layers and ResBlocks built from
MaskedConv2d. 

The model can be modified with additional Resblocks or regular 
masked B layers optionally, however, the A layer must remain at the
start. Use ReLU activation functions in between.
"""
class PixelCNN(nn.Module):
    """
    Parameters:
        num_embeddings -> the num of embeddings of VQVAE
         
    """
    def __init__(self, num_embeddings):
        super(PixelCNN, self).__init__()
        self.num_embeddings = num_embeddings
        self.pixelcnn_model = nn.Sequential(

            MaskedConv2d("A", in_channels = num_embeddings, 
                         out_channels = num_embeddings*4, kernel_size = 9, 
                         padding = "same"),
            nn.ReLU(),
            ResBlock(num_embeddings*4),
            ResBlock(num_embeddings*4),
            MaskedConv2d("B", in_channels = num_embeddings*4, 
                         out_channels = num_embeddings*4, kernel_size = 1,
                         padding = "same"),
            nn.ReLU(),
            MaskedConv2d("B", in_channels = num_embeddings*4, 
                         out_channels = num_embeddings*4, kernel_size = 1,
                         padding = "same"),            
            nn.ReLU(),
      
            nn.Conv2d(num_embeddings*4, num_embeddings, kernel_size = 1, 
                      stride = 1, padding = "valid"),
                        
            )
        
    def forward(self, x):
        return self.pixelcnn_model(x)
    
        
        
        
        
        
        
        
    