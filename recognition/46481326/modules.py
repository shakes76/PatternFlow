# %%
"""Import libraries required for PyTorch"""
import torch # Import PyTorch
import torch.nn as nn # Import PyTorch Neural Network
import torch.nn.functional as F # Import PyTorch Functional
import math # Import Python Math Module

# %%
class Hyperparameters():
    def __init__(self):
        self.rate_learn = 1e-3 # Rate of learn of the optimizer
        self.get_size_batch = 256 # Size of batches for the PyTorch Dataloader(s)
        self.size_image = 64
        self.channels_image = 1 # Number of input channels (from the image)
        self.channels_out = 32
        self.channels_out_res = 2
        self.len_e = 512 # Length of the embedding space
        self.size_e = 64 # Size of the embedding space
        self.loss_terminal = 0.25 # Loss at which the embedding space terminates.
        self.num_epoch = 15000 # Number of training epoch(s)
        self.fn_loss = nn.MSELoss() # Defines the loss function to be Binary Cross Entropy (BCE)
    
# %%
class VQ(nn.Module):
    """Vector Quantizer

    Args:
        len_e (int): Length of the embedding space
        size_e (int): Size of each vector in embedding space
    """
    def __init__(self, len_e, size_e, loss_threshold):
        self.len_e = len_e # Length of the embedding space
        self.size_e = size_e # Size of each vector in embedding space
        
        self.embedding = nn.Embedding(self.len_e, self.size_e) # Create embedding layer
        self.embedding.weight.data.uniform_(-1/self.len_e, 1/self.len_e)
        
        self.loss_threshold = loss_threshold # 
        
    def forward(self, x):
        """Iterate through network

        Args:
            x (_type_): State of network.
        """
        # Convert from a (Batches, Channels, Height, Width) tensor to a (Batches, Height, Width, Channels) tensor.
        x = x.permute(0, 2, 3, 1).contiguous() 
        shape_x = x.shape
        
        x_flat = x.view(-1, self.size_e) # Flatten
        distances = (
            torch.sum(x_flat ^ 2, dim=1, keepdim=True) + 
            torch.sum(self.embedding.weight ^ 2, dim=1) -
            2 * torch.matmul(x_flat, self.embedding.weight.t)
        ) # Compute distances
        
        index_encodings = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(index_encodings.shape[0], self.len_e, device=x.device)
        encodings.scatter_(1, index_encodings, 1)
        
        quantized = torch.matmul(encodings, self.embedding.weight).view(shape_x) # Quantize & unflatten
        
        loss_e = F.mse_loss(quantized.detach(), x)
        loss_q = F.mse_loss(quantized, x.detach())
        loss = loss_q + self.loss_threshold * loss_e
        
        quantized = x + (quantized - x).detach()
        mean_probabilities = torch.mean(encodings, dim=0)
        perplexity = torch.exp(
            -torch.sum(
                mean_probabilities * torch.log(mean_probabilities + 1e-10)
            )
        )
        
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings

# %%
class ResidualBlock(nn.Module):
    """Residual Network Block

    Args:
        nn (_type_): _description_
    """
    def __init__(self, channels_in, channels_out, channels_out_res):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Relu(),
            nn.Conv2d(channels_in, channels_out_res, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels_out_res, channels_out, kernel_size=1, stride=1, bias=False)
        )
    def forward(self, x):
        return x + self.block(x)

class Residual(nn.Module):
    """Residual Network

    Args:
        nn (_type_): _description_
    """
    def __init__(self, channels_in, channels_out, channels_out_res, num_res):
        super(Residual, self).__init__()
        self.num_res = num_res
        self.relu = nn.ReLU()
        self.resnet = nn.ModuleList(
            [Residual(channels_in, channels_out, channels_out_res) for index_res in range(self.num_res)]
            )
    def forward(self, x):
        for index_res in range(self.num_res):
            x = self.resnet[index_res](x)
        x = self.relu(x)
        return x

# %%
class Encoder(nn.Module):
    def __init__(self, channels_in, channels_out, channels_out_res, num_res):
        super(Encoder, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(channels_in, channels_out // 2, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(channels_out // 2, channels_out, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=1, padding=1)
        self.resnet = Residual(channels_in, channels_out, channels_out_res, num_res)
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.resnet(self.conv3(x))
        return x
    
class Decoder(nn.Module):
    def __init__(self, channels_in, channels_out, channels_out_res, num_res, channels_image):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(channels_in, channels_out, kernel_size=3, stride=1, padding=1)
        self.resnet = Residual(channels_out, channels_out, channels_out_res, num_res)
        self.deconv1 = nn.ConvTranspose2d(channels_out, channels_out // 2, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(channels_out // 2, channels_image, kernel_size=4, stride=2, padding=1)
    def forward(self, x):
        x = self.conv1(x)
        x = self.resnet(x)
        x = self.relu(self.deconv1(x))
        x = self.deconv2(x)
        return x
    
# %%
class VQVAE(nn.Module):
    def __init__(self, channels_image, channels_out, channels_out_res, num_res, len_e, size_e, loss_threshold, decay=0):
        super(VQVAE, self).__init__()
        self.encoder = Encoder(channels_image, channels_out, channels_out_res, num_res)
        self.preconv = nn.Conv2d(channels_out, size_e, kernel_size=1, stride=1)
        if (decay > 0):
            pass # May need to change this
        else:
            self.vq = VQ(len_e, size_e, loss_threshold)
        self.decoder = Decoder(size_e, channels_out, channels_out_res, num_res, channels_image)
    def forward(self, x):
        x = self.encoder(x)
        x = self.preconv(x)
        loss, quantized, perplexity, encodings = self.vq(x)
        x_recon = self.decoder(quantized)
        return loss, x_recon, perplexity