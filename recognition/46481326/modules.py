__author__ = "James Chen-Smith"

# %%
"""Import libraries required for PyTorch"""
import torch # Import PyTorch
import torch.nn as nn # Import PyTorch Neural Network
import torch.nn.functional as F # Import PyTorch Functional

# %%
class Hyperparameters():
    def __init__(self):
        self.channels_image = 1 # Number of input channels (from the image)
        self.channels_out = 128 # Number of output channels
        self.channels_out_res = 32 # Number of output channels from residual
        self.fn_loss = nn.MSELoss() # Defines the loss function to be Binary Cross Entropy (BCE)
        self.interval_batch_save_vqvae = 10 # Number of batches per VQVAE save
        self.len_e = 512 # Length of the embedding space
        self.num_epoch_dcgan = 20 # Number of training epoch(s) for DCGAN
        self.num_epoch_vqvae = 2 # Number of training epoch(s) for VQVAE
        self.rate_learn_dcgan = 2e-4 # Rate of learn of the optimizer for DCGAN
        self.rate_learn_vqvae = 1e-3 # Rate of learn of the optimizer for VQVAE
        self.size_batch_dcgan = 256 # Size of batches for the PyTorch Dataloader(s) for DCGAN
        self.size_batch_vqvae = 48 # Size of batches for the PyTorch Dataloader(s) for VQVAE
        self.size_e = 64 # Size of the embedding space
        self.size_image = 64 # Size of the input image
        self.size_z_dcgan = 100 # Size of the latent space for DCGAN
        self.threshold_loss = 0.3 # Loss at which the embedding space terminates.
        self.variance = 0.0338 # Courtesy Aryanman Sharman
# %%
class VQ(nn.Module):
    """Vector Quantizer

    Args:
        len_e (int): Length of the embedding space
        size_e (int): Size of each vector in embedding space
        threshold_loss (float): Loss function beta value
    """
    def __init__(self, len_e, size_e, threshold_loss):
        super(VQ, self).__init__()
        
        self.len_e = len_e # Length of the embedding space
        self.size_e = size_e # Size of each vector in embedding space
        
        self.embedding = nn.Embedding(self.len_e, self.size_e) # Create embedding layer
        self.embedding.weight.data.uniform_(-1/self.len_e, 1/self.len_e)
        
        self.threshold_loss = threshold_loss # 
        
    def forward(self, x):
        # Convert from a (Batches, Channels, Height, Width) tensor to a (Batches, Height, Width, Channels) tensor.
        x = x.permute(0, 2, 3, 1).contiguous() 
        
        x_flat = x.view(-1, self.size_e) # Flatten
        distances = (
            torch.sum(x_flat ** 2, dim=1, keepdim=True) + 
            torch.sum(self.embedding.weight ** 2, dim=1) -
            2 * torch.matmul(x_flat, self.embedding.weight.t())
        ) # Compute distances
        
        encoded_e = torch.argmin(distances, dim=1)
        index_encodings = encoded_e.unsqueeze(1) # Minimum distance
        index_e = torch.zeros(index_encodings.shape[0], self.len_e, device=x.device)
        index_e.scatter_(1, index_encodings, 1) # One-hot encoding
        
        "Quantize & unflatten, multiplying encodings table with embeddings"
        quantized_x = torch.matmul(index_e, self.embedding.weight).view(x.shape)
        
        """Compute loss"""
        loss_e = F.mse_loss(quantized_x.detach(), x) # Read but not update quantized gradient(s)
        loss_q = F.mse_loss(quantized_x, x.detach())
        loss = loss_q + self.threshold_loss * loss_e
        
        quantized_x = x + (quantized_x - x).detach()
        
        return loss, quantized_x.permute(0, 3, 1, 2).contiguous(), index_e, encoded_e
    
    def get_quantized_x(self, x):
        encoded_e = x.unsqueeze(1)
        index_e = torch.zeros(encoded_e.shape[0], self.len_e, device=x.device)
        index_e.scatter_(1, encoded_e, 1)
        quantized_x = torch.matmul(index_e, self.embedding.weight).view(1, 64, 64, 64)
        return quantized_x.permute(0, 3, 1, 2).contiguous()

# %%
class ResidualBlock(nn.Module):
    """Residual Network Block

    Args:
        channels_in (int): Number of channels in
        channels_out (int): Number of channels out
        channel_out_res (int): Number of intermediate channels
    """
    def __init__(self, channels_in, channels_out, channels_out_res):
        super(ResidualBlock, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(channels_in, channels_out_res, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(channels_out_res, channels_out, kernel_size=1, stride=1, bias=False)
    def forward(self, x):
        x_residual = x
        x = self.relu(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x_residual + x

# %%
class Encoder(nn.Module):
    """Encoder

    Args:
        channels_in (int): Number of channels in
        channels_out (int): Number of channels out
        channel_out_res (int): Number of intermediate channels
    """
    def __init__(self, channels_in, channels_out, channels_out_res):
        super(Encoder, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(channels_in, channels_out // 2, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(channels_out // 2, channels_out, kernel_size=4, stride=2, padding=1)
        self.resnet1 = ResidualBlock(channels_out, channels_out, channels_out_res)
        self.resnet2 = ResidualBlock(channels_out, channels_out, channels_out_res)
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.resnet1(x)
        x = self.resnet2(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self, channels_in, channels_out, channels_out_res, channels_image):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(channels_in, channels_out, kernel_size=3, stride=1, padding=1)
        self.resnet1 = ResidualBlock(channels_out, channels_out, channels_out_res)
        self.resnet2 = ResidualBlock(channels_out, channels_out, channels_out_res)
        self.deconv1 = nn.ConvTranspose2d(channels_out, channels_out // 2, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(channels_out // 2, channels_image, kernel_size=4, stride=2, padding=1)
    def forward(self, x):
        x = self.conv1(x)
        x = self.resnet1(x)
        x = self.resnet2(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        return x

# %%
class VQVAE(nn.Module):
    """Vector Quantized Variational Auto Encoder

    Args:
        channels_image (int): Image channels
        channels_out (int): Number of channels out
        channel_out_res (int): Number of intermediate channels
        len_e (int): Length of the codebook
        size_e (int): Length of the codebook
    """
    def __init__(self, channels_image, channels_out, channels_out_res, len_e, size_e, threshold_loss, decay=0):
        super(VQVAE, self).__init__()
        self.encoder = Encoder(channels_image, channels_out, channels_out_res)
        self.preconv = nn.Conv2d(channels_out, size_e, kernel_size=1, stride=1)
        self.vq = VQ(len_e, size_e, threshold_loss) # Vector Quantizer
        self.decoder = Decoder(size_e, channels_out, channels_out_res, channels_image)
    def forward(self, x):
        z = self.encoder(x) # Encode
        z = self.preconv(z) # Pre-resize channel(s)
        loss, quantized_x, _, _ = self.vq(z) # (_, _) = (perplexity, encodings) 
        x_decoded = self.decoder(quantized_x)
        return loss, x_decoded
    
class Discriminator(nn.Module):
    """DCGAN Discriminator

    Args:
        channels_image (int): Image channels
        channels_intermediate (int): Intermediate channel multiplier
    """
    def __init__(self, channels_image, channels_intermediate=128):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Conv2d(channels_image, channels_intermediate, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(channels_intermediate),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels_intermediate, channels_intermediate * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(channels_intermediate * 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels_intermediate * 2, channels_intermediate * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(channels_intermediate * 4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels_intermediate * 4, channels_intermediate * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(channels_intermediate * 8),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels_intermediate * 8, 1, kernel_size=4, stride=2, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.discriminator(x)

class Generator(nn.Module):
    """DCGAN Generator

    Args:
        channels_noise (int): Noise channels
        channels_image (int): Image channels
        channels_intermediate (int): Intermediate channel multiplier
    """
    def __init__(self, channels_noise, channels_image, channels_intermediate=64):
        super(Generator, self).__init__()
        self.generator = nn.Sequential(
            nn.ConvTranspose2d(channels_noise, channels_intermediate * 16, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(channels_intermediate * 16),
            nn.ReLU(),
            nn.ConvTranspose2d(channels_intermediate * 16, channels_intermediate * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels_intermediate * 8),
            nn.ReLU(),
            nn.ConvTranspose2d(channels_intermediate * 8, channels_intermediate * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels_intermediate * 4),
            nn.ReLU(),
            nn.ConvTranspose2d(channels_intermediate * 4, channels_intermediate * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels_intermediate * 2),
            nn.ReLU(),
            nn.ConvTranspose2d(channels_intermediate * 2, channels_image, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid() # Can be tanh as well
        )

    def forward(self, x):
        return self.generator(x)