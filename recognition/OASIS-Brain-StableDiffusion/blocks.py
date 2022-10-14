import torch.nn as nn
import torch
import math


class ConvReluBlock(nn.Module):
    """
    ConvReluBlock object consisting of a double convolution rectified
    linear layer used at every level of the UNET model
    """
    def __init__(self, dim_in, dim_out):
        """
        Block class constructor to initialize the object

        Args:
            dim_in (int): number of channels in the input image
            dim_out (int): number of channels produced by the convolution
        """
        super(ConvReluBlock, self).__init__()
        self.conv1 = nn.Conv2d(dim_in, dim_out, kernel_size=3)
        self.conv2 = nn.Conv2d(dim_out, dim_out, kernel_size=3)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Method to run an input tensor forward through the block
        and returns the resulting output tensor

        Args:
            x (Tensor): input tensor

        Returns:
            Tensor: output tensor
        """
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        return x

class EncoderBlock(nn.Module):
    """
    Encoder block consisting of 5 ConvReluBlocks each
    followed by a max pooling layer
    """
    def __init__(self, dim_in=3, dim_out=1024):
        """
        Encoder Block class constructor to initialize the object

        Args:
            dim_in (int, optional): number of channels in the input image. Defaults to 3.
            dim_out (int, optional): number of channels produced by the convolution. Defaults to 1024.
        """
        super(EncoderBlock, self).__init__()
        self.encoder_block1 = ConvReluBlock(dim_in, 64)
        self.encoder_block2 = ConvReluBlock(64, 128)
        self.encoder_block3 = ConvReluBlock(128, 256)
        self.encoder_block4 = ConvReluBlock(256, 512)
        self.encoder_block5 = ConvReluBlock(512, dim_out)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        """
        Method to run an input tensor forward through the encoder
        and returns the resulting outputs from each encoder layer

        Args:
            x (Tensor): input tensor

        Returns:
            List: list of output tensors from each layer 
        """
        encoder_blocks = [] # list of encoder blocks output
        # BLOCK 1 followed by max pooling
        x = self.encoder_block1(x) 
        encoder_blocks.append(x)
        x = self.pool(x)
        # BLOCK 2 followed by max pooling
        x = self.encoder_block2(x)
        encoder_blocks.append(x)
        x = self.pool(x)
        # BLOCK 3 followed by max pooling
        x = self.encoder_block3(x)
        encoder_blocks.append(x)
        x = self.pool(x)
        # BLOCK 4 followed by max pooling
        x = self.encoder_block4(x)
        encoder_blocks.append(x)
        x = self.pool(x)
        # BLOCK 5 followed by max pooling
        x = self.encoder_block5(x)
        encoder_blocks.append(x)
        x = self.pool(x)
        return encoder_blocks

#Transformer Architecture 
#https://kazemnejad.com/blog/transformer_architecture_positional_encoding/
class PositionalEmbeddingTransformerBlock(nn.Module):
    def __init__(self, dim_in):
        super().__init__()
        self.dim_in = dim_in

    # delete if it doesn't work
    def forward(self, timePos):
        freq = 1.0 / (
            10000 
            ** (torch.arange(self.dim_in, device=timePos.device).float() / self.dim_in)
        )
        positional_encoding_a = torch.sin(timePos.repeat(1, self.dim_in // 2) * freq)
        positional_encoding_b = torch.cos(timePos.repeat(1, self.dim_in // 2) * freq)
        positional_encoding = torch.cat((positional_encoding_a, positional_encoding_b), dim=-1)
        return positional_encoding
    
    # delete if it doesn't work
    def forward2(self, timePos):
        embeddings = math.log(10000) / ((self.dim_in // 2) - 1)
        embeddings = torch.exp(torch.arange(self.dim_in // 2, device=timePos.device) * -embeddings)
        embeddings = timePos[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings