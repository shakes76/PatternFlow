import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision
import math


class ConvReluBlock(nn.Module):
    """
    ConvReluBlock object consisting of a double convolution rectified
    linear layer and a group normalization used at every level of the UNET model
    """
    def __init__(self, dim_in, dim_out, residual_connection=False):
        """
        Block class constructor to initialize the object

        Args:
            dim_in (int): number of channels in the input image
            dim_out (int): number of channels produced by the convolution
            residual_connection (bool, optional): true if this block has a residual connect, false otherwise. Defaults to False.
        """
        super(ConvReluBlock, self).__init__()
        self.residual_connection = residual_connection
        self.conv1 = nn.Conv2d(dim_in, dim_out, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(dim_out, dim_out, kernel_size=3)
        self.relu = nn.ReLU()
        self.gNorm = nn.GroupNorm(1, dim_out)

    def forward(self, x):
        """
        Method to run an input tensor forward through the block
        and returns the resulting output tensor

        Args:
            x (Tensor): input tensor

        Returns:
            Tensor: output tensor
        """
        # Block 1
        x1 = self.conv1(x)
        x2 = self.gNorm(x1)
        x3 = self.relu(x2)

        # Block 2
        x4 = self.conv2(x3)
        x5 = self.gNorm(x4)

        # Handle Residuals
        if (self.residual_connection):
            x6 = F.relu(x + x5)
        else:
            x6 = x5
    
        return x6

class EncoderBlock(nn.Module):
    """
    Encoder block consisting of a max pooling layer followed by 2 ConvReluBlocks
    concatenated with the embedded position tensor
    """
    def __init__(self, dim_in, dim_out, emb_dim=256):
        """
        Encoder Block class constructor to initialize the object

        Args:
            dim_in (int): number of channels in the input image.
            dim_out (int): number of channels produced by the convolution.
            emb_dim (int, optional): number of channels in the embedded layer. Defaults to 256.
        """
        super(EncoderBlock, self).__init__()
        self.encoder_block1 = ConvReluBlock(dim_in, dim_in, residual_connection=True)
        self.encoder_block2 = ConvReluBlock(dim_in, dim_out)
        self.pool = nn.MaxPool2d(2)
        self.embedded_block = nn.Sequential(nn.ReLU(), nn.Linear(emb_dim, dim_out))

    def forward(self, x, position):
        """
        Method to run an input tensor forward through the encoder
        and returns the resulting tensor

        Args:
            x (Tensor): input tensor
            position (Tensor): position tensor

        Returns:
            Tensor: output tensor concatenated with the position tensor
        """
        x = self.pool(x)
        x = self.encoder_block1(x)
        x = self.encoder_block2(x)
        emb_x = self.embedded_block(position)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return emb_x + x

class DecoderBlock(nn.Module):
    """
    Decoder block consisting of an upsample layer followed by 2 ConvReluBlocks
    concatenated with the embedded position tensor
    """
    def __init__(self, dim_in, dim_out, emb_dim=256):
        """
        Decoder Block class constructor to initialize the object

        Args:
            dim_in (int): number of channels in the input image.
            dim_out (int): number of channels produced by the convolution.
            emb_dim (int, optional): number of channels in the embedded layer. Defaults to 256. 
        """
        super(DecoderBlock, self).__init__()
        self.upSample_block = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.decoder_block1 = ConvReluBlock(dim_in, dim_in, residual_connection=True)
        self.decoder_block2 = ConvReluBlock(dim_in, dim_out)
        self.embedded_block = nn.Sequential(nn.ReLU(), nn.Linear(emb_dim, dim_out))
        self.embedded_block = nn.Sequential(nn.ReLU(), nn.Linear(emb_dim, dim_out))

    def forward(self, x, skip_tensor, position):
        """
        Method to run an input tensor forward through the decoder
        and returns the output tensor

        Args:
            x (Tensor): input tensor
            skip_tensor (Tensor): tensor representing the skip connection from the encoder
            position (Tensor): position tensor result of positional encoding

        Returns:
            Tensor: output tensor concatenated with the position tensor
        """
        
        x = self.upSample_block(x)
        x = torch.cat([skip_tensor, x], dim=1)
        x = self.decoder_block1(x)
        x = self.decoder_block2(x)
        emb_x = self.embedded_block(position)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return emb_x + x

class UNet(nn.Module):
    """
    Unet model consisting of a decoding block and an encoding block
    followed by a resulting convolution layer with out dimension 1
    """
    def __init__(self, dim_in_encoder=1, dim_out_encoder=1024, dim_in_decoder=1024, dim_out_decoder=64):
        """
        Unet class constructor to initialize the object

        Args:
            dim_in_encoder (int, optional): number of channels in the input image. Defaults to 3.
            dim_out_encoder (int, optional): number of channels produced by the convolution. Defaults to 1024.
            dim_in_decoder (int, optional): number of channels in the input image. Defaults to 1024.
            dim_out_decoder (int, optional): number of channels produced by the convolution. Defaults to 64.
        """
        super(UNet, self).__init__()
        self.unet_encoder = EncoderBlock(dim_in_encoder, dim_out_encoder)
        self.unet_decoder = DecoderBlock(dim_in_decoder, dim_out_decoder)
        self.unet_head = nn.Conv2d(dim_out_decoder, 1, kernel_size=1)
        self.position_block = PositionalEmbeddingTransformerBlock(32)

    def forward(self, x, position):
        """
        Method to run an input tensor forward through the unet
        and returns the output from all Unet layers

        Args:
            x (Tensor): input tensor
            position (Tensor): position tensor result of positional encoding

        Returns:
            Tensor: output tensor
        """
        position = self.position_block(position)
        encoder_blocks = self.unet_encoder(x, position)
        out = self.unet_decoder(encoder_blocks[::-1][0], encoder_blocks[::-1][1:], position)
        out = self.unet_head(out)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, function):
        super().__init__()
        self.function = function

    def forward(self, x):
        return self.function(x) + x

# unet = UNet()
# print(unet)
# x = torch.randn(5, 1, 256, 256)
# print(len(x))
# print("unet features")
# print(unet(x, torch.Tensor([5, 1, 64, 128])).shape)