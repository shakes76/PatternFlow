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
        self.conv2 = nn.Conv2d(dim_out, dim_out, kernel_size=3, padding=1, bias=False)
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
        return x + emb_x

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

class AttentionBlock(nn.Module):
    """
    Transformer attention block to enhance some parts of the
    data and diminish other parts
    """
    def __init__(self, dims, dim_size):
        """
        Attention Block class constructor to initialize the object

        Args:
            dims (int): number of channels
            dim_size (int): size of channels
        """
        super(AttentionBlock, self).__init__()
        self.dims = dims
        self.dim_size = dim_size
        self.mha_block = nn.MultiheadAttention(dims, 4, batch_first=True)
        self.layer_norm_block = nn.LayerNorm([dims])
        self.a_layer = nn.Sequential(
            nn.LayerNorm([dims]),
            nn.Linear(dims, dims),
            nn.ReLU(),
            nn.Linear(dims, dims),
        )

    def forward(self, x):
        """
        Method to run an input tensor forward through the attention block
        and returns the output tensor

        Args:
            x (Tensor): input tensor

        Returns:
            Tensor: output tensor
        """
        # Restructure the tensor for cross attention
        x = x.view(-1, self.dims, self.dim_size * self.dim_size).swapaxes(1, 2)
        x1 = self.layer_norm_block(x)
        x2, _ = self.mha_block(x1, x1, x1)
        x3 = x2 + x
        x4 = self.a_layer(x3) + x3
        # Return the restructured attention tensor
        return x4.swapaxes(2, 1).view(-1, self.dims, self.dim_size, self.dim_size)

class UNet(nn.Module):
    """
    Unet model consisting of a decoding block, an encoding block,
    cross attention, and residual skip connections
    """
    def __init__(self, dim_in_out=3, m_dim=64, pos_dim=256):
        """
        Unet class constructor to initialize the object

        Args:
            dim_in_out (int, optional): number of channels in the input image. Defaults to 3.
            m_dim (int, optional): dimensional multiplier for generalization. Defaults to 64.
            pos_dim (int, optional): positional dimension. Defaults to 256.
        """
        super(UNet, self).__init__()
        self.pos_dim = pos_dim 
        # Encoding part of the UNet                                     #  in --> out
        self.in_layer = ConvReluBlock(dim_in_out, m_dim)                #   3 --> 64
        self.encoder1 = EncoderBlock(m_dim, m_dim*2)                    #  64 --> 128
        self.attention1 = AttentionBlock(m_dim*2, int(m_dim/2))         # 128 --> 32
        self.encoder2 = EncoderBlock(m_dim*2, m_dim*4)                  # 128 --> 256
        self.attention2 = AttentionBlock(m_dim*4, int(m_dim/4))         # 256 --> 16
        self.encoder3 = EncoderBlock(m_dim*4, m_dim*4)                  # 256 --> 256
        self.attention3 = AttentionBlock(m_dim*4, int(m_dim/8))         # 256 --> 8

        # Bottle neck of the UNet                                       #  in --> out
        self.b1 = ConvReluBlock(m_dim*4, m_dim*8)                       # 256 --> 512
        self.b2 = ConvReluBlock(m_dim*8, m_dim*8)                       # 512 --> 512
        self.b3 = ConvReluBlock(m_dim*8, m_dim*4)                       # 512 --> 256

        # Decoding part of the UNet                                     #  in --> out
        self.decoder1 = DecoderBlock(m_dim*8, m_dim*2)                  # 512 --> 128
        self.attention4 = AttentionBlock(m_dim*2, int(m_dim/4))              # 128 --> 16
        self.decoder2 = DecoderBlock(m_dim*4, m_dim)                    # 256 --> 64
        self.attention5 = AttentionBlock(m_dim, int(m_dim/2))                #  64 --> 32
        self.decoder3 = DecoderBlock(m_dim*2, m_dim)                    # 128 --> 64
        self.attention6 = AttentionBlock(m_dim, m_dim)                  # 64 --> 64
        self.out_layer = nn.Conv2d(m_dim, dim_in_out, kernel_size=1)    # 64 --> 3
    
    def positional_embedding(self, position, dims):
        """
        Calculate the positional tensor using transformer positional embedding

        Args:
            position (Tensor): position tensor result of previous positional encoding   
            dims (int): number of channels

        Returns:
            Tensor: positional embedded tensor
        """
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, dims, 2, device="cuda").float() / dims)
        )
        positional_embedding_a = torch.sin(position.repeat(1, dims // 2) * inv_freq)
        positional_embedding_b = torch.cos(position.repeat(1, dims // 2) * inv_freq)
        positional_embedding = torch.cat([positional_embedding_a, positional_embedding_b], dim=-1)
        return positional_embedding
    
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
        position = position.unsqueeze(-1).type(torch.float)
        position = self.positional_embedding(position, self.pos_dim)

        # Encoder forward step
        x1 = self.in_layer(x)
        x2 = self.encoder1(x1, position)
        x2 = self.attention1(x2)
        x3 = self.encoder2(x2, position)
        x3 = self.attention2(x3)
        x4 = self.encoder3(x3, position)
        x4 = self.attention3(x4)

        # Bottle neck forward step
        x4 = self.b1(x4)
        x4 = self.b2(x4)
        x4 = self.b3(x4)

        # Decoder forward step
        x = self.decoder1(x4, x3, position)
        x = self.attention4(x)
        x = self.decoder2(x, x2, position)
        x = self.attention5(x)
        x = self.decoder3(x, x1, position)
        x = self.attention6(x)
        out = self.out_layer(x)

        return out