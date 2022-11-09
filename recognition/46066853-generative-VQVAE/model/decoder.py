"""
VQVAE decoder implementation
"""

__author__ = "Aryaman Sharma"

import torch
import torch.nn as nn
from .residualBlock import Residual_block

class Decoder(nn.Module):
    '''
    Decoder network
    params:
        in_channels: input channels (from vq)
        num_hidden: hidden convolution channels
        residual_inter: intermediary residual block channels
    '''
    def __init__(self, in_channels, num_hidden, residual_inter):
        super(Decoder, self).__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_hidden,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.residual1 = Residual_block(
            in_channels=num_hidden,
            hidden_inter=residual_inter,
            hidden_final=num_hidden
        )
        self.residual2 = Residual_block(
            in_channels=num_hidden,
            hidden_inter=residual_inter,
            hidden_final=num_hidden
        )
        self.transpose1 = nn.ConvTranspose2d(
            in_channels=num_hidden,
            out_channels=num_hidden//2,
            kernel_size=4,
            stride=2,
            padding=1
        )
        self.transpose2 = nn.ConvTranspose2d(
            in_channels=num_hidden//2,
            out_channels=3,
            kernel_size=4,
            stride=2,
            padding=1
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.residual1(x)
        x = self.residual2(x)
        x = self.transpose1(x)
        x = self.transpose2(x)
        return x