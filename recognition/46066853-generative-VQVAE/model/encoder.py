"""
VQVAE encoder implementation
"""

__author__="Aryaman Sharma"

import torch
import torch.nn as nn
from .residualBlock import Residual_block

class Encoder(nn.Module):
    '''
    Encoder block
    params:
        in_channels = input channels
        num_hidden = hidden blocks for encoder convolution
        residual_inter = intermediary residual block channels
    '''
    def __init__(self, in_channels, num_hidden, residual_inter):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_hidden//2,
            kernel_size=4,
            stride=2,
            padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=num_hidden//2,
            out_channels=num_hidden,
            kernel_size=4,
            stride=2,
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
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.residual1(x)
        x = self.residual2(x)
        return x