"""
Residual block 
"""

__author__ = "Aryaman Sharma"

import torch
import torch.nn as nn

class Residual_block(nn.Module):
    '''
    Create new Residual block
    Params:
        in_channels: Input channels
        hidden_inter: hidden channels for intermediate convolution
        hidden_final: Number of channels for output convolution
    '''
    def __init__(self, in_channels, hidden_inter, hidden_final):
        super(Residual_block, self).__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=hidden_inter,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_inter,
                out_channels=hidden_final,
                kernel_size=1,
                stride=1,
                bias=False
            )
        )

    def forward(self, x):
        # Skip connection
        return x + self.net(x)