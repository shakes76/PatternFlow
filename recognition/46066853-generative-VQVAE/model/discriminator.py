"""
DCGAN discriminator

Code from COMP3710 Lab Demo2 DCGAN by Aryaman Sharma
"""

__author__="Aryaman Sharma"

import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            # Nx3x64x64
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2),
            # Nx128x32x32
            self._step(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            # Nx256x16x16
            self._step(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            self._step(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1),
            # Nx1024x4x4
            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=4, stride=2, padding=0),
            nn.Sigmoid()
            # Nx1x1x1
        )
        
    def _step(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self, x):
        return self.net(x)