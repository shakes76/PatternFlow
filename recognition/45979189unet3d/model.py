import os
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

'''
Decoder layers for the model
'''
class Decoder(nn.Module):
  def __init__(self, in_channels, middle_channels, out_channels):
    super(Decoder, self).__init__()
    self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
    self.conv_relu = nn.Sequential(
        nn.Conv3d(middle_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU()
        )
  def forward(self, x1, x2):
    x1 = self.up(x1)
    x1 = torch.cat((x1, x2), dim=1) #add shortcut layer and input layer together
    x1 = self.conv_relu(x1)
    return x1

'''
The UNet3D class

Architecture
1,64,128,128             ->                 32,64,128,128 -> 6,64,128,128
    32,32,64,64          ->            32,32,64,64
        64,16,32,32      ->      64,16,32,32
            128,8,16,16  ->  128,8,16,16
                     256,4,8,8

'''
class UNet3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool3d(kernel_size=2))

        self.layer2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool3d(kernel_size=2))
        
        self.layer3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool3d(kernel_size=2))
        
        self.layer4 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool3d(kernel_size=2))


        self.decode3 = Decoder(256, 128+128, 128)
        self.decode2 = Decoder(128, 64+64, 64)
        self.decode1 = Decoder(64, 32+32, 32)
        self.decode0 = nn.Sequential(
            nn.ConvTranspose3d(32, 32, kernel_size=2, stride=2),
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.ReLU()
            )
        self.conv_last = nn.Conv3d(32, 6, 1)


    def forward(self, input):
        #size in (Batchsize, channel, height, width, depth)
        e1 = self.layer1(input) # 32,32,64,64
        e2 = self.layer2(e1) # 64,16,32,32
        e3 = self.layer3(e2) # 128,8,16,16
        e4 = self.layer4(e3) # 256,4,8,8
        
        d3 = self.decode3(e4, e3) # 128,8,16,16
        d2 = self.decode2(d3, e2) # 64,16,32,32
        d1 = self.decode1(d2, e1) # 32,32,64,64
        d0 = self.decode0(d1) # 32,64,128,128
        out = self.conv_last(d0) # 6,64,128,128
        
        return out
