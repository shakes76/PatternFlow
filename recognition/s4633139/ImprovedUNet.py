#  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Copyright (c) 2021, H.WAKAYAMA, All rights reserved.
#  File: ImprovedUNet.py
#  Author: Hideki WAKAYAMA
#  Contact: h.wakayama@uq.net.au
#  Platform: macOS Big Sur Ver 11.2.1, Pycharm pro 2021.1
#  Time: 19/10/2021, 15:47
#  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import torch
import torch.nn as nn
import torch.nn.functional as F


class Context(nn.Module):
    """context"""
    def __init__(self, in_channels, out_channels):
        super(Context, self).__init__()
        self.context = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(p=0.3),
            nn.LeakyReLU(negative_slope=0.02, inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.02, inplace=True),
        )

    def forward(self, x):
        x = self.context(x) + x
        return x


class Localization(nn.Module):
    """localization"""
    def __init__(self, in_channels, out_channels):
        super(Localization, self).__init__()
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.02, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.02, inplace=True),
        )

    def forward(self, x):
        return self.localization(x)


class Upsampling(nn.Module):
    """upsampling"""
    def __init__(self, in_channels, out_channels):
        super(Upsampling, self).__init__()
        self.upsampling = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.02, inplace=True),
        )

    def forward(self, x):
        return self.upsampling(x)


class Segment(nn.Module):
    """segmentation layer"""
    def __init__(self, in_channels, out_channels):
        super(Segment, self).__init__()
        self.segment = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.02, inplace=True)
        )

    def forward(self, x):
        return self.segment(x)


class Conv2(nn.Module):
    """convolution stride=2"""
    def __init__(self, in_channels, out_channels):
        super(Conv2, self).__init__()
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.02, inplace=True),
        )

    def forward(self, x):
        return self.conv2(x)


class IUNet(nn.Module):
    """Improved Unet (International MICCAI Brainlesion Workshop(pp. 287-297). Springer, Cham.)"""
    def __init__(self, in_channels=3, out_channels=1, feature_size=[16, 32, 64, 128]):
        super(IUNet, self).__init__()
        self.Conv1 = nn.Conv2d(in_channels=3, out_channels=feature_size[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.Downs = nn.ModuleList()
        self.Convs = nn.ModuleList()
        self.Ups = nn.ModuleList()
        self.Segmentations = nn.ModuleList()

        self.upscale = nn.Upsample(scale_factor=2, mode='nearest')
        self.bottleneck = Context(feature_size[-1] * 2, feature_size[-1] * 2)

        #Downsampling frame
        for feature in feature_size:
            self.Downs.append(Context(feature, feature))
            self.Convs.append(Conv2(feature, feature * 2))

        #Upsampleing frame
        for feature in reversed(feature_size):
            #Upsampling
            self.Ups.append(Upsampling(feature * 2, feature))

            #Localization
            if feature != feature_size[0]:
                self.Ups.append(Localization(feature * 2, feature))
            else:
                self.Ups.append(Localization(feature * 2, feature * 2))

            #Segmentation
            self.Segmentations.append(Segment(feature, 1))

        self.final_conv = nn.Conv2d(feature_size[0] * 2, out_channels, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        skip_connections = []
        segmentation_layers = []
        idxs = [idx for idx in range(0, len(self.Ups),2)]

        x = self.Conv1(x)

        #Downsampling steps
        for i, (context_i, conv_i) in enumerate(zip(self.Downs, self.Convs)):
            x = context_i(x)
            #preserve location
            skip_connections.append(x)
            x = conv_i(x)

        x = self.bottleneck(x) + x
        skip_connections = skip_connections[:: -1]

        #Upsampling steps
        for idx in range(0, len(self.Ups), 2):
            #upsampling
            x = self.Ups[idx](x)

            #localization
            skip_connection = skip_connections[idx // 2]
            concatnate_skip = torch.cat((skip_connection, x), dim=1)
            x = self.Ups[idx + 1](concatnate_skip)

            #segmentation
            if idx == 2 or idx == 4:
                x_segment = self.Segmentations[idx // 2](x)
                segmentation_layers.append(x_segment)

        seg_scale1 = self.upscale(segmentation_layers[0])
        seg_scale2 = self.upscale(segmentation_layers[1] + seg_scale1)
        x = self.final_conv(x)
        x = x + seg_scale2
        output = F.sigmoid(x)

        return output