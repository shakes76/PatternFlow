import torch.nn as nn


class Discriminator(nn.Module):
    """
    DCGAN Discriminator module. Code here was taken from the COMP3710 Lab 2 by Utkarsh Sharma.
    """
    def __init__(self, channels):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(channels, 64, 4, 2, 1), nn.LeakyReLU(0.2),  # B x 32 x 32
            self._segment(64, 64 * 2, 4, 2, 1),  # B x 16 x 16
            self._segment(64 * 2, 64 * 4, 4, 2, 1),  # B x 8 x 8
            self._segment(64 * 4, 64 * 8, 4, 2, 1),  # B x 4 x 4
            nn.Conv2d(64 * 8, 1, 4, 2, 0),  # B x 1 x 1
            nn.Sigmoid())

    def _segment(self, input_shape, output_shape, kernel, stride, pad):
        return nn.Sequential(
            nn.Conv2d(input_shape, output_shape, kernel, stride, pad, bias=False),
            nn.BatchNorm2d(output_shape),
            nn.LeakyReLU(0.2))

    def forward(self, x):
        return self.disc(x)
