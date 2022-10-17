import torch.nn as nn


class Generator(nn.Module):
    """
    DCGAN Generator module. Code here was taken from the COMP3710 Lab 2 by Utkarsh Sharma.
    """
    def __init__(self, noise, channels):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            self._segment(noise, 64 * 16, 4, 1, 0),  # 4x4
            self._segment(64 * 16, 64 * 8, 4, 2, 1),  # 8 x 8
            self._segment(64 * 8, 64 * 4, 4, 2, 1),  # 16 x 16
            self._segment(64 * 4, 64 * 2, 4, 2, 1),  # 32 x 32
            nn.ConvTranspose2d(64 * 2, channels, 4, 2, 1),  # 64 x 64
            nn.Tanh())

    def _segment(self, input_shape, output_shape, kernel, stride, pad):
        return nn.Sequential(
            nn.ConvTranspose2d(input_shape, output_shape, kernel, stride, pad, bias=False),
            nn.BatchNorm2d(output_shape), nn.ReLU())

    def forward(self, x):
        return self.gen(x)

