import torch.nn as nn
from blockEncoder import CompositeBlockEncoder


class Encoder(nn.Module):
    """
    Encoder network for the VQ-VAE.
    Params:
        in_channels: number of input channels (normally this is the number of color channels in the image).
        num_hidden: number of hidden conv. channels.
    """
    def __init__(self, in_channels, num_hidden):
        super(Encoder, self).__init__()
        self._conv_1 = CompositeBlockEncoder(in_channels, num_hidden)
        self._conv_2 = CompositeBlockEncoder(num_hidden, num_hidden // 2)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = self._conv_2(x)
        return x
