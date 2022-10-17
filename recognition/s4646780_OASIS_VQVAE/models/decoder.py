import torch.nn as nn
from blockDecoder import CompositeBlockDecoder


class Decoder(nn.Module):
    """
    Decoder network for the VQ-VAE.
    Params:
        in_channels: number of input channels from the VQ layer.
        num_hidden: number of hidden conv. channels.
    """
    def __init__(self, in_channels, num_hidden):
        super(Decoder, self).__init__()
        self._conv_2 = CompositeBlockDecoder(in_channels, num_hidden)
        self._conv_3 = CompositeBlockDecoder(num_hidden, num_hidden * 2)
        self._conv_4 = nn.Conv2d(num_hidden * 2, 1, kernel_size=3, padding=1, stride=1, bias=False)

    def forward(self, inputs):
        x = self._conv_2(inputs)
        x = self._conv_3(x)
        x = self._conv_4(x)
        return x
