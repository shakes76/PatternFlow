import torch.nn as nn


class CompositeBlockEncoder(nn.Module):
    """
    Create new Composite block for the encoder.
    Params:
        in_channels: number of Input channels.
        out_channels: Number of output channels.
    """
    def __init__(self, in_channels, out_channels):
        super(CompositeBlockEncoder, self).__init__()
        self.block = nn.Sequential(nn.Conv2d(in_channels, out_channels,kernel_size=3, padding=1, stride=1, bias=False),
                                   nn.ReLU(True),
                                   nn.MaxPool2d(kernel_size=2))

    def forward(self, input):
        return self.block(input)
