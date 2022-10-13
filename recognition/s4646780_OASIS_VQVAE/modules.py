import torch.nn as nn


class CompositeBlockEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CompositeBlockEncoder, self).__init__()
        self.block = nn.Sequential(nn.Conv2d(in_channels, out_channels,
                                             kernel_size=3, padding=1, stride=1, bias=False),
                                   nn.ReLU(True),
                                   nn.MaxPool2d(kernel_size=2))

    def forward(self, input):
        return self.block(input)


class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens):
        super(Encoder, self).__init__()
        self._conv_1 = CompositeBlockEncoder(in_channels, num_hiddens)
        self._conv_2 = CompositeBlockEncoder(num_hiddens, num_hiddens // 2)
        # self._conv_3 = CompositeBlockEncoder(num_hiddens//2, num_hiddens//2)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = self._conv_2(x)
        # x = self._conv_3(x)
        return x


class CompositeBlockDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CompositeBlockDecoder, self).__init__()
        self.block = nn.Sequential(nn.Conv2d(in_channels, out_channels,
                                             kernel_size=3, padding=1, stride=1, bias=False),
                                   nn.ReLU(True),
                                   nn.Upsample(scale_factor=2, mode='bilinear'))

    def forward(self, input):
        return self.block(input)


class Decoder(nn.Module):
    def __init__(self, in_channels, num_hiddens):
        super(Decoder, self).__init__()
        # self._conv_1 = CompositeBlockDecoder(in_channels, num_hiddens)
        self._conv_2 = CompositeBlockDecoder(in_channels, num_hiddens)
        self._conv_3 = CompositeBlockDecoder(num_hiddens, num_hiddens * 2)
        self._conv_4 = nn.Conv2d(num_hiddens * 2, 1, kernel_size=3, padding=1, stride=1, bias=False)

    def forward(self, inputs):
        # x = self._conv_1(inputs)
        x = self._conv_2(inputs)
        x = self._conv_3(x)
        x = self._conv_4(x)
        return x
