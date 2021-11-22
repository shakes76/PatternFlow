import math
import torch.nn.functional as F
import torch
import torch.nn as nn


class sampleByFactor(nn.Module):
    """
        Up or down sample can use this class to use in nn.Sequential to build network
    """

    def __init__(self, factor):
        super(sampleByFactor, self).__init__()
        self.scale_up_factor = 2  # follow the origin paper. Progressive (4->8->16...)

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_up_factor, mode='nearest')


class Residual_concat(nn.Module):
    """
        Residual concat to concat last stage's feature map and this stage's feature.
        Thinking from ResNet.
    """

    def __init__(self, last_featureMap, current_featureMap, factor=0.):
        """
        Constructor of residual concat
        Args:
            last_featureMap:  The network structure of last stage used to generate feature map from last stage
            current_featureMap: The network structure of this stage used to generate feature map from this stage
            factor: concat factor
        """
        super(Residual_concat, self).__init__()
        self.last_featureMap = last_featureMap
        self.current_featureMap = current_featureMap
        self.factor = factor

    def increase_dfactor(self, dfactor):
        self.factor = self.factor + dfactor
        self.factor = max(0, min(self.factor, 1.0))

    def get_factor(self):
        return self.factor

    def forward(self, x):
        return self.last_featureMap(x).mul(1.0 - self.factor) + self.current_featureMap(x).mul(self.factor)


class Equalized_learning_rate_Conv(nn.Module):
    """
        kernel's weight all use N(0,1) to init, bias use 0 to init
    """

    def __init__(self, input, output, kernel_size, stride, padding):
        super(Equalized_learning_rate_Conv, self).__init__()
        self.weight_param = nn.Parameter(torch.FloatTensor(output, input, kernel_size, kernel_size).normal_(0.0, 1.0))
        self.bias_param = nn.Parameter(torch.FloatTensor(output).fill_(0))
        self.stride = stride
        self.padding = padding
        self.c = (kernel_size ** 2) * input
        self.gain = math.sqrt(2.)

    def forward(self, x):
        return F.conv2d(input=x,
                        weight=self.weight_param.mul(self.gain / math.sqrt(self.c)),  # scale the weight on runtime
                        bias=self.bias_param,
                        stride=self.stride, padding=self.padding)


class Equalized_learning_rate_Linear(nn.Module):
    """
        Dense layers' weight all use N(0,1) to init, bias use 0 to init
    """

    def __init__(self, input, output):
        super(Equalized_learning_rate_Linear, self).__init__()
        self.weight_param = nn.Parameter(torch.FloatTensor(output, input).normal_(0.0, 1.0))
        self.bias_param = nn.Parameter(torch.FloatTensor(output).fill_(0))
        self.c = input
        self.gain = math.sqrt(2.)

    def forward(self, x):
        N = x.size(0)
        if not self.training:  # in order to avoid confused bug
            x = x.to('cpu')

        return F.linear(input=x.view(N, -1), weight=self.weight_param.mul(self.gain / math.sqrt(self.c)),
                        bias=self.bias_param)


class MiniBatchStd(nn.Module):
    """MiniBatchStd to record some statistics information to guide the model"""
    def __init__(self):
        super(MiniBatchStd, self).__init__()
        self.group_size = 4

    def forward(self, x):
        size = x.size()
        batch_size = size[0]
        mini_group_size = min(batch_size, self.group_size)
        if batch_size % mini_group_size != 0:  # In order to be more robust
            mini_group_size = batch_size
        # G = int(size[0] / mini_group_size)
        if mini_group_size > 1:
            y = x.view(-1, mini_group_size, size[1], size[2], size[3])
            G, S, C, H, W = y.size()
            y = torch.var(y, 1)
            y = torch.sqrt(y + 1e-8)
            y = y.view(G, -1)
            y = torch.mean(y, 1).view(G, 1)
            y = y.expand(G, H * W).view((G, 1, 1, H, W))
            y = y.expand(G, mini_group_size, -1, -1, -1)
            y = y.contiguous().view((-1, 1, H, W))

        return torch.cat([x, y], dim=1)
