import math

import torch
import torch.nn as nn

# Implementing the network described in paper

# Implementing (b) Style-based generator presented at Figure 1
# https://miro.medium.com/max/692/0*nzmrf7VMLsTWt8SX
# from the original StyleGan paper

# "We first map the input to an intermediate latent space W,
# which then controls the generator through adaptive
# instance normalization (AdaIN) at each convolution layer.
# Gaussian noise is added after each convolution, before
# evaluating the nonlinearity. Here “A” stands for a learned
# affine transform, and “B” applies learned per-channel scaling fac-
# tors to the noise input. The mapping network f consists of 8 lay-
# ers and the synthesis network g consists of 18 layers—two for 22
# each resolution (4 − 1024). The output of the last layer is
# converted to RGB using a separate 1 × 1 convolution, similar to
# Karras et al. [30]. "

def scale(module):
    '''
    Scaling weights for ConvLayer, FullyConnected and Synthesis Network blocks
    '''
    name = "weight"
    hook = ScaleWeights(name)
    weight = getattr(module, name)
    module.register_parameter(name + '_orig', nn.Parameter(weight.data))
    del module._parameters[name]
    module.register_forward_pre_hook(hook)
    return module


class ScaleWeights:
    '''
    Apply scaling on a module
    '''
    def __init__(self, name):
        # the name of the attribute to scale
        self.name = name

    def scale(self, module):
        w = getattr(module, self.name + '_orig')
        fin = w.data[0][0].numel() * w.data.size(1)

        return w * math.sqrt(2 / fin)

    def __call__(self, module, _):
        weight = self.scale(module)
        setattr(module, self.name, weight)


class Normalize(nn.Module):
    '''
    Normalisation for Mapping Network f, applied on input vector
    '''
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, keepdim=True, dim=1) + 1e-10)


class FullyConnected(nn.Module):
    '''
    Fully connected layer
    '''
    def __init__(self, in_dim, out_dim):
        super().__init__()

        linear_layer = nn.Linear(in_dim, out_dim)
        linear_layer.weight.data.normal_()
        linear_layer.bias.data.zero_()

        self.linear = scale(linear_layer)

    def forward(self, x):
        return self.linear(x)


class ConvLayer(nn.Module):
    '''
    Convolutional layer
    '''
    def __init__(self, *args, **kwargs):
        super().__init__()
        conv2d = nn.Conv2d(*args, **kwargs)
        # "initialize all weights ... using N(0, 1)" [page 9]
        conv2d.weight.data.normal_()
        conv2d.bias.data.zero_()

        self.conv2d = scale(conv2d)

    def forward(self, x):
        return self.conv2d(x)


class A(nn.Module):
    '''
    Learned affine transform A. This module transforms w into a style vector
    '''

    def __init__(self, latent_dim, channels):
        super().__init__()

        self.transform = FullyConnected(latent_dim, channels * 2)

        self.transform.linear.bias.data[:channels] = 1
        self.transform.linear.bias.data[channels:] = 0

    def forward(self, w):
        return self.transform(w).unsqueeze(2).unsqueeze(3)

class B(nn.Module):
    '''
    "“B” applies learned per-channel scaling factors to the noise input"
    '''

    def __init__(self, channels):
        super().__init__()
        self.w = nn.Parameter(torch.zeros((1, channels, 1, 1)))

    def forward(self, noise):
        return noise * self.w


class AdaIn(nn.Module):
    '''
    Adaptive Instance Normalization
    AdaIN(x,y) = y_s_i ( (x_i - ni(x_i) / sigma(x_i)) + y_b_i
    "Each feature map x_i is normalised separately, then scaled and biased using the corresponding
    scalar components from style y"
    '''

    def __init__(self, x):
        super().__init__()
        self.norm2d = nn.InstanceNorm2d(x)

    def forward(self, image, style):
        factor, bias = style.chunk(2, 1)
        return self.norm2d(image) * factor + bias


class FirstSynthesisNetworkBlock(nn.Module):
    '''
    The first block of the Synthesis network g with constant input. Start from 4 x 4 x 512
    '''

    def __init__(self, channels, latent_dim, input_dim):
        super().__init__()

    def forward(self, w, noise):
        pass

class SynthesisNetworkBlock(nn.Module):
    '''
    General "Synthesis Network g" block
    '''

    def __init__(self, in_channels, out_channels, latent_dim):
        super().__init__()
        pass

    def forward(self, previous_result, w, noise):
        pass


class DiscriminatorBlock(nn.Module):
    '''
    Progressive Discriminator Convolution Block (consists of two ConvLayer blocks)
    '''

    def __init__(self, in_channels, out_channels):
        super().__init__()
        pass

    def forward(self, image):
        pass


class MappingNetwork(nn.Module):
    '''
    Mapping network has number_of_mapping_network_blocks fully connected layers
    Maps latent z to w.
    "...the mapping f is implemented using an 8-layer MLP" [page 2]

    '''

    def __init__(self, number_of_mapping_network_blocks, latent_dim):
        super().__init__()
        layers = []
        layers.append(Normalize())
        for i in range(number_of_mapping_network_blocks):
            layers.append(FullyConnected(latent_dim, latent_dim))
            layers.append(nn.LeakyReLU(0.2))

        self.mapping = nn.Sequential(*layers)

    def forward(self, z):
        return self.mapping(z)


class StyleBasedGenerator(nn.Module):
    '''
    Style-based generator, consists of Mapping Network f and Synthesis Network g
    '''

    def __init__(self, number_of_mapping_network_blocks, latent_dim, input_dim):
        super().__init__()
        pass

    def forward(self, z,
                level=0,  # The number of layers in the progressive GAN
                alpha=-1,
                noise=None):

        pass


class Discriminator(nn.Module):
    '''
    Discriminator (as seen in classic GANs)
    '''

    def __init__(self):
        super().__init__()
        pass

    def forward(self, image,
                level=0,  # The number of layers in the progressive GAN
                alpha=-1):

        pass

