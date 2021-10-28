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


class FullyConnected(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        pass

    def forward(self, x):
        pass


class ConvLayer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        pass

    def forward(self, x):
       pass


class Normalize(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x):
        pass


class A(nn.Module):
    '''
    Learned affine transform A. This module transforms w into a style vector
    '''

    def __init__(self, latent_dim, channels):
        super().__init__()
        pass

    def forward(self, w):
        pass


class AdaIn(nn.Module):
    '''
    Adaptive Instance Normalization
    AdaIN(x,y) = y_s_i ( (x_i - ni(x_i) / sigma(x_i)) + y_b_i
    "Each feature map x_i is normalised separately, then scaled and biased using the corresponding
    scalar components from style y"
    '''

    def __init__(self, x):
        super().__init__()
        pass

    def forward(self, image, style):
        pass


class B(nn.Module):
    '''
    " “B” applies learned per-channel scaling factors to the noise input"
    '''

    def __init__(self, channels):
        super().__init__()
        pass

    def forward(self, noise):
        pass


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
    '''

    def __init__(self, number_of_mapping_network_blocks, latent_dim):
        super().__init__()
        pass

    def forward(self, z):
        pass


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

