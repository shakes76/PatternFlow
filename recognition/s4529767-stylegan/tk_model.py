import torch
import torch.nn as nn
import math

# We first map the input to an intermediate latent space W,
# which then controls the generator through adaptive
# instance normalization (AdaIN) at each convolution layer.
# Gaussian noise is added after each convolution, before
# evaluating the nonlinearity. Here “A” stands for a learned
# affine transform, and “B” applies learned per-channel scaling fac-
# tors to the noise input. The mapping network f consists of 8 lay-
# ers and the synthesis network g consists of 18 layers—two for 22
# each resolution (4 − 1024). The output of the last layer is
# converted to RGB using a separate 1 × 1 convolution, similar to
# Karras et al. [30]. Our generator has a total of 26.2M trainable
# parameters, compared to 23.1M in the traditional generator.


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
        # we initialize all weights of ..., fully-connected, and affine transform layers using N (0, 1)." [page 9]
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
        # "initialize all weights of the convolutional, ... using N(0, 1)" [page 9]
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
        self.weight = nn.Parameter(torch.zeros((1, channels, 1, 1)))

    def forward(self, noise):
        return noise * self.weight


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
        # "Learned affine transformations then specialize w to styles y = (ys, yb) that control adaptive instance
        # normalization (AdaIN) [27, 17, 21, 16] operations after each convolution layer of the synthesis network g."
        ys, yb = style.chunk(2, 1)
        return self.norm2d(image) * ys + yb


class FirstSynthesisNetworkBlock(nn.Module):
    '''
    The first block of the Synthesis network g with constant input. Start from 4 x 4 x 512
    "starting the image synthesis from a learned 4 × 4 × 512 constant tensor" [page 2]
    '''

    def __init__(self, channels, latent_dim, input_dim):
        super().__init__()

        self.adain = AdaIn(channels)
        self.leaky_relu = nn.LeakyReLU(0.2)

        # "synthesis network g consists of 18 layers — two for each resolution (4^2 − 1024^2)."
        self.constant = nn.Parameter(torch.randn(1, channels, input_dim, input_dim))
        self.style1 = A(latent_dim, channels)  # first style generator in the block (learned affine transform)
        self.noise1 = scale(B(channels))

        self.conv = ConvLayer(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)  # conv layer
        self.style2 = A(latent_dim, channels)  # second style generator for the block
        self.noise2 = scale(B(channels))

    def forward(self, w, noise):
        x = self.constant.repeat(noise.shape[0], 1, 1, 1)
        x = x + self.noise1(noise)
        x = self.adain(x, self.style1(w))
        x = self.leaky_relu(x)

        x = self.conv(x)
        x = x + self.noise2(noise)
        x = self.adain(x, self.style2(w))
        x = self.leaky_relu(x)

        return x


class SynthesisNetworkBlock(nn.Module):
    '''
    General Synthesis Network g block
    '''

    def __init__(self, in_channels, out_channels, latent_dim):
        super().__init__()

        self.adain = AdaIn(out_channels)
        # "we use leaky ReLU [41] with α = 0.2 and equalized learning rate [30] for all layers"
        self.leaky_relu = nn.LeakyReLU(0.2)

        # "synthesis network g consists of 18 layers — two for each resolution (4^2 − 1024^2)."
        self.conv1 = ConvLayer(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.noise1 = scale(B(out_channels))
        self.style1 = A(latent_dim, out_channels)  # first style generator in the block (learned affine transform)

        self.conv2 = ConvLayer(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.noise2 = scale(B(out_channels))
        self.style2 = A(latent_dim, out_channels)  # second style generator for the block

    def forward(self, previous_result, w, noise):
        x = self.conv1(previous_result)
        x = x + self.noise1(noise)
        x = self.adain(x, self.style1(w))
        x = self.leaky_relu(x)

        x = self.conv2(x)
        x = x + self.noise2(noise)
        x = self.adain(x, self.style2(w))
        x = self.leaky_relu(x)

        return x


class DiscriminatorBlock(nn.Module):
    '''
    Progressive Discriminator Convolution Block
    '''

    def __init__(self, in_channels, out_channels, layer1_kernel_size, layer1_padding,
                 layer2_kernel_size, layer2_padding):
        super().__init__()

        self.conv = nn.Sequential(
            ConvLayer(in_channels=in_channels, out_channels=out_channels, kernel_size=layer1_kernel_size,
                      padding=layer1_padding),
            nn.LeakyReLU(0.2),
            ConvLayer(in_channels=out_channels, out_channels=out_channels, kernel_size=layer2_kernel_size,
                      padding=layer2_padding),
            nn.LeakyReLU(0.2)
        )

    def forward(self, image):
        return self.conv(image)


class MappingNetwork(nn.Module):
    '''
    Mapping network constists of number_of_mapping_network_blocks fully connected layers. Maps latent z to w.
    "Our mapping network consists of 8 fully-connected layers, and the dimensionality of all input and output
    activations — including z and w — is 512"
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

        self.mapping_network = MappingNetwork(number_of_mapping_network_blocks, latent_dim)

        self.synthesis_network = nn.ModuleList([
            FirstSynthesisNetworkBlock(512, latent_dim, input_dim),
            SynthesisNetworkBlock(512, 512, latent_dim),
            SynthesisNetworkBlock(512, 512, latent_dim),
            SynthesisNetworkBlock(512, 512, latent_dim),
            SynthesisNetworkBlock(512, 256, latent_dim),
            SynthesisNetworkBlock(256, 128, latent_dim),
            SynthesisNetworkBlock(128, 64, latent_dim),
            SynthesisNetworkBlock(64, 32, latent_dim),
            SynthesisNetworkBlock(32, 16, latent_dim)
        ])
        self.to_rgbs = nn.ModuleList([
            ConvLayer(512, 3, 1),
            ConvLayer(512, 3, 1),
            ConvLayer(512, 3, 1),
            ConvLayer(512, 3, 1),
            ConvLayer(256, 3, 1),
            ConvLayer(128, 3, 1),
            ConvLayer(64, 3, 1),
            ConvLayer(32, 3, 1),
            ConvLayer(16, 3, 1)
        ])

    def forward(self, z,
                level,  # The number of layers starting from from 4 x 4
                alpha,
                noise):  # truncation param

        w = [self.mapping_network(latent) for latent in z]

        x = 0

        for i, conv in enumerate(self.synthesis_network):
            current_latent = w[0]

            if i > 0 and level > 0:
                upsample = nn.functional.interpolate(x, scale_factor=2, mode='bilinear',
                                                            align_corners=False)
                x = conv(upsample, current_latent, noise[i])
            else:
                x = conv(current_latent, noise[i])

            if i == level:
                x = self.to_rgbs[i](x)

                if i > 0 and 0 <= alpha < 1:
                    result_prev = self.to_rgbs[i - 1](upsample)
                    x = alpha * x + (1 - alpha) * result_prev

                break

        return x

class Discriminator(nn.Module):
    '''
    Discriminator
    '''

    def __init__(self):
        super().__init__()
        self.number_of_layers = 8

        self.discriminator_blocks = nn.ModuleList([
            DiscriminatorBlock(16, 32, layer1_kernel_size=3, layer1_padding=1, layer2_kernel_size=3, layer2_padding=1),
            DiscriminatorBlock(32, 64, layer1_kernel_size=3, layer1_padding=1, layer2_kernel_size=3, layer2_padding=1),
            DiscriminatorBlock(64, 128, layer1_kernel_size=3, layer1_padding=1, layer2_kernel_size=3, layer2_padding=1),
            DiscriminatorBlock(128, 256, layer1_kernel_size=3, layer1_padding=1, layer2_kernel_size=3, layer2_padding=1),
            DiscriminatorBlock(256, 512, layer1_kernel_size=3, layer1_padding=1, layer2_kernel_size=3, layer2_padding=1),
            DiscriminatorBlock(512, 512, layer1_kernel_size=3, layer1_padding=1, layer2_kernel_size=3, layer2_padding=1),
            DiscriminatorBlock(512, 512, layer1_kernel_size=3, layer1_padding=1, layer2_kernel_size=3, layer2_padding=1),
            DiscriminatorBlock(513, 512, layer1_kernel_size=3, layer1_padding=1, layer2_kernel_size=4, layer2_padding=0)
        ])

        self.from_rgbs = nn.ModuleList([
            ConvLayer(3, 16, 1),
            ConvLayer(3, 32, 1),
            ConvLayer(3, 64, 1),
            ConvLayer(3, 128, 1),
            ConvLayer(3, 256, 1),
            ConvLayer(3, 512, 1),
            ConvLayer(3, 512, 1),
            ConvLayer(3, 512, 1)
        ])

        self.fully_connected = FullyConnected(512, 1)

    def forward(self, image,
                level,  # The number of layers starting from from 4 x 4
                alpha):

        for i in range(level, -1, -1):
            current_layer = self.number_of_layers - i - 1

            if i == level:
                x = self.from_rgbs[current_layer](image)

            if i == 0:
                var = x.var(0, unbiased=False) + 1e-10
                mean_std = torch.sqrt(var).mean().expand(x.size(0), 1, 4, 4)
                x = torch.cat([x, mean_std], 1)

            # convolution
            x = self.discriminator_blocks[current_layer](x)

            if i > 0:
                x = nn.functional.interpolate(x, scale_factor=0.5, mode='bilinear')
                if i == level and 0 <= alpha < 1:
                    result_next = self.from_rgbs[current_layer + 1](image)
                    result_next = nn.functional.interpolate(result_next, scale_factor=0.5, mode='bilinear')

                    x = alpha * x + (1 - alpha) * result_next

        x = x.squeeze(2).squeeze(2)
        return self.fully_connected(x)
