import torch

from torch import nn
from torch.nn import functional as F
from torch.autograd import Function

from math import sqrt
import random


class EqualLearningRate:
    """
    This class is used for equal_learning_rate(), which ams to make a weight
    of a module, such  as linear or conv, to be the same as another.
    """

    def __init__(self, module_name):
        self.name = module_name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()
        return weight * sqrt(2 / fan_in)

    @staticmethod
    def perform(component, name):
        function = EqualLearningRate(name)
        weight = getattr(component, name)
        del component._parameters[name]
        component.register_parameter(name + '_orig', nn.Parameter(weight.data))
        component.register_forward_pre_hook(function)

        return function

    def __call__(self, component, data):
        setattr(component, self.name, self.compute_weight(component))


def equal_learning_rate(component, component_name='weight'):
    """
    To make a weight of a module, such  as linear or conv, to be the same as
    another.
    """
    EqualLearningRate.perform(component, component_name)

    return component


class UpsamplingLayperByFusing(nn.Module):
    def __init__(self, input_c, output_c, filter_dimension, padding=0):
        """
        A fused up-sampling module that used in network.
        """
        super().__init__()

        weight = torch.randn(input_c, output_c,
                             filter_dimension, filter_dimension)
        bias = torch.zeros(output_c)

        fan_in = input_c * filter_dimension * filter_dimension

        self.bias = nn.Parameter(bias)
        self.weight = nn.Parameter(weight)
        self.pad = padding
        self.multi = sqrt(2 / fan_in)

    def forward(self, data):
        pad_array = [1, 1, 1, 1]
        weights = F.pad(self.weight * self.multi, pad_array)
        weights = (
                          weights[:, :, 1:, 1:]
                          + weights[:, :, :-1, 1:]
                          + weights[:, :, 1:, :-1]
                          + weights[:, :, :-1, :-1]
                  ) / 4

        output = F.conv_transpose2d(data, weights,
                                    self.bias, stride=2, padding=self.pad)

        return output


class DownsamplingLayperByFusing(nn.Module):
    """
    A fused down-sampling module that used in network.
    """

    def __init__(self, input_c, output_c, filter_dimension, padding=0):
        super().__init__()

        weight = torch.randn(output_c, input_c,
                             filter_dimension, filter_dimension)
        bias = torch.zeros(output_c)

        fan_in = input_c * filter_dimension * filter_dimension
        self.multiplier = sqrt(2 / fan_in)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

        self.pad = padding

    def forward(self, input):
        pad_array = [1, 1, 1, 1]
        weight = F.pad(self.weight * self.multiplier, pad_array = [1, 1, 1, 1])
        weight = (
                         weight[:, :, 1:, 1:]
                         + weight[:, :, :-1, 1:]
                         + weight[:, :, 1:, :-1]
                         + weight[:, :, :-1, :-1]
                 ) / 4

        out = F.conv2d(input, weight, self.bias, stride=2, padding=self.pad)

        return out


class PixelWiseNormalisation(nn.Module):
    def __init__(self):
        """
        Normalisation process for the input data.
        """
        super().__init__()

    def forward(self, data):
        return data / torch.sqrt(
            torch.mean(data ** 2, dim=1, keepdim=True) + 1e-8)


class BlurFunctionBackward(Function):
    """
    This class is used to return the gradient of the data.
    """
    @staticmethod
    def forward(ctx, gradient_output, filter, flipping_kernel):
        """The forward direction"""
        ctx.save_for_backward(filter, flipping_kernel)
        return F.conv2d(gradient_output, flipping_kernel, padding=1, groups=gradient_output.shape[1])

    @staticmethod
    def backward(ctx, gradgrad_output):
        """The backward direction"""
        filter, flipping_filter = ctx.saved_tensors
        gradient_in = F.conv2d(
            gradgrad_output, filter, padding=1, groups=gradgrad_output.shape[1]
        )
        return gradient_in, None, None


class BlurFunction(Function):
    @staticmethod
    def forward(ctx, input, kernel, kernel_flip):
        ctx.save_for_backward(kernel, kernel_flip)

        output = F.conv2d(input, kernel, padding=1, groups=input.shape[1])

        return output

    @staticmethod
    def backward(ctx, grad_output):
        kernel, kernel_flip = ctx.saved_tensors

        grad_input = BlurFunctionBackward.apply(grad_output, kernel,
                                                kernel_flip)

        return grad_input, None, None


blur = BlurFunction.apply


class Blur(nn.Module):
    def __init__(self, channel):
        super().__init__()

        weight = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]],
                              dtype=torch.float32)
        weight = weight.view(1, 1, 3, 3)
        weight = weight / weight.sum()
        weight_flip = torch.flip(weight, [2, 3])

        self.register_buffer('weight', weight.repeat(channel, 1, 1, 1))
        self.register_buffer('weight_flip',
                             weight_flip.repeat(channel, 1, 1, 1))

    def forward(self, input):
        return blur(input, self.weight, self.weight_flip)
        # return F.conv2d(input, self.weight, padding=1, groups=input.shape[1])


class ConvLayerEqual(nn.Module):
    def __init__(self, *args, **kwargs):
        """ A basic 2D Convolution Layer"""
        super().__init__()

        conv = nn.Conv2d(*args, **kwargs)
        conv.weight.data.normal_()
        conv.bias.data.zero_()
        self.conv = equal_learning_rate(conv)

    def forward(self, input):
        return self.conv(input)


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        linear = nn.Linear(in_dim, out_dim)
        linear.weight.data.normal_()
        linear.bias.data.zero_()

        self.linear = equal_learning_rate(linear)

    def forward(self, input):
        return self.linear(input)


class ConvBlock(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size,
            padding,
            kernel_size2=None,
            padding2=None,
            downsample=False,
            fused=False,
    ):
        super().__init__()

        pad1 = padding
        pad2 = padding
        if padding2 is not None:
            pad2 = padding2

        kernel1 = kernel_size
        kernel2 = kernel_size
        if kernel_size2 is not None:
            kernel2 = kernel_size2

        self.conv1 = nn.Sequential(
            ConvLayerEqual(in_channel, out_channel, kernel1, padding=pad1),
            nn.LeakyReLU(0.2),
        )

        if downsample:
            if fused:
                self.conv2 = nn.Sequential(
                    Blur(out_channel),
                    DownsamplingLayperByFusing(out_channel, out_channel,
                                               kernel2, padding=pad2),
                    nn.LeakyReLU(0.2),
                )

            else:
                self.conv2 = nn.Sequential(
                    Blur(out_channel),
                    ConvLayerEqual(out_channel, out_channel, kernel2,
                                padding=pad2),
                    nn.AvgPool2d(2),
                    nn.LeakyReLU(0.2),
                )

        else:
            self.conv2 = nn.Sequential(
                ConvLayerEqual(out_channel, out_channel, kernel2, padding=pad2),
                nn.LeakyReLU(0.2),
            )

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        return out


class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, in_channel, style_dim):
        super().__init__()

        self.norm = nn.InstanceNorm2d(in_channel)
        self.style = EqualLinear(style_dim, in_channel * 2)

        self.style.linear.bias.data[:in_channel] = 1
        self.style.linear.bias.data[in_channel:] = 0

    def forward(self, input, style):
        style = self.style(style).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1)

        out = self.norm(input)
        out = gamma * out + beta

        return out


class NoiseInjection(nn.Module):
    def __init__(self, channel):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1, channel, 1, 1))

    def forward(self, image, noise):
        return image + self.weight * noise


class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)

        return out


class StyledConvBlock(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size=3,
            padding=1,
            style_dim=512,
            initial=False,
            upsample=False,
            fused=False,
    ):
        super().__init__()

        if initial:
            self.conv1 = ConstantInput(in_channel)

        else:
            if upsample:
                if fused:
                    self.conv1 = nn.Sequential(
                        UpsamplingLayperByFusing(
                            in_channel, out_channel, kernel_size,
                            padding=padding
                        ),
                        Blur(out_channel),
                    )

                else:
                    self.conv1 = nn.Sequential(
                        nn.Upsample(scale_factor=2, mode='nearest'),
                        ConvLayerEqual(
                            in_channel, out_channel, kernel_size,
                            padding=padding
                        ),
                        Blur(out_channel),
                    )

            else:
                self.conv1 = ConvLayerEqual(
                    in_channel, out_channel, kernel_size, padding=padding
                )

        self.noise1 = equal_learning_rate(NoiseInjection(out_channel))
        self.adain1 = AdaptiveInstanceNorm(out_channel, style_dim)
        self.lrelu1 = nn.LeakyReLU(0.2)

        self.conv2 = ConvLayerEqual(out_channel, out_channel, kernel_size,
                                 padding=padding)
        self.noise2 = equal_learning_rate(NoiseInjection(out_channel))
        self.adain2 = AdaptiveInstanceNorm(out_channel, style_dim)
        self.lrelu2 = nn.LeakyReLU(0.2)

    def forward(self, input, style, noise):
        out = self.conv1(input)
        out = self.noise1(out, noise)
        out = self.lrelu1(out)
        out = self.adain1(out, style)

        out = self.conv2(out)
        out = self.noise2(out, noise)
        out = self.lrelu2(out)
        out = self.adain2(out, style)

        return out


class G(nn.Module):
    def __init__(self, code_dim, fused=True):
        super().__init__()

        self.progression = nn.ModuleList(
            [
                StyledConvBlock(512, 512, 3, 1, initial=True),  # 4
                StyledConvBlock(512, 512, 3, 1, upsample=True),  # 8
                StyledConvBlock(512, 512, 3, 1, upsample=True),  # 16
                StyledConvBlock(512, 512, 3, 1, upsample=True),  # 32
                StyledConvBlock(512, 256, 3, 1, upsample=True),  # 64
                StyledConvBlock(256, 128, 3, 1, upsample=True, fused=fused),
                # 128
                StyledConvBlock(128, 64, 3, 1, upsample=True, fused=fused),
                # 256
                StyledConvBlock(64, 32, 3, 1, upsample=True, fused=fused),
                # 512
                StyledConvBlock(32, 16, 3, 1, upsample=True, fused=fused),
                # 1024
            ]
        )

        self.to_rgb = nn.ModuleList(
            [
                ConvLayerEqual(512, 3, 1),
                ConvLayerEqual(512, 3, 1),
                ConvLayerEqual(512, 3, 1),
                ConvLayerEqual(512, 3, 1),
                ConvLayerEqual(256, 3, 1),
                ConvLayerEqual(128, 3, 1),
                ConvLayerEqual(64, 3, 1),
                ConvLayerEqual(32, 3, 1),
                ConvLayerEqual(16, 3, 1),
            ]
        )

        # self.blur = Blur()

    def forward(self, style, noise, step=0, alpha=-1, mixing_range=(-1, -1)):
        out = noise[0]

        if len(style) < 2:
            inject_index = [len(self.progression) + 1]

        else:
            inject_index = sorted(
                random.sample(list(range(step)), len(style) - 1))

        crossover = 0

        for i, (conv, to_rgb) in enumerate(zip(self.progression, self.to_rgb)):
            if mixing_range == (-1, -1):
                if crossover < len(inject_index) and i > inject_index[
                    crossover]:
                    crossover = min(crossover + 1, len(style))

                style_step = style[crossover]

            else:
                if mixing_range[0] <= i <= mixing_range[1]:
                    style_step = style[1]

                else:
                    style_step = style[0]

            if i > 0 and step > 0:
                out_prev = out

            out = conv(out, style_step, noise[i])

            if i == step:
                out = to_rgb(out)

                if i > 0 and 0 <= alpha < 1:
                    skip_rgb = self.to_rgb[i - 1](out_prev)
                    skip_rgb = F.interpolate(skip_rgb, scale_factor=2,
                                             mode='nearest')
                    out = (1 - alpha) * skip_rgb + alpha * out

                break

        return out


class Styled_G(nn.Module):
    def __init__(self, code_dim=512, n_mlp=8):
        super().__init__()

        self.generator = G(code_dim)

        layers = [PixelWiseNormalisation()]
        for i in range(n_mlp):
            layers.append(EqualLinear(code_dim, code_dim))
            layers.append(nn.LeakyReLU(0.2))

        self.style = nn.Sequential(*layers)

    def forward(
            self,
            input,
            noise=None,
            step=0,
            alpha=-1,
            mean_style=None,
            style_weight=0,
            mixing_range=(-1, -1),
    ):
        styles = []
        if type(input) not in (list, tuple):
            input = [input]

        for i in input:
            styles.append(self.style(i))

        batch = input[0].shape[0]

        if noise is None:
            noise = []

            for i in range(step + 1):
                size = 4 * 2 ** i
                noise.append(
                    torch.randn(batch, 1, size, size, device=input[0].device))

        if mean_style is not None:
            styles_norm = []

            for style in styles:
                styles_norm.append(
                    mean_style + style_weight * (style - mean_style))

            styles = styles_norm

        return self.generator(styles, noise, step, alpha,
                              mixing_range=mixing_range)

    def mean_style(self, input):
        style = self.style(input).mean(0, keepdim=True)

        return style


class D(nn.Module):
    def __init__(self, fused=True, from_rgb_activate=False):
        """
        The networl of the discriminator. This models follows the paper carefully.
        This network is also progressive, which supports multiple size of input image.
        """
        super().__init__()

        self.progression = nn.ModuleList(
            [
                ConvBlock(16, 32, 3, 1, downsample=True, fused=fused),  # 512
                ConvBlock(32, 64, 3, 1, downsample=True, fused=fused),  # 256
                ConvBlock(64, 128, 3, 1, downsample=True, fused=fused),  # 128
                ConvBlock(128, 256, 3, 1, downsample=True, fused=fused),  # 64
                ConvBlock(256, 512, 3, 1, downsample=True),  # 32
                ConvBlock(512, 512, 3, 1, downsample=True),  # 16
                ConvBlock(512, 512, 3, 1, downsample=True),  # 8
                ConvBlock(512, 512, 3, 1, downsample=True),  # 4
                ConvBlock(513, 512, 3, 1, 4, 0),
            ]
        )

        def make_from_rgb(out_channel):
            if from_rgb_activate:
                return nn.Sequential(ConvLayerEqual(3, out_channel, 1),
                                     nn.LeakyReLU(0.2))

            else:
                return ConvLayerEqual(3, out_channel, 1)

        self.from_rgb = nn.ModuleList(
            [
                make_from_rgb(16),
                make_from_rgb(32),
                make_from_rgb(64),
                make_from_rgb(128),
                make_from_rgb(256),
                make_from_rgb(512),
                make_from_rgb(512),
                make_from_rgb(512),
                make_from_rgb(512),
            ]
        )

        self.n_layer = len(self.progression) 
        # layers depend on the progress step

        self.linear = EqualLinear(512, 1)

    def forward(self, input, step=0, alpha=-1):
        for i in range(step, -1, -1):
            index = self.n_layer - i - 1

            if i == step:
                out = self.from_rgb[index](input)

            if i == 0:
                out_std = torch.sqrt(out.var(0, unbiased=False) + 1e-8)
                mean_std = out_std.mean()
                mean_std = mean_std.expand(out.size(0), 1, 4, 4)
                out = torch.cat([out, mean_std], 1)

            out = self.progression[index](out)

            if i > 0:
                if i == step and 0 <= alpha < 1:
                    skip_rgb = F.avg_pool2d(input, 2)
                    skip_rgb = self.from_rgb[index + 1](skip_rgb)

                    out = (1 - alpha) * skip_rgb + alpha * out

        out = out.squeeze(2).squeeze(2)
        # print(input.size(), out.size(), step)
        out = self.linear(out)

        return out
