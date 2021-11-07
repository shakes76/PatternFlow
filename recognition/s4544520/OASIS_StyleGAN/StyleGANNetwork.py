import math
import torch
import torch.nn as nn
from collections import OrderedDict
from StyleGANUtils import *

# device setting
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MappingNetwork(nn.Module):
    """
    Mapping network use multiple dense layers to create latent space to learn style.
    """
    def __init__(self,
                 nz,
                 n_latent,
                 num_layers):
        """
        The constructor of mapping network
        Args:
            nz: start of latest's dimension
            n_latent: dense layer's dimension
            num_layers: the number of dense network
        """
        super(MappingNetwork, self).__init__()
        self.denseLayers = nn.ModuleList()

        dim_input = nz
        for i in range(num_layers):
            self.denseLayers.append(Equalized_learning_rate_Linear(dim_input, n_latent))
            dim_input = n_latent

        self.leaky = torch.nn.LeakyReLU(0.2)

    def forward(self, x):
        for layer in self.denseLayers:
            x = layer(x)
            x = self.leaky(x)

        return x


class NoiseProcess(nn.Module):
    """
        Create noise to increase the random of generate images
    """

    def __init__(self):
        super(NoiseProcess, self).__init__()
        self.conv = nn.Conv2d(512, 1, 1, bias=False)
        self.conv.weight.data.fill_(0) # init as zero

    def forward(self, x):
        return self.conv(x)


class AdaIN(nn.Module):
    """
    AdaIn module in StyleGAN. Compare with paper, I choose to add noise also in this moudle.
    """

    def __init__(self, dimIn, dimOut, latent, epsilon=1e-8):
        """
        The constructor of AdaIN
        Args:
            dimIn: Input's dimension
            dimOut: output's dimension
            latent: latent space
            epsilon: setting epsilon
        """
        super(AdaIN, self).__init__()
        self.epsilon = epsilon
        self.addStyleLinear = Equalized_learning_rate_Linear(dimIn, 2 * dimOut)
        self.dimOut = dimOut
        self.latent = latent
        self.dimIn = dimIn
        self.noise = NoiseProcess() # add Noise module
        if self.training:
            self.latent = self.latent.to(device)

    def forward(self, x):
        batchSize, nChannel, width, height = x.size()
        if self.training:
            noise = self.noise(torch.randn(batchSize, nChannel, width, height).cuda())
        else:
            noise = self.noise(torch.randn(batchSize, nChannel, width, height))
        x = noise + x
        tmpX = x.view(batchSize, nChannel, -1)
        mux = tmpX.mean(dim=2).view(batchSize, nChannel, 1, 1)
        varx = torch.clamp((tmpX * tmpX).mean(dim=2).view(batchSize, nChannel, 1, 1) - mux * mux, min=0)
        varx = torch.rsqrt(varx + self.epsilon)
        x = (x - mux) * varx
        styleY = self.addStyleLinear(self.latent)
        yA = styleY[:, : self.dimOut].view(batchSize, self.dimOut, 1, 1)
        yB = styleY[:, self.dimOut:].view(batchSize, self.dimOut, 1, 1)
        return yA * x + yB


class Generator(nn.Module):
    """
    Build Generator.Detail can see the draft in ReadMe.md
    """
    def __init__(self,
                 nc=1,
                 nz=512,
                 size=256,
                 batch_size=12,
                 ):
        super(Generator, self).__init__()
        self.nc = nc  # output's dimension
        self.nz = nz  # latent variable dimension
        self.size = size  # target image's size H W
        self.stages = int(math.log2(self.size / 4)) + 1  # based on paper's progressive growth get stages
        self.current_stage = 1
        self.mapping = MappingNetwork(512, 512, 8)
        self.batch_size = batch_size
        self.z = torch.randn(self.batch_size, 512, 1, 1)
        self.latent_space = self.mapping(self.z)
        self.latent_space = self.latent_space.detach()  # solve can not support deepcopy problem
        # self.model = self.create_init_module()
        self.model = nn.Sequential(
            OrderedDict([
                ("stage_{}".format(self.current_stage), self.start_block()),
                ("to_rgb", self.to_rgb_block(512))

            ])
        )

    def start_block(self):
        layers = []
        dim = 512
        adain00 = AdaIN(self.nz, self.nz, self.latent_space)
        layers.append(adain00)
        conv1 = conv_module(layers, self.nz, dim, kernel_size=3, stride=1, padding=1)
        adain01 = AdaIN(dim, dim, self.latent_space)
        layers.append(adain01)

        return nn.Sequential(*layers)

    def scaled_up_block(self):
        """
        add next stage's module
        """
        layers = []
        layers.append(sampleByFactor(factor=2))  # factor > 1, upsample
        conv1 = conv_module(layers, 512, 512, kernel_size=3, stride=1, padding=1)
        adain00 = AdaIN(512, 512, self.latent_space)
        layers.append(adain00)
        conv2 = conv_module(layers, 512, 512, kernel_size=3, stride=1, padding=1)
        adain01 = AdaIN(512, 512, self.latent_space)
        layers.append(adain01)

        return nn.Sequential(*layers)

    def grow_network(self):
        self.current_stage += 1
        print('growing Generator...\n')
        new_model = copy_previous_layers_except(self.model, ['to_rgb'])
        old_block = nn.Sequential()
        old_to_rgb = copy_previous_layer(self.model, 'to_rgb')
        old_block = nn.Sequential(
            OrderedDict(
                [
                    ("old_to_rgb", old_to_rgb),
                    ("old_upsample", sampleByFactor(factor=2))  # upsample
                ]
            )
        )
        inter_block = self.scaled_up_block()
        new_block = nn.Sequential(
            OrderedDict(
                [
                    ("new_block", inter_block),
                    ("new_to_rgb", self.to_rgb_block(512)),
                ]
            )
        )
        new_model.add_module('residual_module', Residual_concat(old_block, new_block))

        self.model = new_model

    def flush_network(self):
        print('flushing Generator...\n')
        new_block = copy_previous_layer(self.model.residual_module.current_featureMap, 'new_block')
        new_to_rgb = copy_previous_layer(self.model.residual_module.current_featureMap, 'new_to_rgb')
        new_model = copy_previous_layers_except(self.model, ['residual_module'])
        layer_name = 'stage_{}'.format(self.current_stage)
        new_model.add_module(layer_name, new_block)
        new_model.add_module('to_rgb', new_to_rgb)
        self.model = new_model

    def to_rgb_block(self, ndim):
        return Equalized_learning_rate_Conv(input=ndim, output=self.nc, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self,
                 nc=3,
                 nz=512,
                 size=256,
                 ):
        super(Discriminator, self).__init__()
        self.nc = nc
        self.nz = nz
        self.size = size
        self.stages = int(math.log2(self.size / 4)) + 1  # the total number of stages (7 when size=256)
        self.current_stage = self.stages
        self.model = nn.Sequential(
            OrderedDict([
                ('from_rgb', self.from_rgb_block(512)),
                ('stage_{}'.format(self.current_stage), self.last_block())

            ])
        )
        self.result = Equalized_learning_rate_Linear(512, output=1)



    def from_rgb_block(self, ndim):
        layers = []
        layers = conv_module(layers, input=self.nc, output=ndim, kernel_size=1, stride=1, padding=0)
        return nn.Sequential(*layers)

    def last_block(self):
        layers = []
        ndim = 512
        layers.append(MiniBatchStd())
        layers = conv_module(layers, input=ndim + 1, output=ndim, kernel_size=3, stride=1, padding=1)
        layers = conv_module(layers, input=ndim, output=ndim, kernel_size=4, stride=1, padding=0)
        return nn.Sequential(*layers)

    def scaled_down_block(self):
        layers = []

        layers = conv_module(layers, input=512, output=512, kernel_size=3,
                             stride=1, padding=1)
        layers = conv_module(layers, input=512, output=512, kernel_size=3,
                             stride=1, padding=1)
        layers.append(nn.AvgPool2d(kernel_size=2))
        return nn.Sequential(*layers)

    def grow_network(self):
        self.current_stage -= 1
        print('growing Discriminator...\n')
        # old block 
        old_from_rgb = copy_previous_layer(self.model, 'from_rgb')
        old_block = nn.Sequential(
            OrderedDict([
                ('old_downsample', nn.AvgPool2d(kernel_size=2)),
                ('old_from_rgb', old_from_rgb)
            ])
        )
        new_block = nn.Sequential()
        inter_block = self.scaled_down_block()
        new_block = nn.Sequential(
            OrderedDict([
                ('new_from_rgb', self.from_rgb_block(512)),
                ('new_block', inter_block)
            ])
        )
        new_model = nn.Sequential()
        new_model.add_module('residual_module', Residual_concat(old_block, new_block))
        # copy the trained layers except "to_rgb"
        for item in self.model.named_children():
            if item[0] != 'from_rgb':
                new_model.add_module(item[0], item[-1])
                new_model[-1].load_state_dict(item[-1].state_dict())
        del self.model
        self.model = new_model

    def flush_network(self):
        print('flushing Discriminator\n')
        new_block = copy_previous_layer(self.model.residual_module.current_featureMap, 'new_block')
        new_from_rgb = copy_previous_layer(self.model.residual_module.current_featureMap, 'new_from_rgb')
        layer_name = 'stage_{}'.format(self.current_stage)
        new_model = nn.Sequential(
            OrderedDict(
                [
                    ('from_rgb', new_from_rgb),
                    (layer_name, new_block)
                ])
        )
        for item in self.model.named_children():
            if item[0] != 'residual_module':
                new_model.add_module(item[0], item[-1])
                new_model[-1].load_state_dict(item[-1].state_dict())
        self.model = new_model

    def forward(self, x):
        x = self.model(x)
        result = self.result(x)
        return result


def copy_previous_layer(module, layer):
    """
    copy previous layer to do progressive growth
    Args:
        module: nn.Sequential layer
        layer: exact layer

    Returns: a new nn.Sequential

    """
    new_module = nn.Sequential()
    for childModule in module.named_children():
        if childModule[0] == layer:
            new_module.add_module(childModule[0], childModule[1])
            new_module[-1].load_state_dict(childModule[1].state_dict())
    return new_module


def copy_previous_layers_except(module, not_copy_list):
    """
    copy layers except some layers
    Args:
        module: nn.Sequential layers
        not_copy_list:  not copy  layers

    Returns: a new nn.Sequential

    """
    new_module = nn.Sequential()
    for childModule in module.named_children():
        if childModule[0] not in not_copy_list:
            new_module.add_module(childModule[0], childModule[1])
            new_module[-1].load_state_dict(childModule[1].state_dict())
    return new_module


def conv_module(layers, input, output, kernel_size, stride, padding):
    """
    Style GAN's conv (Equalized learning rate init )
    Args:
        layers: layers list
        input: input channels
        output: output channels
        kernel_size: kernel size
        stride: stride
        padding: padding

    Returns: conv modules with conv and no linear layers
    """
    layers.append(Equalized_learning_rate_Conv(input, output, kernel_size, stride, padding))
    layers.append(nn.LeakyReLU(0.2))
    return layers
