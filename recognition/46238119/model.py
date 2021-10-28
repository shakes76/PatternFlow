"""
Author: Yaxian Shi
Student No.: 46238119
This contains the models of Generator and Discriminator of StyleGAN2.
-- Equalized learning rate;
-- Modulate and demodulate layers;
-- Skip connection in Generator and residual connection in Discriminator.
"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class EqualizedLinear(nn.Module):
  """
  Equalized lr linear layer:
  -- apply He's initialization
  -- initialize bias to 0
  -- dymanically renormalize the weights in each layer with per-layer normalization factor
  """
  def __init__(self, 
               in_dim,            # number of input channel
               out_dim,           # number of output channel
               lr_mul = 1.0,      # multiplier of learning rate
               bias = True,       # apply bias or not
               bias_value = 0.0   # the initial value of bias
               ):
    super().__init__()
    
    self.weight = nn.Parameter(torch.randn([out_dim, in_dim]) / lr_mul)    # He's initialization with lr_mul
    if bias:
      self.bias = nn.Parameter(torch.ones([out_dim]) * bias_value)
    else:
      self.bais = None
    
    self.weight_factor = np.sqrt(2.0/in_dim) * lr_mul    # per-layer normalization constant * lr_mul
    self.bias_factor = lr_mul

  def forward(self, x):
    w = self.weight * self.weight_factor
    # Question here: if bias is None, how to operate???
    b = self.bias * self.bias_factor
    x = F.linear(x, w, bias=b)
    return x

class EqualizedConv2d(nn.Module):
  """
  Equalized lr Conv2d layer without activation func:
  -- apply He's initialization
  -- initialize bias to 0
  -- dymanically renormalize the weights in each layer with per-layer normalization factor
  """
  def __init__(self, 
               in_dim,        # number of input channel
               out_dim,       # number of output channel
               kernel_size,   # kernel size
               padding=0,     # padding
               bias = True    # apply bias or not
               ):
    super().__init__()

    self.padding = padding
    # equalized weights
    self.weight = nn.Parameter(torch.randn([out_dim, in_dim, kernel_size, kernel_size]))
    self.c = np.sqrt(2.0/in_dim * (kernel_size ** 2))
    # bias
    if bias:
      self.bias = nn.Parameter(torch.zeros([out_dim]))
    else:
      self.bias = None
    
  def forward(self, x):
    w = self.weight * self.c
    b = self.bias
    x = F.conv2d(x, w, bias=self.bias, padding=self.padding)
    return x

class ModulatedConv2d(nn.Module):
  """
  Convolution layer with weight modulation and demodulation:
  -- Modulate each input feature map of the convolution by style vector.
  -- Demodulate each output feature map by normalizing.
  """
  def __init__(self, 
               in_dim,                  # the number of input channel
               out_dim,                 # the number of output channel
               kernel_size,             # kernel size
               demodulate = True,       # whether to demodulate weight
               eps = 1e-8               # epsilon for demodulation to avoid numerical issues
               ):
    super().__init__()
    self.in_dim = in_dim
    self.kernel_size = kernel_size
    self.demodulate = demodulate
    self.eps = eps
    # calculate padding
    self.padding = kernel_size // 2
    # weight
    self.weight = nn.Parameter(torch.randn([out_dim, in_dim, kernel_size, kernel_size]))
    self.c = np.sqrt(2.0/in_dim * (kernel_size ** 2))
    
  def forward(self, x, s):
    """
    x: input feature map with shape [batch_size, in_dim, height, width]
    s: style vector to scale input feature map with shape [batch_size, in_dim]
    """
    # get shape of input x
    b, c, h, w = x.shape
    weight = self.weight

    # normalize w and s
    if self.demodulate:
      weight = weight * weight.square().mean([1,2,3], keepdim=True).rsqrt()
      s = s * s.square().mean().rsqrt()

    # get weight and reshape it to [1, out_dim, in_dim, k, k]
    weight = weight[None, :, :, :, :]

    # reshape scale vector to [batch_size, 1, in_dim, 1, 1]
    s = s[:, None, :, None, None]

    # modulation, with size [batch_size, out_dim, in_dim, k, k]
    weight = weight * s

    # Demodulate
    if self.demodulate:
      # standard deviation
      sigma_inversed =  (weight.square().sum(dim=(2,3,4), keepdim=True) + self.eps).rsqrt()
      weight = weight * sigma_inversed

    # employ grouped convolutions by reshape weights insteads of N samples with one group.
    # reshape x
    x = x.reshape(1, -1, h, w)
    # reshape weight
    weight = weight.reshape(-1, self.in_dim, self.kernel_size, self.kernel_size)
    # use grouped convolution
    x = F.conv2d(x, weight, padding=self.padding, groups=b)
    # reshape x
    x = x.reshape(b, -1, h, w)
    return x

class BlurPool(nn.Module):
  """
  Bilinear filter to anti-aliasing.
  ref: https://github.com/adobe/antialiased-cnns/blob/master/antialiased_cnns/blurpool.py
  """
  def __init__(self):
    super().__init__()
    # get the blurring kernel and normalize it
    a = np.array([1., 2., 1.])
    kernel = torch.Tensor(a[:,None]*a[None,:])
    kernel = kernel/torch.sum(kernel)
    kernel = kernel[None,None,:,:]
    #print(kernel, kernel.shape)
    self.kernel = nn.Parameter(kernel, requires_grad=False)
    # replication padding layer
    self.padding = nn.ReplicationPad2d(1)

  def forward(self, x):
    # Get shape of the input feature and reshape it
    b, c, h, w = x.shape
    x = x.reshape(-1, 1, h, w)

    # padding
    x = self.padding(x)

    # anti-aliase x using kernel
    x = F.conv2d(x, self.kernel)
    return x.view(b, c, h, w)

class UpsampleLayer(nn.Module):
  """
  Scales the image up by 2 and by Bilinear filtering layer.
  """
  def __init__(self):
    super().__init__()
    # Upsampling layer
    self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
    # anti-aliasing layer
    self.blurpool = BlurPool()

  def forward(self, x): 
    x = self.upsample(x)
    return self.blurpool(x)

class DownsampleLayer(nn.Module):
  """
  Downsample the image up by 2 and by Bilinear filtering layer.
  """
  def __init__(self):
    super().__init__()
    # anti-aliasing layer
    self.blurpool = BlurPool()

  def forward(self, x): 
    _, _, h, w = x.shape
    n_h = h // 2
    n_w = w // 2
    x = self.blurpool(x)
    x = F.interpolate(x, (n_h, n_w), mode='bilinear', align_corners=False)
    return x

class ToRGB(nn.Module):
  """
  Transform feature map to an RGB image using 1×1 convolution.
  """
  def __init__(self, 
               in_dim,              # number of input channel
               out_dim,             # number of output channel
               w_dim,               # dimension of latent w
               kernel_size = 1      # kernel size
               ):
    super().__init__()

    # style vector using EqualizedLinear with bias initialied to 1
    self.affine = EqualizedLinear(in_dim=w_dim, out_dim=in_dim, bias_value=1.0)

    # weight modulated layer
    self.modulated = ModulatedConv2d(in_dim, out_dim, kernel_size, demodulate=False)
    # bias
    self.bias = nn.Parameter(torch.zeros([out_dim]))

    # weight gan (changed on 10.25)
    self.c = 1 / np.sqrt(in_dim * (kernel_size ** 2))

  def forward(self, x, w):
    """
    x: input feature map with shape [batch_size, in_dim, height, width]
    w: w vector with shape [batch_size, w_dim] to transform to affine vector
    """
    # calculate the style vector, style vector with shape [batch_size, in_dim] (10.25)
    style = self.affine(w) * self.c

    # weight modulate conv2d layer, the output with shape [batch_size, 3, h, w]
    x = self.modulated(x, style)

    # add bias
    x += self.bias[None, :, None, None]

    #return self.activation(x)
    return x

class StyleLayer(nn.Module):
  """
  StyleLayer has a weight modulation and demodulation convolution layer.
  """
  def __init__(self, 
               in_dim,              # number of input channel
               out_dim,             # number of output channel
               w_dim,               # dimension of latent w
               use_noise = True,    # use noise or not
               kernel_size = 3      # kernel size
               ):
    super().__init__()
    self.use_noise = use_noise

    # calculate padding
    self.padding = kernel_size // 2

    # style vector using EqualizedLinear with bias initialied to 1
    self.affine = EqualizedLinear(in_dim=w_dim, out_dim=in_dim, bias_value=1.0)

    # weight modulated layer
    self.modulated = ModulatedConv2d(in_dim=in_dim, out_dim=out_dim, kernel_size=kernel_size)

    # bias
    self.bias = nn.Parameter(torch.zeros([out_dim]))

    # noise scale(dimension????)
    if use_noise:
      self.noise_scale = nn.Parameter(torch.zeros(1))

    # leaky ReLu activation function
    self.activation = nn.LeakyReLU(0.2, True)

  def forward(self, x, w):
    """
    x: input feature map with shape [batch_size, in_dim, height, width]
    w: w vector with shape [batch_size, w_dim] to transform to affine vector
    generate noise: noise with shape [batch_size, 1, height, width]
    """
    # get configs of x's shape
    b, c, x_h, x_w = x.shape

    # calculate the style vector, style vector with shape [batch_size, in_dim]
    style = self.affine(w)

    # weight modulate conv2d layer, the output with shape [batch_size, out_dim, h, w]
    x = self.modulated(x, style)

    # generate noise
    noise = None
    if self.use_noise:
      noise = torch.randn([b, 1, x_h, x_w], device=x.device)
      noise = self.noise_scale[None, :, None, None] * noise
      x += noise

    # add bias
    x += self.bias[None, :, None, None]
    return self.activation(x)

class StyleBlock(nn.Module):
  """
  Contain two sytlelayers without upsample layer, output feature maps and RGB image.
  """
  def __init__(self, 
               in_dim,                # the number of input channel
               out_dim,               # the number of output channel
               w_dim                 # the deminsion of latent vector
               ):
    super().__init__()

    # the first stylelayer
    self.stylelayer1 = StyleLayer(in_dim, out_dim, w_dim)
    # the second stylelayer
    self.stylelayer2 = StyleLayer(out_dim, out_dim, w_dim)
    # toRGB layer
    self.torgb = ToRGB(out_dim, 1, w_dim)
    # upsample layer
    self.upsamplelayer = UpsampleLayer()

  def forward(self, x, w, img):
    """
    x: input feature map with shape [batch_size, in_dim, height, width]
    w: w vector with shape [batch_size, w_dim] to transform to affine vector
    img: input img with shape [batch_size, 3, height/2, width/2]
    """
    # two stylelayers
    # output feature map with shape [batch_size, out_dim, h, w]
    x = self.stylelayer1(x, w)
    # output feature map with shape [batch_size, out_dim, h, w]
    x = self.stylelayer2(x, w)
    # transform to RGB image
    if img is not None:
      img = self.upsamplelayer(img)    
    rgb = self.torgb(x, w)
    if img is not None:
      rgb  = rgb + img
    
    #print(x.shape, rgb.shape)
    return x, rgb

class MappingNetwork(nn.Module):
  """
  A non-linear mapping network. Affine transform latent vector z to w. 
  
  Architiecture: Consist of 8-layer MLP using leakyRelu with a = 0.2, lr' = 0.01*lr.
  Initialize all weihts of FC, affine transform layers using N(0,1), bias = 0.

  input: latent vector z with dimension 512.

  output: w with dimension 512, control adaptive instance normalization operation in network G.
  """

  def __init__(self, 
               z_dim,                 # the dimension of latent vector z
               w_dim,                 # the dimension of w
               num_w,                 # the number of latent w to output
               num_layers = 8,        # the number of linear layers for mapping network
               lr_mul = 0.01          # lr multiplier for the mapping layers
               ):
    super().__init__()
    self.num_w = num_w

    self.Layers = nn.ModuleList()

    in_dim = z_dim
    for i in range(num_layers):
      self.Layers.append(EqualizedLinear(in_dim=in_dim, out_dim=w_dim, lr_mul=lr_mul))
      in_Dim = w_dim

    self.activation = nn.LeakyReLU(0.2, True)

  def forward(self, x):
    """
    x: input latent vector
    """
    # normalize x
    x = x * (x.square().mean(1, keepdim=True) + 1e-8).rsqrt()
    # mapping x to w
    for layer in self.Layers:
      x = self.activation(layer(x))
    # output latent vector with number of num_w
    if self.num_w is not None:
      x = x.unsqueeze(0).repeat([self.num_w, 1, 1])

    return x

class GeneratorNetwork(nn.Module):
    """
    The input of generator network is a constant.
    The generator network consis of sytleblocks. The resolution of each styleblock is doubled.
    The output of generator network is an image, summed by each upsampled output image of styleblocks.
    """
  def __init__(self, 
                 w_dim,                   # the dimension of latent vector w
                 img_resolution,          # the resolution of output image
                 img_dim = 3,             # the dimension of output image
                 dim_max = 512            # maximum number of channel in layers
                 ):
    super().__init__()
    self.w_dim = w_dim
    self.img_resolution = img_resolution
    self.img_dim = img_dim

    # calculate the log2 of image resolution, 8
    img_res_log2 = int(np.log2(img_resolution))

    # calculate the resolution of each block: [4, ... , 256]
    block_res = [2 ** i for i in range(2, img_res_log2 + 1)]        
        
    # calculate the number of channel for each block, max channle is 512, min channel is 32
    # channles = [32, 64, 128, 256, 512, 512, 512]
    # block_dim = {4: 512, 8: 512, 16: 512, 32: 256, 64: 128, 128: 64, 256: 32}
    channels = [min(dim_max, res * 8) for res in block_res]
    channels = channels[::-1]
    block_dim = {}
    for i in range(len(block_res)):
      block_dim[block_res[i]] = channels[i]
    # the number of blocks, 7
    self.num_blocks = len(block_dim)

    # initial constant input c with shape [1, 512, 4, 4]
    #self.initial_constant = nn.Parameter(torch.randn([1, block_dim[4], 4, 4]))
    self.initial_constant = nn.Parameter(torch.randn([block_dim[4], 4, 4]))

    # the first style layer with 4*4 resolution and ToRGB layer
    self.style_layer0 = StyleLayer(in_dim=block_dim[4], out_dim=block_dim[4], w_dim=w_dim)
    self.to_rgb = ToRGB(in_dim=block_dim[4], out_dim=self.img_dim, w_dim=w_dim)

    # upsample layer before each styleblock except the first styleblock
    self.upsamplelayer = UpsampleLayer()

    # styleblocks with resolution from 8*8 to 256*256
    self.blocks = nn.ModuleList()
    for res in block_res[1:]:
      in_dim = block_dim[res // 2]
      out_dim = block_dim[res]
      block = StyleBlock(in_dim=in_dim, out_dim=out_dim, w_dim=w_dim)
      self.blocks.append(block)

  def forward(self, w):
    """
    w: latent vectors from mapping network. Each block will have each w. 
        w with shape [num_blocks, batch_size, w_dim]
    Each block will have two noises for each conv layer.
    """
    # batch size
    batch_size = w.shape[1]

    # adjust the shape of constant to match batch size
    x = self.initial_constant.unsqueeze(0).repeat([w.shape[1], 1, 1, 1])

    # The first style block
    x = self.style_layer0(x, w[0])
        
    # output the first rgb image
    rgb = self.to_rgb(x, w[0])

    # output of the rest blocks
    for i in range(1, self.num_blocks):
      # upsample the feature map
      x = self.upsamplelayer(x)
      # output new feature map and rgb
      x, rgb = self.blocks[i - 1](x, w[i], rgb)

      return rgb

class Generator(nn.Module):
  """
  Comebime Mapping network with GeneratorNetwork.
  """
  def __init__(self, 
               z_dim,                 # the dimension of latent z
               w_dim,                 # the dimension of latent w
               img_resolution,        # output img resolution
               img_dim = 3            # the dimension of output img
               ):
    super().__init__()
    self.z_dim = z_dim
    self.w_dim = w_dim
    self.img_resolution = img_resolution
    self.img_dim = img_dim

    # calculate the number of w needed
    num_w = int(np.log2(img_resolution) - 1)
    # mapping network
    self.mapping = MappingNetwork(z_dim=z_dim, w_dim=w_dim, num_w=num_w)
    # generator network
    self.generatornetwork = GeneratorNetwork(w_dim=w_dim, img_resolution=img_resolution, img_dim=img_dim) 

  def forward(self, z):
    """
    z: input latent vector
    """
    # mapping z to w
    w = self.mapping(z)
    # get the img
    img = self.generatornetwork(w)

    return img


