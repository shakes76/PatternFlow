"""
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
