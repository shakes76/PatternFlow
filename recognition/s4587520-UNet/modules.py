#Create Model
import torch
from torch import nn

class UNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv16 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding="same")
    self.conv32 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding="same")
    self.softmax = nn.Softmax2d()
    self.context16 = Context_Module(16)
    self.context32 = Context_Module(32)
    self.context64 = Context_Module(64)
    self.context128 = Context_Module(128)
    self.context256 = Context_Module(256)
    self.stride32 = Stride2_Conv(32)
    self.stride64 = Stride2_Conv(64)
    self.stride128 = Stride2_Conv(128)
    self.stride256 = Stride2_Conv(256)
    self.upsample16 = Upsampling_Module(16)
    self.upsample32 = Upsampling_Module(32)
    self.upsample64 = Upsampling_Module(64)
    self.upsample128 = Upsampling_Module(128)
    self.local16 = Localisation_Module(16)
    self.local32 = Localisation_Module(32)
    self.local64 = Localisation_Module(64)
    self.local128 = Localisation_Module(128)
    self.seg32_a = Segmentation_Module(32)
    self.seg32_b = Segmentation_Module(32)
    self.seg64 = Segmentation_Module(64)
    self.upscale = nn.UpsamplingNearest2d(scale_factor=2)
    

  def forward(self, x):
    x = self.conv16(x).detach()
    x1 = self.context16(x).detach()
    x = self.stride32(x1).detach()
    x2 = self.context32(x).detach()
    x = self.stride64(x2).detach()
    x3 = self.context64(x).detach()
    x = self.stride128(x3).detach()
    x4 = self.context128(x).detach()
    x = self.stride256(x4).detach()
    x = self.context256(x).detach()
    x = self.upsample128(x, x4).detach()
    x = self.local128(x).detach()
    x = self.upsample64(x, x3).detach()
    x5 = self.local64(x).detach()
    x = self.upsample32(x5, x2).detach()
    x6 = self.local32(x).detach()
    x = self.upsample16(x6, x1).detach()
    x = self.conv32(x).detach()
    
    #Segmentation Layers
    x5 = self.seg64(x5).detach()
    x5 = simple_size_fix(self.upscale(x5), x6).detach()
    x6 = torch.add(self.seg32_a(x6), x5).detach()
    x6 = simple_size_fix(self.upscale(x6), x).detach()
    x = torch.add(self.seg32_b(x), x6).detach()
    
    #Output
    x = self.softmax(x).detach()

    return x


#Context Module: 2 3x3 convs connected by 0.3 dropout, adds input after module
class Context_Module(nn.Module):
  def __init__(self, output_filters):
    super().__init__()
    self.stack = nn.Sequential(
      nn.InstanceNorm2d(output_filters),
      nn.LeakyReLU(negative_slope=0.01),
      nn.Conv2d(output_filters, output_filters, kernel_size=3, padding="same"),
      nn.Dropout(p=0.3),
      nn.InstanceNorm2d(output_filters),
      nn.LeakyReLU(negative_slope=0.01),
      nn.Conv2d(output_filters, output_filters, kernel_size=3, padding="same")
    )

  def forward(self, x):
    #Return result of module elemnt-wise summed with input
    return torch.add(x, self.stack(x))


#Repeat each feature voxel twice in each dimension, then 3x3 conv
class Upsampling_Module(nn.Module):
  def __init__(self, output_filters):
    super().__init__()
    self.stack = nn.Sequential(
      nn.UpsamplingNearest2d(scale_factor=2), #Padding value is applied in both directions on each dimensions
      nn.Conv2d((2*output_filters), output_filters, kernel_size=3, padding="same")
    )

  def forward(self, main, cat):
    #main is the signal to be processed and cat is the signal to be concatenated
    out = self.stack(main)
    out = simple_size_fix(out, cat)
    return torch.cat((out, cat), dim=1)

#Performs a 3x3 conv with stride 2 at set output filters
class Stride2_Conv(nn.Module):
  def __init__(self, output_filters):
    super().__init__()
    self.stack = nn.Sequential(
      nn.Conv2d(int(output_filters/2), output_filters, kernel_size=3, stride=2, padding=1)
    )

  def forward(self, x):
    return self.stack(x)
  
#Localisation module
#3x3 conv followed by 1x1 conv that halves features
class Localisation_Module(nn.Module):
  def __init__(self, output_filters):
    super().__init__()
    self.stack = nn.Sequential(
        nn.Conv2d(2*output_filters, 2*output_filters, kernel_size=3, padding="same"),
        nn.Conv2d(2*output_filters, output_filters, kernel_size=1)
    )
  def forward(self, x):
    return self.stack(x)

#Segmentation module
#3x3 conv followed by 1x1 conv that halves features
class Segmentation_Module(nn.Module):
  def __init__(self, input_filters):
    super().__init__()
    self.stack = nn.Sequential(
        nn.Conv2d(input_filters, 2, kernel_size=3, padding="same"),
    )
  def forward(self, x):
    return self.stack(x)


#Fixes rounding errors
def simple_size_fix(x, desired):
  pad_width = desired.size(3) - x.size(3)
  pad_height = desired.size(2) - x.size(2)
  x = nn.functional.pad(x, [0, pad_width, 0, pad_height])
  return x

