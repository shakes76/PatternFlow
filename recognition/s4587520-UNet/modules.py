#Create Model
from contextvars import Context
import torch
from torch import nn

class UNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.input = nn.Sequential(
        nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3),
    )
    self.context16 = Context_Module(16)
    self.context32 = Context_Module(32)
    self.context64 = Context_Module(64)
    self.context128 = Context_Module(128)
    self.contect256 = Context_Module(256)
    self.stride32 = Stride2_Conv(32)
    self.stride64 = Stride2_Conv(64)
    self.stride128 = Stride2_Conv(128)
    self.stride256 = Stride2_Conv(256)
    self.upsample16 = Upsampling_Module(16)
    self.upsample32 = Upsampling_Module(32)
    self.upsample64 = Upsampling_Module(64)
    self.upsample128 = Upsampling_Module(128)
    
    

  def forward(self, x):
    x = self.input(x)
    x1 = self.context16(x)
    x2 = self.stride32(x1)
    x2 = self.context32(x2)
    x3 = self.stride64(x2)
    x3 = self.context64(x3)
    x4 = self.stride128(x3)
    x4 = self.context128(x4)
    x5 = self.stride256(x4)
    x5 = self.context256(x5)
    
    prediction = self.softmax(x)
    return prediction


#Context Module: 2 3x3 convs connected by 0.3 dropout, adds input after module
class Context_Module(nn.Module):
  def __init__(self, output_filters):
    super().__init__()
    self.stack = nn.Sequential(
      nn.Conv2d(output_filters, output_filters, kernel_size=3),
      nn.Dropout(p=0.3),
      nn.Conv2d(output_filters, output_filters, kernel_size=3)
    )

  def forward(self, x):
    #Return result of module elemnt-wise summed with input
    return torch.sum(x, self.stack(x))


#Repeat each feature voxel twice in each dimension, then 3x3 conv
class Upsampling_Module(nn.Module):
  def __init__(self, output_filters):
    super().__init__()
    self.stack = nn.Sequential(
      nn.ReplicationPad3d(padding=1), #Padding value is applied in both directions on each dimensions
      nn.Conv2d(output_filters, output_filters, kernel_size=3)
    )

  def forward(self, main, cat):
    #main is the signal to be processed and cat is the singal to be concatenated
    return torch.cat((self.stack(main), cat))

#Performs a 3x3 conv with stride 2 at set output filters
class Stride2_Conv(nn.Module):
  def __init__(self, output_filters):
    super().__init__()
    self.stack = nn.Sequential(
      nn.Conv2d(2*output_filters, output_filters, kernel_size=3, stride=2)
    )

  def forward(self, x):
    #main is the signal to be processed and cat is the singal to be concatenated
    return self.stack(x)
  
  #Localisation module
  #3x3 conv followed by 1x1 conv that halves features
  class Localisation_Module(nn.Module):
    def __init__(self, output_filters):
      super().__init__()
      self.stack = nn.Sequential(
          nn.Conv2d(2*output_filters, 2*output_filters, kernel_size=3),
          nn.Conv2d(2*output_filters, output_filters, kernel_size=1)
      )
    def forward(self, x):
      return self.stack(x)
