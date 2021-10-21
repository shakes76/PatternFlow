
import torch.nn as nn


class DarkNet53(nn.Module):
  """
    Build up DarkNet53 according the DarkNet structure and downsampling the image.
  """

  def __init__(self, layers):
    super().__init__()
    self.feature_map = 32
    self.inputs = nn.Sequential(
      nn.Conv2d(3, self.feature_map, kernel_size=3, stride=1, padding=1, bias=False),
      nn.BatchNorm2d(self.feature_map),
      nn.LeakyReLU(0.1)
    )

    self.fms_layers = [32, 64, 128, 256, 512, 1024]

    self.layer1 = self.make_res_layer([self.fms_layers[0], self.fms_layers[1]], layers[0])

    self.layer2 = self.make_res_layer([self.fms_layers[1], self.fms_layers[2]], layers[1])

    self.layer3 = self.make_res_layer([self.fms_layers[2], self.fms_layers[3]], layers[2])

    self.layer4 = self.make_res_layer([self.fms_layers[3], self.fms_layers[4]], layers[3])

    self.layer5 = self.make_res_layer([self.fms_layers[4], self.fms_layers[5]], layers[4])

    self.layers_out_filters = [64, 128, 256, 512, 1024]

  def make_res_layer(self, num_feature_maps, blocks):
    layers = []

    # 3*3 conv
    layers.append(nn.Sequential(
      nn.Conv2d(self.feature_map, num_feature_maps[1], 3, 2, 1, bias=False),
      nn.BatchNorm2d(num_feature_maps[1]),
      (nn.LeakyReLU(0.1)
       )))

    # Residual
    self.feature_map = num_feature_maps[1]

    for i in range(0, blocks):
      layers.append(BasicBlock(self.feature_map, num_feature_maps))
    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.inputs(x)

    x = self.layer1(x)
    x = self.layer2(x)
    concat1 = self.layer3(x)
    concat2 = self.layer4(concat1)
    concat3 = self.layer5(concat2)

    return concat1, concat2, concat3


class BasicBlock(nn.Module):
  def __init__(self, inplanes, planes):
    super().__init__()

    self.conv = nn.Sequential(
      nn.Conv2d(inplanes, planes[0], 1, 1, 0, bias=False),
      nn.BatchNorm2d(planes[0]),
      nn.LeakyReLU(0.1),
      nn.Conv2d(planes[0], planes[1], 3, 1, 1, bias=False),
      nn.BatchNorm2d(planes[1]),
      nn.LeakyReLU(0.1)
    )

  def forward(self, x):
    skip_conv = x
    fx = self.conv(x)
    hx = fx + skip_conv
    return hx


def darknet53():
  """
   The Residual Block repeat [1, 2, 8, 8, 4] times.
  """
  return DarkNet53([1, 2, 8, 8, 4])
