import torch
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

        self.feature_maps = [32, 64, 128, 256, 512, 1024]

        self.layer1 = self.make_res_layer([self.feature_maps[0], self.feature_maps[1]], layers[0])

        self.layer2 = self.make_res_layer([self.feature_maps[1], self.feature_maps[2]], layers[1])

        self.layer3 = self.make_res_layer([self.feature_maps[2], self.feature_maps[3]], layers[2])

        self.layer4 = self.make_res_layer([self.feature_maps[3], self.feature_maps[4]], layers[3])

        self.layer5 = self.make_res_layer([self.feature_maps[4], self.feature_maps[5]], layers[4])

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


class YoloBody(nn.Module):
    """
    This class is main doing the upsampling for the three different size of anchors boxes.
    """

    def __init__(self, anchors_mask, num_classes):
        super().__init__()
        # The Residual Block repeat [1, 2, 8, 8, 4] times.
        self.darknet53 = DarkNet53([1, 2, 8, 8, 4])
        # [32, 64, 128, 256, 512, 1024]
        fms = self.darknet53.feature_maps
        # The total feature maps is 3 *(1(ISIC) +4 + 1)
        self.y3, self.conv_2_l3 = make_last_layers([fms[-2], fms[-1]], fms[-1],
                                                   len(anchors_mask[0]) * (num_classes + 4 + 1))

        self.y3_conv = conv2d(512, 256, 1)
        self.y3_upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.y2, self.conv_2_l2 = make_last_layers([fms[-3], fms[-2]], fms[-2] + 256,
                                                   len(anchors_mask[1]) * (num_classes + 4 + 1))

        self.y2_conv = conv2d(256, 128, 1)
        self.y2_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.y1, self.conv_2_l1 = make_last_layers([fms[-4], fms[-3]], fms[-3] + 128,
                                                   len(anchors_mask[2]) * (num_classes + 4 + 1))

    def forward(self, x):
        y1, y2, y3 = self.darknet53(x)
        # foward
        y3_up = self.y3(y3)  # conv*5
        y3_ = self.conv_2_l3(y3_up)  # conv*2

        conv_y3 = self.y3_conv(y3_up)
        upsample_y3 = self.y3_upsample(conv_y3)

        new_x3 = torch.cat([upsample_y3, y2], 1)

        y2_up = self.y2(new_x3)
        y2_ = self.conv_2_l2(y2_up)

        conv_y2 = self.y2_conv(y2_up)
        upsample_y2 = self.y2_upsample(conv_y2)

        new_x2 = torch.cat([upsample_y2, y1], dim=1)

        y1_ = self.conv_2_l1(self.y1(new_x2))

        return y3_, y2_, y1_


def conv2d(in_channel, out_channel, ks):
    """
        return the Sequential for upsampling
    """
    pad = (ks - 1) // 2 if ks else 0
    return nn.Sequential(*(
        (nn.Conv2d(in_channel, out_channel, kernel_size=ks, stride=1, padding=pad, bias=False)),
        (nn.BatchNorm2d(out_channel)),
        (nn.LeakyReLU(0.1)),
    ))


def make_last_layers(feature_map, in_channel, out_channel):
    """
    return the conv layer *5 for up sampling use and conv layer*2 for out.
    """
    # conv2D*5
    conv_5 = nn.Sequential(
        conv2d(in_channel, feature_map[0], 1),
        conv2d(feature_map[0], feature_map[1], 3),
        conv2d(feature_map[1], feature_map[0], 1),
        conv2d(feature_map[0], feature_map[1], 3),
        conv2d(feature_map[1], feature_map[0], 1),
    )
    # conv2D with 3*3 and 1*1
    conv_2 = nn.Sequential(
        conv2d(feature_map[0], feature_map[1], 3),
        nn.Conv2d(feature_map[1], out_channel, 1, 1, 0, bias=True)
    )

    return conv_5, conv_2
