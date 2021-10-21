from nets.darknet import darknet53
import torch
import torch.nn as nn


class YoloBody(nn.Module):
    """
    This class is main doing the upsampling for the three different size of anchors boxes.
    """

    def __init__(self, anchors_mask, num_classes):
        super().__init__()

        self.darknet53 = darknet53()
        # [32, 64, 128, 256, 512, 1024]
        fms = self.darknet53.fms_layers
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
