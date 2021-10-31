import torch
from torch import nn
from enum import Enum

"""
Assume input width/height to be 416/480/512 (could be any the multiple of 32)
"""

##########################################################
#   CONSTANTS
##########################################################


class ModelSize(Enum):
    Tiny = (0.33, 0.375)
    S = (0.33, 0.50)
    M = (0.67, 0.75)
    L = (1.00, 1.00)
    X = (1.33, 1.25)


##########################################################
#   YOLOModel
#   = PAFPN + DetectionHead
##########################################################


class YOLOModel(nn.Module):
    def __init__(self, num_classes, model_size: ModelSize) -> None:
        self.backbone = PAFPN(model_size)
        self.head = DetectionHead(model_size)

    def forward(self, x):
        features = self.backbone.forward(x)
        detection = self.head.forward(features)
        return detection


##########################################################
#   PAFPN: Path Aggregation feature pyramid network
#   = Modified CSPNet + Feature pyramid
#   3 UpSampling + 3 DownSampling
##########################################################


class PAFPN(nn.Module):
    def __init__(self, model_size: ModelSize) -> None:
        super().__init__()
        depth, width = model_size.value
        in_channels = (1024, 512, 256)
        self.backbone = CSPNet(model_size)
        self.up_conv0 = ConvBlock(
            in_channels[0] * width, in_channels[1] * width, kernel_size=1, stride=1
        )
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")
        self.csp0 = CSPLayer(
            int(in_channels[1] * width * 2),
            int(in_channels[1] * width),
            round(depth * 3),
            shortcut=False,
        )
        self.up_conv1 = ConvBlock(
            int(in_channels[1] * width),
            int(in_channels[2] * width),
            kernel_size=1,
            stride=1,
        )
        self.csp1 = CSPLayer(
            int(in_channels[2] * width * 2),
            int(in_channels[2] * width),
            round(depth * 3),
            False,
        )
        self.down_conv0 = ConvBlock(
            int(in_channels[2] * width),
            int(in_channels[2] * width),
            kernel_size=3,
            stride=2,
        )
        self.csp2 = CSPLayer(
            int(in_channels[2] * width * 2),
            int(in_channels[1] * width),
            round(depth * 3),
            shortcut=False,
        )
        self.down_conv1 = ConvBlock(
            int(in_channels[2] * width),
            int(in_channels[2] * width),
            kernel_size=3,
            stride=2,
        )
        self.csp3 = CSPLayer(
            int(in_channels[1] * width * 2),
            int(in_channels[0] * width),
            round(depth * 3),
            shortcut=False,
        )

    def forward(self, x):
        """3, 4, 5 are different level in the hierarchy,
        5 being finest & deepest, 3 being largest & shallowest"""

        feature3, feature4, feature5 = self.backbone(input)
        # up 1
        left_out5 = self.up_conv0(feature5)
        f_out0 = self.upsample(left_out5)
        f_out0 = torch.cat([f_out0, feature4], 1)
        f_out0 = self.csp0(f_out0)

        # up 2
        left_out4 = self.up_conv1(f_out0)
        f_out1 = self.upsample(left_out4)
        f_out1 = torch.cat([f_out1, feature3], 1)
        output_3 = self.csp1(f_out1)  # there is no left_out3

        # down 1
        p_out1 = self.down_conv0(output_3)
        p_out1 = torch.cat([p_out1, left_out4], 1)
        output_4 = self.csp2(p_out1)

        # down 2
        p_out0 = self.down_conv1(output_4)
        p_out0 = torch.cat([p_out0, left_out5], 1)
        output_5 = self.csp3(p_out0)

        return (output_3, output_4, output_5)


##########################################################
#   Modified CSPDarkNet53
#   Input -> Focus -> stem
#   -> CSP Layer * 4 (last layer go through SPP bottleneck)
#   -> extract last 3 layers of "Feature Pyramid"
#   => Feature Pyramid (in PAFPN)
##########################################################


class CSPNet(nn.Module):
    """Generate a CSPNet with a corresponding size given by trainer.

    Args:
        size (ModelSize): controls the depth & width of the model
    """

    def __init__(self, size: ModelSize) -> None:
        super().__init__()
        depth, width = size.value
        base_channels = int(width * 64)  # 64
        base_depth = max(round(3 * depth), 1)  # 3
        # stem + focus 512,512,3 => 256,256,64
        self.focus = FocusLayer(in_channels=base_channels, kernel_size=3)
        # dark2: 256,256,64 => 128,128,128
        self.dark2 = ResBlock(base_channels, base_channels * 2, depth=base_depth)
        # dark 3: 128,128,128 => 64,64,256
        self.dark3 = ResBlock(
            base_channels * 2, base_channels * 4, depth=base_depth * 3
        )
        # dark4: 64,64,256 => 32,32,512
        self.dark4 = ResBlock(
            base_channels * 4, base_channels * 8, depth=base_depth * 3
        )
        # dark 5: 32,32,512 => 16,16,1024
        self.dark5 = ResBlock(
            base_channels * 8, base_channels * 16, depth=base_depth, is_last=True
        )

    def forward(self, x):
        x = self.focus(x)
        x = self.dark2(x)
        # dark2 = x # we don't need the result of first res_block
        x = self.dark3(x)
        feature3 = x
        x = self.dark4(x)
        feature4 = x
        x = self.dark5(x)
        feature5 = x
        return (feature3, feature4, feature5)


##########################################################
#   CSPNet blocks
##########################################################
class FocusLayer(nn.Module):
    """Lossless interlaced down-sampling:
    Quadruple channels & Half the width and the height"""

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1) -> None:
        super().__init__()
        self.stem = ConvBlock(in_channels, kernel_size)

    def forward(self, x):
        x = torch.cat(
            [
                x[..., ::2, ::2, :],
                x[..., 1::2, ::2, :],
                x[..., ::2, 1::2, :],
                x[..., 1::2, 1::2, :],
            ],
            axis=-1,
        )


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, depth, is_last=False):
        layers = []
        layers.append(ConvBlock(in_channels, out_channels, kernel_size=3, stride=2))
        if is_last:
            layers.append(
                SPPBottleneck(in_channels=out_channels, out_channels=out_channels)
            )
        layers.append(
            CSPLayer(
                in_channels=out_channels,
                out_channels=out_channels,
                num_bottleneck=depth,
            )
        )
        self.module = nn.Sequential(*layers)

    def forward(self, x):
        return self.module(x)


class CSPLayer(nn.Module):
    def __init__(
        self, in_channels, out_channels, shortcut=True, num_bottleneck=1, expansion=0.5
    ):
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvBlock(in_channels, hidden_channels, kernel_size=1, stride=1)
        self.conv2 = ConvBlock(in_channels, hidden_channels, kernel_size=1, stride=1)
        self.conv3 = ConvBlock(
            2 * hidden_channels, out_channels, kernel_size=1, stride=1
        )
        self.layers = nn.Sequential(
            *[
                Bottleneck(hidden_channels, hidden_channels, shortcut=shortcut)
                for _ in range(num_bottleneck)
            ]
        )

    def forward(self, x):
        # left
        y = self.conv1(x)
        y = self.layers(y)
        # right
        y_2 = self.conv2(x)
        # concat
        y = torch.cat((y, y_2), axis=1)
        # fusion
        return self.conv3(y)


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, shortcut=True, expansion=0.5) -> None:
        super.__init__()
        inner_channels = int(out_channels * expansion)
        self.layers = nn.Sequential(
            [
                ConvBlock(in_channels, inner_channels, kernel_size=1, stride=1),
                ConvBlock(inner_channels, out_channels, 3, stride=1),
            ]
        )
        self.shortcut = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.layers(x)
        if self.shortcut:
            y = y + x
        return y


class SPPBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=(5, 9, 13)):
        super.__init__()
        hidden_channels = in_channels // 2
        self.conv1 = ConvBlock(in_channels, hidden_channels, 1, stride=1)
        self.modules = nn.Sequential(
            *[
                nn.MaxPool2d(
                    kernel_size=kernel_size,
                    stride=1,
                    padding=kernel_size // 2,
                )
                for kernel_size in kernel_sizes
            ]
        )
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = ConvBlock(conv2_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [pool(x) for pool in self.modules], axis=1)
        x = self.conv2(x)
        return x


##########################################################
#   General Building blocks
##########################################################


class ConvBlock(nn.Module):
    """Conv2D -> BatchNorm -> ACT"""

    def __init__(
        self, in_channels, out_channels, kernel_size=1, stride=1, act: str = "silu"
    ) -> None:
        super().__init__()
        padding = "valid" if stride == 2 else "same"
        activation = (
            nn.LeakyReLU(inplace=True)
            if act.lower() != "silu"
            else nn.SiLU(inplace=True)
        )
        self.layers = nn.Sequential(
            *[
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    padding=padding,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                activation,
            ]
        )

        def forward(self, x):
            return self.layers(x)


##########################################################
#   PAFPN: Path Aggregation feature pyramid network
##########################################################
class DetectionHead(nn.Module):
    def __init__(
        self,
        in_channels=[256, 512, 1024],
        num_classes=50,
        width=1.0,
        strides=[8, 16, 32],
    ) -> None:
        super().__init__()

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()

        for i in range(len(in_channels)):
            self.stems.append(
                ConvBlock(
                    in_channels=int(in_channels[i] * width),
                    out_channels=int(256 * width),
                    kernel_size=1,
                    stride=1,
                )
            )
            self.cls_convs.append(
                nn.Sequential(
                    *[
                        ConvBlock(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                        ),
                        ConvBlock(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                        ),
                    ]
                )
            )
            self.cls_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=num_classes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )

            self.reg_convs.append(
                nn.Sequential(
                    *[
                        ConvBlock(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            kernel_size=3,
                            stride=1
                        ),
                        ConvBlock(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            kernel_size=3,
                            stride=1
                        ),
                    ]
                )
            )
            self.reg_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=4,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.obj_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )

    def forward(self, inputs):
        """ inputs:
                feature3  64, 64, 256
                feature4  32, 32, 512
                feature5  16, 16, 1024
        """
        outputs = []
        for k, x in enumerate(inputs):
            # cross-channel fusion with 1x1 kernel
            x = self.stems[k](x)
            # extract feature
            cls_feat = self.cls_convs[k](x)
            # predict class
            cls_output = self.cls_preds[k](cls_feat)

            # extract feature
            reg_feat = self.reg_convs[k](x)
            # predict bbox
            reg_output = self.reg_preds[k](reg_feat)

            # predict if object exist
            obj_output = self.obj_preds[k](reg_feat)

            output = torch.cat([reg_output, obj_output, cls_output], 1)
            outputs.append(output)
        return outputs
