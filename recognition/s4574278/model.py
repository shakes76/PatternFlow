from enum import Enum
import torch
import torchvision
from torch import nn
from typing import List

import data

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


class YoloxModel(nn.Module):
    def __init__(
        self, classes: List[str], model_size: ModelSize, input_shape=(512, 512)
    ) -> None:
        super().__init__()
        self.num_classes = len(classes)
        self.input_shape = input_shape
        self.backbone = PAFPN(model_size)
        self.head = DetectionHead(self.num_classes, width=model_size.value[1])

    def forward(self, x):
        # PAFPN
        features = self.backbone.forward(x)
        return self.head(features)

    def init_state(self, file=None, map_location=None):
        if file is None:
            self.apply(self._init_weight)
            print("Model generated")
        if map_location is None:
            map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.load_state_dict(torch.load(file, map_location=map_location))
            print(f"Model loaded from {file}")

    @staticmethod
    def _init_weight(module: nn.Module):
        for m in module.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(module.weight.data, gain=2e-2)

    def save_state(self, file):
        torch.save(self.state_dict(), file)

    def post_process(self, outputs) -> torch.Tensor:
        """
            Apply NMS => select bounding boxes from all the candidates, and
            Restore Image size
        """
        grids = []
        strides = []
        grid_shapes = [output.shape[-2:] for output in outputs]
        #   outputs
        #   n, 4 + 1 + num_classes, 64, 64 => n, 4 + 1 + num_classes, 4096
        #   Same for 32, 32 & 16, 16 layers
        #   concat together  4096 + 1024 + 256 => n, 5376, 5 + num_classes
        outputs = torch.cat([output.flatten(start_dim=2) for output in outputs], dim=2).permute(0, 2, 1)

        # compress to probability 
        outputs[:, :, 4:] = torch.sigmoid(outputs[:, :, 4:])
        for h, w in grid_shapes:
            # Build the grid
            ys, xs = torch.meshgrid([torch.arange(h), torch.arange(w)])
            # ---------------------------#
            #   1, 4096, 2
            #   1, 1024, 2
            #   1, 256, 2
            # ---------------------------#   
            grid = torch.stack((xs, ys), 2).view(1, -1, 2)
            shape = grid.shape[:2]

            grids.append(grid)
            strides.append(torch.full((shape[0], shape[1], 1), self.input_shape[0] / h))

        # concat grids and strides
        grids = torch.cat(grids, dim=1).type(outputs.type())
        strides = torch.cat(strides, dim=1).type(outputs.type())

        # apply offset & stride to bbox = (the predicted offsets + grid anchor coordinate) x stride size
        outputs[..., :2] = (outputs[..., :2] + grids) * strides
        # use Exponential function to restore the width and height
        outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides

        # Normalization by dividing by image shape 
        outputs[..., [0, 2]] = outputs[..., [0, 2]] / self.input_shape[1]
        outputs[..., [1, 3]] = outputs[..., [1, 3]] / self.input_shape[0]

        # # Restore image shape
        # outputs[..., [0,1,2,3]]=bbox_to_corner()
        return outputs


class YoloXDecoder:
    def __init__(self, model_weights_path, classes: List[str], model_size=ModelSize.S, input_shape=None, confidence=0.5,
                 nms_threshold=0.5):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YoloxModel(classes, model_size, input_shape).to(device)
        self.model.init_state(model_weights_path)
        self.model.evel()
        self.classes = classes
        self.num_classes = len(self.classes)
        self.input_shape = self.model.input_shape
        self.nms_threshold = nms_threshold
        self.confidence_bar = confidence

    def see(self, image):
        image_shape = image.shape()[0:2]
        images = torch.unsqueeze(image, 0)
        is_resized, _ = data.resize(images, self.input_shape, [])
        with torch.no_grad():
            outputs = self.model(is_resized)
            outputs = self.model.post_process(outputs)
        boxes = self.nms(
            outputs,
            self.num_classes,
            self.input_shape,
            image_shape,
            is_resized,
            confidence_bar=self.confidence_bar,
            nms_threshold=self.nms_threshold,
        )
        return boxes[0]

    def nms(self, pred, num_classes, input_shape, image_shape, is_resized, confidence_bar=0.5,
            nms_threshold=0.4):
        # n, n_points, n_class+5
        boxes_corner = bbox_to_corner(pred)

        output = [None for _ in range(len(pred))]
        # Loop through inputs
        for i, image_pred in enumerate(pred):
            #   cls_p  [n_points, 1]    confidence
            #   cls_index   [n_points, 1]    class
            cls_p, cls_index = torch.max(image_pred[:, 5:5 + num_classes], 1, keepdim=True)

            # filter by confidence
            conf_mask = (image_pred[:, 4] * cls_p[:, 0] >= confidence_bar).squeeze()

            if not image_pred.size(0):
                continue
            # If still have more boxes
            # remember: n_points x 7
            # 7 => x1, y1, x2, y2, obj_conf, cls_p, cls_index
            detections = torch.cat((image_pred[:, :5], cls_p, cls_index.float()), 1)
            detections = detections[conf_mask]

            nms_out_index = torchvision.ops.batched_nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                detections[:, 6],
                nms_threshold,
            )

            output[i] = detections[nms_out_index]
        return output

    def mAP(self, targets):
        """Calculate IOU nms for 0.5 and 0.75"""
        pass


##########################################################
#   PAFPN: Path Aggregation feature pyramid network
#   = Modified CSPNet + Feature pyramid
#   3 UpSampling + 3 DownSampling
##########################################################


class PAFPN(nn.Module):
    def __init__(self, model_size: ModelSize) -> None:
        super().__init__()
        in_channels = (1024, 512, 256)
        self.backbone = CSPNet(model_size)
        depth, width = model_size.value
        self.up_conv0 = ConvBlock(
            int(in_channels[0] * width),
            int(in_channels[1] * width),
            kernel_size=1,
            stride=1,
        )
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")
        self.csp0 = CSPLayer(
            int(in_channels[1] * width * 2),
            int(in_channels[1] * width),
            shortcut=False,
            num_bottleneck=round(depth * 3),
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
            False,
            num_bottleneck=round(depth * 3),
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
            shortcut=False,
            num_bottleneck=round(depth * 3),
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
            shortcut=False,
            num_bottleneck=round(depth * 3),
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
#   PAFPN: Path Aggregation feature pyramid network
##########################################################


class DetectionHead(nn.Module):
    def __init__(self, num_classes=50, width=1.0, in_channels=[256, 512, 1024]) -> None:
        super().__init__()

        self.cls_conv_layer = nn.ModuleList()
        self.reg_conv_layer = nn.ModuleList()
        self.cls_predictor = nn.ModuleList()
        self.box_predictor = nn.ModuleList()
        self.obj_predictor = nn.ModuleList()
        self.stems = nn.ModuleList()

        # Loop through 3 levels of Heads (default)
        for i in range(len(in_channels)):
            self.stems.append(
                ConvBlock(
                    in_channels=int(in_channels[i] * width),
                    out_channels=int(256 * width),
                    kernel_size=1,
                    stride=1,
                )
            )
            self.cls_conv_layer.append(
                nn.Sequential(
                    *[
                        ConvBlock(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            kernel_size=3,
                            stride=1,
                        ),
                        ConvBlock(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            kernel_size=3,
                            stride=1,
                        ),
                    ]
                )
            )
            self.cls_predictor.append(
                nn.Conv2d(
                    int(256 * width),
                    num_classes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )

            self.reg_conv_layer.append(
                nn.Sequential(
                    *[
                        ConvBlock(
                            int(256 * width),
                            int(256 * width),
                            kernel_size=3,
                            stride=1,
                        ),
                        ConvBlock(
                            int(256 * width),
                            int(256 * width),
                            kernel_size=3,
                            stride=1,
                        ),
                    ]
                )
            )
            self.box_predictor.append(
                nn.Conv2d(
                    int(256 * width),
                    4,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.obj_predictor.append(
                nn.Conv2d(
                    int(256 * width),
                    1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )

    def forward(self, inputs):
        """
        inputs:
            feature3  64, 64, 256
            feature4  32, 32, 512
            feature5  16, 16, 1024
        outputs:
            feature3 64, 64, (4 + 1 + num_classes)
            feature3 32, 32, (4 + 1 + num_classes)
            feature3 16, 16, (4 + 1 + num_classes)
        """
        outputs = []
        for i, x in enumerate(inputs):
            # cross-channel fusion with 1x1 kernel 
            # => end_channel: 256
            x = self.stems[i](x)

            # predict class
            class_features = self.cls_conv_layer[i](x)
            # => 
            #   num_classes, 64, 64
            #   num_classes, 32, 32
            #   num_classes, 16, 16
            class_output = self.cls_predictor[i](class_features)

            # extract features for box and obj
            box_features = self.reg_conv_layer[i](x)

            # predict bbox
            # 4, 64, 64
            # 4, 32, 32
            # 4, 16, 16
            box_output = self.box_predictor[i](box_features)

            # predict if object exist
            obj_output = self.obj_predictor[i](box_features)

            output = torch.cat([box_output, obj_output, class_output], 1)
            outputs.append(output)

        return outputs


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
        self.focus = FocusLayer(
            in_channels=3, out_channels=base_channels * 4, kernel_size=3
        )
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
        self.stem = ConvBlock(in_channels * 4, out_channels, kernel_size, stride)

    def forward(self, x):
        x = torch.cat(
            [
                # pixel top-left
                x[..., ::2, ::2],
                # pixel bottom-left
                x[..., 1::2, ::2],
                # pixel top-right
                x[..., ::2, 1::2],
                # pixel bottom-right
                x[..., 1::2, 1::2],
            ],
            dim=-1,
        )
        return self.stem(x)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, depth, is_last=False):
        super().__init__()
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
    ) -> None:
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvBlock(in_channels, hidden_channels, kernel_size=1, stride=1)
        self.conv2 = ConvBlock(in_channels, hidden_channels, kernel_size=1, stride=1)
        self.layers = nn.Sequential(
            *[
                Bottleneck(hidden_channels, hidden_channels, shortcut=shortcut)
                for _ in range(num_bottleneck)
            ]
        )
        # stack together
        self.conv3 = ConvBlock(
            hidden_channels * 2, out_channels, kernel_size=1, stride=1
        )

    def forward(self, x):
        # left
        y = self.conv1(x)
        y = self.layers(y)
        # right
        y_2 = self.conv2(x)
        # concat
        y = torch.cat((y, y_2), dim=1)
        # fusion
        return self.conv3(y)


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, shortcut=True, expansion=0.5) -> None:
        super().__init__()
        inner_channels = int(out_channels * expansion)
        self.layers = nn.Sequential(
            *[
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
        super().__init__()
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
        x = torch.cat([x] + [pool(x) for pool in self.modules], dim=1)
        x = self.conv2(x)
        return x


##########################################################
#   General Building blocks
##########################################################


class ConvBlock(nn.Module):
    """Conv2D -> BatchNorm -> ACT"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        bias=False,
        act: str = "silu",
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2
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
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    bias=bias,
                ),
                nn.BatchNorm2d(out_channels),
                activation,
            ]
        )

        def forward(self, x):
            return self.layers(x)


##########################################################
# Helpers
##########################################################


def bbox_to_corner(boxes: torch.Tensor):
    """convert cxcywh (center-x, center-y, width, height) bbox to xyxy(top-left, bottom-right)
    Input: 
        batch, boxes, [cx, cy, w, h]
        center_x : boxes[:, :, 0],
        center_y : boxes[:, :, 1],
        width    : boxes[:, :, 2],
        height   : boxes[:, :, 3],
    """
    half_width = boxes[:, 2] / 2
    half_height = boxes[:, :, 3] / 2
    boxes_corner = boxes.detach().clone().cuda()
    boxes_corner[:, :, 0] = boxes[:, :, 0] - half_width
    boxes_corner[:, :, 1] = boxes[:, :, 1] - half_height
    boxes_corner[:, :, 2] = boxes[:, :, 0] + half_width
    boxes_corner[:, :, 3] = boxes[:, :, 1] + half_height
    return boxes_corner
