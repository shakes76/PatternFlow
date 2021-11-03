##########################################################
# Loss
##########################################################
from enum import Enum

import torch
from torch import nn


class YoloxLoss(nn.Module):
    def __init__(self, num_classes, strides=(8, 16, 32)):
        super().__init__()
        self.num_classes = num_classes
        self.strides = strides

        self.bce_w_logits_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = GIoU(loss_reduction="none")
        # grids = [tensor([0.]), tensor([0.]), tensor([0.])]
        self.grids = [torch.zeros(1)] * len(strides)

    def forward(self, inputs):
        # inputs is the 3 layers of features from 3 to 5
        # see model.py YoloxModel.forward()
        x_offset = []
        y_offset = []
        results = []
        expanded_strides = []

        # Inputs:
        #    [[batch_size, num_classes + 5, 16, 16]
        #
        for k, (stride, output) in enumerate(zip(self.strides, inputs)):
            output, grid = self.get_output_and_grid(output, k, stride)
            x_offset.append(grid[:, :, 0])
            y_offset.append(grid[:, :, 1])
            expanded_strides.append(torch.ones_like(grid[:, :, 0]) * stride)
            results.append(output)

        return self.get_losses(
            x_offset, y_offset, expanded_strides, torch.cat(results, dim=1)
        )


class LossType(Enum):
    IoU = 0
    GIoU = 1


class GIoU(nn.Module):
    def __init__(self, reduction="none", loss_type: LossType = LossType.GIoU):
        super(GIoU, self).__init__()
        self.reduction = reduction
        self.type = loss_type

    def forward(self, boxes, boxes_gt):
        assert boxes.shape[0] == boxes_gt.shape[0]

        boxes = boxes.view(-1, 4)
        boxes_gt = boxes_gt.view(-1, 4)
        tl = torch.max(
            (boxes[:, :2] - boxes[:, 2:] / 2), (boxes_gt[:, :2] - boxes_gt[:, 2:] / 2)
        )
        br = torch.min(
            (boxes[:, :2] + boxes[:, 2:] / 2), (boxes_gt[:, :2] + boxes_gt[:, 2:] / 2)
        )

        area_boxes = torch.prod(boxes[:, 2:], 1)
        area_gt = torch.prod(boxes_gt[:, 2:], 1)

        en = (tl < br).type(tl.type()).prod(dim=1)
        area_intersect = torch.prod(br - tl, 1) * en
        area_union = area_boxes + area_gt - area_intersect
        iou = (area_intersect) / (area_union + 1e-16)

        if self.type == LossType.IoU:
            loss = 1 - iou ** 2
        elif self.type == LossType.GIoU:
            center_tl = torch.min(
                (boxes[:, :2] - boxes[:, 2:] / 2), (boxes_gt[:, :2] - boxes_gt[:, 2:] / 2)
            )
            center_br = torch.max(
                (boxes[:, :2] + boxes[:, 2:] / 2), (boxes_gt[:, :2] + boxes_gt[:, 2:] / 2)
            )
            area_c = torch.prod(center_br - center_tl, 1)
            giou = iou - (area_c - area_union) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        # by default there no reduction
        return loss
