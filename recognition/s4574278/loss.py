##########################################################
# Loss
##########################################################
from enum import Enum

import torch
from torch import nn
from torch.nn import functional as F


class YoloxLoss(nn.Module):
    def __init__(self, num_classes, strides=(8, 16, 32)):
        super().__init__()
        self.num_classes = num_classes
        self.strides = strides

        self.bce_w_logits_loss = nn.BCEWithLogitsLoss(reduction="sum")
        self.iou_loss = GIoU(loss_reduction="num")
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
        #   f5: [n(batch_size), num_classes + 1 + 4, 16, 16]
        #   f4: [n, num_classes + 1 + 4,, 32, 32]
        #   f3: [n, num_classes + 1 + 4,, 64, 64]
        # Output:
        #   16x16 = 256 =>[n, 256, num_classes + 1 + 4]
        #   32x32= 1,024 =>[n, 1024, num_classes + 1 + 4]
        #   64x64 = 4,096 =>[n, 4096, num_classes + 1 + 4]
        # x_offsets => [n, 4096]
        for i, (stride, output) in enumerate(zip(self.strides, inputs)):
            output, grid = self.build_grid(i, output, stride)
            x_offset.append(grid[:, :, 0])
            y_offset.append(grid[:, :, 1])
            expanded_strides.append(torch.ones_like(grid[:, :, 0]) * stride)
            results.append(output)

        losses = self.calc_losses(x_offset, y_offset, expanded_strides, torch.cat(results, dim=1))
        return losses

    def build_grid(self, i, output, stride):
        current_grid = self.grids[i]
        rows, cols = output.shape[-2:]
        if current_grid.shape[2:4] != output.shape[2:4]:
            # build new grid if dimension dismatch
            ys, xs = torch.meshgrid([torch.arange(rows), torch.arange(cols)])
            current_grid = torch.stack((xs, ys), 2).view(1, rows, cols, 2).type(output.type())
            self.grids[i] = current_grid
        current_grid = current_grid.view(1, -1, 2)

        output = output.flatten(start_dim=2).permute(0, 2, 1)
        output[..., :2] = (output[..., :2] + current_grid) * stride
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride
        return output, current_grid

    def calc_losses(self, x_offset, y_offset, expanded_strides, labels, outputs):
        # n x n_points x 4
        bbox_preds = outputs[:, :, :4]
        # n x n_points x 1
        obj_preds = outputs[:, :, 4:5]
        # n x n_points x n_classes
        cls_preds = outputs[:, :, 5:]
        # 4096 + 1024 + 256
        total_n_points = outputs.shape[1]
        # concat 3 layers of n_points into to 1 x n_points
        # 4096 + 1024 + 256
        x_offset = torch.cat(x_offset, 1)
        y_offset = torch.cat(y_offset, 1)
        expanded_strides = torch.cat(expanded_strides, 1)

        cls_targets = []
        reg_targets = []
        obj_targets = []
        fg_masks = []

        num_fg = 0.0
        for batch_idx in range(outputs.shape[0]):
            num_gt = len(labels[batch_idx])
            if num_gt == 0:
                cls_target = outputs.new_zeros((0, self.num_classes))
                reg_target = outputs.new_zeros((0, 4))
                obj_target = outputs.new_zeros((total_n_points, 1))
                fg_mask = outputs.new_zeros(total_n_points).bool()
            else:
                # -----------------------------------------------#
                #   gt_bboxes_per_image     [num_gt, num_classes]
                #   gt_classes              [num_gt]
                #   bboxes_preds_per_image  [n_anchors_all, 4]
                #   cls_preds_per_image     [n_anchors_all, num_classes]
                #   obj_preds_per_image     [n_anchors_all, 1]
                # -----------------------------------------------#
                gt_bboxes_per_image = labels[batch_idx][..., :4]
                gt_classes = labels[batch_idx][..., 4]
                bboxes_preds_per_image = bbox_preds[batch_idx]
                cls_preds_per_image = cls_preds[batch_idx]
                obj_preds_per_image = obj_preds[batch_idx]

                gt_matched_classes, fg_mask, pred_ious_this_matching, matched_gt_inds, num_fg_img = self.get_assignments(
                    num_gt, total_n_points, gt_bboxes_per_image, gt_classes, bboxes_preds_per_image,
                    cls_preds_per_image, obj_preds_per_image,
                    expanded_strides, x_offset, y_offset,
                )
                torch.cuda.empty_cache()
                num_fg += num_fg_img
                cls_target = F.one_hot(gt_matched_classes.to(torch.int64),
                                       self.num_classes).float() * pred_ious_this_matching.unsqueeze(-1)
                obj_target = fg_mask.unsqueeze(-1)
                reg_target = gt_bboxes_per_image[matched_gt_inds]
            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.type(cls_target.type()))
            fg_masks.append(fg_mask)

        cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)

        num_fg = max(num_fg, 1)
        loss_iou = self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)
        loss_obj = self.bce_w_logits_loss(obj_preds.view(-1, 1), obj_targets)
        loss_cls = self.bce_w_logits_loss(cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets)
        reg_weight = 5.0
        loss = reg_weight * loss_iou + loss_obj + loss_cls

        return loss / num_fg


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
