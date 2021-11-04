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
        self.iou_loss = GIoU(loss_reduction="sum")
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
            n_gt = len(labels[batch_idx])
            if n_gt == 0:
                cls_target = outputs.new_zeros((0, self.num_classes))
                reg_target = outputs.new_zeros((0, 4))
                obj_target = outputs.new_zeros((total_n_points, 1))
                fg_mask = outputs.new_zeros(total_n_points).bool()
            else:
                # gt_bboxes_per_image    n_gt, num_classes
                # gt_classes             n_gt
                # bboxes_preds_per_image n_anchors, 4
                # cls_preds_per_image    n_anchors, num_classes
                # obj_preds_per_image    n_anchors, 1
                gt_bboxes_per_image = labels[batch_idx][..., :4]
                gt_classes = labels[batch_idx][..., 4]
                bboxes_preds_per_image = bbox_preds[batch_idx]
                cls_preds_per_image = cls_preds[batch_idx]
                obj_preds_per_image = obj_preds[batch_idx]

                gt_matched_classes, fg_mask, pred_ious_this_matching, matched_gt_inds, num_fg_img = self.sim_oat(
                    n_gt, total_n_points, gt_bboxes_per_image, gt_classes, bboxes_preds_per_image,
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

    @torch.no_grad()
    def sim_oat(self, n_gt, n_anchors, gt_boxes_per_image, gt_classes, bboxes_preds_per_image, cls_preds_per_image,
                obj_preds_per_image, expanded_strides, x_offset, y_offset):
        """
        Logic from official YOLOX, it consists of 4 steps, 
            1. Define a set of positive cases by look into the pixels around center points
            2. Calculate the pair-wise reg/cls loss of samples against each gt (Loss-aware)
            3. From each GT determine a dynamic K (can be manually stated, from 5 ~ 15)
            4. From the bigger picture, eliminate double-assigned samples
        """
        # fg_mask: [n_anchors]
        # is_in_centers_of_boxes: [n_gt, len(fg_mask)]
        fg_mask, is_in_centers_of_boxes = self.check_in_boxes(gt_boxes_per_image, expanded_strides, x_offset,
                                                              y_offset, n_anchors, n_gt)

        # fg_mask: [n_anchors]
        # bboxes_preds_per_image : [fg_mask, 4]
        # cls_preds : [fg_mask, num_classes]
        # obj_preds : [fg_mask, 1]
        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]
        cls_preds = cls_preds_per_image[fg_mask]
        obj_preds = obj_preds_per_image[fg_mask]
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]

        # pair_wise_ious      [n_gt, fg_mask]
        pair_wise_ious = self.bboxes_iou(gt_boxes_per_image, bboxes_preds_per_image, False)
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

        # cls_preds: n_gt, fg_mask, num_classes
        # gt_cls_per_image: n_gt, fg_mask, num_classes
        cls_preds = cls_preds.float().unsqueeze(0).repeat(n_gt, 1, 1).sigmoid_() * obj_preds.unsqueeze(0).repeat(
            n_gt, 1, 1).sigmoid_()
        gt_cls_per_image = F.one_hot(gt_classes.to(torch.int64), self.num_classes).type(torch.half).unsqueeze(1). \
            repeat(1, num_in_boxes_anchor, 1)
        pair_wise_cls_loss = F.binary_cross_entropy(cls_preds.sqrt_(), gt_cls_per_image, reduction="none").sum(-1)
        del cls_preds

        costs = pair_wise_cls_loss + 3.0 * pair_wise_ious_loss + 100000.0 * (~is_in_centers_of_boxes).half()

        num_fg, gt_matched_classes, pred_ious, matched_gt_idx = self.dynamic_k_matching(costs,
                                                                                                       pair_wise_ious,
                                                                                                       gt_classes,
                                                                                                       n_gt, fg_mask)
        del pair_wise_cls_loss, costs, pair_wise_ious, pair_wise_ious_loss
        return gt_matched_classes, fg_mask, pred_ious, matched_gt_idx, num_fg


    def check_in_boxes(self, gt_boxes_per_image, expanded_strides, x_offset, y_offset, total_num_anchors, n_gt, center_radius = 2.5):
        # expanded_strides_per_image : n_anchors
        # {x/y}_centers_per_image: n_gt, n_anchors
        expanded_strides_per_image  = expanded_strides[0]
        x_centers_per_image         = ((x_offset[0] + 0.5) * expanded_strides_per_image).unsqueeze(0).repeat(n_gt, 1)
        y_centers_per_image         = ((y_offset[0] + 0.5) * expanded_strides_per_image).unsqueeze(0).repeat(n_gt, 1)

        # gt_bboxes_per_image_x       [n_gt, n_anchors]
        gt_bboxes_per_image_l = (gt_boxes_per_image[:, 0] - 0.5 * gt_boxes_per_image[:, 2]).unsqueeze(1).repeat(1, total_num_anchors)
        gt_bboxes_per_image_r = (gt_boxes_per_image[:, 0] + 0.5 * gt_boxes_per_image[:, 2]).unsqueeze(1).repeat(1, total_num_anchors)
        gt_bboxes_per_image_t = (gt_boxes_per_image[:, 1] - 0.5 * gt_boxes_per_image[:, 3]).unsqueeze(1).repeat(1, total_num_anchors)
        gt_bboxes_per_image_b = (gt_boxes_per_image[:, 1] + 0.5 * gt_boxes_per_image[:, 3]).unsqueeze(1).repeat(1, total_num_anchors)


        # bbox_deltas: n_gt, n_anchors, 4
        b_l = x_centers_per_image - gt_bboxes_per_image_l
        b_r = gt_bboxes_per_image_r - x_centers_per_image
        b_t = y_centers_per_image - gt_bboxes_per_image_t
        b_b = gt_bboxes_per_image_b - y_centers_per_image
        bbox_deltas = torch.stack([b_l, b_t, b_r, b_b], 2)

        # is_in_boxes : n_gt, n_anchors
        # is_in_boxes_all: n_anchors
        is_in_boxes     = bbox_deltas.min(dim=-1).values > 0.0
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0

        gt_bboxes_per_image_l = (gt_boxes_per_image[:, 0]).unsqueeze(1).repeat(1, total_num_anchors) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_r = (gt_boxes_per_image[:, 0]).unsqueeze(1).repeat(1, total_num_anchors) + center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_t = (gt_boxes_per_image[:, 1]).unsqueeze(1).repeat(1, total_num_anchors) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_b = (gt_boxes_per_image[:, 1]).unsqueeze(1).repeat(1, total_num_anchors) + center_radius * expanded_strides_per_image.unsqueeze(0)

        # center_deltas: n_gt, n_anchors, 4
        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        center_deltas       = torch.stack([c_l, c_t, c_r, c_b], 2)

        # is_in_centers: n_gt, n_anchors
        # is_in_centers_all: n_anchors
        is_in_centers       = center_deltas.min(dim=-1).values > 0.0
        is_in_centers_all   = is_in_centers.sum(dim=0) > 0


        # is_in_boxes_anchor: n_anchors
        # is_in_boxes_and_center: n_gt, is_in_boxes_anchor
        is_in_boxes_anchor      = is_in_boxes_all | is_in_centers_all
        is_in_boxes_and_center  = is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]
        return is_in_boxes_anchor, is_in_boxes_and_center

def dynamic_k_matching(self, costs, pair_wise_ious, gt_classes, n_gt, fg_mask):
    # Inputs
    #   costs : n_gt, fg_mask
    #   pair_wise_ious : n_gt, fg_mask
    #   gt_classes : n_gt
    #   fg_mask : n_anchors
    
    # matching_matrix : n_gt, fg_mask
    matching_matrix = torch.zeros_like(costs)

    # select n_candidate_k points with maximum iou
    # then sum and calculate
    n_candidate_k = min(10, pair_wise_ious.size(1))
    # topk_ious      : n_gt, n_candidate_k
    topk_ious, _ = torch.topk(pair_wise_ious, n_candidate_k, dim=1)
    # dynamic_ks     : n_gt
    dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)

    for gt_idx in range(n_gt):
        # select k points with minimum cost for each gt
        _, pos_idx = torch.topk(costs[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False)
        matching_matrix[gt_idx][pos_idx] = 1.0
    del topk_ious, dynamic_ks, pos_idx

    # anchor_matching_gt: fg_mask
    anchor_matching_gt = matching_matrix.sum(0)
    if (anchor_matching_gt > 1).sum() > 0:
        
        # When a anchor point matching mulitple gt, we only keep the one with smallest cost
        _, cost_argmin = torch.min(costs[:, anchor_matching_gt > 1], dim=0)
        matching_matrix[:, anchor_matching_gt > 1] *= 0.0
        matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
    # fg_mask_inboxes  [fg_mask]
    fg_mask_inboxes = matching_matrix.sum(0) > 0.0
    # num_fg = num of feature point for positive samples
    num_fg = fg_mask_inboxes.sum().item()

    # update fg_mask
    fg_mask[fg_mask.clone()] = fg_mask_inboxes

    # Find the classes matched by the anchor point
    matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
    gt_matched_classes = gt_classes[matched_gt_inds]

    pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[fg_mask_inboxes]
    return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds

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

        # if no reduction
        return loss
