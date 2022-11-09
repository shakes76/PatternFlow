import torch
import torch.nn as nn
import math
import numpy as np


# code is referenced from https://github.com/bubbliiiing/yolo3-pytorch/blob/master/nets/yolo_training.py
# some adjusted small parts
class YOLOLoss(nn.Module):
    def __init__(self, anchors, num_classes, input_shape, anchors_mask):
        super().__init__()
        self.anchors = anchors
        self.num_classes = num_classes
        self.bs_iou = []
        self.avr_iou = []
        self.bbox_attrs = 5 + num_classes
        self.input_shape = input_shape
        self.anchors_mask = anchors_mask

        self.ignore_threshold = 0.5


    def clip_by_tensor(self, t, t_min, t_max):
        t = t.float()
        result = (t >= t_min).float() * t + (t < t_min).float() * t_min
        result = (result <= t_max).float() * result + (result > t_max).float() * t_max
        return result

    def MSELoss(self, pred, target):
        return torch.pow(pred - target, 2)

    def BCELoss(self, pred, target):
        epsilon = 1e-7
        pred = self.clip_by_tensor(pred, epsilon, 1.0 - epsilon)
        output = - target * torch.log(pred) - (1.0 - target) * torch.log(1.0 - pred)
        return output

    def forward(self, current_anchor, input, targets=None):

        bs, height, width = input.size(0), input.size(2), input.size(3)

        # Get the stride, for the 13*13 feature layer, every feature point correspond 32 pixes
        # in the original picture
        # eg. input = 416*416, stride height = 416/13 = 32

        scaled_pixel_height = self.input_shape[0] / height
        scaled_pixel_width = self.input_shape[1] / width
        scaled_anchors = []
        for w, h in self.anchors:
            scaled_anchors.append((w / scaled_pixel_width, h / scaled_pixel_height)
                                 )

        # adjust the dim
        prediction = input.view(bs, len(self.anchors_mask[current_anchor]), self.bbox_attrs, height, width).permute(0,
                                                                                                                    1,
                                                                                                                    3,
                                                                                                                    4,
                                                                                                                    2).contiguous()
        # adjust the center anchor box with in 0-1
        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])

        w = prediction[..., 2]
        h = prediction[..., 3]

        conf = torch.sigmoid(prediction[..., 4])

        pred_cls = torch.sigmoid(prediction[..., 5:])

        y_true, noobj_mask, box_loss_scale,iou = self.get_target(current_anchor, targets, scaled_anchors, height, width)

        noobj_mask = self.get_ignore(current_anchor, x, y, h, w, targets, scaled_anchors, height, width, noobj_mask)


        y_true = y_true.cuda()
        noobj_mask = noobj_mask.cuda()
        box_loss_scale = box_loss_scale.cuda()

        box_loss_scale = 2 - box_loss_scale

        loss_x = torch.sum(self.BCELoss(x, y_true[..., 0]) * box_loss_scale * y_true[..., 4])
        loss_y = torch.sum(self.BCELoss(y, y_true[..., 1]) * box_loss_scale * y_true[..., 4])

        loss_w = torch.sum(self.MSELoss(w, y_true[..., 2]) * 0.5 * box_loss_scale * y_true[..., 4])
        loss_h = torch.sum(self.MSELoss(h, y_true[..., 3]) * 0.5 * box_loss_scale * y_true[..., 4])

        loss_conf = torch.sum(self.BCELoss(conf, y_true[..., 4]) * y_true[..., 4]) + \
                    torch.sum(self.BCELoss(conf, y_true[..., 4]) * noobj_mask)

        loss_cls = torch.sum(self.BCELoss(pred_cls[y_true[..., 4] == 1], y_true[..., 5:][y_true[..., 4] == 1]))

        loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls
        num_pos = torch.sum(y_true[..., 4])
        num_pos = torch.max(num_pos, torch.ones_like(num_pos))
        return loss, num_pos,iou

    def calculate_iou(self, _box_a, _box_b):

        b1_x1, b1_x2 = _box_a[:, 0] - _box_a[:, 2] / 2, _box_a[:, 0] + _box_a[:, 2] / 2
        b1_y1, b1_y2 = _box_a[:, 1] - _box_a[:, 3] / 2, _box_a[:, 1] + _box_a[:, 3] / 2

        b2_x1, b2_x2 = _box_b[:, 0] - _box_b[:, 2] / 2, _box_b[:, 0] + _box_b[:, 2] / 2
        b2_y1, b2_y2 = _box_b[:, 1] - _box_b[:, 3] / 2, _box_b[:, 1] + _box_b[:, 3] / 2

        box_a = torch.zeros_like(_box_a)
        box_b = torch.zeros_like(_box_b)
        box_a[:, 0], box_a[:, 1], box_a[:, 2], box_a[:, 3] = b1_x1, b1_y1, b1_x2, b1_y2
        box_b[:, 0], box_b[:, 1], box_b[:, 2], box_b[:, 3] = b2_x1, b2_y1, b2_x2, b2_y2

        A = box_a.size(0)
        B = box_b.size(0)

        max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2), box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
        min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2), box_b[:, :2].unsqueeze(0).expand(A, B, 2))
        inter = torch.clamp((max_xy - min_xy), min=0)
        inter = inter[:, :, 0] * inter[:, :, 1]

        area_a = ((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
        area_b = ((box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]

        union = area_a + area_b - inter
        return inter / union

    def get_target(self, current_anchor, targets, anchors, height, width):

        bs = len(targets)

        noobj_mask = torch.ones(bs, len(self.anchors_mask[current_anchor]), height, width, requires_grad=False)

        box_loss_scale = torch.zeros(bs, len(self.anchors_mask[current_anchor]), height, width, requires_grad=False)

        y_true = torch.zeros(bs, len(self.anchors_mask[current_anchor]), height, width, self.bbox_attrs,
                             requires_grad=False)
        for b in range(bs):
            if len(targets[b]) == 0:
                continue
            batch_target = torch.zeros_like(targets[b])

            batch_target[:, [0, 2]] = targets[b][:, [0, 2]] * width
            batch_target[:, [1, 3]] = targets[b][:, [1, 3]] * height
            batch_target[:, 4] = targets[b][:, 4]
            batch_target = batch_target.cpu()

            gt_box = torch.FloatTensor(torch.cat((torch.zeros((batch_target.size(0), 2)), batch_target[:, 2:4]), 1))

            anchor_shapes = torch.FloatTensor(
                torch.cat((torch.zeros((len(anchors), 2)), torch.FloatTensor(anchors)), 1))

            best_ns = torch.argmax(self.calculate_iou(gt_box, anchor_shapes), dim=-1)
            self.bs_iou.append(torch.max(self.calculate_iou(gt_box, anchor_shapes)))
            for t, best_n in enumerate(best_ns):
                if best_n not in self.anchors_mask[current_anchor]:
                    continue

                k = self.anchors_mask[current_anchor].index(best_n)

                i = torch.floor(batch_target[t, 0]).long()
                j = torch.floor(batch_target[t, 1]).long()

                c = batch_target[t, 4].long()

                noobj_mask[b, k, j, i] = 0

                y_true[b, k, j, i, 0] = batch_target[t, 0] - i.float()
                y_true[b, k, j, i, 1] = batch_target[t, 1] - j.float()
                y_true[b, k, j, i, 2] = math.log(batch_target[t, 2] / anchors[best_n][0])
                y_true[b, k, j, i, 3] = math.log(batch_target[t, 3] / anchors[best_n][1])
                y_true[b, k, j, i, 4] = 1
                y_true[b, k, j, i, c + 5] = 1

                box_loss_scale[b, k, j, i] = batch_target[t, 2] * batch_target[t, 3] / width / height


        iou_ = torch.tensor(self.bs_iou).sum() / len(self.bs_iou)
        self.bs_iou = []
        return y_true, noobj_mask, box_loss_scale,iou_

    def get_ignore(self, current_anchor, x, y, h, w, targets, scaled_anchors, height, width, noobj_mask):

        bs = len(targets)

        FloatTensor = torch.cuda.FloatTensor
        LongTensor = torch.cuda.LongTensor

        grid_x = torch.linspace(0, width - 1, width).repeat(height, 1).repeat(
            int(bs * len(self.anchors_mask[current_anchor])), 1, 1).view(x.shape).type(FloatTensor)
        grid_y = torch.linspace(0, height - 1, height).repeat(width, 1).t().repeat(
            int(bs * len(self.anchors_mask[current_anchor])), 1, 1).view(y.shape).type(FloatTensor)

        scaled_anchors_l = np.array(scaled_anchors)[self.anchors_mask[current_anchor]]
        anchor_w = FloatTensor(scaled_anchors_l).index_select(1, LongTensor([0]))
        anchor_h = FloatTensor(scaled_anchors_l).index_select(1, LongTensor([1]))

        anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, height * width).view(w.shape)
        anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, height * width).view(h.shape)

        pred_boxes_x = torch.unsqueeze(x.data + grid_x, -1)
        pred_boxes_y = torch.unsqueeze(y.data + grid_y, -1)
        pred_boxes_w = torch.unsqueeze(torch.exp(w.data) * anchor_w, -1)
        pred_boxes_h = torch.unsqueeze(torch.exp(h.data) * anchor_h, -1)
        pred_boxes = torch.cat([pred_boxes_x, pred_boxes_y, pred_boxes_w, pred_boxes_h], dim=-1)

        for b in range(bs):

            pred_boxes_for_ignore = pred_boxes[b].view(-1, 4)

            if len(targets[b]) > 0:
                batch_target = torch.zeros_like(targets[b])

                batch_target[:, [0, 2]] = targets[b][:, [0, 2]] * width
                batch_target[:, [1, 3]] = targets[b][:, [1, 3]] * height
                batch_target = batch_target[:, :4]

                anch_ious = self.calculate_iou(batch_target, pred_boxes_for_ignore)

                anch_ious_max, _ = torch.max(anch_ious, dim=0)
                anch_ious_max = anch_ious_max.view(pred_boxes[b].size()[:3])
                noobj_mask[b][anch_ious_max > self.ignore_threshold] = 0
        return noobj_mask
