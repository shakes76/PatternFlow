from typing import List
import math
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches
import torch

##########################################################
# Visualization
##########################################################


def draw_bbox(image, boxes):
    tableau_light = (
        np.array(
            [
                (158, 218, 229),
                (219, 219, 141),
                (199, 199, 199),
                (247, 182, 210),
                (196, 156, 148),
                (197, 176, 213),
                (255, 152, 150),
                (152, 223, 138),
                (255, 187, 120),
                (174, 199, 232),
            ]
        )
        / 255
    )

    plt.imshow(image)

    def plot_bbox(bbox, color):
        plt.gca().add_patch(
            patches.Rectangle(
                xy=(bbox[0], bbox[1]),
                width=bbox[2] - bbox[0],
                height=bbox[3] - bbox[1],
                fill=False,
                edgecolor=color,
                linewidth=2,
            )
        )

    for i, box in enumerate(boxes):
        plot_bbox(box, tableau_light[i % 10])


##########################################################
# Helpers
##########################################################


def bbox_to_center(boxes: List):
    """convert (top-left,bottom-right) bbox to (center-x, center-y, width, height)"""
    x_min, y_min, x_max, y_max = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min
    return torch.stack((center_x, center_y, width, height), axis=-1)


def bbox_to_corner(boxes: List):
    """convert (center-x, center-y, width, height) bbox to (top-left, bottom-right)"""
    center_x, center_y, width, height = (
        boxes[:, 0],
        boxes[:, 1],
        boxes[:, 2],
        boxes[:, 3],
    )
    x_min = center_x - 0.5 * width
    y_min = center_y - 0.5 * height
    x_max = center_x + 0.5 * width
    y_max = center_y + 0.5 * height
    return torch.stack((x_min, y_min, x_max, y_max), axis=-1)


##########################################################
# Loss
##########################################################


def ciou(boxes_pred, boxes_target):
    """Copied from
    https://github.com/Zzh-tju/CIoU/blob/master/layers/modules/multibox_loss.py"""
    boxes_pred = torch.sigmoid(boxes_pred)
    boxes_target = torch.sigmoid(boxes_target)
    rows = boxes_pred.shape[0]
    cols = boxes_target.shape[0]
    ciou_arr = torch.zeros((rows, cols))
    if rows * cols == 0:
        return ciou_arr
    exchange = False
    if boxes_pred.shape[0] > boxes_target.shape[0]:
        boxes_pred, boxes_target = boxes_target, boxes_pred
        ciou_arr = torch.zeros((cols, rows))
        exchange = True
    w1 = torch.exp(boxes_pred[:, 2])
    h1 = torch.exp(boxes_pred[:, 3])
    w2 = torch.exp(boxes_target[:, 2])
    h2 = torch.exp(boxes_target[:, 3])
    area1 = w1 * h1
    area2 = w2 * h2
    center_x1 = boxes_pred[:, 0]
    center_y1 = boxes_pred[:, 1]
    center_x2 = boxes_target[:, 0]
    center_y2 = boxes_target[:, 1]

    inter_l = torch.max(center_x1 - w1 / 2, center_x2 - w2 / 2)
    inter_r = torch.min(center_x1 + w1 / 2, center_x2 + w2 / 2)
    inter_t = torch.max(center_y1 - h1 / 2, center_y2 - h2 / 2)
    inter_b = torch.min(center_y1 + h1 / 2, center_y2 + h2 / 2)
    inter_area = torch.clamp((inter_r - inter_l), min=0) * torch.clamp(
        (inter_b - inter_t), min=0
    )

    c_l = torch.min(center_x1 - w1 / 2, center_x2 - w2 / 2)
    c_r = torch.max(center_x1 + w1 / 2, center_x2 + w2 / 2)
    c_t = torch.min(center_y1 - h1 / 2, center_y2 - h2 / 2)
    c_b = torch.max(center_y1 + h1 / 2, center_y2 + h2 / 2)

    inter_diag = (center_x2 - center_x1) ** 2 + (center_y2 - center_y1) ** 2
    c_diag = torch.clamp((c_r - c_l), min=0) ** 2 + torch.clamp((c_b - c_t), min=0) ** 2

    union = area1 + area2 - inter_area
    u = (inter_diag) / c_diag
    iou = inter_area / union
    v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(w2 / h2) - torch.atan(w1 / h1)), 2)
    with torch.no_grad():
        S = (iou > 0.5).float()
        alpha = S * v / (1 - iou + v)
    ciou_arr = iou - u - alpha * v
    ciou_arr = torch.clamp(ciou_arr, min=-1.0, max=1.0)
    if exchange:
        ciou_arr = ciou_arr.T
    return torch.sum(1 - ciou_arr)
