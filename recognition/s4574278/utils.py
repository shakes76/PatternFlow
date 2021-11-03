import math
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches
import torch
from torch import nn

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

    plt.imshow(image, interpolation="nearest")
    plt.axis("off")

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

class YoloxLoss(nn.Module):    
    def __init__(self, num_classes, strides=[8, 16, 32]):
        super().__init__()
        self.num_classes = num_classes
        self.strides = strides

        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = ciou
        self.grids = [torch.zeros(1)] * len(strides)

    def forward(self, inputs):
        results = []
        x_offset = []
        y_offset = []
        expanded_strides = []

        for k, (stride, output) in enumerate(zip(self.strides, inputs)):
            output, grid = self.get_output_and_grid(output, k, stride)
            x_offset.append(grid[:, :, 0])
            y_offset.append(grid[:, :, 1])
            expanded_strides.append(torch.ones_like(grid[:, :, 0]) * stride)
            results.append(output)

        return self.get_losses(x_offset, y_offset, expanded_strides, torch.cat(results, dim=1))


##########################################################
# Helpers
##########################################################


def resize(image_path, target_size, boxes, dtype=torch.float16):
    """resize image and annotation box"""
    image = Image.open(image_path)
    # Size of image
    image_height, image_width = image.size
    width, height = target_size
    # resize
    scale = min(width / image_width, height / image_height)
    new_width = int(image_width * scale)
    new_height = int(image_height * scale)
    delta_x = (width - new_width) // 2
    delta_y = (height - new_height) // 2

    # fill black background
    image = image.resize((new_width, new_height))
    new_image = Image.new("RGB", (width, height), (0, 0, 0))
    new_image.paste(image, (delta_x, delta_y))
    image_tensor = torch.tensor(new_image, dtype)

    # Adjust BBox accordingly
    if len(boxes) > 0:
        # new x min and x max
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * new_width / image_width + delta_x
        # new y min and y max
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * new_height / image_height + delta_y
        # x_min y_min must >= 0
        boxes[:, 0:2][boxes[:, 0:2] < 0] = 0
        # adjust the x_max, y_max according to the width and height
        boxes[:, 2][boxes[:, 2] > width] = width
        boxes[:, 3][boxes[:, 3] > height] = height
        # drop the boxes if any dimension collapse to zero
        box_w = boxes[:, 2] - boxes[:, 0]
        box_h = boxes[:, 3] - boxes[:, 1]
        boxes = boxes[torch.logical_and(box_w > 1, box_h > 1)]

    return image_tensor, boxes
