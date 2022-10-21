import torch
import numpy as np
import os
from PIL import Image

def get_IOU(box_a, box_b):
    x_left = max(box_a[0], box_b[0])
    y_top = max(box_a[1], box_b[1])
    x_right = min(box_a[0] + box_a[2], box_b[0] + box_b[2])
    y_bottom = min(box_a[1] + box_a[3], box_b[1] + box_b[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersect_area = (x_right - x_left) * (y_bottom - y_top)

    box_a_area = (box_a[2] * box_a[3])
    box_b_area = (box_b[2] * box_b[3])

    iou = intersect_area / float(box_a_area + box_b_area - intersect_area)

    return iou

def calculate_Image_IOU(test_image_path, label_path):
    # Import model
    model = torch.hub.load('yolov5', 'custom', 'runs/train/exp2/weights/best.pt')
    iou_list = []
    for dirname, _, filenames in os.walk(test_image_path):
        for filename in filenames:
            label_filename = filename.replace('.jpg', '.txt')
            file_path = os.path.join(dirname, filename)
            img = Image.open(file_path)
            prediction = model(img)
            prediction = prediction.xywh[0]

            # If no bounding boxes
            if not prediction.empty:
                min_x = prediction.loc[0]["xcenter"]/img.shape[0]
                max_y = prediction.loc[0]["ycenter"]/img.shape[1]
                width = prediction.loc[0]["width"]/img.shape[0]
                height = prediction.loc[0]["height"]/img.shape[1]
                
                # Open label .txt
                label_val = open(label_path + label_filename).read().split()

                # Convert center coords to left most and top most
                label_min_x = float(label_val[1]) - float(label_val[3])/2
                label_min_y = float(label_val[2]) - float(label_val[4])/2
                
                iou = get_IOU([min_x, max_y, width, height], [label_min_x, label_min_y, float(label_val[3]), float(label_val[4])])
                iou_list.append(iou)

    return min(iou_list), np.average(iou_list)
