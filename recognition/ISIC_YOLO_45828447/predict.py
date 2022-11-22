# The prediction functionality is done through YOLOv5's detect.py, which is used inside the jupyter notebook, the following are just calculations
# for determining some of the metrics used to determine the accuracy of the prediction
import torch
import numpy as np
import os
from PIL import Image

def get_IOU(box_a, box_b):
    if (box_a[0] - box_a[2]/2 > box_b[0] - box_b[2]/2):
        temp = box_a
        box_a = box_b
        box_b = temp
    x_left = max(box_a[0] - box_a[2]/2, box_b[0] - box_b[2]/2)
    y_top = max(box_a[1] - box_a[3]/2, box_b[1] - box_b[3]/2)
    x_right = min(box_a[0] + box_a[2]/2, box_b[0] + box_b[2]/2)
    y_bottom = min(box_a[1] + box_a[3]/2, box_b[1] + box_b[3]/2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersect_area = (x_right - x_left) * (y_bottom - y_top)

    box_a_area = (box_a[2] * box_a[3])
    box_b_area = (box_b[2] * box_b[3])

    iou = intersect_area / float(box_a_area + box_b_area - intersect_area)

    return iou

def calculate_Image_IOU(test_image_path, label_path):
    # Import model
    model = torch.hub.load('ultralytics/yolov5', 'custom', '/kaggle/working/yolov5/runs/train/exp5/weights/best.pt')
    iou_list = []
    for dirname, _, filenames in os.walk(test_image_path):
        for filename in filenames:
            label_filename = filename.replace('.jpg', '.txt')
            file_path = os.path.join(dirname, filename)
            img = Image.open(file_path)
            prediction = model(img)
            prediction = prediction.pandas().xywh[0]

            # If no bounding boxes
            if not prediction.empty:
                center_x = prediction.loc[0]["xcenter"]/np.array(img).shape[0]
                center_y = prediction.loc[0]["ycenter"]/np.array(img).shape[1]
                width = prediction.loc[0]["width"]/np.array(img).shape[0]
                height = prediction.loc[0]["height"]/np.array(img).shape[1]

                # Open label .txt
                label_val = open(label_path + label_filename).read().split()

                iou = get_IOU([center_x, center_y, width, height], [float(label_val[1]), float(label_val[2]), float(label_val[3]), float(label_val[4])])
                iou_list.append(iou)

    return min(iou_list), np.average(iou_list)
