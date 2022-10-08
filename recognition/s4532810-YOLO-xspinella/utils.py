import gdown
from zipfile import ZipFile
import os
import PIL
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import torchvision.transforms as T

"""
Function library for the patternflow YOLO project
"""

def Draw_Box(img_fp: str, box_spec: list, out_fp: str):
    """
    Draws the specified box on the given image
    :param img_fp: filepath to the image
    :param box_spec: box size/location/class specification as:
                        [class, centre_x, centre_y, width, height]
    :param out_fp: output file location
    """
    ### open image with cv2, save image size, define box specs ###
    img = cv2.imread(img_fp)
    height, width, _ = img.shape
    class_id, c_x, c_y, w, h = box_spec
    class_id = int(class_id)

    ### define specifications for drawing box for different classes ###
    class_list = ['Not Melanoma', 'Melanoma']
    colours = [(0, 255, 0), (0, 255, 255)]
    label = class_list[class_id]
    colour = colours[class_id] 

    ### redefine box location in cv2 format, un-normalise co-ords ###
    # This box format appears to be: [mid of left edge, mid of top edge, width, height]
    box = [int((c_x - 0.5*w)* width), int((c_y - 0.5*h) * height), int(w*width), int(h*height)]
    cv2.rectangle(img, box, color=colour)   # bounding box
    cv2.rectangle(img, (box[0], box[1] - 20), (box[0] + box[2], box[1]), colour, -1)    # label box 
    cv2.putText(img, class_list[class_id], (box[0], box[1] - 5), cv2.FONT_HERSHEY_DUPLEX, .5, (0,0,0))
    cv2.imwrite(out_fp, img)