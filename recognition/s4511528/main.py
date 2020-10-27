"""
An improved uNet for ISICs dataset segmentation

@author foolish li jia min
"""

import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
from model import build_model


# Load images
images = [cv2.imread(file) for file in glob.glob('Downloads/ISIC2018_Task1-2_Training_Input_x2/*.jpg')]
masks = [cv2.imread(file, cv2.IMREAD_GRAYSCALE) for file in glob.glob('Downloads/ISIC2018_Task1_Training_GroundTruth_x2/*.png')]

width = 256
height = 256
channels = 3

# Resize and reshape the image dataset
for i in range(len(images)):
    images[i] = cv2.resize(images[i],(height,width))
    images[i] = images[i]/255


for i in range(len(masks)):
    masks[i] = cv2.resize(masks[i],(height,width))
    masks[i] = masks[i]/255
    masks[i][masks[i] > 0.5] = 1
    masks[i][masks[i] <= 0.5] = 0

X = np.zeros([2594, height, width, channels])
y = np.zeros([2594, height, width])

for i in range(len(images)):
    X[i] = images[i]

for i in range(len(masks)):
    y[i] = masks[i]
        
y = y[:, :, :, np.newaxis]