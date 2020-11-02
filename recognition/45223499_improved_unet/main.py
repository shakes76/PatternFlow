import tensorflow as tf
from matplotlib import image
import imageio
import os
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import PIL
import PIL.Image
import random
import cv2
import glob

# download images
img_height =256
img_width = 256
imag_channels = 3
imag_input = "C:/Users/s4522349/Downloads/ISIC2018_Task1-2_Training_Input_x2/"
output = "C:/Users/s4522349/Downloads/ISIC2018_Task1_Training_GroundTruth_x2/"
imag_input = pathlib.Path(imag_input)
imag_output = pathlib.Path(output)

# list files
def lis_files(path, names):   
    lis = []
    for name in names:
        image = os.path.join(path, name)
        image = image.replace('\\', '/')
        lis += [image]
    return lis
image_name = os.listdir(imag_input)[1:2595]
list_input = lis_files(imag_input, image_name)
imag_output = os.listdir(imag_output)[1:2595]
list_output = lis_files(output, imag_output)

# divide dataset into train, test and validation datasets
train_X = list_input[:1558]
val_X = list_input[1558:2076]
test_X = list_input[2076:2594]
train_y = list_output[:1558]
val_y = list_output[1558:2076]
test_y = list_output[2076:2594]

