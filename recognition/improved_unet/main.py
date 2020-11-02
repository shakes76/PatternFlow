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

