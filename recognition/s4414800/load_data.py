"""
Loading the data locally, generate numpy file so that data 
can be easily used by other high-performance compilers.

@author Yin Peng
@email yin.peng@uqconnect.edu.au
"""

import cv2
import glob
import numpy as np

# The original image size is 511 X3 84, which is scaled down to 256 X 192
IMG_HEIGHT = 256
IMG_WIDTH = 192

def getData(path):
    data = []
    # Get all image path
    image_path = glob.glob(path)
    for path in image_path:
        # IMREAD_GRAYSCALE 
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_WIDTH,IMG_HEIGHT))
        if img is not None:
            data.append(img)

    return np.array(data)

mask_data = getData("ISIC2018_Task1_Training_GroundTruth_x2/*.png")
input_data = getData("ISIC2018_Task1-2_Training_Input_x2/*.jpg")

# Save the data for loading
np.save("mask_data.npy",mask_data)
np.save("input_data.npy",input_data)
