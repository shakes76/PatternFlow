"""
 iterates through directories, and resizes images to 64x64x3 to save space, allowing to upload data quickly for training.
"""
import os
import numpy as np
import json
import cv2
import matplotlib.pyplot as plt
import data


START_DIR = "data/AKOA_Analysis/"

label = 0
for dir in os.listdir(START_DIR):
    for file in os.listdir(START_DIR + "/" + dir):
        filename = f"{START_DIR}/{dir}/{file}"
        newfilename = f"data/resize/{dir}/{file}"
        image = cv2.imread(filename)
        image = cv2.resize(image, (64, 64))
        cv2.imwrite(newfilename, image)