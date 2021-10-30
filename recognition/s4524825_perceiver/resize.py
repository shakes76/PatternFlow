import os
import numpy as np
import json
import cv2
import matplotlib.pyplot as plt
import data

#iterates through directories, and resizes images to 64x64x3

START_DIR = "data/AKOA_Analysis/"

label = 0
for dir in os.listdir(START_DIR):
    for file in os.listdir(START_DIR + "/" + dir):
        filename = f"{START_DIR}/{dir}/{file}"
        newfilename = f"data/resize/{dir}/{file}"
        image = cv2.imread(filename)
        # plt.imshow(image)
        # plt.imshow(cv2.resize(image, (32, 32)))
        # plt.show()
        image = cv2.resize(image, (64, 64))
        # plt.imshow(image)
        # print("normal")
        # plt.show()
        # plt.imshow(data.random_flip(image))
        # print("flip")
        # plt.show()
        # print(filename, newfilename)
        cv2.imwrite(newfilename, image)