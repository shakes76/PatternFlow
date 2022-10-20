import os
from pathlib import Path
import numpy as np
from PIL import Image

"""
dataset.py
Contains the data loader for loading and preprocessing the data
"""

# Path to dataset directory
path = "C:/Users/Eric/Documents/uni/2022_sem2/comp3710/report/AD_NC"

# Load all the training images
def load_dataset():
    train_images = []
    for f in os.listdir(Path(path + "/train/AD").absolute()):
        img = np.array(Image.open(Path(path + "/train/AD/" + f).absolute()))
        train_images.append(img)
    for f in os.listdir(Path(path + "/train/NC").absolute()):
        img = np.array(Image.open(Path(path + "/train/NC/" + f).absolute()))
        train_images.append(img)
    test_images = []
    for f in os.listdir(Path(path + "/test/AD").absolute()):
        img = np.array(Image.open(Path(path + "/test/AD/" + f).absolute()))
        test_images.append(img)
    for f in os.listdir(Path(path + "/test/NC").absolute()):
        img = np.array(Image.open(Path(path + "/test/NC/" + f).absolute()))
        test_images.append(img)

    print("Data successfully loaded")
    return (np.array(train_images), np.array(test_images))