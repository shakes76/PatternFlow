# reads the data

#file path: ../AD_NC/test
import os
from pathlib import Path
import numpy as np
from PIL import Image

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
    return (train_images, test_images)
    
"""
train_images = []
for f in os.listdir(os.path.join(root_path, 'Train')):
    img = np.array(Image.load(img))
    train_images(img)
"""