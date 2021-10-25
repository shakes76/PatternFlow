import tensorflow as tf
import numpy as np
import os
import cv2

#helpers
def load_list_from_txt(txt_path, image_dir_path):
    """
        Loads list of images in txt file denoted as:
            `image path relative to image_dir_path` label
    """
    images = []
    labels = []
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.replace('\n', '').split(' ')
            images.append(os.path.join(image_dir_path, line[0]))
            labels.append(int(line[1]))

    return images, labels


#training iterators etc.

def training_data_iterator():
    images, labels = load_list()
