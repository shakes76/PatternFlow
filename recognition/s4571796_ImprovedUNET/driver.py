#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Joshua Yu Xuan Soo"
__studentID__ = "s4571796"
__email__ = "s4571796@uqconnect.edu.au"

"""
Driver Script for Image Segmentation on ISIC Melanoma Dataset by utilizing
an Improved UNET Model as defined in the imports.

References: 
    https://arxiv.org/pdf/1802.10508v1.pdf

"""
# Imports
from model import unet_model
import os
import glob
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Check for GPU -> Use Anaconda3\python.exe
# GPU Verified
"""
print("You are using TensorFlow version", tf.__version__)
if len(tf.config.list_physical_devices('GPU')) > 0:
        print("You have a GPU enabled.")
else:
    print("Please enable a GPU first.")
"""

# Global Variables
classes = 2 # Number of features as Output
# Maintain Aspect Ratio of Image
height = 96 # 96 Pixels
width = 128 # 128 Pixels

def format_images_png(directory=None):
    """ Performs a formatting that returns an original folder containing
    images into a numpy array in the form of (E, H, W) only if a directory
    is provided. Requires images to all be in .png format. Does not 
    perform any operations otherwise.

    Arguments:
    directory: The directory containing the images in png format // ("dir")

    Returns:
    images: Numpy Array consisting of all images from a directory
    """
    # Represent training image information as a list form
    images = []

    if directory != None:
        # Open all files in specific folder in directory
        for directory_path in glob.glob(directory):
            for img_path in glob.glob(os.path.join(directory_path, "*.png")):
                # Open the image
                img = cv2.imread(img_path, 0)       
                # Resize image as per limitations of GPU
                img = cv2.resize(img, (width, height))
                # Append images to list
                images.append(img)
       
        # Convert training data list form to a numpy array
        images = np.array(images)

    return images

def format_images_jpg(directory=None):
    """ Performs a formatting that returns an original folder containing
    images into a numpy array in the form of (E, H, W) only if a directory
    is provided. Requires images to all be in .png format. Does not 
    perform any operations otherwise.

    Arguments:
    directory: The directory containing the images in png format // ("dir")

    Returns:
    images: Numpy Array consisting of all images from a directory
    """
    # Represent training image information as a list form
    images = []

    if directory != None:
        # Open all files in specific folder in directory
        for directory_path in glob.glob(directory):
            for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
                # Open the image
                img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)       
                # Resize image as per limitations of GPU
                img = cv2.resize(img, (width, height))
                # Append images to list
                images.append(img)
       
        # Convert training data list form to a numpy array
        images = np.array(images)

    return images

def main():
    # Loading the Directories containing the Images
    images = format_images_jpg("ISIC_Images")
    masks = format_images_png("ISIC_Labels")

    # Create Training Set, Validation Set, Test Set, choose arbitrary random state value
    # Training (0.7), Validation (0.15), Test (0.15)
    train_images, val_images, train_masks, val_masks = train_test_split(images, masks, test_size=0.3, random_state=42)
    val_images, test_images, val_masks, test_masks = train_test_split(val_images, val_masks, test_size=0.5, random_state=42)

if __name__ == "__main__":
    main()