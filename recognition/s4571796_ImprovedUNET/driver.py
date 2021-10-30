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
from tensorflow.keras.utils import to_categorical
import os
import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
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

def map_images(masks):
    """ Takes in a set of images and masks, and performs a mapping
    which converts the images and masks into a 4D Array. Also maps
    each of the values in the array representing the masks into either
    of 2 prominent values representing the feature set. [0, 255]

    Obtain 2 Prominent Feature Values by taking the 4 numbers with the
    largest occurences in the array. Refer to Lines 133 to 143 below
    2 Feature Peak Values: 0, 255
    Midpoint Formula: (x + y) / 2
    Midpoints: 127.5 (128)
    Define Ranges to convert values to
       Range (0, 128) set to 0
       Range (129, 255) set to 255

    Arguments:
    masks: The set of masks represented in a 3D numpy array

    Requires:
    images and masks have to come from the same set, i.e Training, Test

    Returns:
    images: The set of images represented in a 4D numpy array
    masks: The set of masks represented in a 4D numpy array with 2 values
    representing the 2 features
    """
    encoder = LabelEncoder()

    # Get the first mask as an example
    # firstmask = masks[0]
    # print(firstmask.shape)

    # # Change the 2D Array of size 64*64 into a single 1D Array of size 4096
    # firstmask = np.ravel(firstmask)

    # # Print the frequency of occurences for each of the values of range 0 to 255
    # unique, counts = np.unique(firstmask, return_counts=True)
    # frequencies = np.asarray((unique, counts)).T
    # print(frequencies)

    # Store Original Size of Array into 3 Variables: 
    # Number of Images (n), Height (h), Width (w)
    n, h, w = masks.shape

    # Convert Numpy Array consisting of Training Masks to 1D
    masks = masks.reshape(-1,1)

    # Iterate through all pixels in the entire training set and replace values
    # while using Numpy's fast time complexity methods.
    masks[(masks > 0)&(masks < 129)] = 0
    masks[(masks >= 129)&(masks <= 255)] = 255

    # Transform (0, 255) into (0, 1)
    masks = encoder.fit_transform(masks)

    # Validate that the array is supposed to only consist of 4 feature values,
    # Correct Values: 0, 1
    # print(np.unique(masks))

    # Transform the Train Masks back into the original n, h and w
    masks = masks.reshape(n, h, w)

    # Perform Addition of Channel Dimension of Train Masks
    masks = np.expand_dims(masks, axis=3)

    return masks

def get_model(output_classes, height, width, input_classes):
    """ Gets the imported model (Improved UNET) with specific parameters

    Arguments:
    output_classes: Number of features as Output
    height: The height of each image
    width: The width of each image
    input_classes: Number of features as Input

    Returns:
    model: A UNET model with the specified parameters
    """
    return unet_model(num_channels=output_classes, image_height=height, image_width=width, image_channels=input_classes)

def plot_graphs(history, epoch, type=None):
    """ Given the history of the model, plot a graph for viewing.

    Arguments:
    history: The history of the model during training
    epoch: Number of epochs
    type: The graph to be plotted, i.e. Loss or Accuracy

    Returns: 
    plot: A plot of the type of graph specified displayed
    """
    # Define x-axis as Epochs
    epochs = range(1, epoch+1)

    # Plot loss graph
    if type == "loss":
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        plt.plot(epochs, loss, 'g', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')

    # Plot accuracy graph
    elif type == "accuracy":
        accuracy = history.history['accuracy']
        val_accuracy = history.history['val_accuracy']
        epochs = range(1, len(accuracy) + 1)
        plt.plot(epochs, accuracy, 'g', label='Training Accuracy')
        plt.plot(epochs, val_accuracy, 'b', label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Number of Epochs')
        plt.ylabel('Accuracy')

    plt.legend()
    plt.grid()
    plt.show()

def dice_coef(y_true, y_pred):
    """ Returns the Dice Score for binary image segmentation

    Arguments:
    y_true: The true array
    y_pred: The predicted array

    Returns:
    score: The dice score according to the dice formula
    """
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    smooth = 0.0001
    score = (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)
    return score


def dice_coef_multilabel(y_true, y_pred, labels):
    """ Returns the Dice Score for multiclass image segmentation

    Arguments:
    y_true: The true array
    y_pred: The predicted array
    labels: The number of classes for the output

    Returns:
    score: The total Dice Score divided by the number of classes
    """
    # Initialize Dice Score as 0
    dice = 0
    # Iterate through all classes
    for index in range(labels):
        coeff = dice_coef(y_true[:,:,:,index], y_pred[:,:,:,index])
        dice += coeff
        print("The dice score for class " + str(index) + " is " + str(coeff))
    # Return the Dice Score
    score = dice/labels
    return score

def main():
    # Loading the Directories containing the Images
    images = format_images_jpg("ISIC_Images")
    masks = format_images_png("ISIC_Labels")

    # Create Training Set, Validation Set, Test Set, choose arbitrary random state value
    # Training (0.7), Validation (0.15), Test (0.15)
    train_images, val_images, train_masks, val_masks = train_test_split(images, masks, test_size=0.3, random_state=42)
    val_images, test_images, val_masks, test_masks = train_test_split(val_images, val_masks, test_size=0.5, random_state=42)

    # Change the values of Masks to [0,1]
    train_masks = map_images(train_masks)
    val_masks = map_images(val_masks)
    test_masks = map_images(test_masks)

    # Convert the Masks to 2 channels
    train_masks = to_categorical(train_masks, num_classes=classes)
    val_masks = to_categorical(val_masks, num_classes=classes)
    test_masks = to_categorical(test_masks, num_classes=classes)

if __name__ == "__main__":
    main()