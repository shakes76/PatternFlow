from matplotlib import image
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
from glob import glob
import os
#Contains the data loader for loading and preprocessing data

path = "C:/Users/danie/Downloads/ISIC DATA/"
img_height = 128
img_width = 128
img_channels = 3

def load_data(path, split = 0.2):
    """
    Loads a ISIC data set
    Param: path - should reference a folder which contains 2 subfolders one with images, and one with masks
    Param: split - the ratio of valid and testing data: default is 0.2 i.e 0.6/0.2/0.2 train/test/valid
    Returns: Raw data in the form: (train_x, train_y), (valid_x, valid_y), (test_x, test_y)
    """

    #Adds only the jpg files to the list i.e. NOT THE SUPERPIXEL images
    images = sorted(glob(os.path.join(path, "ISIC-2017_Training_Data", "*.jpg")))
    #Adds only the png files to the list: in this case it is all files, but filters for files that shouldn't be there
    masks = sorted(glob(os.path.join(path, "ISIC-2017_Training_Part1_GroundTruth", "*.png")))

    test_size = int(len(images) * split)

    train_x, valid_x = train_test_split(images, test_size=test_size, random_state=42)
    train_y, valid_y = train_test_split(masks, test_size=test_size, random_state=42)

    train_x, test_x = train_test_split(train_x, test_size=test_size, random_state=42)
    train_y, test_y = train_test_split(train_y, test_size=test_size, random_state=42)

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)

def decode_image(images):
    """
    Decodes an image - by reading it with the appropriate number of channels alongside standarding and centering the image
    Param: images - the set of images that needs to be decoded
    Returns: the corresponding set of standardized, centered, decoded images
    """
    image = tf.io.read_file(images)
    image = tf.image.decode_jpeg(image, channels=img_channels) #3 img channels i.e. rgb
    image = tf.image.resize(image, (img_width, img_height))
    image -= tf.math.reduce_mean(image)
    image = tf.divide(image, 255.0)
    return image

def decode_mask(masks):
    """
    Decodes an mask - by reading it with the appropriate number of channels
    Param: masks - the set of masks that needs to be decoded
    Returns: the corresponding set decoded masks
    """
    mask = tf.io.read_file(masks)
    mask = tf.image.decode_png(mask, channels=1) #black and white masks
    mask = tf.image.resize(mask, (img_width, img_height))
    return mask
