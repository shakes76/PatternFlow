from sklearn.model_selection import train_test_split
import os
from glob import glob
import cv2
import numpy as np
import tensorflow as tf
#Contains the data loader for loading and preprocessing data


def load_data(path):
    """
    Loads a ISIC data set
    Param: path - should reference a folder which contains 2 subfolders one with images, and one with masks
    Returns: Raw data in the form: (train_x, train_y), (valid_x, valid_y), (test_x, test_y)
    """

    #Instantiate empty arrays of the processed images
    processed_images = []
    processed_masks = []
    
    #Adds only the jpg files to the list i.e. NOT THE SUPERPIXEL images
    images = sorted(glob(os.path.join(path, "ISIC-2017_Training_Data", "*.jpg")))
    #Adds only the png files to the list: in this case it is all files, but filters for files that shouldn't be there
    masks = sorted(glob(os.path.join(path, "ISIC-2017_Training_Part1_GroundTruth", "*.png")))

    #Process Data
    for image, mask in zip(images, masks):
        processed_image = process_image(image, 128, 128)
        processed_mask = process_image(mask, 128, 128)

        #Add to list of completed processed images
        processed_images.append(processed_image)
        processed_masks.append(processed_mask)
    print("Images processed")
    
    #Conver to numpy arrays for convenience and to save memory
    processed_images = np.array(processed_images)
    processed_masks = np.array(processed_masks)

    #One-hot encode the masks
    processed_masks = tf.keras.utils.to_categorical(processed_masks)
    
    return processed_images, processed_masks

def process_image(image, width, height):
    """
    Processes an image and resizes to a set size
    Param: image - the path of the image which is to be processed
    Param: (tuple) size - the size of the image
    Returns: a processed black and white image that is normalized and resized
    """
    processed_image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    processed_image = cv2.resize(processed_image, (width, height))
    #Normalize image
    processed_image = processed_image / 255.0   # type: ignore
    return processed_image

def train_test_valid(images, masks, split=0.2):
    """
    Splits two given image sets (of equal length) into training, validation and
    testing data for each of the sets
    Param: images - the images set to be split
    Param: masks - the masks set to be split
    Param: split - A split ratio, to indicate how much data shall be used for testing
           0 < split < 1 - default split = 0.2 i.e. 80% of the data will be used for 
           training and 20% for testing and validation
    Returns: Split data in the form: (train_x, train_y), (valid_x, valid_y), (test_x, test_y)
    """
    test_size = int(len(images) * split)

    train_x, valid_x = train_test_split(images, test_size=test_size, random_state=42)
    train_y, valid_y = train_test_split(masks, test_size=test_size, random_state=42)

    train_x, test_x = train_test_split(train_x, test_size=test_size, random_state=42)
    train_y, test_y = train_test_split(train_y, test_size=test_size, random_state=42)

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)
    

if __name__ == "__main__":
    print(load_data("C:/Users/danie/Downloads/ISIC DATA/"))