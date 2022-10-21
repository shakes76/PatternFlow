"""
Laterality classification of the OAI AKOA knee data set.

@author Jonathan Godbold, s4533974.

Usage of this file is strictly for The University of Queensland.
Date: 27/10/2020.

Description:
Imports the OASIS dataset and cleans the data for the driver script.
Formats the data in the form of 3 tensors of images, 3 tensors of labels.

"""

# Import data passing Libraries.
import numpy as np
import os
from PIL import *

# Import model building Libraries.
import tensorflow as tf

def generate_paths():
    """
    Create two lists, one for the list of all image files, and one for the unique ID's.
    Format returned: list of all image paths, list of ID's of people in the train list, "" validate list, "" test list.
    """
    unique_list = []
    subset_path_AKOA = []
    train_path_AKOA = os.path.expanduser("/Users/jonathan/Desktop/2020 S2/COMP3710/AKOA_Analysis/")
    for path in os.listdir(train_path_AKOA):
        if '.png' in path:
            # Format the string to extract the ID for the patient.
            unique_id = path
            unique_id = unique_id.split("OAI")
            unique_id = unique_id[1]
            unique_id = unique_id.split("_")
            unique_id = unique_id[0]
            unique_list.append(unique_id)
            subset_path_AKOA.append(os.path.join(train_path_AKOA, path))
        
    unique_list = set(unique_list) # Remove any duplicates from the list.
    # There are 101 different patients. Use 70 to train, 20 to validate, and 21 to test.
    size = len(unique_list) 
    train_list = [] # Declare our three sets: train, validate and test which comprises of 101 patients total.
    validate_list = []
    test_list = []
    counter = 0
    for i in unique_list:
        # 70 unique ID's for training.
        if (counter <= 70):
            train_list.append(i)
            counter += 1
        elif (counter > 70 and counter <= 90):
        # 20 unique ID's to validate.
            validate_list.append(i)
            counter += 1
        else:
            # 21 unique ID's to validate.
            test_list.append(i)
            counter += 1
    return subset_path_AKOA, train_list, validate_list, test_list

def generate_sets(train_list, validate_list, test_list, subset_path_AKOA):
    """
    Prepare the file names for the test, train and validation sets.
    Format returned: three lists of separate sets containing each file name that corresponds to that set.
    """
    # Create a list for all the paths of each type of image.
    train_images_src = []
    for i in train_list:
        for j in subset_path_AKOA:
            if (i in j):
                train_images_src.append(j)
            
    validate_images_src = []
    for i in validate_list:
        for j in subset_path_AKOA:
            if (i in j):
                validate_images_src.append(j)
            
    test_images_src = []
    for i in test_list:
        for j in subset_path_AKOA:
            if (i in j):
                test_images_src.append(j)

    return train_images_src, validate_images_src, test_images_src

def loadData(train_images_src, validate_images_src, test_images_src):
    """
    Load images as numpy arrays.
    Format returned: three lists containing training images, validation images, and testing images.
    """
    train_images = [np.array((Image.open(path))) for path in train_images_src]
    print("Training images loaded.")
    validate_images = [np.array((Image.open(path))) for path in validate_images_src]
    print("Validation images loaded.")
    test_images = [np.array((Image.open(path))) for path in test_images_src]
    print("Test images loaded.")
    return train_images, validate_images, test_images

def loadLabels(train_images_src, validate_images_src, test_images_src):
    """
    Loads the corresponding Y labels for the images in each of the three sets.
    Very basic idea, if image name has "Right" or "Left" add 0 or 1 respectively.
    Format returned: three lists which contain the labels for each of the sets.
    """
    # Set up our labels.
    train_images_y = []
    for file in train_images_src:
        if ("RIGHT" in file):
            train_images_y.append(1)
        else:
            train_images_y.append(0)
        
    validate_images_y = []
    for file in validate_images_src:
        if ("RIGHT" in file):
            validate_images_y.append(1)
        else:
            validate_images_y.append(0)
        
    test_images_y = []
    for file in test_images_src:
        if ("RIGHT" in file):
            test_images_y.append(1)
        else:
            test_images_y.append(0) 

    return train_images_y, validate_images_y, test_images_y

def formatData(train_images, validate_images, test_images, train_images_y, validate_images_y, test_images_y):
    """
    Formats the data.
    Normalises the X_test sets.
    Converts the Y-labels from lists to NumPy arrays.
    Converts all data into tensors.
    Format returned: Three tensors of X-lables, and three tensorts of Y-labels.
    """
    # Normalise our data.
    train_images = [x / 255 for x in train_images]
    validate_images = [x / 255 for x in validate_images]
    test_images = [x / 255 for x in test_images]

    # Set the Y-Labels.
    train_images_y = np.array(train_images_y)
    validate_images_y = np.array(validate_images_y)
    test_images_y = np.array(test_images_y)

    # Transfer all data into tensorflow.
    train_images = tf.convert_to_tensor(train_images)
    validate_images = tf.convert_to_tensor(validate_images)
    test_images = tf.convert_to_tensor(test_images)

    train_images_y = tf.convert_to_tensor(train_images_y)
    validate_images_y = tf.convert_to_tensor(validate_images_y)
    test_images_y = tf.convert_to_tensor(test_images_y)

    return train_images, validate_images, test_images, train_images_y, validate_images_y, test_images_y
