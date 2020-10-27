"""
Laterality classification of the OAI AKOA knee data set.

@author Jonathan Godbold, s4533974.

Usage of this file is strictly for The University of Queensland.
Date: 27/10/2020.

Description:
Imports the OASIS dataset and cleans the data for the driver script.

"""

# Import Libraries.
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import *

def generate_paths():
    """
    Create two lists, one for the list of all image files, and one for the unique ID's.
    Format returned: list of all image paths, list of ID's of people in the train list, "" validate list, "" test list.
    """
    unique_list = []
    subset_path_AKOA = []
    train_path_AKOA = os.path.expanduser("H:/Desktop/AKOA_Analysis/")
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
    return generate_sets(subset_path_AKOA, train_list, validate_list, test_list)

def generate_sets(train_list, validate_list, test_list, subset_path_AKOA):
    """
    Prepare the file names for the test, train and validation sets.
    Format returned: three lists of separate sets containing each file name that corresponds to that set.
    """
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
