
import tensorflow as tf
import numpy as np
import os
import cv2
import random

"""
Note: All the code used in this file has been inspired by
https://www.youtube.com/watch?v=j-3vuBynnOE
"""

"""
Class for loading data. This class will load all image data
and store it an a npy array for later use.
"""
class LoadData:

    """
    constructor of the class that calls the train and validate methods.
    These methods will load all image data as train and test sets
    """
    def __init__(self):
        self.load_training_data() # Calls method to load training data
        self.load_testing_data() # Calls method to test training data

    """
    Method for loading training data set. It loads all images from a predefined
    directory and loads and stores all image data as a npy array.
    """
    def load_training_data(self):
        DATADIR = "/home/Student/s4644467/PatternFlow/recognition/AD_NC/train"
        CATEGORIES = ["AD", "NC"] # Looping through all the categories
        count = 0 # keeps count of how many images are loaded
        training_data = [] # list containing all the training data
        for category in CATEGORIES:
            path = os.path.join(DATADIR, category)
            class_num = CATEGORIES.index(category)
            for img in os.listdir(path):
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                training_data.append([img_arr, class_num])
                count += 1
                if (count % 100 == 0): # Keeps track of how many images are loaded
                    print(f"Loaded {count} train images", flush=True)
        print("Loaded " + str(count) + " train images", flush=True)
        random.shuffle(training_data) # Shuffles image data
        
        X_train = []
        y_train = []
        
        # Appends image to their labels
        for features, label in training_data:
            X_train.append(features)
            y_train.append(label)
        
        # Reshapes to given image size
        X_train = np.array(X_train).reshape(-1, 240, 256, 1)
        print(X_train.shape)

        # Saves image data as numpy arrays to access later
        np.save("X_train.npy", X_train)
        np.save("y_train.npy", y_train)

    
    """
    Method for loading test data set. It loads all images from a predefined
    directory and loads and stores all image data as a npy array.
    """
    def load_testing_data(self):
        DATADIR = "/home/Student/s4644467/PatternFlow/recognition/AD_NC/test"
        CATEGORIES = ["AD", "NC"] # Looping through all the categories
        count = 0 # keeps count of how many images are loaded
        training_data = [] # list containing all the training data
        for category in CATEGORIES:
            path = os.path.join(DATADIR, category)
            class_num = CATEGORIES.index(category)
            for img in os.listdir(path):
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                training_data.append([img_arr, class_num])
                count += 1
                if (count % 100 == 0): # Keeps track of how many images are loaded
                    print(f"Loaded {count} test images", flush=True)
                
        print("Loaded " + str(count) + " test images", flush=True)
        random.shuffle(training_data) # Shuffles image data
        
        X_train = []
        y_train = []

        # Appends image to their labels
        for features, label in training_data:
            X_train.append(features)
            y_train.append(label)

        # Reshapes to given image size
        X_train = np.array(X_train).reshape(-1, 240, 256, 1)
        print(X_train.shape)

        # Saves image data as numpy arrays to access later
        np.save("X_test.npy", X_train)
        np.save("y_test.npy", y_train)