"""
dataset.py

Data loader for loading and preprocessing data.

Author: Joshua Wang (Student No. 46965611)
Date Created: 11 Oct 2022
"""
import os
import numpy as np
import sklearn
from keras.utils import img_to_array, load_img
from sklearn.model_selection import train_test_split

class DataLoader():
    def __init__(self, directory="C:/AD_NC"):
        self.directory = directory

    def load_data(self):
        """
        Loads the dataset that will be used and shuffles the data.
        """
        print("Loading data...")

        # Read data from folders in directory
        test_AD = self.read_data(self.directory + "/test/AD")
        test_NC = self.read_data(self.directory + "/test/NC")
        train_AD = self.read_data(self.directory + "/train/AD")
        train_NC = self.read_data(self.directory + "/test/NC")

        # Join datasets together
        X = np.array(test_AD + test_NC + train_AD + train_NC)
        y = np.concatenate((np.ones(len(test_AD)), np.zeros(len(test_NC)),
                np.ones(len(train_AD)), np.zeros(len(train_NC))))

        # Shuffle dataset and get training, testing and validation sets
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                test_size=0.2, shuffle=True)
        X_train, X_val, y_train, y_val = train_test_split(X_train,
                y_train, test_size=0.2)

        print("Finished laoding data.")
        
        return X_train, X_test, X_val, y_train, y_test, y_val
    
    def read_data(self, path):
        """
        Reads images from a given path into arrays.
        """
        set = []
        for id in os.listdir(path):
            set.append(img_to_array(load_img(path + "/" + id)))
        return set