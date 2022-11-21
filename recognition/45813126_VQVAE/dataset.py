"""
File: dataset.py
Author: Georgia Spanevello
Student ID: 45813126
Description: Contains the class to load and process the OASIS dataset.
"""

from matplotlib import image
import glob
import numpy as np


class Dataset:
    def __init__(self):
        """ Gather data. REQUIREMENT - Change path variable to direct the program to downloaded and extracted data """
        path = "C:/Users/Georgia Spanevello/.keras/datasets/oasis/keras_png_slices_data"
        self.train_data_path = path + "/keras_png_slices_train"
        self.train_labels_path = path + "/keras_png_slices_seg_train"
        self.test_data_path = path + "/keras_png_slices_test"
        self.test_labels_path = path + "/keras_png_slices_seg_test"
        self.valid_data_path = path + "/keras_png_slices_validate"
        self.valid_labels_path = path + "/keras_png_slices_seg_validate"

        self.train_data = self.load_process_images(self.train_data_path)
        self.train_labels = self.load_process_labels(self.train_labels_path)
        self.test_data = self.load_process_images(self.test_data_path)
        self.test_labels = self.load_process_labels(self.test_labels_path)
        self.valid_data = self.load_process_images(self.valid_data_path)
        self.valid_labels = self.load_process_labels(self.valid_labels_path)

    def load_process_images(self, path):
        """ Load and process the data at the given path """
        images = []

        # Read then store all images
        for file in glob.glob(path + "/*.png"):
            images.append(image.imread(file))
        data = np.array(images, dtype = np.float32)

        # Residual extraction then normalise
        data = (data - np.mean(data)) / np.std(data)
        data = (data - np.amin(data)) / np.amax(data - np.amin(data))

        return data[:, :, :, np.newaxis]
    
    def load_process_labels(self, path):
        """ Load and process (one-hot encoding) the labels at the given path """
        images = []

        # Read then store all labels
        for file in glob.glob(path + "/*.png"):
            curr_image = image.imread(file)
            one_hot_enc = np.zeros(curr_image.shape)
            for i, value in enumerate(np.unique(curr_image)):
                one_hot_enc[:, :][curr_image == value] = i
            images.append(one_hot_enc)
        labels = np.array(images, dtype = np.uint8)

        # Process labels using one-hot encoding
        one_hot_encs = []
        num_classes = 4
        for i in range(labels.shape[0]):
            curr_image = labels[i]
            one_hot_enc = np.zeros((curr_image.shape[0], curr_image.shape[1], num_classes), dtype = np.uint8)
            for i, value in enumerate(np.unique(curr_image)):
                one_hot_enc[:, :, i][curr_image == value] = 1
                one_hot_encs.append(one_hot_enc)

        return np.array(one_hot_encs)
