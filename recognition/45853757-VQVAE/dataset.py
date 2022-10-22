import numpy as np
#from PIL import Image
#import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
import os
import matplotlib.pyplot as plt

def preprocess_data(training_data, validation_data, testing_data):
    """
    Normalise each data set and find the variance of training data

    Params: 

    Returns: 
    """
    training_data = np.array(training_data)
    training_data = training_data.astype('float16') / 255.
    training_data = training_data.reshape((len(training_data), np.prod(training_data.shape[1:])))

    validation_data = np.array(validation_data)
    validation_data = validation_data.astype('float16') / 255.
    validation_data = validation_data.reshape((len(validation_data), np.prod(validation_data.shape[1:])))

    testing_data = np.array(testing_data)
    testing_data = testing_data.astype('float16') / 255.
    testing_data = testing_data.reshape((len(testing_data), np.prod(testing_data.shape[1:])))

    variance = np.var(training_data / 255.)

    return training_data, validation_data, testing_data, variance


def load_data():
    """
    Loads the data to be used for training, validating and testing the model.

    Params: None

    Returns: Three normalised data sets of images for training, validation and testing and the variance of the training dataset.
    """
    # Initialise three empty lists for our data to be stored appropriately
    training_data = []
    validation_data = []
    testing_data = []

    # Create tuple pairs of the directories for the files with the images, and
    # which list they should be sorted into
    location_and_data_category = [["D:/keras_png_slices_data/keras_png_slices_train", \
            training_data], ["D:/keras_png_slices_data/keras_png_slices_vaildate", \
            validation_data], ["D:/keras_png_slices_data/keras_png_slices_test", \
            testing_data]]

    # Find and store the individual directories for each image in each file
    for dataset in location_and_data_category:
        for file_name in os.listdir(dataset[0]):
            dataset[1].append(img_to_array(load_img(os.path.join(dataset[0], file_name), color_mode="grayscale")))
        #dataset[1] = np.array(dataset[1]).squeeze()
    
    return preprocess_data(training_data, validation_data, testing_data)
