import os
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array


def preprocess_data(training_data, validation_data, testing_data):
    " Normalises each data set and finds the variance of training data "

    training_data = np.array(training_data)
    training_data = training_data.astype('float16') / 255.

    validation_data = np.array(validation_data)
    validation_data = validation_data.astype('float16') / 255.
   
    testing_data = np.array(testing_data)
    testing_data = testing_data.astype('float16') / 255.

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

    # Create list pairs of the directories for the files with the images, and
    # which list they should be sorted into
    location_and_data_category = [["D:/keras_png_slices_data/keras_png_slices_train", \
            training_data], ["D:/keras_png_slices_data/keras_png_slices_vaildate", \
            validation_data], ["D:/keras_png_slices_data/keras_png_slices_test", \
            testing_data]]

    # Find and store each image in each file into the correct list
    for dataset in location_and_data_category:
        for file_name in os.listdir(dataset[0]):
            dataset[1].append(img_to_array(load_img(os.path.join(dataset[0], file_name), color_mode="grayscale")))
    
    return preprocess_data(training_data, validation_data, testing_data)
