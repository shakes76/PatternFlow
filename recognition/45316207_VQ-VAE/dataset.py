"""
dataset.py

Alex Nicholson (45316207)
11/10/2022

Contains the data loader for loading and preprocessing your data

"""


import glob
import numpy as np
import PIL
import os

import matplotlib.pyplot as plt


def load_dataset(max_images=None, verbose=False):
    """
    Loads the OASIS dataset of brain MRI images

        Parameters:
            (optional) max_images (int): The maximum number of images of the dataset to be used (default=None)
            (optional) verbose (bool): Whether a description of the dataset should be printed after it has loaded

        Returns:
            train_data_scaled (ndarray): Numpy array of scaled image data for training (9,664 images max)
            test_dat_scaleda (ndarray): Numpy array of scaled image data testing (1,120 images max)
            validate_data_scaled (ndarray): Numpy array of scaled image data validation (544 images max)
            data_variance (int): Variance of the test dataset
    """

    print("Loading dataset...")

    # File paths
    images_path = "keras_png_slices_data/"
    test_path = images_path + "keras_png_slices_test/"
    train_path = images_path + "keras_png_slices_train/"
    validate_path = images_path + "keras_png_slices_validate/"
    dataset_paths = [test_path, train_path, validate_path]

    # Set up the lists we will load our data into
    test_data = []
    train_data = []
    validate_data = []
    datasets = [test_data, train_data, validate_data]

    # Load all the images into numpy arrays
    for i in range(0, len(dataset_paths)):
        # Get all the png files in this dataset_path directory
        images_list = glob.glob(os.path.join(dataset_paths[i], "*.png"))

        images_collected = 0 
        for img_filename in images_list:
            # Stop loading in images if we hit out max image limit
            if max_images and images_collected >= max_images:
                break

            # Open the image
            img = PIL.Image.open(img_filename)
            # Convert image to numpy array
            data = np.asarray(img)
            datasets[i].append(data)

            # Close the image (not strictly necessary)
            del img
            images_collected = images_collected + 1

    # Convert the datasets into numpy arrays
    train_data = np.array(train_data)
    test_data = np.array(test_data)
    validate_data = np.array(validate_data)

    # Preprocess the data
    train_data = np.expand_dims(train_data, -1)
    test_data = np.expand_dims(test_data, -1)
    validate_data = np.expand_dims(validate_data, -1)
    # Scale the data into values between -0.5 and 0.5 (range of 1 centred about 0)
    train_data_scaled = (train_data / 255.0) - 0.5
    test_data_scaled = (test_data / 255.0) - 0.5
    validate_data_scaled = (validate_data / 255.0) - 0.5

    # Get the dataset variance
    data_variance = np.var(train_data / 255.0)

    if verbose == True:
        # Debug dataset loading    
        print(f"###train_data ({type(train_data)}): {np.shape(train_data)}###")
        print(f"###test_data ({type(test_data)}): {np.shape(test_data)}###")
        print(f"###train_data_scaled ({type(train_data_scaled)}): {np.shape(train_data_scaled)}###")
        print(f"###test_data_scaled ({type(test_data_scaled)}): {np.shape(test_data_scaled)}###")
        print(f"###data_variance ({type(data_variance)}): {data_variance}###")
        print('')

        print(f"###validate_data ({type(validate_data)}): {np.shape(validate_data)}###")
        print(f"###validate_data_scaled ({type(validate_data_scaled)}): {np.shape(validate_data_scaled)}###")

        print('')
        print('')

    return (train_data_scaled, validate_data_scaled, test_data_scaled, data_variance)


if __name__ == "__main__":
    # Run a test
    load_dataset(max_images=1000)