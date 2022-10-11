"""
dataset.py

Alex Nicholson (45316207)
11/10/2022

Contains the data loader for loading and preprocessing your data

"""

# from IPython import display

import glob
import imageio
# import matplotlib.pyplot as plt
import numpy as np
import PIL
from PIL import Image
# import tensorflow as tf
# import tensorflow_probability as tfp
# import time
import os


"""
Loads the OASIS dataset of brain MRI images

    Parameters:
        (optional) max_images (int): The maximum number of images of the dataset to be used (default=None)

    Returns:
        train_data (ndarray): Numpy array of grayscale images for training (9,664 images max)
        test_data (ndarray): Numpy array of grayscale images for testing (1,120 images max)
        validate_data (ndarray): Numpy array of grayscale images for validation (544 images max)

"""
def load_dataset(max_images=None):
    print("Loading dataset...")

    # # Download the dataset and unzip
    # if not os.path.exists("/content/keras_png_slices_data/"):
    #     print("Not yet downloaded")
    #     os.system("wget https://cloudstor.aarnet.edu.au/plus/s/tByzSZzvvVh0hZA/download")
    #     os.system("unzip /content/download")
    #     os.system("mkdir /content/out/")

    # File paths
    images_path = "keras_png_slices_data/"
    test_path = images_path + "keras_png_slices_validate/"
    train_path = images_path + "keras_png_slices_train/"
    validate_path = images_path + "keras_png_slices_test/"
    dataset_paths = [test_path, train_path, validate_path]

    # Set up the lists we will load our data into
    test_data = []
    train_data = []
    validate_data = []
    datasets = [test_data, train_data, validate_data]

    # Load all the images into numpy arrays
    for i in range(0, len(dataset_paths)):
        print(dataset_paths[i])

        # Get all PNG files in the dataset_path directory
        images_list = glob.glob(os.path.join(dataset_paths[i], "*.png"))

        images_collected = 0
        for img_filename in images_list:
            # Break if we hit out image limit
            if max_images and images_collected >= max_images:
                break

            # Open the image
            img = PIL.Image.open(images_list[i])
            # Convert image to numpy array
            data = np.asarray(img)
            datasets[i].append(data)
            # Close the image (not strictly necessary)
            del img
            images_collected = images_collected + 1

    # Convert the datasets into numpy arrays
    test_data = np.array(test_data)
    train_data = np.array(train_data)
    validate_data = np.array(validate_data)

    # # Conversion stunt to the existing variable names
    # train_images = train_data
    # test_images = test_data

    # Debug dataset loading
    print('train_data shape:', train_data.shape)
    print('test_data shape:', test_data.shape)
    print('validate_data shape:', validate_data.shape)
    print('')
    # print('train_images shape:', train_images.shape)
    # print('test_images shape:', test_images.shape)

    return (train_data, test_data, validate_data)


if __name__ == "__main__":
    load_dataset(max_images=1000)