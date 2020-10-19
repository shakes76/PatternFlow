import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import glob
from PIL import Image

from generator import Generator
from discriminator import Discriminator

import time


# Define some global variables and useful functions
EPOCHS = 500
BATCH_SIZE = 256

# Get all of the unlabelled images as there is no need for the partitioned
# data for this task
TEST = glob.glob("./keras_png_slices_data/keras_png_slices_test/*.png")
TRAIN = glob.glob("./keras_png_slices_data/keras_png_slices_train/*.png")
VALIDATE = glob.glob("./keras_png_slices_data/keras_png_slices_validate/*.png")
IMAGE_NAMES = TEST + TRAIN + VALIDATE

# Clear any previous models
tf.keras.backend.clear_session()

def load_images(filenames):
    """
    Loads in images for the current batch of filenames. This input is a sublist
    of IMAGE_NAMES of size BATCH_SIZE, and returns a 4D array of images with
    shape (BATCH_SIZE, x_size, y_size, colour_channels)
    """

    # Initialise the results
    total = []

    # Iterate through each filename
    for i in range(len(filenames)):
        # Load the images
        image = Image.open(filenames[i])
        
        # Cast to an array
        image = np.array(image)

        # Add it to the list of images
        total.append(image)

    # Cast the result list to an array itself
    total = np.array(total)

    # Normalise the data
    total = total / 255.0

    # Add an axis to make this array 4D
    total = total[:, :, :, np.newaxis]

    return total


def generate_samples(generator, epoch):
    """
    Creates nine sample images with a given generator and saves them in a file. 
    Named per epoch, as one is generated at the end of each epoch to check on 
    the progress of training.
    """

    # Intialise the figure
    fig = plt.figure(figsize=(15,15))

    # Generate the seeds for each image
    seed = tf.random.normal([9, 100])

    for i in range(1, 10):
        # Choose the i'th image
        plt.subplot(3, 3, i)

        # Generate the image
        image = generator(seed)

        # Display the image
        plt.imshow(image[i-1])
        plt.axis('off')

    plt.savefig("./generated_images/Epoch-{}.png".format(epoch+1))