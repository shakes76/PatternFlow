import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models, backend
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import IPython.display as display
import pathlib
import glob

#Load the data from the given file path
def FetchData(path):
    images = sorted(glob.glob(path))
    return images
#Convert the array to the format I want
def squeeze(array):
    Array = np.asarray(array)
    Array = np.squeeze(Array)
    return Array

def ImageExtract(images):
    """
    Pre-process the images for the VQVAE Model to train on
    Return the np array of the correct dimensions needed (number of images, image height, image width)
"""
    Array = []
    for x in images:
        #Extract images of correct type and sizes
        sample_image = tf.io.read_file(str(x))
        sample_image = tf.image.decode_png(sample_image)
        sample_image = tf.image.convert_image_dtype(sample_image, tf.float32)
        sample_image = tf.image.resize(sample_image, [256, 256])
        y = tf.shape(sample_image)[1]
        y = y // 2
        #add needed axis
        image = sample_image[:, y:, :]
        image = tf.cast(sample_image, tf.float32)
        image = np.squeeze(image)
        Array.append([image])
        #Convert to appropriate np array
        Array2 = squeeze(Array)
    return Array

def combine(train, test):
    """
    Combine the given train and test set together into a single larger test set
    Scale image pixels
    Return combined constructed array
    """
    Oasis = np.concatenate([train, test], axis=0)
    Oasis = np.expand_dims(Oasis, -1).astype("float32") / 255
    return Oasis

def main():
    train_images = FetchData('D:\\University\\2022 Sem 2\\COMP3710\\Project Report\\BranchFOlder\\keras_png_slices_data\\keras_png_slices_data\\keras_png_slices_train\\*')
    test_images = FetchData('D:\\University\\2022 Sem 2\\COMP3710\\Project Report\\BranchFOlder\\keras_png_slices_data\\keras_png_slices_data\\keras_png_slices_test\\*')
    validate_images = FetchData('D:\\University\\2022 Sem 2\\COMP3710\\Project Report\\BranchFOlder\\keras_png_slices_data\\keras_png_slices_data\\keras_png_slices_validate\\*')

    train = ImageExtract(train_images)
    test = ImageExtract(test_images)
    validate = ImageExtract(validate_images)

    Oasis = combine(train, test)

if __name__ =="__main__":
    main();
