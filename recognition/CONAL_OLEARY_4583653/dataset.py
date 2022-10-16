import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf


BUFFER_SIZE = 1500
BATCH_SIZE = 32


def normalise(images):
    """
      Takes in a numpy array of images and normalises it and converts it to a TensorFlow Dataset
    """
    return tf.data.Dataset.from_tensor_slices((images / 255).astype("float32")).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


def extractFiles(pathname):
    """
      Takes a directory pathname as a string and returns a numpy array of all the images within the directory
    """
    files = glob.glob(pathname + "/*")
    images = []
    for filename in files:
        img = np.array(Image.open(str(filename)))
        img = np.expand_dims(img, axis=-1)
        images.append(img)
    images = np.array(images)
    return images


def getImages(pathname):
    """
      Takes a directory pathname as a string and returns a normalised TensorFlow Dataset of all the images within the directory
    """
    return normalise(extractFiles(pathname))


def getDatasets():
    """
      Gets the OASIS Datasets
    """
    slice_train = getImages(
        "/content/data/keras_png_slices_data/keras_png_slices_train/")
    slice_test = getImages(
        "/content/data/keras_png_slices_data/keras_png_slices_test/")
    slice_val = getImages(
        "/content/data/keras_png_slices_data/keras_png_slices_val/")
    return [slice_train, slice_test, slice_val]
