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
    return tf.data.Dataset.from_tensor_slices(images / 255).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


def extractFiles(pathname):
    """
      Takes a directory pathname as a string and returns a numpy array of all the images within the directory
    """
    files = glob.glob(pathname + "/*")
    images = np.array([np.array(Image.open(str(filename)))
                      for filename in files])
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
    seg_train = getImages(
        "/Users/Patrick/Downloads/keras_png_slices_data/keras_png_slices_seg_train")
    seg_test = getImages(
        "/Users/Patrick/Downloads/keras_png_slices_data/keras_png_slices_seg_test")
    seg_val = getImages(
        "/Users/Patrick/Downloads/keras_png_slices_data/keras_png_slices_seg_validate")
    slice_train = getImages(
        "/Users/Patrick/Downloads/keras_png_slices_data/keras_png_slices_train/")
    slice_test = getImages(
        "/Users/Patrick/Downloads/keras_png_slices_data/keras_png_slices_test/")
    slice_val = getImages(
        "/Users/Patrick/Downloads/keras_png_slices_data/keras_png_slices_val/")
    return [seg_train, seg_test, seg_val, slice_train, slice_test, slice_val]
