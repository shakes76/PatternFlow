"""
dataset.py 
Contains functions for returning the training, validation and testing splits
from the ADNI MRI dataset. Dataset is already preprocessed, downloaded 
from UQ blackboard.
"""

import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.utils import image_dataset_from_directory

from IPython.display import display

def downloadDataSet():
  """Download the preprocessed version of the ADNI dataset for Alzheimers 
  disease from the blackboard course help site. Returns string path to file"""
  url = "https://cloudstor.aarnet.edu.au/plus/s/L6bbssKhUoUdTSI/download"
  directory = keras.utils.get_file(origin=url, extract=True)
  return os.path.join(directory, "../AD_NC")

def getTraining(datasetPath):
    """Returns normalised training set"""
    directory = os.path.join(datasetPath, "train")
    training = image_dataset_from_directory(directory, labels="inferred", image_size=(240, 256), batch_size=32,
                validation_split=0.3, subset="training", color_mode="grayscale", label_mode=None, seed=1)

    training = training.map(lambda x: x / 255.0)

    # Downsample and add targets
    training = training.map(lambda x: (tf.image.resize(x, (240 // 4, 256 // 4), method="guassian"), x))
    return training

def getValidation(datasetPath):
    """Returns noramlised validation set"""
    directory = os.path.join(datasetPath, "train")
    validation = image_dataset_from_directory(directory, labels="inferred", image_size=(240, 256), batch_size=32,
                validation_split=0.3, subset="validation", color_mode="grayscale", label_mode=None, seed=1)

    validation = validation.map(lambda x: x / 255.0)
    
    # Downsample and add targets
    validation = validation.map(lambda x: (tf.image.resize(x, (240 // 4, 256 // 4), method="gaussian"), x))
    return validation

def getTest(datasetPath):
    """Returns normalized test set"""
    directory = os.path.join(datasetPath, "test")
    training = image_dataset_from_directory(directory, labels="inferred", image_size=(240, 256), batch_size=32,
                color_mode="grayscale", label_mode=None)

    normalisedData = training.map(lambda x: x / 255.0)
    return normalisedData



def preview_data(dataset):
    """Function to construct a matplotlib figure to preview 
    some given images in a dataset
    """
    plt.figure(figsize=(10, 20))
    for batch in dataset.take(1):
        for image in batch:
            display(keras.preprocessing.image.array_to_img(image))