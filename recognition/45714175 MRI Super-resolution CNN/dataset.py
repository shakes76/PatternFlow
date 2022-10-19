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

def downloadDataSet():
  """Download the preprocessed version of the ADNI dataset for Alzheimers 
  disease from the blackboard course help site. Returns string path to file"""
  url = "https://cloudstor.aarnet.edu.au/plus/s/L6bbssKhUoUdTSI/download"
  directory = keras.utils.get_file(origin=url, extract=True)
  return os.path.join(directory, "../AD_NC")

def getTraining(datasetPath):
    """Returns normalised training set"""
    directory = os.path.join(datasetPath, "train")
    training = image_dataset_from_directory(directory, labels="inferred", image_size=(128, 128), batch_size=32,
                validation_split=0.3, subset="training", color_mode="grayscale")

    normalisedData = training.map(lambda x: x / 255.0)
    return normalisedData

def getValidation(datasetPath):
    """Returns noramlised validation set"""
    directory = os.path.join(datasetPath, "train")
    training = image_dataset_from_directory(directory, labels="inferred", image_size=(128, 128), batch_size=32,
                validation_split=0.3, subset="validation", color_mode="grayscale")

    normalisedData = training.map(lambda x: x / 255.0)
    return normalisedData

def getTest(datasetPath):
    """Returns normalized test set"""
    directory = os.path.join(datasetPath, "test")
    training = image_dataset_from_directory(directory, labels="inferred", image_size=(128, 128), batch_size=32,
                color_mode="grayscale")

    normalisedData = training.map(lambda x: x / 255.0)
    return normalisedData