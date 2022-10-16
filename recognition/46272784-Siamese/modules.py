# This file contains the source code of the components of my model as functions or classes
import os
import sys
sys.path.insert(1, os.getcwd())
from dataset import loadFile

import random
import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

def getLabel(x, y):
    if x[1] == y[1]:
        # Same label
        return x[0], y[0], 1.0
    # Different Label
    return x[0], y[0], 0.0

def generatePairs(ds):
    # zip the dataset with itself
    pairs = tf.data.Dataset.zip((ds, ds))
    # assigning label
    return pairs.map(getLabel)


def main():
    # Code for testing the functions
    t, v = loadFile('F:/AI/COMP3710/data/AD_NC/')
    d = generatePairs(t)
    print(len(d)*8)

if __name__ == "__main__":
    main()
        
    
