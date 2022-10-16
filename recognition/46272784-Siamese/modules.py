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
from tensorflow.keras import layers, Model
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

def makeCNN():
    inputs = layers.Input(shape=(256, 240, 1))
    
    net = layers.Conv2D(64, kernel_size=(4, 4), strides=(2,2), padding='same')(inputs)
    net = layers.BatchNormalization()(net)
    net = layers.LeakyReLU(0.2)(net)
    
    net = layers.Conv2D(128, kernel_size=(4, 4), strides=(2,2), padding='same')(net)
    net = layers.BatchNormalization()(net)
    net = layers.LeakyReLU(0.2)(net)
    
    net = layers.Conv2D(128, kernel_size=(4, 4), strides=(2,2), padding='same')(net)
    net = layers.BatchNormalization()(net)
    net = layers.LeakyReLU(0.2)(net)
    
    dense = layers.Flatten()(net)
    dense = layers.BatchNormalization()(dense)
    out = layers.Dense(32, activation="tanh")(dense)
    
    
    return Model(inputs=inputs, outputs=out, name='CNN')

def makeSiamese(cnn):
    input_1 = layers.Input((256, 240, 1))
    input_2 = layers.Input((256, 240, 1))
    
    tower_1 = cnn(input_1)
    tower_2 = cnn(input_2)
    # TODO Finish Siamese


def main():
    # Code for testing the functions
    t, v = loadFile('F:/AI/COMP3710/data/AD_NC/')
    d = generatePairs(t)
    print(len(d)*8)

if __name__ == "__main__":
    main()
        
    
