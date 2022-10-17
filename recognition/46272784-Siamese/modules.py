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
    EPS = 1e-8
    
    input_1 = layers.Input((256, 240, 1))
    input_2 = layers.Input((256, 240, 1))
    
    tower_1 = cnn(input_1)
    tower_2 = cnn(input_2)
    # Merging the two networks outputs (EPS is used to avoied 0 distance)
    distance = lambda v: tf.math.sqrt(tf.math.maximum(tf.math.reduce_sum(tf.math.square(v[0] - v[1]), axis=1, keepdims=True), EPS))
    merge_layer = layers.Lambda(distance)([tower_1, tower_2])
    normal_layer = layers.BatchNormalization()(merge_layer)
    # 1 if same class, 0 if not
    output_layer = layers.Dense(1, activation="sigmoid")(normal_layer)
    
    return Model(inputs=[input_1, input_2], outputs=output_layer, name='Siamese')


def loss(margin=1):
    def contrastive_loss(y_true, y_pred):
        return tf.math.reduce_mean(
            (1 - y_true) * tf.math.square(y_pred) + y_true * tf.math.square(tf.math.maximum(margin - (y_pred), 0))
        )
    return contrastive_loss

def main():
    # Code for testing the functions
    t, v = loadFile('F:/AI/COMP3710/data/AD_NC/')
    d = generatePairs(t)
    print(len(d)*8)
    
    cnn = makeCNN()
    print(cnn.summary())
    siamese = makeSiamese(cnn)
    print(siamese.summary())

if __name__ == "__main__":
    main()
        
    
