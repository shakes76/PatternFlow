# This file contains the source code of the components of my model as functions or classes
import os
import sys

from numpy.random.mtrand import normal
sys.path.insert(1, os.getcwd())
from dataset import loadFile

import random
import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow import data
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import resnet
import matplotlib.pyplot as plt

def generatePairs(ad, nc, batch=8):
    # DataGenerator for weak augmentation
    # datagen = keras.preprocessing.image.ImageDataGenerator(rotation_range=25,
    #                                                        width_shift_range=0.2,
    #                                                        height_shift_range=0.2)
    # datagen.fit(ad)
    # ad = datagen.flow(ad)
    
    # datagen.fit(nc)
    # nc = datagen.flow(nc)
    
    ad = ad.unbatch()
    nc = nc.unbatch()
    # Zipping the data into pairs and give them labels
    diff1 = (data.Dataset.zip((ad, nc))).map(lambda im1, im2: (im1, im2, 1.))
    diff2 = (data.Dataset.zip((nc, ad))).map(lambda im1, im2: (im1, im2, 1.))
    same1 = (data.Dataset.zip((ad, ad))).map(lambda im1, im2: (im1, im2, 0.))
    same2 = (data.Dataset.zip((nc, nc))).map(lambda im1, im2: (im1, im2, 0.))
    # Sample (concatinate) all four image-label pair datasets
    combined_ds = data.experimental.sample_from_datasets([diff1, diff2, same1, same2])
    combined_ds = combined_ds.batch(batch_size=batch)
    return combined_ds
    

def makeCNN():
    # This CNN is almost the same as the one presented in the paper 
    input = layers.Input(shape=(256, 240, 1))
    conv = layers.Conv2D(64, 10, activation='relu', name='c0')(input)
    pool = layers.MaxPooling2D(2)(conv)
    norm = layers.BatchNormalization()(pool)
    
    conv = layers.Conv2D(128, 7, activation='relu', name='c1')(norm)
    pool = layers.MaxPooling2D(2)(conv)
    norm = layers.BatchNormalization()(pool)
    
    conv = layers.Conv2D(128, 4, activation='relu', name='c2')(norm)
    pool = layers.MaxPooling2D(2)(conv)
    norm = layers.BatchNormalization()(pool)
    
    conv = layers.Conv2D(256, 4, activation='relu', name='c3')(norm)
    norm = layers.BatchNormalization()(conv)
    
    flat = layers.Flatten(name='flat')(norm)
    # dense = layers.Dense(4096, activation='sigmoid')(flat)
    # dense = layers.Dense(1024, activation='sigmoid', name='d0', kernel_regularizer=tf.keras.regularizers.l2(1e-3))(flat)
    out = layers.Dense(512, activation='sigmoid', name='out', kernel_regularizer=tf.keras.regularizers.l2(1e-3))(flat)
    
    return Model(inputs=input, outputs=out, name='embeddingCNN')
    

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
    output_layer = layers.Dense(1, activation="sigmoid", name='out2')(normal_layer)
    
    return Model(inputs=[input_1, input_2], outputs=output_layer, name='Siamese')


def loss(margin=1):
    def contrastive_loss(y_true, y_pred):
        return tf.math.reduce_mean(
            (1 - y_true) * tf.math.square(y_pred) + y_true * tf.math.square(tf.math.maximum(margin - (y_pred), 0))
        )
        
    def crossentropy(y_true, y_pred):
        tf.keras.losses.BinaryCrossentropy(y_true, y_pred)
    return contrastive_loss

def main():
    # Code for testing the functions
    t_a, t_n, v_a, v_n = loadFile('F:/AI/COMP3710/data/AD_NC/')
    d = generatePairs(t_a, t_n)
    for p in d:
        print(p)
        break
    print(len(d)*8)
    
    # cnn = makeCNN()
    # print(cnn.summary())
    # siamese = makeSiamese(cnn)
    # print(siamese.summary())

if __name__ == "__main__":
    main()
        
    
