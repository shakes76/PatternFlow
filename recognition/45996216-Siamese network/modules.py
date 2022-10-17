"""
COMP3170
Jialiang Hou
45996216
Create a Siamese network modules
"""
import dataset
import random
import tensorflow as tf
import os
import tensorflow.keras.backend as K
import tensorflow
import numpy as np


def SiameseNetwork():
    # define the structure of the layers
    seq_conv_model = [
        # tf.keras.layers.Reshape( input_shape=(240,256,1) , target_shape=(120,128,1)),
        # convolutional layer
        tf.keras.layers.Conv2D(32, kernel_size=(8, 8), strides=1, activation='relu'),
        # pool layer
        tf.keras.layers.MaxPooling2D(pool_size=(6, 6), strides=1),

        tf.keras.layers.Conv2D(64, kernel_size=(6, 6), strides=1, activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(4, 4), strides=1),

        tf.keras.layers.Conv2D(128, kernel_size=(4, 4), strides=1, activation='relu'),

        # flat the image
        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(256, activation=tf.keras.activations.sigmoid)
    ]

    # define an instance of model
    seq_model = tf.keras.Sequential(seq_conv_model)

    # there are two input in Siamese network
    input_x1 = tf.keras.layers.Input(shape=(60, 64, 1))
    input_x2 = tf.keras.layers.Input(shape=(60, 64, 1))

    # two inputs all should go through the model
    output_x1 = seq_model(input_x1)
    output_x2 = seq_model(input_x2)

    # calculate the distance between the two outputs
    # the l1 distance should be minimized if they are in same label, should be maximized if they are different
    distance_euclid = tf.keras.layers.Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))([output_x1, output_x2])

    # input the distance to a dense layer
    outputs = tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid)(distance_euclid)
    # finish the model
    model = tf.keras.models.Model([input_x1, input_x2], outputs)

    model.compile(loss=tf.keras.losses.binary_crossentropy, optimizer=tf.keras.optimizers.Adam(lr=0.0001),
                  metrics=['accuracy'])

    # print the structure of model
    model.summary()
    return model
