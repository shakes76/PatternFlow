#imports
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)
import numpy as np
import os

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, concatenate, Dense, Flatten
from tensorflow.keras.layers import MaxPool2D, AveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.layers import LeakyReLU, Conv2DTranspose, Reshape

# Load the dataset

# Preprocess data
train_images = (train_images - 127.5) / 127.5 # Normalize the images to [-1, 1]


# Batch and shuffle the data

# Hyperparameters


# Build the networks
def generator_network(input_shape, activation):
    '''
    Receive random noise with gaussian distribution.
    Outputs image.
    '''

def discriminator_network(input_shape, activation):
    '''
    Receive generator output image and real images from dataset.
    Outputs binary classficiation: real vs fake.
    '''
# Build the model

# Phase 1: Discriminator
# Optimise the discriminator weights only.
# Phase 2: Generator
# Produce fake images.
# Feed these fake images all labeled as 1(Real).
# This results in the generator creating images to fool the discriminator.
# Never sees real images from the dataset.

# Model Collapse
# Generator will often reach equilibrium faster.
# This means it will just generate a couple of images
# it knows will fool the discriminator.

#DCGAN's deal better with model collapse.
#Deep = More Complex = Slower Equilibrium

# Train

# Test

# Visual Results