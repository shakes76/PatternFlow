'''
    File name: Model_DCGAN.py
    Author: Bin Lyu(45740165)
    Date created: 11/03/2020
    Date last modified: 11/04/2020
    Python Version: 4.7.4
    Move model part into this new file
'''

import tensorflow as tf
from os import listdir
from numpy import asarray, load, zeros, ones, savez_compressed
from numpy.random import randn, randint
from PIL import Image
from matplotlib import pyplot
import glob
from keras.optimizers import Adam
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Conv2D, Conv2DTranspose, Dropout
from keras.layers import Reshape, LeakyReLU, BatchNormalization
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


# create discriminator model, a binary classification CNN model
def define_discriminator(ishape=(80,80,3)):
    model = Sequential()
    # normal
    model.add(Conv2D(128, (5,5), padding='same', input_shape=ishape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization()
    # downsample to 40x40
    model.add(Conv2D(128, (5,5), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # downsample to 20x30
    model.add(Conv2D(128, (5,5), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # downsample to 10x10
    model.add(Conv2D(128, (5,5), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # downsample to 5x5
    model.add(Conv2D(128, (5,5), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # use binary classification activation function 
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    # compile model, adam stochastic gradient descent
    # learning rate=0.0002, momentum=0.5
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    return model

# create generator model, generate 80×80 color image from a point in latent space
def define_generator(latent_size):
    model = Sequential()
    # foundation for 5x5 feature maps
    n_nodes = 128 * 5 * 5
    # use fully connected layer to interpret the point in latent space
    # gan tries to map the input distribution in latent space to generate new output
    model.add(Dense(n_nodes, input_dim=latent_size))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((5, 5, 128)))
    # upsample to 10x10
    # use transpose convolutional layer to increase the area of activations to 4 times
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # upsample to 20x20
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # upsample to 40x40
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # upsample to 80x80
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # output layer 80x80x3, scale to [-1,1]
    model.add(Conv2D(3, (5,5), activation='tanh', padding='same'))
    # no need to compile due to this model is not trained directly
    model.summary()
    return model

    # define the combined generator and discriminator model
# train the generator model weight 
def define_gan(g_model, d_model):
    # make weights in the discriminator not trainable
    # ensure only train and update the model weights in the generator
    d_model.trainable = False
    
    model = Sequential()
    # combine generator and discriminator model together
    model.add(g_model)
    model.add(d_model)
    model.summary()
    
    # compile gan model, adam stochastic gradient descent
    # learning rate=0.0002, momentum=0.5
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model
