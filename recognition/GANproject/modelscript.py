#note: the discriminator only works for image size 256 x 256, be sure to change input dimensions if working with a different image
#generator is designed to take a latent space of 256

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *

###defining the generator###
def define_generator():

    model = tf.keras.Sequential()

    model.add(Dense(32*32*256, input_shape=(256,)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((32, 32,256)))

    model.add(Conv2DTranspose(256, (4,4), strides=(2,2), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    
    
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    
    
    model.add(Conv2DTranspose(64, (4,4), strides=(2,2), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    

    model.add(Conv2D(64, (3,3), strides=(1,1), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))


    model.add(Conv2D(1, (3,3),strides=(1,1), padding='same', use_bias=False))
    return model

###defining the discriminator###
def define_discriminator(input=(256,256,1)):

    model = tf.keras.Sequential()
    
    model.add(Conv2D(32, (4,4), strides=(2, 2), padding='same',input_shape=input))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    
    
    model.add(Conv2D(64, (4,4), strides=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))

    model.add(Conv2D(128, (4,4), strides=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    
    model.add(Flatten())
    model.add(Dense(1))
    return model

