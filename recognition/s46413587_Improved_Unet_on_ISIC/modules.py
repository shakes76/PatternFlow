import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp
import tensorflow as tf

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Conv2DTranspose, BatchNormalization, Activation, Dropout
from tensorflow.keras.models import Model, load_model

import dataset

img_h = dataset.img_h
img_w = dataset.img_w
b_size = dataset.b_size

#Convolutional 'block' with two layers - four of these pairs go in a U-Net, and they are split into these pairs so 
# snapshots can be taken for upsampling 
def conv_block(tensor, nfilters, size=3, padding='same', initializer="he_normal"):
    block = Conv2D(filters=nfilters, kernel_size=(size, size), padding=padding, kernel_initializer=initializer)(tensor)
    block = BatchNormalization()(block)
    block = Activation("relu")(block)
    block = Conv2D(filters=nfilters, kernel_size=(size, size), padding=padding, kernel_initializer=initializer)(block)
    block = BatchNormalization()(block)
    block = Activation("relu")(block)
    return block

#Upsampling blocks that replace pooling and 'undo' the above convolutions
def deconv_block(tensor, residual, nfilters, size=3, padding='same', strides=(2, 2)):
    block = Conv2DTranspose(nfilters, kernel_size=(size, size), strides=strides, padding=padding)(tensor)
    block = concatenate([block, residual], axis=3)
    block = conv_block(block, nfilters)
    return block

#Dice similarity coefficient calculator - Metric for the model
def dice_similarity(x, y):
    """
    Returns:
        int: dice coefficient
    """
    return 2 * (tf.keras.backend.sum(tf.keras.backend.flatten(x) * tf.keras.backend.flatten(y)) + 1) / \
           (tf.keras.backend.sum(tf.keras.backend.flatten(x) + tf.keras.backend.flatten(y)) + 1)

#The inverse of the above function to be used as a loss function for the model
def dice_loss(x, y):
    """
    Returns:
        int:  dice co-efficient loss
    """
    return 1 - dice_similarity(x, y)

#The overarching U-Net structure
def Unet(h, w, filters):
# down
    input = Input(shape=(h, w, 3), name='image_input')
    conv1_snapshot = conv_block(input, nfilters=filters)
    conv1_out = MaxPooling2D(pool_size=(2, 2))(conv1_snapshot) 
    conv2_snapshot = conv_block(conv1_out, nfilters=filters*2)
    conv2_out = MaxPooling2D(pool_size=(2, 2))(conv2_snapshot)
    conv3_snapshot = conv_block(conv2_out, nfilters=filters*4)
    conv3_out = MaxPooling2D(pool_size=(2, 2))(conv3_snapshot)
    conv4_snapshot = conv_block(conv3_out, nfilters=filters*8)
    conv4_out = MaxPooling2D(pool_size=(2, 2))(conv4_snapshot)
    conv4_out = Dropout(0.5)(conv4_out)
    conv5 = conv_block(conv4_out, nfilters=filters*16)
    conv5 = Dropout(0.5)(conv5)
# up
    deconv6 = deconv_block(conv5, residual=conv4_snapshot, nfilters=filters*8)
    deconv6 = Dropout(0.5)(deconv6)
    deconv7 = deconv_block(deconv6, residual=conv3_snapshot, nfilters=filters*4)
    deconv7 = Dropout(0.5)(deconv7) 
    deconv8 = deconv_block(deconv7, residual=conv2_snapshot, nfilters=filters*2)
    deconv9 = deconv_block(deconv8, residual=conv1_snapshot, nfilters=filters)
    output_layer = Conv2D(filters=2, kernel_size=(1, 1), activation='softmax')(deconv9)

    model = Model(inputs=input, outputs=output_layer, name='Unet')
    return model

#Make the Model
model = Unet(img_w,img_h, 64)
