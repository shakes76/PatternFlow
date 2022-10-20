import tensorflow as tf
from keras.models import Model
from keras.layers import Conv2D, MaxPool2D, Dense, Concatenate, UpSampling2D, Input, Activation, add, BatchNormalization, Dropout

def context_module(x, filters, kernel_size=(3, 3), padding="same",strides=1):
    block = Conv2D(filters, kernel_size, strides, padding)(x)
    block = BatchNormalization()(block)
    block = Activation('relu')(block)
    block = Conv2D(filters, kernel_size, strides, padding)(block)
    block = BatchNormalization()(block)
    block = Activation('relu')(block)
    block = Dropout(rate=0.3)(block)
    return block
