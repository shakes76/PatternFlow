import tensorflow as tf
from keras.models import Model
from keras.layers import Conv2D, MaxPool2D, Dense, Concatenate, UpSampling2D, Input, Activation, add, \
    BatchNormalization, Dropout


def context_module(input_image, filters, kernel_size=(3, 3), padding="same", strides=1):
    block = Conv2D(filters, kernel_size, strides, padding)(input_image)
    block = BatchNormalization()(block)
    block = Activation('relu')(block)
    block = Conv2D(filters, kernel_size, strides, padding)(block)
    block = BatchNormalization()(block)
    block = Activation('relu')(block)
    block = Dropout(rate=0.3)(block)
    return block


def localization_module(input_image, filters):
    block = Conv2D(filters, kernel_size=(3, 3), padding="same", strides=1)(input_image)
    block = Conv2D(filters, kernel_size=(1, 1), padding="same", strides=1)(block)
    return block


def upsampling_module(input_image, filters):
    block = UpSampling2D((2, 2))(input_image)
    block = Conv2D(filters, kernel_size=(3, 3), strides=1, padding="same")(block)
    return block


def improved_Unet(input_image):
    enc1_1 = Conv2D(filters=16, kernel_size=(3, 3), padding="same", strides=1)(input_image)
    enc1_2 = context_module(enc1_1, filters=16)
    enc1 = add([enc1_1, enc1_2])

    enc2_1 = Conv2D(filters=32, kernel_size=(3, 3), padding="sane", strides=2)(enc1)
    enc2_2 = context_module(enc2_1, filters=32)
    enc2 = add([enc2_1, enc2_2])
