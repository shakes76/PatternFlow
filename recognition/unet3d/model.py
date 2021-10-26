import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.layers import Conv3D, Conv3DTranspose, MaxPool3D, BatchNormalization, Dropout, concatenate

def get_model(width=256, height=256, depth=128, start_neurons=8):
    input_layer = keras.Input((width, height, depth, 1))
    num_class = 6

    # downsampling block 1
    conv1 = Conv3D(start_neurons * 1, kernel_size=3, activation="relu", padding="same")(input_layer)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv3D(start_neurons * 2, kernel_size=3, activation="relu", padding="same")(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPool3D(pool_size=2)(conv1)
    pool1 = Dropout(0.25)(pool1)

    # downsampling block 2
    conv2 = Conv3D(start_neurons * 2, kernel_size=3, activation="relu", padding="same")(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv3D(start_neurons * 4, kernel_size=3, activation="relu", padding="same")(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPool3D(pool_size=2)(conv2)
    pool2 = Dropout(0.25)(pool2)

    # downsampling block 3
    conv3 = BatchNormalization()(pool2)
    conv3 = Conv3D(start_neurons * 8, kernel_size=3, activation="relu", padding="same")(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPool3D(pool_size=2)(conv3)
    pool3 = Dropout(0.25)(pool3)
    
    # Middle
    convm = Conv3D(start_neurons * 8, kernel_size=3, activation="relu", padding="same")(pool3)
    convm = BatchNormalization()(convm)
    convm = Conv3D(start_neurons * 16, kernel_size=3, activation="relu", padding="same")(convm)
    convm = BatchNormalization()(convm)

    deconv3 = Conv3DTranspose(start_neurons * 8, kernel_size=3, strides=2, padding="same")(convm)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(0.5)(uconv3)
    uconv3 = Conv3D(start_neurons * 8, kernel_size=3, activation="relu", padding="same")(uconv3)
    uconv3 = Conv3D(start_neurons * 8, kernel_size=3, activation="relu", padding="same")(uconv3)

    deconv2 = Conv3DTranspose(start_neurons * 2, kernel_size=3, strides=2, padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Dropout(0.5)(uconv2)
    uconv2 = Conv3D(start_neurons * 2, kernel_size=3, activation="relu", padding="same")(uconv2)
    uconv2 = Conv3D(start_neurons * 2, kernel_size=3, activation="relu", padding="same")(uconv2)

    deconv1 = Conv3DTranspose(start_neurons * 1, kernel_size=3, strides=2, padding="same")(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Dropout(0.5)(uconv1)
    uconv1 = Conv3D(start_neurons * 1, kernel_size=3, activation="relu", padding="same")(uconv1)
    uconv1 = Conv3D(start_neurons * 1, kernel_size=3, activation="relu", padding="same")(uconv1)

    output_layer = Conv3D(num_class, kernel_size=1, padding="same", activation="softmax")(uconv1)

    model = keras.Model(input_layer, output_layer)

    return model

  