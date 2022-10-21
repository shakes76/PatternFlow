from keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, BatchNormalization, Conv2DTranspose, concatenate
from keras.layers import Input, Activation, SeparableConv2D, BatchNormalization, UpSampling2D
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
import tensorflow as tf


def unet():
    conv_args = {
    "activation": "relu",
    "kernel_initializer": "he_normal",
    "padding": "same",
    }

    inputs = Input((256,256,3))

    skip_1, skip_2, skip_3, skip_4, x = contraction(inputs, conv_args)
    x = bottleneck(x, conv_args)
    x = expansion(x, skip_1, skip_2, skip_3, skip_4, conv_args)
    output = output_layer(x)

    model = tf.keras.Model(inputs=inputs, outputs=output)

    return model


def contraction(input, conv_args):
    x = Conv2D(32, 3, **conv_args)(input)
    x = Conv2D(32, 3, **conv_args)(x)
    x = BatchNormalization()(x, training=False)
    skip_1 = x
    x = MaxPool2D(pool_size = (2,2))(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    

    x = Conv2D(64, 3, **conv_args)(x)
    x = Conv2D(64, 3, **conv_args)(x)
    x = BatchNormalization()(x, training=False)
    skip_2 = x
    x = MaxPool2D(pool_size = (2,2))(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = Conv2D(128, 3, **conv_args)(x)
    x = Conv2D(128, 3, **conv_args)(x)
    x = BatchNormalization()(x, training=False)
    skip_3 = x
    x = MaxPool2D(pool_size = (2,2))(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = Conv2D(256, 3, **conv_args)(x)
    x = Conv2D(256, 3, **conv_args)(x)
    x = BatchNormalization()(x, training=False)
    skip_4 = x
    x = MaxPool2D(pool_size = (2,2))(x)
    output = tf.keras.layers.Dropout(0.3)(x)

    return skip_1, skip_2, skip_3, skip_4, output

def bottleneck(input, conv_args):
    x = Conv2D(512, 3, **conv_args)(input)
    output = Conv2D(512, 3, **conv_args)(x)
    return output

def expansion(input, skip_1, skip_2, skip_3, skip_4, conv_args):
    x = Conv2DTranspose(256, (3,3), strides=(2,2), padding='same')(input)
    x = concatenate([x, skip_4], axis=3)
    x = Conv2D(256, 3, **conv_args)(x)
    x = Conv2D(256, 3, **conv_args)(x)

    x = Conv2DTranspose(128, (3,3), strides=(2,2), padding='same')(x)
    x = concatenate([x, skip_3], axis=3)
    x = Conv2D(128, 3, **conv_args)(x)
    x = Conv2D(128, 3, **conv_args)(x)

    x = Conv2DTranspose(64, (3,3), strides=(2,2), padding='same')(x)
    x = concatenate([x, skip_2], axis=3)
    x = Conv2D(64, 3, **conv_args)(x)
    x = Conv2D(64, 3, **conv_args)(x)

    x = Conv2DTranspose(32, (3,3), strides=(2,2), padding='same')(x)
    x = concatenate([x, skip_1], axis=3)
    x = Conv2D(32, 3, **conv_args)(x)
    output = Conv2D(32, 3, **conv_args)(x)
    return output

def output_layer(input):
    output = Conv2D(1, 1, padding="same", activation = "sigmoid")(input)
    return output