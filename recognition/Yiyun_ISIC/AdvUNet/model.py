"""
Improved UNet implementation

Reference: https://arxiv.org/abs/1802.10508v1
"""
import tensorflow as tf
from tensorflow.keras import layers, models

def encoder_module(input, num_filters, strides=(1, 1)):
    conv = layers.Conv2D(num_filters, (3, 3), strides,
                         padding="same", activation=layers.LeakyReLU(0.01))(input)

    # context module (pre-activation residual blocks)
    ctx1 = layers.BatchNormalization()(conv)
    ctx1 = layers.Activation(layers.LeakyReLU(0.01))(ctx1)
    ctx1 = layers.Conv2D(num_filters, (3, 3), padding="same")(ctx1)
    ctx_drop = layers.Dropout(0.3)(ctx1)
    ctx2 = layers.BatchNormalization()(ctx_drop)
    ctx2 = layers.Activation(layers.LeakyReLU(0.01))(ctx2)
    ctx2 = layers.Conv2D(num_filters, (3, 3), padding="same")(ctx2)

    # element-wise sum
    sum = layers.Add()([conv, ctx2])
    return sum

def decoder_module():
    pass


class AdvUNet:
    def __init__(self, input_shape=(256, 256, 3)):
        inputs = layers.Input(input_shape)
        down1 = encoder_module(inputs, 16)
        down2 = encoder_module(down1, 32, strides=(2, 2))
        down3 = encoder_module(down2, 64, strides=(2, 2))
        down4 = encoder_module(down3, 128, strides=(2, 2))
