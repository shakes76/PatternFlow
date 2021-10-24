"""
Improved UNet implementation (2D version)

Reference: https://arxiv.org/abs/1802.10508v1
"""
import tensorflow as tf
from tensorflow.keras import layers, models


def encoder_module(input, num_filters, strides=(1, 1)):
    conv = layers.Conv2D(num_filters, (3, 3), strides,
                         padding="same", activation=layers.LeakyReLU(0.01))(input)

    # context module (pre-activation residual blocks)
    # todo: replace with instance normalization
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


def decoder_module(input, encode_output, num_filters, localization_module=True):
    # upsampling module
    up = layers.UpSampling2D((2, 2))(input)
    conv1 = layers.Conv2D(num_filters, (3, 3), padding="same",
                          activation=layers.LeakyReLU(0.01))(up)
    concat = layers.Concatenate()([conv1, encode_output])

    if not localization_module:
        return concat

    # localization module
    conv2 = layers.Conv2D(num_filters, (3, 3), padding="same",
                          activation=layers.LeakyReLU(0.01))(concat)
    conv2 = layers.Conv2D(num_filters, (1, 1), padding="same",
                          activation=layers.LeakyReLU(0.01))(conv2)
    return conv2


def build_model(input_shape):
    inputs = layers.Input(input_shape)

    # downsampling
    down1 = encoder_module(inputs, 16)
    down2 = encoder_module(down1, 32, strides=(2, 2))  # todo: strides=2?
    down3 = encoder_module(down2, 64, strides=(2, 2))
    down4 = encoder_module(down3, 128, strides=(2, 2))
    down5 = encoder_module(down3, 256, strides=(2, 2))

    # upsampling
    up1 = decoder_module(down5, down4, 128)
    up2 = decoder_module(up1, down3, 64)
    up3 = decoder_module(up2, down2, 32)
    up4 = decoder_module(up3, down1, 16, localization_module=False)
    conv = layers.Conv2D(32, (3, 3), padding="same",
                         activation=layers.LeakyReLU(0.01))(up4)

    # segmentation layers
    seg1 = layers.Conv2D(1, (1, 1), padding="same",
                         activation=layers.LeakyReLU(0.01))(up2)
    seg1 = layers.UpSampling2D((2, 2), interpolation="bilinear")(seg1)
    seg2 = layers.Conv2D(1, (1, 1), padding="same",
                         activation=layers.LeakyReLU(0.01))(up3)
    seg2 = layers.Add()([seg2, seg1])
    seg2 = layers.UpSampling2D((2, 2), interpolation="bilinear")(seg2)
    seg3 = layers.Conv2D(1, (1, 1), padding="same",
                         activation=layers.LeakyReLU(0.01))(conv)  # todo: how many classes?
    seg3 = layers.Add()([seg3, seg2])

    outputs = layers.Activation("sigmoid")(seg3)  # todo: softmax?
    model = models.Model(inputs, outputs, name="AdvUNet")
    return model


class AdvUNet:
    def __init__(self, input_shape=(256, 256, 3)):
        self.model = build_model(input_shape)

    
