import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers, models, activations


# method for created the improved unet model
def improved_unet(width, height, channels):
    inputs = layers.Input((width, height, channels))

    # encoding path
    en1 = conv_relu_block(inputs, 16)
    en1 = context_module(en1, 16)
    en2 = encoder_block(en1, 32)
    en3 = encoder_block(en2, 64)
    en4 = encoder_block(en3, 128)
    en5 = encoder_block(en4, 256)

    # decoding path
    de1 = decoder_block(en5, en4, 128)
    de2 = decoder_block(de1, en3, 64)
    de3 = decoder_block(de2, en2, 32)
    de4 = upsampling_module(de3, 16)
    de4 = layers.concatenate([en1, de4])
    de4 = layers.Conv2D(32, (3, 3), padding='same')(de4)
    de4 = segmentation_layer(de4)

    # segmentation output
    seg1 = segmentation_layer(de2)
    seg1 = upsampling_module(seg1, 32)
    seg2 = segmentation_layer(de3)
    seg2 = layers.Add()([seg1, seg2])
    seg2 = upsampling_module(seg1, 32)
    seg3 = de4
    seg3 = layers.Add()([seg2, seg3])

    outputs = layers.Conv2D(1, (1, 1), activation="sigmoid")(seg3)

    model = models.Model(inputs, outputs)
    return model


# methods for different modules of the unet


def conv_relu_block(input_layer, conv_depth, conv_size=(3, 3), strides=1):
    # convolutions with leaky relu, and normalisation
    conv = layers.Conv2D(conv_depth, conv_size, padding='same', strides=strides)(input_layer)
    conv = tfa.layers.InstanceNormalization()(conv)
    conv = layers.LeakyReLU(alpha=0.01)(conv)
    return conv


def context_module(input_layer, conv_depth):
    # context module contains 2 convolutions with a dropout of 0.3 in between
    # with added element-wise sum
    cont = conv_relu_block(input_layer, conv_depth)
    cont = layers.Dropout(0.3)(cont)
    cont = conv_relu_block(cont, conv_depth)
    cont = layers.Add()([input_layer, cont])
    return cont


def localization_module(input_layer, conv_depth):
    # a localization module contains one 3x3 convolution
    # and a 1x1 convolution that halves the number of features
    loc = conv_relu_block(input_layer, conv_depth)
    loc = conv_relu_block(loc, conv_depth, (1, 1))
    return loc


def upsampling_module(input_layer, conv_depth):
    up = layers.UpSampling2D((2, 2))(input_layer)
    up = conv_relu_block(up, conv_depth)
    return up


def segmentation_layer(input_layer):
    seg = layers.Activation('sigmoid')(input_layer)
    return seg


# method for blocks in the encoder path
def encoder_block(input_layer, conv_depth):
    en = conv_relu_block(input_layer, conv_depth, strides=2)
    en = context_module(en, conv_depth)
    return en


def decoder_block(input_layer, concat, conv_depth):
    de = upsampling_module(input_layer, conv_depth)
    de = layers.concatenate([concat, de])
    de = localization_module(de, conv_depth)
    return de
