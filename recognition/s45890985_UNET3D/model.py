import tensorflow as tf
from tensorflow.keras import datasets, layers, models, callbacks

def relu_conv_block(input_layer, conv_size):
    conv = layers.Conv3D(conv_size, (3,3,3), padding= 'same', activation = 'relu') (input_layer)
    conv = layers.BatchNormalization()(conv)
    conv = layers.Relu()(conv)

    return conv

def encoder_block(input_layer, conv_size):
    en = relu_conv_block(input_layer,conv_size)
    en = layers.Dropout(0.2)(en)
    en = relu_conv_block(en,conv_size)
    en = layers.MaxPooling3D(pool_size=(2,2,2))(en)

    return en


