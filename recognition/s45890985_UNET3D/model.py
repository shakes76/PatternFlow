import tensorflow as tf
from tensorflow.keras import datasets, layers, models, callbacks

def relu_conv_block(input_layer, conv_size):
    conv = layers.Conv3D(conv_size, (3,3,3), padding= 'same', activation = 'relu') (input_layer)
    conv = layers.BatchNormalization()(conv)
    conv = layers.Relu()(conv)

    return conv

def encoder_block(input_layer, conv_size, drop_out):
    en = relu_conv_block(input_layer,conv_size)
    en = layers.Dropout(drop_out)(en)
    en = relu_conv_block(en,conv_size)
    en = layers.MaxPooling3D(pool_size=(2,2,2))(en)

    return en

def decoder_block(input_layer, concat_layer, conv_size, drop_out):
    de = layers.Conv3DTranspose(conv_size, (2,2),strides=(2,2,2), padding= 'same')(input_layer)
    de = layers.concatenate([de, concat_layer])
    de = relu_conv_block(de, conv_size)
    de = layers.Dropout(drop_out)(de)
    de = relu_conv_block(de, conv_size)

    return de




