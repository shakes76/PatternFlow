import tensorflow as tf
import numpy as np

# this code is greatly inspired by the work at the following website:
# https://github.com/VidushiBhatia/U-Net-Implementation/blob/main/U_Net_for_Image_Segmentation_From_Scratch_Using_TensorFlow_v4.ipynb

def EncoderBlock(inputs, n_filters=32, dropout_prob=0.3, max_pooling=True):
    """
    Creates an architecture for learning using multiple convolution layers, max pooling, and relu activation. 
    Dropout can be added for regularization to prevent overfitting. 
    
    :return: activation values for the next layer
    """
    conv = tf.keras.layers.Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='HeNormal')(inputs)
    conv = tf.keras.layers.Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='HeNormal')(conv)

    conv = tf.keras.layers.BatchNormalization()(conv, training=False)

    if dropout_prob > 0:     
        conv = tf.keras.layers.Dropout(dropout_prob)(conv)
    if max_pooling:
        next_layer = tf.keras.layers.MaxPooling2D(pool_size = (2,2))(conv)    
    else:
        next_layer = conv

    skip_connection = conv    

    return next_layer, skip_connection

def DecoderBlock(prev_layer_input, skip_layer_input, n_filters=32):
    """
    A decoder block to assist the encoderer.
    
    :return: decoded layer output
    """
    up = tf.keras.layers.Conv2DTranspose(n_filters, (3,3), strides=(2,2), padding='same')(prev_layer_input)

    merge = tf.keras.layers.concatenate([up, skip_layer_input])

    conv = tf.keras.layers.Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='HeNormal')(merge)
    conv = tf.keras.layers.Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='HeNormal')(conv)
    
    return conv

def unet_full(input_size=(128, 128, 3), n_filters=32, n_classes=3):
    """
    Combines encoder and decoder blocks to form the overall UNet architecture.

    :return: final model
    """
    # this input represents the size of one image
    inputs = tf.keras.Input(input_size)
    
    # multiple encoder convolutional blocks with increasing number of filters
    layer_block1 = EncoderBlock(inputs, n_filters, dropout_prob=0, max_pooling=True)
    layer_block2 = EncoderBlock(layer_block1[0], n_filters*2,dropout_prob=0, max_pooling=True)
    layer_block3 = EncoderBlock(layer_block2[0], n_filters*4,dropout_prob=0, max_pooling=True)
    layer_block4 = EncoderBlock(layer_block3[0], n_filters*8,dropout_prob=0.3, max_pooling=True)
    layer_block5 = EncoderBlock(layer_block4[0], n_filters*16, dropout_prob=0.3, max_pooling=False) 
    
    # multiple decoder blocks with decreasing number of filters
    layer_block6 = DecoderBlock(layer_block5[0], layer_block4[1],  n_filters * 8)
    layer_block7 = DecoderBlock(layer_block6, layer_block3[1],  n_filters * 4)
    layer_block8 = DecoderBlock(layer_block7, layer_block2[1],  n_filters * 2)
    layer_block9 = DecoderBlock(layer_block8, layer_block1[1],  n_filters)

    # the model is completed with one 3x3 convolutional layer and a final 1x1 layer to resize the image
    conv10 = tf.keras.layers.Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(layer_block9)
    conv11 = tf.keras.layers.Conv2D(n_classes, 1, padding='same')(conv10)
    
    # finalise the model
    model = tf.keras.Model(inputs=inputs, outputs=conv11)

    return model

