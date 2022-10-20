import tensorflow as tf
import numpy as np

def EncoderMiniBlock(inputs, n_filters=32, dropout_prob=0.3, max_pooling=True):
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

def DecoderMiniBlock(prev_layer_input, skip_layer_input, n_filters=32):
    """
    A decoder block to assist the encoderer.
    
    :return: decoded layer output
    """
    up = tf.keras.layers.Conv2DTranspose(n_filters, (3,3), strides=(2,2), padding='same')(prev_layer_input)

    merge = tf.keras.layers.concatenate([up, skip_layer_input])

    conv = tf.keras.layers.Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='HeNormal')(merge)
    conv = tf.keras.layers.Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='HeNormal')(conv)
    
    return conv

def unet(input_size=(128, 128, 3), n_filters=32, n_classes=255):
    """
    Combines encoder and decoder blocks to form the overall UNet architecture.

    :return: final model
    """
    # Input size represent the size of 1 image (the size used for pre-processing) 
    inputs = tf.keras.Input(input_size)
    
    # layer includes multiple convolutional mini blocks with different maxpooling, dropout and filter parameters
    # Observe that the filters are increasing as we go deeper into the network which will increasse the # channels of the image 
    layer_block1 = EncoderMiniBlock(inputs, n_filters, dropout_prob=0, max_pooling=True)
    layer_block2 = EncoderMiniBlock(layer_block1[0], n_filters*2,dropout_prob=0, max_pooling=True)
    layer_block3 = EncoderMiniBlock(layer_block2[0], n_filters*4,dropout_prob=0, max_pooling=True)
    layer_block4 = EncoderMiniBlock(layer_block3[0], n_filters*8,dropout_prob=0.3, max_pooling=True)
    layer_block5 = EncoderMiniBlock(layer_block4[0], n_filters*16, dropout_prob=0.3, max_pooling=False) 
    
    # layer includes multiple mini blocks with decreasing number of filters
    # Observe the skip connections from the layer are given as input to the layer
    # Recall the 2nd output of layer block was skip connection, hence layer_block[1] is used
    layer_block6 = DecoderMiniBlock(layer_block5[0], layer_block4[1],  n_filters * 8)
    layer_block7 = DecoderMiniBlock(layer_block6, layer_block3[1],  n_filters * 4)
    layer_block8 = DecoderMiniBlock(layer_block7, layer_block2[1],  n_filters * 2)
    layer_block9 = DecoderMiniBlock(layer_block8, layer_block1[1],  n_filters)

    # Complete the model with 1 3x3 convolution layer (Same as the prev Conv Layers)
    # Followed by a 1x1 Conv layer to get the image to the desired size. 
    # Observe the number of channels will be equal to number of output classes
    conv10 = tf.keras.layers.Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(layer_block9)
    conv11 = tf.keras.layers.Conv2D(n_classes, 1, padding='same')(conv10)
    
    # Define the model
    model = tf.keras.Model(inputs=inputs, outputs=conv11)

    return model

