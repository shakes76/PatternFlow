"""
ISICs dataset by an improved Unet to make image segment.
ISICs data set concludes thousands of Skin Lesion images. 
This recognition algorithm aims to automatically do Lesion Segmentation through an improved unet model

@author Xiaoqi Zhuang
@email x.zhuang@uqconnect.edu.au
"""
import tensorflow as tf
from tensorflow import keras
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, Flatten, MaxPooling2D, BatchNormalization, Conv2DTranspose
from keras.optimizers import SGD, Adam
tf.random.Generator = None
import tensorflow_addons as tfa

#The activation function is "leaky ReLe" which the alpha is 1e-2.
leakyRELU =tf.keras.layers.LeakyReLU(alpha=1e-2)

def context_modules(previous_layer, numFilters):
    """
    The context module:
    A pre-activation residual block with two 3x3 convolutional layers and a dropout layer (pdrop = 0.3) in between.
    @param previous_layer, the layer before the context module
    @numFilters, the number of output filters for every convolutional layer
    @return, the layer after the context module
    """
    
    l1 = tfa.layers.InstanceNormalization()(previous_layer)
    l2 = tf.keras.layers.Activation("relu")(l1)
    l3 = tf.keras.layers.Conv2D(numFilters, (3, 3), activation = leakyRELU, padding="same")(l2)
    l4 = tfa.layers.InstanceNormalization()(l3)
    l5 = tf.keras.layers.Activation("relu")(l4)
    l6 = tf.keras.layers.Conv2D(numFilters, (3, 3), activation = leakyRELU, padding="same")(l5)
    l6 = tf.keras.layers.Dropout(0.3)(l6)
    
    return l6

def upSampling(previous_layer, numFilters):
    """
    Upsampling the low resolution feature maps:
    A simple upscale that repeats the feature voxels twice in each spatial dimension, 
    followed by a 3x3 convolution that halves the number of feature maps.
    @param previous_layer, the layer before the Upsampling
    @numFilters, the number of output filters for the convolutional layer
    @return, the layer after the upSampling module
    """
    
    l1 = tf.keras.layers.UpSampling2D()(previous_layer)
    l2 = tf.keras.layers.Conv2D(numFilters,(3,3), activation = leakyRELU, padding="same")(l1)
    
    return l2

def localization(previous_layer, numFilters):
    """
    A localization module consists of a 3x3x3 convolution followed by a 1x1 convolution 
    that halves the number of feature maps.
    @param previous_layer, the layer before the localization module
    @numFilters, the number of output filters for every convolutional layer
    @return, the layer after the localization module
    """
    
    l1 = tf.keras.layers.Conv2D(numFilters, (3, 3), activation = leakyRELU, padding="same")(previous_layer)
    l2 = tf.keras.layers.Conv2D(numFilters, (1, 1), activation = leakyRELU, padding="same")(l1)
    
    return l2

def improvedUnet():
    inputs = tf.keras.layers.Input((256, 256, 3))

    #Encode path
    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation = leakyRELU, padding="same")(inputs)
    c2 = context_modules(c1, 16)
    c3 = tf.keras.layers.Add()([c1, c2])

    c4 = tf.keras.layers.Conv2D(32, (3, 3),activation = leakyRELU, strides = 2, padding="same")(c3)
    c5 = context_modules(c4, 32)
    c6 = tf.keras.layers.Add()([c4, c5])

    c7 = tf.keras.layers.Conv2D(64, (3, 3),activation = leakyRELU, strides = 2, padding="same")(c6)
    c8 = context_modules(c7, 64)
    c9 = tf.keras.layers.Add()([c7, c8])

    c10 = tf.keras.layers.Conv2D(128, (3, 3),activation = leakyRELU, strides = 2, padding="same")(c9)
    c11 = context_modules(c10, 128)
    c12 = tf.keras.layers.Add()([c10, c11])


    c13 = tf.keras.layers.Conv2D(256, (3, 3),activation = leakyRELU, strides = 2, padding="same")(c12)
    c14 = context_modules(c13, 256)
    c15 = tf.keras.layers.Add()([c13, c14])

    #Decode path
    c16 = upSampling(c15, 128)
    c17 = tf.keras.layers.concatenate([c16, c12])

    c18 = localization(c17, 128)
    c19 = upSampling(c18, 64)
    c20 = tf.keras.layers.concatenate([c19, c9])

    c21 = localization(c20, 64)
    s1 = tf.keras.layers.Conv2D(1, (1, 1), activation = leakyRELU, padding="same")(c21)
    s1 = tf.keras.layers.UpSampling2D(interpolation = "bilinear")(s1)

    c23 = upSampling(c21, 32)
    c24 = tf.keras.layers.concatenate([c23, c6])
    c25 = localization(c24, 32)
    s2 = s1 = tf.keras.layers.Conv2D(1, (1, 1), activation = leakyRELU, padding="same")(c25)
    s3 = tf.keras.layers.Add()([s1, s2])
    s3 = tf.keras.layers.UpSampling2D(interpolation = "bilinear")(s2)

    c27 = upSampling(c25, 16)
    c28 = tf.keras.layers.concatenate([c27, c3])

    c29 = tf.keras.layers.Conv2D(32, (3, 3), activation = leakyRELU, padding='same')(c28)
    s4 = tf.keras.layers.Conv2D(1, (1, 1), activation = leakyRELU, padding="same")(c29) 
    s5 = tf.keras.layers.Add()([s3, s4])

    outputs = tf.keras.activations.sigmoid(s5)
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])
    
    return model