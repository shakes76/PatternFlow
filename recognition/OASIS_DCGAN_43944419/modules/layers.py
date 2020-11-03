'''
Layers module for Tensorflow
'''
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Activation, BatchNormalization, Conv2D, Conv2DTranspose, Dense, Dropout, Flatten, Input, LeakyReLU, ReLU, Reshape
from tensorflow.keras.initializers import GlorotNormal, GlorotUniform

# Convolutional layer with batch normalization and LeakyReLU activation layers.
def Discriminator_Norm_Dilated_Conv2D(input, 
                                      filters, 
                                      dropout,
                                      kernel_size=(3,3), 
                                      strides=(2,2), 
                                      activation=LeakyReLU(alpha=0.2), 
                                      padding='same',
                                      use_bias=True,
                                      **kwargs):
  
  conv_layer = Conv2D(filters,
                      kernel_size,
                      strides=strides,
                      padding=padding,
                      activation=None,
                      use_bias=use_bias,
                    **kwargs)(input)

  norm_layer = BatchNormalization()(conv_layer)

  dropout_layer = Dropout(dropout)(norm_layer)

  layer = activation(dropout_layer)

  return layer

# Fully connected layer with batch normalization.
def FullyConnected(input, 
                   units, 
                   activation=ReLU(),
                   use_bias=True,
                   kernel_initializer=GlorotNormal(),
                   **kwargs):
  
  dense_layer = Dense(units,
                      activation=None,
                      use_bias=use_bias,
                      kernel_initializer=kernel_initializer,
                      **kwargs)(input)

  norm_layer = BatchNormalization()(dense_layer)

  layer = activation(norm_layer)

  return layer


# Fractional-Strided convolution for Generator Network
# with batch normalization and ReLU activation.
def Generator_Norm_Conv2DTranspose(input, 
                                   filters, 
                                   kernel_size=(3,3), 
                                   strides=(2,2), 
                                   padding='same', 
                                   activation=ReLU(),
                                   use_bias=True,
                                   kernel_initializer=GlorotNormal(), 
                                   **kwargs):
  
  conv_layer = Conv2DTranspose(filters,
                               kernel_size=kernel_size,
                               strides=strides,
                               padding=padding,
                               activation=None, 
                               use_bias=use_bias,
                               kernel_initializer=kernel_initializer)(input)
  
  norm_layer = BatchNormalization()(conv_layer)

  layer = activation(norm_layer)

  return layer




 
