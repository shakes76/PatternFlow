'''
Layers module for Tensorflow

@author Peter Ngo

7/11/2020
'''
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Activation, BatchNormalization, Conv2D, Conv2DTranspose, Dense, Dropout, Flatten, Input, LeakyReLU, ReLU, Reshape
from tensorflow.keras.initializers import GlorotNormal, GlorotUniform


def Discriminator_Norm_Dropout_Conv2D(input, 
                                      filters, 
                                      dropout,
                                      kernel_size=(3,3), 
                                      strides=(2,2), 
                                      activation=LeakyReLU(alpha=0.2), 
                                      padding='same',
                                      use_bias=False,
                                      kernel_initializer=GlorotNormal(),
                                      **kwargs):
  """
  Create a convolutional layer with batch normalization, dropout and LeakyReLU.

  :param input:
      Input layer
  :param filters:
      Total number of filters
  :param dropout:
      Percentage of input units to drop
  :param kernel_size:
      Size of the kernel
  :param strides:
      Stride height and width of convolution
  """
  #Convolution layer with weights initialized from Glorot normal initializer.
  conv_layer = Conv2D(filters,
                      kernel_size,
                      strides=strides,
                      padding=padding,
                      activation=None,
                      use_bias=use_bias,
                      kernel_initializer=kernel_initializer,
                    **kwargs)(input)
  #Transform the outputs from the conv_layer to have mean≈0 and std≈1.
  norm_layer = BatchNormalization()(conv_layer)
  #Drop a percentage of the outputs.
  dropout_layer = Dropout(dropout)(norm_layer)
  #LeakyReLU with a negative slope coefficient of 0.2.
  layer = activation(dropout_layer)

  return layer


def FullyConnected(input, 
                   units, 
                   activation=ReLU(),
                   use_bias=False,
                   kernel_initializer=GlorotNormal(),
                   reshape_shape=(1,1,1),
                   **kwargs):
  """
  Create a dense layer with batch normalization and reshape to allow for convolutions.

  :param input:
      Input layer
  :param units:
      Total number of units in the layer
  :param activation:
      ReLU activation
  :param kernel_initializer:
      Initiate weights with a Glorot normal initializer
  :param reshape_shape:
      Reshape the outputs of dense layer
  """
  dense_layer = Dense(units,
                      activation=None,
                      use_bias=use_bias,
                      kernel_initializer=kernel_initializer,
                      **kwargs)(input)
  #Transform the outputs from the dense layer to have mean≈0 and std≈1. 
  norm_layer = BatchNormalization()(dense_layer)
  #Relu activation
  layer = activation(norm_layer)
  #Reshape to allow for convolution operations.
  layer = Reshape(target_shape=reshape_shape)(layer)

  return layer


def Generator_Norm_Conv2DTranspose(input, 
                                   filters, 
                                   kernel_size=(3,3), 
                                   strides=(2,2), 
                                   padding='same', 
                                   activation=ReLU(),
                                   use_bias=False,
                                   kernel_initializer=GlorotNormal(), 
                                   **kwargs):
  """
  Create a Fractional-Strided conv layer with batch norm and ReLU.

  :param input:
      Input layer
  :param filters:
      Total number of filters
  :param kernel_size:
      Size of the kernel
  :param strides:
      Stride height and width of convolution
  """
  #Upsample the input's height and width by 2.
  conv_layer = Conv2DTranspose(filters,
                               kernel_size=kernel_size,
                               strides=strides,
                               padding=padding,
                               activation=None, 
                               use_bias=use_bias,
                               kernel_initializer=kernel_initializer)(input)
  #Transform the outputs from the conv_layer to have mean≈0 and std≈1.
  norm_layer = BatchNormalization()(conv_layer)
  #ReLU activation function.
  layer = activation(norm_layer)

  return layer


def Generator_Tanh_Conv2DTranspose(input, 
                        filters, 
                        kernel_size=(3,3), 
                        strides=(2,2), 
                        padding='same', 
                        activation=Activation('tanh'),
                        use_bias=False,
                        kernel_initializer=GlorotNormal(), 
                        **kwargs):
  """
  Create a Fractional-Strided conv layer with Tanh activation.

  :param input:
      Input layer
  :param filters:
      Total number of filters
  :param kernel_size:
      Size of the kernel
  :param strides:
      Stride height and width of convolution
  """
  #Upsample the input to be 256x256x1
  conv_layer = Conv2DTranspose(filters,
                               kernel_size=kernel_size,
                               strides=strides,
                               padding=padding,
                               activation=None, 
                               use_bias=use_bias,
                               kernel_initializer=kernel_initializer)(input)
  #Set Outputs values to [−1,1]
  layer = activation(conv_layer)

  return layer


def Flatten_Dense(input,
              **kwargs):
  """
  Create an output layer for the discriminator.

  :param input:
      Input layer
  """
  #Flatten the input tensor.
  flatten_layer = Flatten()(input)
  #Output a single scalar with no activation function.
  layer = Dense(1)(flatten_layer)

  return layer




 
