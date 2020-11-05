'''
Layers module for Tensorflow
'''
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Activation, BatchNormalization, Conv2D, Conv2DTranspose, Dense, Dropout, Flatten, Input, LeakyReLU, ReLU, Reshape
from tensorflow.keras.initializers import GlorotNormal, GlorotUniform

# Dilated convolution for Generator Network
# with batch normalization and ReLU activation.
# Maybe helps with denoising the segmentation between
# black/grey background vs skull, skull vs cerebral fluid?
def Generator_Dilated_Conv2D(input, 
                                   filters, 
                                   kernel_size=(3,3), 
                                   strides=(1,1), 
                                   padding='same',
                                   dilation_rate=(1,1),
                                   activation=ReLU(),
                                   use_bias=True,
                                   kernel_initializer=GlorotNormal(), 
                                   **kwargs):
  
  conv_layer = Conv2D(filters,
                               kernel_size=kernel_size,
                               strides=strides,
                               padding=padding,
                               activation=None, 
                               use_bias=use_bias,
                               kernel_initializer=kernel_initializer)(input)
  
  norm_layer = BatchNormalization()(conv_layer)

  layer = activation(norm_layer)

  return layer



# Convolutional layer with batch normalization, dropout and LeakyReLU activation layers.
# Dropout before activation layer for computational efficiency.
def Discriminator_Norm_Dropout_Conv2D(input, 
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
                   reshape_shape=(1,1,1),
                   **kwargs):
  
  dense_layer = Dense(units,
                      activation=None,
                      use_bias=use_bias,
                      kernel_initializer=kernel_initializer,
                      **kwargs)(input)

  norm_layer = BatchNormalization()(dense_layer)

  layer = activation(norm_layer)

  layer = Reshape(target_shape=reshape_shape)(layer)

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




# Generator output convolutional layer with a Tanh activation.
def Generator_Tanh_Conv2DTranspose(input, 
                        filters, 
                        kernel_size=(3,3), 
                        strides=(2,2), 
                        padding='same', 
                        activation=Activation('tanh'),
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
  
  layer = activation(conv_layer)

  return layer


# Discriminator dense layer with flattened input and no activation.
def Flatten_Dense(input,
              **kwargs):

  flatten_layer = Flatten()(input)

  layer = Dense(1)(flatten_layer)

  return layer




 
