'''
Layers module for Tensorflow
'''
from tensorflow.keras.layers import Input, Activation, ReLU, LeakyReLU, Conv2D, Conv2DTranspose, BatchNormalization, Flatten, Dense, Reshape
from tensorflow.keras.initializers import GlorotNormal

#layers
def Norm_Conv2D(input_layer, 
                             n_filters, 
                             kernel_size=(3, 3),
                             strides=(1, 1),
                             activation=ReLU(), 
                             use_bias=True,
                             kernel_initializer=GlorotNormal(),
                             **kwargs):
    """
    Create a single convolution layer with batch norm

    :param input_layer:
        The input layer
    :param n_filters:
        The number of filters
    :param kernel_size:
        The size of the kernel filter
    :param strides:
        The stride number during convolution
    """
    #Create a 2D convolution layer 
    conv_layer = Conv2D(n_filters, 
                   kernel_size=kernel_size,
                   strides=strides,
                   padding='same',
                   activation=None,
                   use_bias=use_bias,
                   kernel_initializer=kernel_initializer, 
                   **kwargs)(input_layer)
    #adaptive batch normalization layer 
    norm_layer = BatchNormalization()(conv_layer)
    #Acitvation function
    layer = activation(norm_layer) 
    
    return layer

def Norm_Conv2DTranspose(input_layer, 
                             n_filters, 
                             kernel_size=(3, 3),
                             strides=(1, 1),
                             activation=ReLU(), 
                             use_bias=True,
                             kernel_initializer=GlorotNormal(),
                             **kwargs):
    """
    Create a single convolution transpose layer with batch norm

    :param input_layer:
        The input layer
    :param n_filters:
        The number of filters
    :param kernel_size:
        The size of the kernel filter
    :param strides:
        The stride number during convolution
    """
    #Create a 2D convolution layer 
    conv_layer = Conv2DTranspose(n_filters, 
                   kernel_size=kernel_size,
                   strides=strides,
                   padding='same',
                   activation=None,
                   use_bias=use_bias,
                   kernel_initializer=kernel_initializer, 
                   **kwargs)(input_layer)
    #adaptive batch normalization layer 
    norm_layer = BatchNormalization()(conv_layer)
    #Acitvation function
    layer = activation(norm_layer) 
    
    return layer
