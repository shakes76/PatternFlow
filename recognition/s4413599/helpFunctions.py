from tensorflow.keras.layers import UpSampling2D, Conv2D, Dropout, LeakyReLU
'''
    This file contains some Help functions 
    Used to help network to be constructed easier
    Author: Anqi Yan S4413599
'''

# Leaky ReLu with a negative slope of 10^-2 in the improved UNET
slope = pow(10, -2)
def get_Conv2D(input_layer, n_filters, kernel_size=(3, 3), strides=(1, 1), activation=LeakyReLU(alpha=slope),
               padding='same'):
    '''
    :param input_layer:
        The input layer is layer from the last layer
    :param n_filters:
        Number of filters
    :param kernel_size:
        Kernel size in the convolutional layer
    :param strides:
        Srides in the convolutional layer
    :param activation:
        Activation function
    :param padding:
        Padding in tht convolutional layer
    :return:
        Constructed convolution layer
    '''
    return Conv2D(n_filters, kernel_size=kernel_size, strides=strides, activation=activation, padding=padding)(input_layer)


def get_contextModule(input_layer, n_filters):
    '''
    :param input_layer:
        The input layer is layer from the last layer
    :param n_filters:
        Number of filters
    :return:
        Constructed context Module
    '''
    conv = get_Conv2D(input_layer, n_filters)
    dropout = Dropout(0.3)(conv)
    return get_Conv2D(dropout, n_filters)


def get_unsamplingModule(input, n_filters):
    '''
    :param input:
        The input layer is layer from the last layer
    :param n_filters:
        Number of filters
    :return:
    '''
    up_sample = UpSampling2D(size=(2, 2))(input)
    return get_Conv2D(up_sample, n_filters)


def get_local_module(input, n_filters):
    '''
    :param input:
        The input layer is layer from the last layer
    :param n_filters:
        Number of filters
    :return:
    '''
    # 3 * 3 convolutional
    conv = get_Conv2D(input, n_filters, kernel_size=(3, 3))
    return get_Conv2D(conv, n_filters, kernel_size=(1, 1))