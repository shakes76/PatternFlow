import tensorflow as tf
from keras.layers import Dense, LeakyReLU, Lambda, Reshape, Conv2D, UpSampling2D, GaussianNoise, Input

def context_module(residual_block):
    """
    Creates a context module for the network, as defined by Isensee et al.

    Parameters:
        residual_block: input layer of dimensionality n by n

    Returns:
        a tensor with more extracted features of dimensionality n by n

    Reference:
        https://arxiv.org/abs/1802.10508v1
    """
    pass

def reduction_module(residual_block):
    """
    Creates a reduction module for the network, defined as a convolutional layer with stride 2 by
    Isensee et al.

    Parameters:
        residual_block: input layer of dimensionality n by n

    Returns:
        a tensor of dimensionality n/2 by n/2
    
    Reference:
        https://arxiv.org/abs/1802.10508v1
    """
    pass

def upsampling_module(residual_block):
    """
    Creates an upsampling module for the network, which spatially repeats the features twice then
    halves the number of feature maps with a single convolution.

    Parameters:
        residual_block: input layer of dimensionality n by n

    Returns:
        a tensor of dimensionality n*2 by n*2

    Reference:
        https://arxiv.org/abs/1802.10508v1
    """
    pass

def localization_module(residual_block):
    """
    Creates a localization module that concatenates the features from the upsampled features of the
    input to the features of the corresponding context module in the same level of the network.

    Paramters:
        residual_block: input layer of dimensionality n by n

    Returns:
        a tensor of dimensionality n by n
    
    Reference:
        https://arxiv.org/abs/1802.10508v1
    """
    pass

def segmentation_module(residual_block):
    """
    Creates a segmentation module that takes input and applies a leaky ReLU activation for later
    usage. Segmentation modules are summed together to form the final network output.

    Parameters:
        residual_block: input layer of dimensionality n by n

    Returns:
        a tensor from activation of dimensionality n by n
    
    Reference:
        https://arxiv.org/abs/1802.10508v1
    """
    pass

def context_aggregation_pathway(input):
    """
    Creates the full encoding part of the network. This includes context and reduction modules.

    Parameters:
        input: raw input for network of dimensionality N by N

    Returns:
        an encoded tensor with a latent space dimensionality n by n
    
    Reference:
        https://arxiv.org/abs/1802.10508v1
    """
    pass

def localization_pathway():
    """
    Creates the full upsampling and segmentation part of the network. This includes upsampling and
    segmentation modules.

    Parameters:
        encoded_block: input of dimensionality n by n

    Returns:
        a tensor of the same dimensionality as the original input to the context aggregation pathway
    
    Reference:
        https://arxiv.org/abs/1802.10508v1
    """
    pass

def improved_unet():
    """
    Creates the full improved unet network. This includes the encoding from the context aggregation
    pathway and the upsampling & segmentation from the localization pathway.

    Parameters:
        input: raw input for the network

    Returns:
        a tensor of labels derived from segmentation
    """