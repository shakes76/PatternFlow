import tensorflow as tf
from keras.layers import Dense, LeakyReLU, Lambda, Reshape, Conv2D, UpSampling2D, GaussianNoise, Input, Dropout, Add, Concatenate

def context_module(residual_block, filters, kernel_size=3, rate=0.3, seed=69):
    """
    Creates a context module for the network, which extracts some new features at one UNet level.

    Parameters:
        residual_block: input layer of dimensionality n by n

    Returns:
        a tensor with more extracted features of dimensionality n by n

    Reference:
        https://arxiv.org/abs/1802.10508v1
    """
    context_conv = Conv2D(filters=filters, kernel_size=kernel_size, padding='same')(residual_block)
    context_drop = Dropout(rate=rate, seed=seed)(context_conv)
    context_conv = Conv2D(filters=filters, kernel_size=kernel_size, padding='same')(context_drop)
    context_add = Add()([residual_block, context_conv])
    return context_add

def reducing_module(residual_block, filters, kernel_size=3, strides=2):
    """
    Creates a reducing module for the network, which encodes the data down one level.

    Parameters:
        residual_block: input layer of dimensionality n by n

    Returns:
        a tensor of dimensionality n/2 by n/2
    
    Reference:
        https://arxiv.org/abs/1802.10508v1
    """
    reducing_conv = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same')(residual_block)
    return reducing_conv

def upsampling_module(residual_block, concat_block, filters, kernel_size=3, size=2):
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
    upsampling_upsm = UpSampling2D(size=size)(residual_block)
    upsampling_conv = Conv2D(filters=filters, kernel_size=kernel_size, padding='same')(upsampling_upsm)
    upsampling_cnct = Concatenate()([concat_block, upsampling_conv])
    return upsampling_cnct

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
    Creates the full encoding part of the network. This includes context and reducing modules.

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