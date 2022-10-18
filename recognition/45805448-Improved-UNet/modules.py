import array
import tensorflow as tf
from keras import Model
from keras.layers import LeakyReLU, Conv2D, UpSampling2D, Input, Dropout, Add, Concatenate
from keras.optimizers import Adam

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

def upsampling_module(residual_block, context_block, filters, kernel_size=3, size=2):
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
    upsampling_cnct = Concatenate()([context_block, upsampling_conv])
    return upsampling_cnct

def localization_module(residual_block, filters, kernel_size=3):
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
    localization_conv = Conv2D(filters=filters, kernel_size=kernel_size, padding='same')(residual_block)
    localization_conv = Conv2D(filters=filters, kernel_size=1, padding='same')(localization_conv)
    return localization_conv

def segmentation_module(residual_block, filters, prev_segmentation=None, upsample=True, size=2):
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
    segmentation_conv = Conv2D(filters=filters, kernel_size=1, padding='same')(residual_block)
    if prev_segmentation != None:
        segmentation_add = Add()([prev_segmentation, segmentation_conv])
        if upsample:
            segmentation_upsm = UpSampling2D(size=size)(segmentation_add)
            return segmentation_upsm
        else:
            return segmentation_add
    else:
        segmentation_upsm = UpSampling2D(size=size)(segmentation_conv)
        return segmentation_upsm

def context_aggregation_pathway(input, filters=16, num_levels=5):
    """
    Creates the full encoding part of the network. This includes context and reducing modules.

    Parameters:
        input: raw input for network of dimensionality N by N

    Returns:
        an encoded tensor with a latent space dimensionality n by n
    
    Reference:
        https://arxiv.org/abs/1802.10508v1
    """
    contexts = []

    reducing_layers = reducing_module(input, filters, strides=1) # Start convolution, doesn't reduce dimensionality
    context_layers = context_module(reducing_layers, filters)
    contexts.append(context_layers)
    filters = filters * 2

    # Starting at level = 1
    for _ in range(1, num_levels):
        reducing_layers = reducing_module(context_layers, filters)
        context_layers = context_module(reducing_layers, filters)
        contexts.append(context_layers)
        filters = filters * 2

    contexts.pop() # Don't need last context for localization
    
    return context_layers, contexts, filters

def localization_pathway(encoded, contexts: list, filters=256, num_levels=5, segmentations=3):
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
    segmentation_layers = None
    filters = filters // 2
    upsampling_layers = upsampling_module(encoded, contexts.pop(), filters)

    # Starting at level = max_level - 1
    for level in range((num_levels - 1) - 1, 0):
        localization_layers = localization_module(upsampling_layers, filters)
        if level < segmentations:
            segmentation_layers = segmentation_module(localization_layers, segmentation_layers)
        filters = filters // 2
        upsampling_layers = upsampling_module(localization_layers, contexts.pop(), filters)
        
    localization_layers = reducing_module(upsampling_layers, filters, strides=1) # End convolution, doesn't reduce dimensionality
    segmentation_layers = segmentation_module(localization_layers, segmentation_layers, upsample=False)

    return segmentation_layers


def improved_unet(input, filters=16, num_levels=5, segmentations=3):
    """
    Creates the full improved unet network. This includes the encoding from the context aggregation
    pathway and the upsampling & segmentation from the localization pathway.

    Parameters:
        input: raw input for the network

    Returns:
        a tensor of labels derived from segmentation
    """
    input_layer = Input(shape=input.shape)
    encoded, contexts, filters = context_aggregation_pathway(input_layer, filters, num_levels)
    output = localization_pathway(encoded, contexts, filters, num_levels, segmentations)

    improved_unet = tf.keras.Model(inputs=input_layer, outputs=output)
    improved_unet.compile(optimizer=Adam(), loss='binary_crossentropy')

    return improved_unet