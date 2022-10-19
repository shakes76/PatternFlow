import array
import tensorflow as tf
from keras import Model
from keras.layers import LeakyReLU, Conv2D, UpSampling2D, Input, Dropout, Add, Concatenate
from keras.optimizers import Adam

def convolution(filters, kernel_size=3, strides=1, padding='same', activation=LeakyReLU(alpha=0.01)):
    """
    Default convolution layer configuration.

    Returns:
        a convolution layer
    """
    return Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, activation=activation)

def dropout(rate=0.3, seed=69):
    """
    Default dropout layer configuration.

    Returns:
        a dropout layer
    """
    return Dropout(rate=rate, seed=seed)

def upsampling(size=2):
    """
    Default upsampling layer configuration.

    Returns:
        an upsampling layer
    """
    return UpSampling2D(size=size)

def context_module(residual_block, filters):
    """
    Creates a context module for the network, which extracts some new features at one UNet level.

    Parameters:
        residual_block: input layer of dimensionality n by n

    Returns:
        a tensor with more extracted features of dimensionality n by n

    Reference:
        https://arxiv.org/abs/1802.10508v1
    """
    context_conv = convolution(filters)(residual_block)
    context_drop = dropout()(context_conv)
    context_conv = convolution(filters)(context_drop)
    context_add = Add()([residual_block, context_conv])
    return context_add

def reducing_module(residual_block, filters):
    """
    Creates a reducing module for the network, which encodes the data down one level.

    Parameters:
        residual_block: input layer of dimensionality n by n

    Returns:
        a tensor of dimensionality n/2 by n/2
    
    Reference:
        https://arxiv.org/abs/1802.10508v1
    """
    reducing_conv = convolution(filters, strides=2)(residual_block)
    return reducing_conv

def upsampling_module(residual_block, context_block, filters):
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
    upsampling_upsm = upsampling()(residual_block)
    upsampling_conv = convolution(filters)(upsampling_upsm)
    upsampling_cnct = Concatenate()([context_block, upsampling_conv])
    return upsampling_cnct

def localization_module(residual_block, filters):
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
    localization_conv = convolution(filters)(residual_block)
    localization_conv = convolution(filters, kernel_size=1)(localization_conv)
    return localization_conv

def segmentation_module(residual_block, prev_segmentation=None, filters=16, upsample=True):
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
    segmentation_conv = convolution(filters, kernel_size=1)(residual_block)
    if prev_segmentation != None:
        segmentation_add = Add()([prev_segmentation, segmentation_conv])
        if upsample:
            segmentation_upsm = upsampling()(segmentation_add)
            return segmentation_upsm
        else:
            return segmentation_add
    else:
        segmentation_upsm = upsampling()(segmentation_conv)
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
    # Initialize context aggregation pathway
    contexts = []
    start_layer = convolution(filters)(input)
    context_layers = context_module(start_layer, filters)
    contexts.append(context_layers)

    # Continue building context aggregation pathway, starting at level = 1
    for level in range(1, num_levels):
        reducing_layers = reducing_module(context_layers, filters * 2**level)
        context_layers = context_module(reducing_layers, filters * 2**level)
        contexts.append(context_layers)

    # Don't need last context for localization
    contexts.pop()
    
    return context_layers, contexts

def localization_pathway(encoded, contexts: list, filters=16, num_levels=5, segmentations=3):
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
    # Initialize localization pathway
    segmentation_layers = None
    upsampling_layers = upsampling_module(encoded, contexts.pop(), filters * 2**((num_levels - 1) - 1))

    # Continue building localization pathway, starting at level = max_level - 1
    for level in range((num_levels - 2), 0, -1):
        localization_layers = localization_module(upsampling_layers, filters * 2**level)
        if level < segmentations:
            segmentation_layers = segmentation_module(localization_layers, segmentation_layers)
        upsampling_layers = upsampling_module(localization_layers, contexts.pop(), filters * 2**(level - 1))
        
    # Complete localization pathway
    convolution_layer = convolution(filters * 2)(upsampling_layers)
    segmentation_layers = segmentation_module(convolution_layer, segmentation_layers, upsample=False)

    finish_layer = convolution(2, kernel_size=1, activation='softmax')(segmentation_layers)

    return finish_layer


def improved_unet(input_shape, filters=16, num_levels=5, segmentations=3):
    """
    Creates the full improved unet network. This includes the encoding from the context aggregation
    pathway and the upsampling & segmentation from the localization pathway.

    Parameters:
        input: raw input for the network

    Returns:
        a tensor of labels derived from segmentation
    """
    input = Input(shape=input_shape)
    encoded, contexts = context_aggregation_pathway(input, filters, num_levels)
    print(tf.keras.Model(inputs=input, outputs=encoded).summary())
    output = localization_pathway(encoded, contexts, filters, num_levels, segmentations)

    improved_unet = tf.keras.Model(inputs=input, outputs=output)
    improved_unet.compile(optimizer=Adam(), loss='binary_crossentropy')

    return improved_unet