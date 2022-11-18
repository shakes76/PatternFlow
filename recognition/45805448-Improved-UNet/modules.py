from keras import Model
from keras.backend import flatten, sum
from keras.layers import LeakyReLU, Conv2D, UpSampling2D, Input, Dropout, Add, Concatenate
from keras.optimizers import Adam

# Default values for ISIC 2016 Training Dataset
NUM_FILTERS = 16
NUM_LEVELS = 5
NUM_SEGMENTATIONS = 3

def convolution(input, filters, kernel_size=3, strides=1, padding='same', activation=LeakyReLU(alpha=0.01)):
    """
    Default convolution layer configuration.

    Returns:
        a convolution layer
    """
    return Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, activation=activation)(input)

def dropout(input, rate=0.3, seed=69):
    """
    Default dropout layer configuration.

    Returns:
        a dropout layer
    """
    return Dropout(rate=rate, seed=seed)(input)

def upsampling(input, size=2):
    """
    Default upsampling layer configuration.

    Returns:
        an upsampling layer
    """
    return UpSampling2D(size=size)(input)

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
    context_conv = convolution(residual_block, filters)
    context_drop = dropout(context_conv)
    context_conv = convolution(context_drop, filters)
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
    reducing_conv = convolution(residual_block, filters, strides=2)
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
    upsampling_upsm = upsampling(residual_block)
    upsampling_conv = convolution(upsampling_upsm, filters)
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
    localization_conv = convolution(residual_block, filters)
    localization_conv = convolution(localization_conv, filters, kernel_size=1)
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
    segmentation_conv = convolution(residual_block, filters, kernel_size=1)
    if prev_segmentation != None:
        segmentation_add = Add()([prev_segmentation, segmentation_conv])
        if upsample:
            segmentation_upsm = upsampling(segmentation_add)
            return segmentation_upsm
        else:
            return segmentation_add
    else:
        segmentation_upsm = upsampling(segmentation_conv)
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
    start_layer = convolution(input, filters)
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
    segmentation modules. The output is ran through a softmax activation to determine the labels
    for the individual pixels.

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
    convolution_layer = convolution(upsampling_layers, filters * 2)
    segmentation_layers = segmentation_module(convolution_layer, segmentation_layers, upsample=False)

    finish_layer = convolution(segmentation_layers, 2, kernel_size=1, activation='softmax')

    return finish_layer

def dice_coefficient(ground_truth, softmax_out):
    """
    Multiclass Dice loss function as defined by Isensee et al. This metric is used to counteract the
    class imbalance in the data that other loss functions such as categorical crossentropy may
    struggle with. 
    
    In other words, an image with too much background may be classed with too much respect to that, 
    and miss the smaller occurences of other labels (such as the actual skin cancer) to minimise the
    loss.

    Parameters:
        ground_truth: actual locations of the skin cancer
        softmax_out: predicted locations of the cancer, extracted from the softmax activation in the
            final layer of the improved UNet model

    Returns:
        customised loss between the ground truth of the image and the predicted labels

    Reference:
        https://arxiv.org/abs/1802.10508v1
    """
    ground_truth_flat = flatten(ground_truth)
    softmax_out_flat = flatten(softmax_out)
    multiclass_summation = sum(ground_truth_flat * softmax_out_flat) / (sum(ground_truth_flat) + sum(softmax_out_flat))
    return 2 * multiclass_summation

def improved_unet(input_shape, batch_size, filters=NUM_FILTERS, num_levels=NUM_LEVELS, segmentations=NUM_SEGMENTATIONS):
    """
    Creates the full improved unet network. This includes the encoding from the context aggregation
    pathway and the upsampling & segmentation from the localization pathway.

    Parameters:
        input: raw input for the network

    Returns:
        a tensor of labels derived from segmentation
    """
    input = Input(shape=input_shape, batch_size=batch_size)
    encoded, contexts = context_aggregation_pathway(input, filters, num_levels)
    output = localization_pathway(encoded, contexts, filters, num_levels, segmentations)

    improved_unet = Model(inputs=input, outputs=output)
    improved_unet.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=[dice_coefficient])

    return improved_unet