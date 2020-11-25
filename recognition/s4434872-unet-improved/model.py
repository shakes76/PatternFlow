"""
Improved UNet Model module.

@author Dhilan Singh (44348724)

Created: 07/11/2020
"""
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Activation, ReLU, LeakyReLU, Conv2D, BatchNormalization, Flatten, Dense, Dropout, UpSampling2D, concatenate, Add
from tensorflow_addons.layers import InstanceNormalization

# Activation used throughout, with negative slope of 10^-2 
leakyReLu = LeakyReLU(alpha=1e-2)
# Use dropout rate of 0.3 throughout
dropout_rate = 0.3
# Input Image Parameters for model
image_pixel_rows = 256 
image_pixel_cols = 256
image_channels = 1

def conv2D_layer(input_layer, 
                 n_filters, 
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 activation=leakyReLu, 
                 use_bias=True,
                 kernel_initializer='he_normal',
                 batch_normalization=False,
                 instance_normalization=False,
                **kwargs):
    """
    Create a 2D convolutional layer with normalization.

    @param input_layer:
        The input layer.
    @param n_filters:
        The number of filters.
    @param kernel_size:
        The size of the kernel filter.
    @param strides:
        The stride number during convolution.
    @param activation:
        Keras activation layer to use.
    @param batch_normalization:
        If true, apply batch normalization.
    @param instance_normalization:
        If true, apply instance normalization.

    Reference: Adapted from Shakes lecture code layers.py
    """
    # Create a 2D convolution layer 
    conv_layer = Conv2D(n_filters, 
                        kernel_size=kernel_size,
                        strides=strides,
                        padding='same',
                        activation=None,
                        use_bias=use_bias,
                        kernel_initializer=kernel_initializer, 
                        **kwargs)(input_layer)
    
    # Apply chosen normalization method
    if batch_normalization:
        # Apply Batch normalization layer 
        norm_layer = BatchNormalization()(conv_layer)
    elif instance_normalization:
        # Apply Instance normalization
        norm_layer = InstanceNormalization()(conv_layer)

    # Activation function
    layer = Activation(activation=activation)(norm_layer)
    
    return layer


def context_module(input, n_filters, activation=leakyReLu):
    """
    The activations in the context pathway are computed by context modules.
    Each context module is a pre-activation residual block with two
    3x3x convolutional layers and a dropout layer (pdrop = 0.3) in between.
    """
    conv1 = conv2D_layer(input, n_filters, kernel_size=(3, 3), strides=(1, 1), activation=activation, instance_normalization=True)
    dropout = Dropout(rate=dropout_rate)(conv1)
    conv2 = conv2D_layer(dropout, n_filters, kernel_size=(3, 3), strides=(1, 1), activation=activation, instance_normalization=True)
    return conv2

def upsampling_module(input, n_filters, activation=leakyReLu):
    """
    Upsampling the low resolution feature maps, is done by means of a simple 
    upscale that repeats the feature voxels twice in each spatial dimension, 
    followed by a 3x3 convolution that halves the number of feature maps. 
    Compared to the more frequently employed transposed convolution this 
    approach delivers similar performance while preventing checkerboard
    artifacts in the network output.
    """
    up_sample = UpSampling2D(size=(2,2))(input)
    conv1 = conv2D_layer(up_sample, n_filters, kernel_size=(3, 3), strides=(1, 1), activation=activation, instance_normalization=True)
    return conv1

def localization_module(input, n_filters, activation=leakyReLu):
    """
    A localization module consists of a 3x3 convolution followed by a 1x1 
    convolution that halves the number of feature maps of the input.
    """
    conv1 = conv2D_layer(input, n_filters, kernel_size=(3, 3), strides=(1, 1), activation=activation, instance_normalization=True)
    conv2 = conv2D_layer(conv1, n_filters, kernel_size=(1, 1), strides=(1, 1), activation=activation, instance_normalization=True)
    return conv2

# improved unet model
def improved_unet_model(output_channels, n_filters=16, 
                        input_size=(image_pixel_rows, image_pixel_cols, image_channels)):
    """
    Improved UNet network based on https://arxiv.org/abs/1802.10508v1.

    Comprises of a context aggregation pathway that encodes increasingly abstract 
    representations of the input as we progress deeper into the network, followed 
    by a localization pathway that recombines these representations with shallower 
    features to precisely localize the structures of interest.


    output_channels: Correspond to the classes a pixel can be. Here 4.
    f: Initial number of filters used in convolutional layers. Starts at 16.
    
    Reference: https://arxiv.org/abs/1802.10508v1
    """
    # Input Image
    inputs = Input(shape=input_size)

    # Context Pathway Level 1
    conv1 = conv2D_layer(inputs, n_filters, kernel_size=(3, 3), strides=(1, 1), activation=leakyReLu, instance_normalization=True)
    context1 = context_module(conv1, n_filters)
    rescon1 = Add()([conv1, context1])

    # Context Pathway Level 2
    down_conv1 = conv2D_layer(rescon1, 2*n_filters, kernel_size=(3, 3), strides=(2, 2), activation=leakyReLu, instance_normalization=True)
    context2 = context_module(down_conv1, 2*n_filters)
    rescon2 = Add()([down_conv1, context2])

    # Context Pathway Level 3
    down_conv2 = conv2D_layer(rescon2, 4*n_filters, kernel_size=(3, 3), strides=(2, 2), activation=leakyReLu, instance_normalization=True)
    context3 = context_module(down_conv2, 4*n_filters)
    rescon3 = Add()([down_conv2, context3])

    # Context Pathway Level 4
    down_conv3 = conv2D_layer(rescon3, 8*n_filters, kernel_size=(3, 3), strides=(2, 2), activation=leakyReLu, instance_normalization=True)
    context4 = context_module(down_conv3, 8*n_filters)
    rescon4 = Add()([down_conv3, context4])

    # Context Pathway Level 5 (Base Part 1)
    down_conv4 = conv2D_layer(rescon4, 16*n_filters, kernel_size=(3, 3), strides=(2, 2), activation=leakyReLu, instance_normalization=True)
    context5 = context_module(down_conv4, 16*n_filters)
    rescon5 = Add()([down_conv4, context5])

    # Localization Pathway Level 5 (Base Part 2)
    up_sample5 = upsampling_module(rescon5, 8*n_filters)

    # Localization Pathway Level 4
    skip4 = concatenate([up_sample5, rescon4])
    localize4 = localization_module(skip4, 8*n_filters)
    up_sample4 = upsampling_module(localize4, 4*n_filters)
    
    # Localization Pathway Level 3
    skip3 = concatenate([up_sample4, rescon3])
    localize3 = localization_module(skip3, 4*n_filters)
    up_sample3 = upsampling_module(localize3, 2*n_filters)
    # Segmentation Layer 3 (At 1/4 of the size of original image)
    seg_map_3 = conv2D_layer(localize3, output_channels, kernel_size=(1, 1), strides=(1, 1), activation=leakyReLu, instance_normalization=True)
    seg_map_3_upscale = UpSampling2D(interpolation='bilinear')(seg_map_3)

    # Localization Pathway Level 2
    skip2 = concatenate([up_sample3, rescon2])
    localize2 = localization_module(skip2, 2*n_filters)
    up_sample2 = upsampling_module(localize2, n_filters)
    # Segmentation Layer 2 (At 1/2 of the size of original image)
    seg_map_2 = conv2D_layer(localize2, output_channels, kernel_size=(1, 1), strides=(1, 1), activation=leakyReLu, instance_normalization=True)
    # Add the upscaled segmentation layer 3 to segmentation layer 2
    seg_map_3_2 = Add()([seg_map_3_upscale, seg_map_2])
    seg_map_3_2_upscale = UpSampling2D(interpolation='bilinear')(seg_map_3_2)

    # Localization Pathway Level 1
    skip1 = concatenate([up_sample2, rescon1])
    conv_out = conv2D_layer(skip1, 2*n_filters, kernel_size=(3, 3), strides=(1, 1), activation=leakyReLu, instance_normalization=True)
    # Segmentation Layer 1 (At the same size of original image)
    seg_map_1 = conv2D_layer(conv_out, output_channels, kernel_size=(1, 1), strides=(1, 1), activation=leakyReLu, instance_normalization=True)
    # Add the upscaled segmentation sum of layer 3 and 2 to segmentation layer 1
    seg_map_full = Add()([seg_map_3_2_upscale, seg_map_1])

    if output_channels > 2:
        # Multiclass classification, so use softmax to output class predctions that
        # sum to 1 (i.e. probabilities).
        outputs = Activation('softmax')(seg_map_full)
    else:
        # Problem involves binary classification of pixels, so can use a sigmoid (0 or 1).
        outputs = Activation('sigmoid')(seg_map_full)
    
    # Complete Model
    return Model(inputs=inputs, outputs=outputs)