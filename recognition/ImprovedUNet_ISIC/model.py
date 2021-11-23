# Filename: model.py
# Author: Navin Sivasankaran
# Date created: 6/11/2020
# Date last modified (addressing feedback): 26/11/2020
# Python Version: 3.7.7
# Description: The script containing the Improved U-Net model. Called automatically by main.py

#Import libraries
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *

"""
The following function will determine the Dice coefficient, which is a spatial overlap index.
This measures the overlap between 2 samples and will be used as the metric to determine the performance of the model.
The link "https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2" by Ekin Tu (2019)
assisted with the understanding of the Dice coefficient and the code behind it.

Parameters:
    y_actual: Actual segmentation image (Ground truth)
    y_predicted: Predicted segmentation image (label)

Returns:
    Dice coefficient
"""
def dice_coefficient(y_actual, y_predicted):
    y_actual = tf.keras.backend.flatten(y_actual)
    y_predicted = tf.keras.backend.flatten(y_predicted)
    intersection_y = tf.keras.backend.sum(y_actual * y_predicted)
    union_y = tf.keras.backend.sum(y_actual) + tf.keras.backend.sum(y_predicted)
    return ((2.0*intersection_y + 1e-7) / (union_y + 1e-7))

"""
The following function will determine the Dice distance.
This value will show how much of the predicted segmentation image fails to match the actual image.

Parameters:
    y_actual: Actual segmentation image (Ground truth)
    y_predicted: Predicted segmentation image (label)

Returns:
    Dice loss
"""
def dice_loss(y_actual, y_predicted):
    return 1 - dice_coefficient(y_actual, y_predicted)

"""
Context module for the Improved U-Net model, for the encoder phase.
Contains a pre-activation residual block with 2 3x3x3 convolutional layers
and a dropout layer in between (refer to the README file for more information).

Parameters:
    layer: The tensor that is passed in for the context module to be continued on.
    filter_size: Numebr of convolution filters

Returns:
    layer: The tensor, after the context module has been completed on it
"""
def context_module_unet(layer, filter_size):
    layer = BatchNormalization()(layer)
    layer = Conv2D(filter_size, (3, 3), padding = 'same')(layer)
    layer = LeakyReLU(alpha = 0.3)(layer)
    layer = Dropout(0.3)(layer)
    layer = Conv2D(filter_size, (3, 3), padding = 'same')(layer)
    layer = LeakyReLU(alpha = 0.3)(layer)

    return layer

"""
Upsampling module for the Improved U-Net model, for the decoder phase.
Contains a up-sampling layer, followed by a 3x3x3 convolution layer halving the number of feature maps
(refer to the README file for more information).

Parameters:
    layer: The tensor that is passed in for the context module to be continued on.
    filter_size: Numebr of convolution filters

Returns:
    layer: The tensor, after the upsampling module has been completed on it
"""
def upsample_module_unet(layer, filter_size):
    layer = UpSampling2D()(layer)
    layer = Conv2D(filter_size, (3, 3), padding = 'same')(layer)
    layer = LeakyReLU(alpha = 0.3)(layer)
    
    return layer
    
"""
Localisation module for the Improved U-Net model, for the decoder phase.
Used to transfer information that have been encoded by lower levels to higher spatial resolutions.
Contains a 3x3x3 convolution layer, followed by a 1x1x1 one
(refer to the README file for more information).

Parameters:
    layer: The tensor that is passed in for the context module to be continued on.
    filter_size: Numebr of convolution filters

Returns:
    layer: The tensor, after the upsampling module has been completed on it
"""
def localisation_module_unet(layer, filter_size):
    layer = Conv2D(filter_size, (3, 3), padding = 'same')(layer)
    layer = LeakyReLU(alpha = 0.3)(layer)
    layer = Conv2D(filter_size, (1, 1), padding = 'same')(layer)
    layer = LeakyReLU(alpha = 0.3)(layer)

    return layer

"""
Segmentation layer module for the Improved U-Net model, for the decoder phase.
Used for deep supervision in the localisation module
(refer to the README file for more information).

Parameters:
    localise_module_a: Second localisation module run in the U-Net
    localise_module_b: Third localisation module run in the U-Net
    conv_a: Second-last convolutional layer run in the U-Net

Returns:
    sum_b: An element-wise summation of the combined
"""
def segmentation_layer(localise_module_a, localise_module_b, conv_a):
    segment_1 = Conv2D(1, (1, 1), padding = 'same')(localise_module_a)
    segment_1 = LeakyReLU(alpha = 0.3)(segment_1)

    upsample_a = UpSampling2D()(segment_1)
    
    segment_2 = Conv2D(1, (1, 1), padding = 'same')(localise_module_b)
    segment_2 = LeakyReLU(alpha = 0.3)(segment_2)
    sum_a = add([upsample_a, segment_2])
    
    upsample_b = UpSampling2D()(sum_a)
    segment_3 = Conv2D(1, (1, 1), padding = 'same')(conv_a)
    segment_3 = LeakyReLU(alpha = 0.3)(segment_3)

    sum_b = add([upsample_b, segment_3])
    
    return sum_b

def unet_model():
    inputs = Input((256, 256, 1))
    
    conv2D_1 = Conv2D(8, (3, 3), padding = 'same')(inputs)
    conv2D_1 = LeakyReLU(alpha = 0.3)(conv2D_1)
    
    cont_1 = context_module_unet(conv2D_1, 8)
    sum_1 = add([conv2D_1, cont_1])
    
    conv2D_2 = Conv2D(16, (3, 3), padding = 'same', strides = 2)(sum_1)
    conv2D_2 = LeakyReLU(alpha = 0.3)(conv2D_2)
    cont_2 = context_module_unet(conv2D_2, 16)
    sum_2 = add([conv2D_2, cont_2])
    
    conv2D_3 = Conv2D(32, (3, 3), padding = 'same', strides = 2)(sum_2)
    conv2D_3 = LeakyReLU(alpha = 0.3)(conv2D_3)
    cont_3 = context_module_unet(conv2D_3, 32)
    sum_3 = add([conv2D_3, cont_3])
    
    conv2D_4 = Conv2D(64, (3, 3), padding = 'same', strides = 2)(sum_3)
    conv2D_4 = LeakyReLU(alpha = 0.3)(conv2D_4)
    cont_4 = context_module_unet(conv2D_4, 64)
    sum_4 = add([conv2D_4, cont_4])
    
    conv2D_5 = Conv2D(128, (3, 3), padding = 'same', strides = 2)(sum_4)
    conv2D_5 = LeakyReLU(alpha = 0.3)(conv2D_5)
    cont_5 = context_module_unet(conv2D_5, 128)
    sum_5 = add([conv2D_5, cont_5])
    
    upsample_1 = upsample_module_unet(sum_5, 64)
    concatenate_1 = concatenate([upsample_1, sum_4])
    
    localise_1 = localisation_module_unet(concatenate_1, 64)
    upsample_2 = upsample_module_unet(localise_1, 32)
    concatenate_2 = concatenate([upsample_2, sum_3])
    
    localise_2 = localisation_module_unet(concatenate_2, 32)
    upsample_3 = upsample_module_unet(localise_2, 16)
    concatenate_3 = concatenate([upsample_3, sum_2])
    
    localise_3 = localisation_module_unet(concatenate_3, 16)
    upsample_4 = upsample_module_unet(localise_3, 8)
    concatenate_4 = concatenate([upsample_4, sum_1])
    
    conv2D_6 = Conv2D(16, (3, 3), padding = 'same')(concatenate_4)
    conv2D_6 = LeakyReLU(alpha = 0.3)(conv2D_6)
    
    segmentation_1 = segmentation_layer(localise_2, localise_3, conv2D_6)
    
    conv2D_final = Conv2D(1, (1, 1), activation = 'sigmoid', padding = 'same')(segmentation_1)

    model = tf.keras.Model(inputs=inputs, outputs=conv2D_final)
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[dice_coefficient])

    return model

