import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers
"""
Return a context module made of two 3x3 convolutions with a dropout of 0.3 between them
Parameters:
    input: The layer prior to this module
    filters: The number of filters for this module
Returns:
    Context module
"""
def context_module(input, filters):
    conv1 = tfa.layers.InstanceNormalization()(input)
    conv1 = layers.Conv2D(filters, (3, 3), padding = "same", activation = LeakyReLU(alpha = 0.01))(conv1)
    dropout = layers.Dropout(0.3) (conv1)
    conv2 = tfa.layers.InstanceNormalization()(dropout)
    conv2 = layers.Conv2D(filters, (3, 3), padding = "same", activation = LeakyReLU(alpha = 0.01))(conv2)
    return conv2
	
"""
Return an upsampling module with a 3x3 convolution
Parameters:
    input: The layer prior to this module
    filters: The number of filters for this module
Returns:
    Upsampling module
"""
def upsampling_module(input, filters):
    up = layers.UpSampling2D((2, 2))(input)
    up = layers.Conv2D(filters, (3, 3), padding = "same", activation = LeakyReLU(alpha = 0.01))(up)
    return up
	
"""
Return a localization module made of two 3x3 convolutions
Parameters:
    input: The layer prior to this module
    filters: The number of filters for this module
Returns:
    Localization module
"""
def localization_module(input, filters):
    conv1 = layers.Conv2D(filters, (3, 3), padding = "same", activation = LeakyReLU(alpha = 0.01))(input)
    conv2 = layers.Conv2D(filters, (1, 1), padding = "same", activation = LeakyReLU(alpha = 0.01))(conv1)
    return conv2