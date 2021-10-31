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
	
def improved_model(input_size = (256, 256, 1)):
    
    input_layer = layers.Input(shape=(input_size))
    
    #Encoder
    conv1 = Conv2D(16, (3, 3), padding = "same")(inputs)
    conv_module1 = context_module(conv1, 16)
    add1 = layers.Add()([conv1, conv_module1])
    
    conv2 = Conv2D(32, (3, 3), strides = 2, padding = "same")(add1)
    conv_module2 = context_module(conv2, 16)
    add2 = layers.Add()([conv2, conv_module2])
    
    conv3 = Conv2D(64, (3, 3), strides = 2, padding = "same")(add2)
    conv_module3 = context_module(conv3, 16)
    add3 = layers.Add()([conv3, conv_module3])
    
    conv4 = Conv2D(128, (3, 3), strides = 2, padding = "same")(add3)
    conv_module4 = context_module(conv4, 16)
    add4 = layers.Add()([conv4, conv_module4])
    
    conv5 = Conv2D(256, (3, 3), strides = 2, padding = "same")(add4)
    conv_module5 = context_module(conv5, 16)
    add5 = layers.Add()([conv5, conv_module5])
    