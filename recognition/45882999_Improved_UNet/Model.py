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
    conv1 = layers.Conv2D(filters, (3, 3), padding = "same", activation = layers.LeakyReLU(alpha = 0.01))(conv1)
    dropout = layers.Dropout(0.3) (conv1)
    conv2 = tfa.layers.InstanceNormalization()(dropout)
    conv2 = layers.Conv2D(filters, (3, 3), padding = "same", activation = layers.LeakyReLU(alpha = 0.01))(conv2)
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
    up = layers.Conv2D(filters, (3, 3), padding = "same", activation = layers.LeakyReLU(alpha = 0.01))(up)
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
    conv1 = layers.Conv2D(filters, (3, 3), padding = "same", activation = layers.LeakyReLU(alpha = 0.01))(input)
    conv2 = layers.Conv2D(filters, (1, 1), padding = "same", activation = layers.LeakyReLU(alpha = 0.01))(conv1)
    return conv2
	
"""
Creates an improved UNet model based on the paper “Brain Tumor Segmentation 
and Radiomics Survival Prediction: Contribution to the BRATS 2017 Challenge,”
Parameters:
    input_shape: the shape of the data that is inputted
Returns: 
    An improved UNet model
"""
def improved_model(input_size = (256, 256, 1)):
    
    input_layer = layers.Input(shape=(input_size))
    
    #Encoder
    conv1 = layers.Conv2D(16, (3, 3), padding = "same")(input_layer)
    conv_module1 = context_module(conv1, 16)
    add1 = layers.Add()([conv1, conv_module1])
    
    conv2 = layers.Conv2D(32, (3, 3), strides = 2, padding = "same")(add1)
    conv_module2 = context_module(conv2, 32)
    add2 = layers.Add()([conv2, conv_module2])
    
    conv3 = layers.Conv2D(64, (3, 3), strides = 2, padding = "same")(add2)
    conv_module3 = context_module(conv3, 64)
    add3 = layers.Add()([conv3, conv_module3])
    
    conv4 = layers.Conv2D(128, (3, 3), strides = 2, padding = "same")(add3)
    conv_module4 = context_module(conv4, 128)
    add4 = layers.Add()([conv4, conv_module4])
    
    conv5 = layers.Conv2D(256, (3, 3), strides = 2, padding = "same")(add4)
    conv_module5 = context_module(conv5, 256)
    add5 = layers.Add()([conv5, conv_module5])
    
	#Decoder
    up_module1 = upsampling_module(add5, 128)
    concat1 = layers.concatenate([up_module1, add4])
    local_module1 = localization_module(concat1, 128)
    
    up_module2 = upsampling_module(local_module1, 64)
    concat2 = layers.concatenate([up_module2, add3])
    local_module2 = localization_module(concat2, 64)
    
    up_module3 = upsampling_module(local_module2, 32)
    concat3 = layers.concatenate([up_module3, add2])
    local_module3 = localization_module(concat3, 32)
    
    up_module4 = upsampling_module(local_module3, 16)
    concat4 = layers.concatenate([up_module4, add1])
    
    conv6 = layers.Conv2D(32, (3, 3), padding = "same")(concat4)
    
    seg1 = layers.Conv2D(1, (1, 1), padding = "same")(local_module2)
    seg1 = layers.UpSampling2D(size = (2, 2))(seg1)
    
    seg2 = layers.Conv2D(1, (1, 1), padding = "same")(local_module3)
    add6 = layers.Add()([seg1, seg2])
    add6 = layers.UpSampling2D(size = (2, 2))(add6)
    
    seg3 = layers.Conv2D(1, (1, 1), padding = "same")(conv6)
    add7 = layers.Add()([add6, seg3])
    
    outputs = layers.Conv2D(1, (1, 1), activation = "sigmoid")(add7)
    model = tf.keras.Model(inputs = input_layer, outputs = outputs)
    
    return model