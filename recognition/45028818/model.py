
import tensorflow as tf
import matplotlib as plt

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
# Tensorflow addons for instance normalization as described in Improved Unet Paper
import tensorflow_addons as tfa


# Constants
INSTANCE_NORMALIZATION_ARGS = dict(
    axis=3,                             # Axis being normalised
    center=True,                        # Signal to add beta as an offset to the normalised tensor
    scale=True,                         # Signal to multiply by gamma
    beta_initializer='random_uniform',  
    gamma_initializer='random_uniform') 

LEAKY_ALPHA = 0.01


def context_module(input, out_filter):
    # First Convolution block
    c1 = Conv2D(filters=out_filter, kernel_size=(3,3), padding='same')(input)
    c2 = tfa.layers.InstanceNormalization(**INSTANCE_NORMALIZATION_ARGS)(c1)
    c3 = LeakyReLU(alpha=LEAKY_ALPHA)(c2)
    
    # DropOut
    c4 = Dropout(0.3)(c3)
    
    # Secound Convolution block
    c5 = Conv2D(filters=out_filter, kernel_size=(3,3), padding='same')(c4)
    c6 = tfa.layers.InstanceNormalization(**INSTANCE_NORMALIZATION_ARGS)(c5)
    c7 = LeakyReLU(alpha=LEAKY_ALPHA)(c6)
    
    # Preactivation residual add
    c8 = Add()([input,c7])
    
    return c8

# Module that recombines the features following concatenation and reduces the number of feature maps for memory
def localization_module(input, out_filter):
    # First Convolution block
    l1 = Conv2D(filters=out_filter, kernel_size=(3,3), padding='same')(input)
    l2 = tfa.layers.InstanceNormalization(**INSTANCE_NORMALIZATION_ARGS)(l1)
    l3 = LeakyReLU(alpha=LEAKY_ALPHA)(l2)
    
    # Secound Convolution block, of shape (1x1x1)
    l4 = Conv2D(filters=out_filter, kernel_size=(1,1), padding='same')(l3)
    l5 = tfa.layers.InstanceNormalization(**INSTANCE_NORMALIZATION_ARGS)(l4)
    l6 = LeakyReLU(alpha=LEAKY_ALPHA)(l5)
    
    return l6

# Upsamples features from a lower 'level' of the UNet to a higher spatial information
def upsampling_module(input, out_filter):
    # Upsample 
    u1 = UpSampling2D(size=(2, 2))(input)
    
    # Convolutional block
    u2 = Conv2D(filters=out_filter, kernel_size=(3,3), padding='same')(u1)
    u3 = tfa.layers.InstanceNormalization(**INSTANCE_NORMALIZATION_ARGS)(u2)
    u4 = LeakyReLU(alpha=LEAKY_ALPHA)(u3)
    
    return u4

# Connects context_modueles to reduce the resolution of the feature maps and allow for more features while aggregating
def context_connector(input, out_filter):
    cc1 = Conv2D(filters=out_filter, kernel_size=(3,3), strides=2, padding='same')(input)
    cc2 = tfa.layers.InstanceNormalization(**INSTANCE_NORMALIZATION_ARGS)(cc1)
    cc3 = LeakyReLU(alpha=LEAKY_ALPHA)(cc2)
    return cc3

def improved_unet(input_size = (512,512,3)):
    input = Input(shape=input_size)
    
    # Context Pathway
    # Layer 1
    x1 = Conv2D(filters=16, kernel_size=(3,3), padding='same')(input)
    x2 = tfa.layers.InstanceNormalization(**INSTANCE_NORMALIZATION_ARGS)(x1)
    x3 = LeakyReLU(alpha=LEAKY_ALPHA)(x2) 
    x4 = context_module(x3, 16)
    
    # Layer 2
    x5 = context_connector(x4, 32)
    x6 = context_module(x5, 32)
    
    # Layer 3
    x7 = context_connector(x5, 64)
    x8 = context_module(x7, 64)
    
    # Layer 4
    x9 = context_connector(x8, 128)
    x10 = context_module(x9, 128)
    
    # Layer 5.1
    x11 = context_connector(x10, 256)
    x12 = context_module(x11, 256)
    
    # Begin Localization Pathway
    # Layer 5.2
    x13 = upsampling_module(x12, 128)
    
    # Layer 4
    x14 = Concatenate()([x10, x13])
    x15 = localization_module(x14, 128)
    x16 = upsampling_module(x15, 64)
    
    # Layer 3
    x17 = Concatenate()([x8, x16])
    x18 = localization_module(x17, 64) # Segmentation 1 from here
    x19 = upsampling_module(x18, 32)
    
    # Layer 3: Segmentation
    seg1 = Activation('sigmoid')(x18)
    seg1 = upsampling_module(seg1, 32)
    
    # Layer 2
    x20 = Concatenate()([x6, x19])
    x21 = localization_module(x20, 32) # Segmentation 2 from here
    x22 = upsampling_module(x21, 16)
    
    # Layer 2: Segmentation
    seg2 = Activation('sigmoid')(x21)
    seg3 = Add()([seg1,seg2])
    seg3 = upsampling_module(seg3, 32)
    
    # Layer 1
    x23 = Concatenate()([x4, x22])
    x24 = Conv2D(filters=32, kernel_size=(3,3), padding='same')(x23)
    x25 = tfa.layers.InstanceNormalization(**INSTANCE_NORMALIZATION_ARGS)(x24)
    x26 = LeakyReLU(alpha=LEAKY_ALPHA)(x25) 
    
    # Layer 1: Segmentation
    seg4 = Activation('sigmoid')(x26)
    segFinal = Add()([seg3,seg4])
    
    # Output
    output = Conv2D(filters=1, kernel_size=(1,1), activation='sigmoid', padding='same')(segFinal)
    
    uNet = Model(inputs=input, outputs=output)
    
    return uNet
    

