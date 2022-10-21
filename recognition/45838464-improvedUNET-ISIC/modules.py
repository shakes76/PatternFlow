import tensorflow as tf
from tensorflow.keras.layers import Conv2D,LeakyReLU,Dropout,UpSampling2D,Input, concatenate, Add
from tensorflow.keras import Model
import tensorflow_addons as tfa
import numpy as np

IMAGE_SIZE = 128 # latent space will have an encoding for every pixel
CHANNELS = 3 #RGB
BATCH_SIZE = 1
KERNEL_SIZE = 3
STRIDE = 2
FIRST_DEPTH = 16
DROPOUT = 0.3
activation_func = LeakyReLU(alpha=0.01)

def context_module(input, depth): 
    """
    From "Brain Tumor Segmentation and Radiomics
    Survival Prediction: Contribution to the BRATS
    2017 Challenge" -> "Each context module is in fact a pre-activation residual block [13] with two
    3x3x3 convolutional layers and a dropout layer (pdrop = 0.3) in between. 
    """
    block = tfa.layers.InstanceNormalization()(input)
    block = Conv2D(depth, KERNEL_SIZE, padding="same", activation=activation_func)(block)
    block = Dropout(DROPOUT)(block)
    block = tfa.layers.InstanceNormalization()(block)
    block = Conv2D(depth, KERNEL_SIZE, padding='same', activation=activation_func)(block)
    return block

def localization_module(input, depth):
    """
    "A localization module
    consists of a 3x3x3 convolution followed by a 1x1x1 convolution that halves the
    number of feature maps."
    """
    block = Conv2D(depth, KERNEL_SIZE, padding = 'same', activation=activation_func)(input)
    block = Conv2D(depth, (1, 1), padding = 'same', activation=activation_func)(block)
    return block

def encoding_layer(input, depth, stride):
    """
    Building block for the encoder network as decribed in the paper.
    """
    conv = Conv2D(depth, KERNEL_SIZE, padding = 'same',activation=activation_func, strides=stride)(input)
    contxt = context_module(conv, depth)
    add = Add()([conv, contxt])

    return add

def decoding_layer(input, add, depth):
    """
    Decoding building block as described in the paper.
    "... which is done by
    means of a simple upscale that repeats the feature voxels twice in each spatial
    dimension, followed by a 3x3x3 convolution that halves the number of feature
    maps"
    
    """
    block = UpSampling2D(size=(2, 2))(input)
    block = Conv2D(depth, KERNEL_SIZE, activation=activation_func, padding = 'same')(block)
    print(block.shape)
    print(add.shape)

    conc = concatenate([block, add])
    loc = localization_module(conc, depth)

    return loc

def improved_uNET():
    """
    Improved UNet Architecture build from block defined above.
    """
    
    input_layer = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS))
    
    ### Encoder ###
    sum1 = encoding_layer(input_layer, FIRST_DEPTH*2, stride=(1, 1))
    sum2 = encoding_layer(sum1, FIRST_DEPTH*2, stride=(2, 2))
    sum3 = encoding_layer(sum2, FIRST_DEPTH*(2**2), stride=(2, 2))
    sum4 = encoding_layer(sum3, FIRST_DEPTH*(2**3), stride=(2, 2))
    sum5 = encoding_layer(sum4, FIRST_DEPTH*(2**4), stride=(2, 2))

    ### Decoder ###
    loc1 = decoding_layer(sum5, sum4, FIRST_DEPTH*(2**3))
    
    loc2 = decoding_layer(loc1, sum3, FIRST_DEPTH*(2**2))
    
    seg1 = Conv2D(3, (1, 1), padding = 'same')(loc2)
    seg1 = UpSampling2D((2, 2))(seg1)

    loc3 = decoding_layer(loc2, sum2, FIRST_DEPTH*2)

    seg2 = Conv2D(3, (1, 1), padding = 'same')(loc3)
    seg2 = Add()([seg1, seg2])
    seg2 = UpSampling2D((2, 2))(seg2)

    lastup = UpSampling2D((2, 2))(loc3)
    lastup = Conv2D(FIRST_DEPTH, KERNEL_SIZE, padding = 'same', activation=activation_func)(lastup)
    lastconc = concatenate([lastup, sum1])

    lastconv = Conv2D(FIRST_DEPTH*2, KERNEL_SIZE, strides = (1, 1), padding = 'same')(lastconc)
    seg3 = Conv2D(3, (1, 1), padding = 'same')(lastconv)
    
    final_seg = Add()([seg2, seg3])

    # softmax (one hot encoded)
    output_layer = Conv2D(3, (1, 1), activation='sigmoid')(final_seg)
    model = Model(name="Improved uNET", inputs=input_layer, outputs=output_layer)
    
    return model
 

