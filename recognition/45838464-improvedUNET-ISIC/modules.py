import tensorflow as tf
from tensorflow.keras.layers import Conv2D,LeakyReLU,Dropout,UpSampling2D,Input, BatchNormalization, ReLU, concatenate, Add
from tensorflow.keras import Model
import numpy as np

IMAGE_SIZE = 256 # latent space will have an encoding for every pixel
CHANNELS = 3 #RGB
BATCH_SIZE = 1
KERNEL_SIZE = 3
STRIDE = 2
FIRST_DEPTH = 16
DROPOUT = 0.3

def context_module(input, depth): 
    """
    From "Brain Tumor Segmentation and Radiomics
Survival Prediction: Contribution to the BRATS
2017 Challenge" -> "Each context module is in fact a pre-activation residual block [13] with two
3x3x3 convolutional layers and a dropout layer (pdrop = 0.3) in between. 
    """
    block = BatchNormalization()(input)
    block = ReLU()(block)
    block = Conv2D(depth, KERNEL_SIZE, padding="same")(block)
    block = Dropout(DROPOUT)(block)
    block = BatchNormalization()(block)
    block = ReLU()(block)
    block = Conv2D(depth, KERNEL_SIZE, padding='same')
    return block

def element_wise_sum(down_sample, context):
    """
    Maybe use Add frunction from keras.layer
    """
    pass

def upsampling_module(input, depth):
    """
    "... which is done by
means of a simple upscale that repeats the feature voxels twice in each spatial
dimension, followed by a 3x3x3 convolution that halves the number of feature
maps"
    
    """
    block = UpSampling2D(size=(2, 2))(input)
    block = Conv2D(depth, KERNEL_SIZE, activation=LeakyReLU(alpha=0.01), padding = 'same')(block)
    return block

def localization_module(input, depth):
    """
    A localization module
consists of a 3x3x3 convolution followed by a 1x1x1 convolution that halves the
number of feature maps
    """
    block = Conv2D(depth, KERNEL_SIZE, activation=LeakyReLU(alpha=0.01), padding = 'same')(input)
    block = BatchNormalization()(block)
    block = Conv2D(depth, 1, activation=LeakyReLU(alpha=0.01), padding = 'same')(block)
    block = BatchNormalization()(block)
    return block

def segmentation(input):
    
    return Conv2D(FIRST_DEPTH, (1, 1), (1, 1))(input)


def improved_uNET(input):
    
    input_layer = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS), batch_size=BATCH_SIZE)
    
    ### Encoder ###
    conv1 = Conv2D(FIRST_DEPTH, KERNEL_SIZE, padding = 'same')(input_layer)
    context1 = context_module(conv1, FIRST_DEPTH)
    sum1 = Add()(conv1, context1)

    conv2 = Conv2D(FIRST_DEPTH*2, KERNEL_SIZE, STRIDE, padding = 'same')(sum1)
    context2 = context_module(conv2, FIRST_DEPTH*2)
    sum2 = Add()(conv2, context2)

    conv3 = Conv2D(FIRST_DEPTH*(2**2), KERNEL_SIZE, STRIDE, padding = 'same')(sum2)
    context3 = context_module(conv3, FIRST_DEPTH*(2**2))
    sum3 = Add()(conv3, context3)

    conv4 = Conv2D(FIRST_DEPTH*(2**3), KERNEL_SIZE, STRIDE,  padding = 'same')(sum3)
    context4 = context_module(conv4, FIRST_DEPTH*(2**3))
    sum4 = Add()(conv4, context4)

    conv5 = Conv2D(FIRST_DEPTH*(2**4), KERNEL_SIZE, STRIDE, padding = 'same')(sum4)
    context5 = context_module(conv5, FIRST_DEPTH*(2**4))
    sum5 = Add()(conv5, context5)

    ### Decoder ###
    upsamp1 = upsampling_module(sum5, FIRST_DEPTH*(2**3))
    conc1 = concatenate([sum4, upsamp1])
    loc1 = localization_module(conc1, FIRST_DEPTH*(2**3))
    
    upsamp2 = upsampling_module(loc1, FIRST_DEPTH*(2**2))
    conc2 = concatenate([sum3, upsamp2])
    loc2 = localization_module(conc2, FIRST_DEPTH*(2**2))

    seg1 = segmentation(loc2)
    seg1 = UpSampling2D()((2,2))(seg1)

    upsamp3 = upsampling_module(loc2, FIRST_DEPTH*2)
    conc3 = concatenate([sum2, upsamp3])
    loc3 = localization_module(conc3, FIRST_DEPTH*2)

    seg2 = segmentation(loc3)
    seg2 = Add()(seg1, seg2)
    seg2 = UpSampling2D((2,2))(seg2)

    upsamp4 = upsampling_module(loc3, FIRST_DEPTH)
    conc4 = concatenate([sum1, upsamp4])

    lastconv = Conv2D(FIRST_DEPTH*2, KERNEL_SIZE, stride = (1, 1), activation=LeakyReLU(alpha=0.01), padding = 'same')(conc4)
    seg3 = segmentation(lastconv, FIRST_DEPTH)
    
    final_seg = Add(seg2, seg3)

    # softmax 
    output_layer = Conv2D(2,KERNEL_SIZE, activation='softmax', padding='same')(final_seg)
    model = Model(name="Improved uNET", inputs=input_layer, outputs=output_layer)
    return model




    

