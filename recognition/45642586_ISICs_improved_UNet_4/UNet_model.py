"""
Improved_UNet for ISIC2018 data set.

COMP3710 Project:
    Question 4: Segment the ISICs data set with the Improved UNet [1] with all labels having a minimum Dice similarity
                coefficient of 0.8 on the test set. [Normal Difficulty]


@author: Xiao Sun
@Student Id: 45642586
"""

# This is Improved UNet model module.

import os
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras import layers, models, Input, Model
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD
import tensorflow as tf




# layers

def UNet_context_module(filters, inp, layer_name):
    # Each context_module consists of two 3x3 conv layers and a dropout(0.3) in between.
    
    x1 = layers.Conv2D(filters, kernel_size =3, padding = 'same')(inp)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.LeakyReLU(alpha=0.01)(x1)
    x1 = layers.Dropout(.3)(x1)
    x2 = layers.Conv2D(filters, kernel_size =3, padding = 'same')(x1)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.LeakyReLU(alpha=0.01)(x2)

    return x2
    
def UNet_upsampling_module(filters, inp):
    # ...It is like a layer that combines the UpSampling2D and Conv2D layers into one layer. 
    
    # what twice means in paper (Answer from Piazza: kernel size 2 by 2)?
    x1 = layers.UpSampling2D(size=(2,2))(inp)
    x2 = layers.Conv2D(filters, kernel_size =3, padding = 'same')(x1)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.LeakyReLU(alpha=0.01)(x2)
    
    return x2
    
    
def UNet_localization_module(filters, inp):
    # A localization module consists of a 3x3x3 convolution followed by a 1x1x1 convolution that halves the
    # number of feature maps.
    
    x1 = layers.Conv2D(filters, kernel_size =3, padding = 'same')(inp)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.LeakyReLU(alpha=0.01)(x1)
    
    x1 = layers.Dropout(.3)(x1)
    x2 = layers.Conv2D(filters, kernel_size =1, padding = 'same')(x1)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.LeakyReLU(alpha=0.01)(x2)
    
    return x2
    
    
# Build networks

def Improved_UNet_model(filters=16, input_layer = Input((256,256,3))):
    
    
    # block 1:
    block1_x1 = layers.Conv2D(filters, kernel_size =3, padding = 'same')(input_layer)
    #block1_x1 = layers.BatchNormalization()(block1_x1)
    block1_x1 = layers.LeakyReLU(alpha=0.01)(block1_x1)
    #block1_x1 = layers.Dropout(0.3)(block1_x1)
    
    block1_x2 = UNet_context_module(filters, block1_x1, "context_module1")
    
    output_b1 = layers.Add()([block1_x1, block1_x2])
    
    
    # block 2:
    block2_x1 = layers.Conv2D(filters*2, kernel_size =3, strides = 2, padding = 'same')(output_b1)
    #block2_x1 = layers.BatchNormalization()(block2_x1)
    block2_x1 = layers.LeakyReLU(alpha=0.01)(block2_x1)
    #block2_x1 = layers.Dropout(0.3)(block2_x1)
    
    block2_x2 = UNet_context_module(filters*2, block2_x1, "context_module2")
    
    output_b2 = layers.Add()([block2_x1, block2_x2])
    
    
    # block 3:
    block3_x1 = layers.Conv2D(filters*4, kernel_size =3, strides = 2, padding = 'same')(output_b2)
    #block3_x1 = layers.BatchNormalization()(block3_x1)
    block3_x1 = layers.LeakyReLU(alpha=0.01)(block3_x1)
    #block3_x1 = layers.Dropout(0.3)(block3_x1)
    
    block3_x2 = UNet_context_module(filters*4, block3_x1, "context_module3")
    
    output_b3 = layers.Add()([block3_x1, block3_x2])
    
    
    # block 4:
    block4_x1 = layers.Conv2D(filters*8, kernel_size =3, strides = 2, padding = 'same')(output_b3)
    #block4_x1 = layers.BatchNormalization()(block4_x1)
    block4_x1 = layers.LeakyReLU(alpha=0.01)(block4_x1)
    #block4_x1 = layers.Dropout(0.3)(block4_x1)
    
    block4_x2 = UNet_context_module(filters*8, block4_x1, "context_module4")
    
    output_b4 = layers.Add()([block4_x1, block4_x2])
    
    
    # block 5:
    block5_x1 = layers.Conv2D(filters*16, kernel_size =3, strides = 2, padding = 'same')(output_b4)
    #block5_x1 = layers.BatchNormalization()(block5_x1)
    block5_x1 = layers.LeakyReLU(alpha=0.01)(block5_x1)
    #block5_x1 = layers.Dropout(0.3)(block5_x1)
    
    block5_x2 = UNet_context_module(filters*16, block5_x1, "context_module5")
    
    output_b5 = layers.Add()([block5_x1, block5_x2])
    
    
    # up_block 6:
    block6_x1 = UNet_upsampling_module(filters*8, output_b5)
    # connection
    output_b6 = layers.concatenate([output_b4, block6_x1])
    
    
    # up_block 7:
    block7_x1 = UNet_localization_module(filters*8, output_b6)
    block7_x2 = UNet_upsampling_module(filters*4, block7_x1)
    # connection
    output_b7 = layers.concatenate([output_b3, block7_x2])
    
    
    # up_block 8:
    block8_x1 = UNet_localization_module(filters*4, output_b7)
    block8_x2 = UNet_upsampling_module(filters*2, block8_x1)
    # connection
    output_b8 = layers.concatenate([output_b2, block8_x2])
    
    
    # up_block 9:
    block9_x1 = UNet_localization_module(filters*2, output_b8)
    block9_x2 = UNet_upsampling_module(filters, block9_x1)
    # connection
    output_b9 = layers.concatenate([output_b1, block9_x2])
    
    # upscale
    segmentation_1 = layers.Conv2D(1, kernel_size =3, padding = 'same')(block7_x1)
    segmentation_1 = layers.UpSampling2D(size=(8,8))(segmentation_1)
    segmentation_2 = layers.Conv2D(1, kernel_size =3, padding = 'same')(block8_x1)
    segmentation_2 = layers.UpSampling2D(size=(4,4))(segmentation_2)
    final_block_output = layers.Conv2D(1, kernel_size =3, padding = 'same')(output_b9)
    
    # combine different level's output as the final output.
    output = layers.Add()([segmentation_1, segmentation_2, final_block_output])
    #output = layers.BatchNormalization()(output)
    output = layers.Activation('sigmoid')(output)
    
    improved_unet_model = Model(input_layer, output, name="improved_unet_model")
    improved_unet_model.summary()
    
    return improved_unet_model
    
