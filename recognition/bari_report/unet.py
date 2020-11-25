"""
Created on Oct 29, 2020

@author: s4542006, Md Abdul Bari
"""

from keras.models import Model
from keras.layers import Input, BatchNormalization, Activation, Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate


def get_conv_block(tensor, n_filters, size=3, padding='same', 
               initializer="he_normal"):
    """two consecutive convolutional layers followed by batch normalization
    
    Parameters:
        tensor [float] : multi-dimensional array having input image data
        n_filters (int) : number of channels in the convolutional block
        size (int) : size of the kernel used for convolution of the images
        padding (str) : type of padding to build convolutional blocks
        initializer (str) : type of kernel initializer to be used 
        
    Return: blocks in expansion path for the UNet architecture
    """
    x = Conv2D(filters=n_filters, kernel_size=(size, size), padding=padding, 
               kernel_initializer=initializer)(tensor)
    x = Activation("relu")(x)
    x = Conv2D(filters=n_filters, kernel_size=(size, size), padding=padding, 
               kernel_initializer=initializer)(x)
    x = Activation("relu")(x)
    return x


def get_deconv_block(tensor, residual, n_filters, size=2, padding='same', 
                 strides=(2, 2), initializer="he_normal"):
    """first create deconvolutional layer and joined with corresponding layers 
    in the contraction path and finally adds convolutional block.
    
    Parameters:
        tensor [float] : multi-dimensional array having input image data
        residual [float] : corresponding tensor in contration path
        n_filters (int) : number of channels in the convolutional block
        size (int) : size of the kernel and strides
        padding (str) : type of padding to build convolutional blocks
        strides (int, int) : number of cells for horizontal & vertical strides
        initializer (str) : type of kernel initializer to be used 
        
    Return: blocks in expansion path for the UNet architecture 
    """
    y = Conv2DTranspose(n_filters, kernel_size=(size, size), padding=padding, 
                        strides=strides, kernel_initializer=initializer)(tensor)
    y = concatenate([y, residual], axis=3)
    y = get_conv_block(y, n_filters)
    return y


def my_unet(n_filters=32, size=(2,2), drop_out=0.10):
    """build and return UNet model contraction and expansio path
    
    Parameters:
        n_filters (int): number of filters for each convolutional block
        size (int, int) : size of pool and strides
        drop_out (float) : proportion of units to drop from proving output
            to limit overfitting the training inputs.
            
    Return: UNet model
    """
    #input 
    img_height = img_width = 256
    input_layer = Input(shape=(img_height, img_width, 1), name='image_input')
    
    # contraction path
    conv1 = get_conv_block(input_layer, n_filters=n_filters*1)
    conv1_max = MaxPooling2D(pool_size=size, strides=size)(conv1)
    conv1_out = Dropout(drop_out*1)(conv1_max)
    
    conv2 = get_conv_block(conv1_out, n_filters=n_filters*2)
    conv2_max = MaxPooling2D(pool_size=size, strides=size)(conv2)
    conv2_out = Dropout(drop_out*2)(conv2_max)
    
    conv3 = get_conv_block(conv2_out, n_filters=n_filters*4)
    conv3_max = MaxPooling2D(pool_size=size, strides=size)(conv3)
    conv3_out = Dropout(drop_out*3)(conv3_max)
    
    conv4 = get_conv_block(conv3_out, n_filters=n_filters*8)
    conv4_max = MaxPooling2D(pool_size=size, strides=size)(conv4)
    conv4_out = Dropout(drop_out*4)(conv4_max)
    
    conv5 = get_conv_block(conv4_out, n_filters=n_filters*16)
    conv5_out = Dropout(drop_out*5)(conv5)
    
    # expansion path
    deconv6 = get_deconv_block(conv5_out, residual=conv4, n_filters=n_filters*8)
    deconv6 = Dropout(drop_out*4)(deconv6) 
    
    deconv7 = get_deconv_block(deconv6, residual=conv3, n_filters=n_filters*4)
    deconv7 = Dropout(drop_out*3)(deconv7)
    
    deconv8 = get_deconv_block(deconv7, residual=conv2, n_filters=n_filters*2)
    deconv8 = Dropout(drop_out*2)(deconv8)
    
    deconv9 = get_deconv_block(deconv8, residual=conv1, n_filters=n_filters*1)
    deconv9 = Dropout(drop_out*1)(deconv9)
    
    # output
    output_layer = Conv2D(1, kernel_size=(1, 1))(deconv9)
    output_layer = Activation('sigmoid')(output_layer)
    # create and return an instance of the UNET model
    model = Model(inputs=input_layer, outputs=output_layer, name='my_unet')
    return model


if __name__ == "__main__":
    print("This module provides utility functions training and testing UNET",
          "and is not meant to be executed on its own.")