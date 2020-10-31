"""
Created on Oct 29, 2020

@author: s4542006, Md Abdul Bari
"""

from keras.models import Model
from keras.layers import Input, BatchNormalization, Activation, Dropout, LeakyReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate


def conv_block(tensor, n_filters, size=3, padding='same', 
               initializer="he_normal", batch_norm=True, alpha=0.2):
    """two consecutive convolutional layers followed by batch normalization"""
    x = Conv2D(filters=n_filters, kernel_size=(size, size), padding=padding, 
               kernel_initializer=initializer)(tensor)
    x = LeakyReLU(alpha=alpha)(x)
    if batch_norm:
        x = BatchNormalization()(x)
    x = Conv2D(filters=n_filters, kernel_size=(size, size), padding=padding, 
               kernel_initializer=initializer)(x)
    x = LeakyReLU(alpha=alpha)(x)
    if batch_norm:
        x = BatchNormalization()(x)  
    return x

def deconv_block(tensor, residual, n_filters, size=3, padding='same', 
                 strides=(2, 2), initializer="he_normal"):
    """first create deconvolutional layer and joined with corresponding layers 
    in the contraction path and finally adds convolutional block."""
    y = Conv2DTranspose(n_filters, kernel_size=(size, size), padding=padding, 
                        strides=strides, kernel_initializer=initializer )(tensor)
    y = concatenate([y, residual], axis=3)
    y = conv_block(y, n_filters)
    return y

def build_unet(img_dim=256, n_filters=16, drop_out=0.10, batch_norm=True):
    # contraction 
    img_height = img_width = img_dim
    input_layer = Input(shape=(img_height, img_width, 1), name='image_input')
    
    conv1 = conv_block(input_layer, n_filters=n_filters*1)
    conv1_max = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv1_out = Dropout(drop_out*1)(conv1_max)
    
    conv2 = conv_block(conv1_out, n_filters=n_filters*2)
    conv2_max = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv2_out = Dropout(drop_out*2)(conv2_max)
    
    conv3 = conv_block(conv2_out, n_filters=n_filters*4)
    conv3_max = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv3_out = Dropout(drop_out*3)(conv3_max)
    
    conv4 = conv_block(conv3_out, n_filters=n_filters*8)
    conv4_max = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv4_out = Dropout(drop_out*4)(conv4_max)
    
    conv5 = conv_block(conv4_out, n_filters=n_filters*16)
    
    # expansion 
    deconv6 = deconv_block(conv5, residual=conv4, n_filters=n_filters*8)
    deconv6 = Dropout(drop_out*4)(deconv6) 
    
    deconv7 = deconv_block(deconv6, residual=conv3, n_filters=n_filters*4)
    deconv7 = Dropout(drop_out*3)(deconv7)
    
    deconv8 = deconv_block(deconv7, residual=conv2, n_filters=n_filters*2)
    deconv8 = Dropout(drop_out*2)(deconv8)
    
    deconv9 = deconv_block(deconv8, residual=conv1, n_filters=n_filters*1)
    deconv9 = Dropout(drop_out*1)(deconv9)
    
    # output
    output_layer = Conv2D(1, kernel_size=(1, 1))(deconv9)
    output_layer = BatchNormalization()(output_layer)
    output_layer = Activation('sigmoid')(output_layer)
    # create and return an instance of the UNET model
    model = Model(inputs=input_layer, outputs=output_layer, name='my_unet')
    return model