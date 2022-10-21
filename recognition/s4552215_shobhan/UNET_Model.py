# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 15:59:52 2020

@author: s4552215
"""


from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add

def conv2d_block(input_tensor, n_filters, kernel_size = 3, batchnorm = True):
    
    # 1st layer
    layer = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),
              kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    if batchnorm:
        layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)
    
    # 2nd layer
    layer = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),
              kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    if batchnorm:
        layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)
    
    return layer

# To build  U-NET model
def unet_gen(inp_img, n_fil = 16, drop = 0.1, batch = True):
    
    # Contracting Path
    c1 = conv2d_block(inp_img, n_fil * 1, kernel_size = 3, batchnorm = batch)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(drop)(p1)
    
    c2 = conv2d_block(p1, n_fil * 2, kernel_size = 3, batchnorm = batch)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(drop)(p2)
    
    c3 = conv2d_block(p2, n_fil * 4, kernel_size = 3, batchnorm = batch)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(drop)(p3)
    
    c4 = conv2d_block(p3, n_fil * 8, kernel_size = 3, batchnorm = batch)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(drop)(p4)
    
    c5 = conv2d_block(p4, n_filters = n_fil * 16, kernel_size = 3, batchnorm = batch)
    
    # Expansive Path
    u6 = Conv2DTranspose(n_fil * 8, (3, 3), strides = (2, 2), padding = 'same')(c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(drop)(u6)
    c6 = conv2d_block(u6, n_fil * 8, kernel_size = 3, batchnorm = batch)
    
    u7 = Conv2DTranspose(n_fil * 4, (3, 3), strides = (2, 2), padding = 'same')(c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(drop)(u7)
    c7 = conv2d_block(u7, n_fil * 4, kernel_size = 3, batchnorm = batch)
    
    u8 = Conv2DTranspose(n_fil * 2, (3, 3), strides = (2, 2), padding = 'same')(c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(drop)(u8)
    c8 = conv2d_block(u8, n_fil * 2, kernel_size = 3, batchnorm = batch)
    
    u9 = Conv2DTranspose(n_fil * 1, (3, 3), strides = (2, 2), padding = 'same')(c8)
    u9 = concatenate([u9, c1])
    u9 = Dropout(drop)(u9)
    c9 = conv2d_block(u9, n_fil * 1, kernel_size = 3, batchnorm = batch)
    
    outputs = Conv2D(2, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[inp_img], outputs=[outputs])
    return model
