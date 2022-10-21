#!/usr/bin/env python
# coding: utf-8

# In[6]:


import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import Input, LeakyReLU, Conv2D as conv2D, Dropout, Add, UpSampling2D, concatenate

def improved_unet(W, H):
    input_size = (W, H, 3)
    inputs = Input(input_size)
    # Down sampling
    conv1 = conv2D(16, 3, activation = tf.keras.layers.LeakyReLU(alpha=0.01), padding = 'same')(inputs)
    cont1 = conv2D(16, 3, activation = LeakyReLU(alpha=0.01), padding = 'same')(conv1)
    cont1 = Dropout(0.5)(cont1)
    cont1 = conv2D(16, 3, activation = LeakyReLU(alpha=0.01), padding = 'same')(cont1)
    conc1 = Add()([conv1, cont1]) #W*H*16

    conv2 = conv2D(32, 3, strides = (2,2), activation = LeakyReLU(alpha=0.01), padding = 'same')(conc1)
    cont2 = conv2D(32, 3, activation = LeakyReLU(alpha=0.01), padding = 'same')(conv2)
    cont2 = Dropout(0.5)(cont2)
    cont2 = conv2D(32, 3, activation = LeakyReLU(alpha=0.01), padding = 'same')(cont2)
    conc2 = Add()([conv2, cont2]) #W/2*H/2*32

    conv3 = conv2D(64, 3, strides = (2,2), activation = LeakyReLU(alpha=0.01), padding = 'same')(conc2)
    cont3 = conv2D(64, 3, activation = LeakyReLU(alpha=0.01), padding = 'same')(conv3)
    cont3 = Dropout(0.4)(cont3)
    cont3 = conv2D(64, 3, activation = LeakyReLU(alpha=0.01), padding = 'same')(cont3)
    conc3 = Add()([conv3, cont3]) #W/4*H/4*64

    conv4 = conv2D(128, 3, strides = (2,2), activation = LeakyReLU(alpha=0.01), padding = 'same')(conc3)
    cont4 = conv2D(128, 3, activation = LeakyReLU(alpha=0.01), padding = 'same')(conv4)
    cont4 = Dropout(0.4)(cont4)
    cont4 = conv2D(128, 3, activation = LeakyReLU(alpha=0.01), padding = 'same')(cont4)
    conc4 = Add()([conv4, cont4]) #W/8*H/8*128
 
    conv5 = conv2D(256, 3, strides = (2,2), activation = LeakyReLU(alpha=0.01), padding = 'same')(conc4)
    cont5 = conv2D(256, 3, activation = LeakyReLU(alpha=0.01), padding = 'same')(conv5)
    cont5 = Dropout(0.4)(cont5)
    cont5 = conv2D(256, 3, activation = LeakyReLU(alpha=0.01), padding = 'same')(cont5)
    conc5 = Add()([conv5, cont5]) #W/16*H/16*256
    
    uconv5 = UpSampling2D(size = (2,2))(conc5)
    uconv5 = conv2D(128, 3, activation = LeakyReLU(alpha=0.01), padding = 'same')(uconv5)
     
    # Up sampling
    uconv4 = concatenate([uconv5, conc4])
    lconv4 = conv2D(128, 3, activation = LeakyReLU(alpha=0.01), padding = 'same')(uconv4)
    lconv4 = conv2D(128, 1, activation = LeakyReLU(alpha=0.01), padding = 'same')(lconv4)
    uconv4 = UpSampling2D(size = (2,2))(uconv4)
    uconv4 = Dropout(0.4)(uconv4)
    uconv4 = conv2D(64, 3, activation = LeakyReLU(alpha=0.01), padding = 'same')(uconv4)
     
    
    uconv3 = concatenate([uconv4, conc3])
    lconv3 = conv2D(64, 3, activation = LeakyReLU(alpha=0.01), padding = 'same')(uconv3)
    lconv3 = conv2D(64, 1, activation = LeakyReLU(alpha=0.01), padding = 'same')(lconv3)
    uconv3 = UpSampling2D(size = (2,2))(lconv3)
    uconv3 = conv2D(32, 3, activation = LeakyReLU(alpha=0.01), padding = 'same')(uconv3)

    uconv2 = concatenate([uconv3, conc2])   
    lconv2 = conv2D(32, 3, activation = LeakyReLU(alpha=0.01), padding = 'same')(uconv2)
    lconv2 = conv2D(32, 1, activation = LeakyReLU(alpha=0.01), padding = 'same')(lconv2)
    
   
    uconv2 = UpSampling2D(size = (2,2))(uconv2)
    uconv2 = conv2D(16, 3, activation = LeakyReLU(alpha=0.01), padding = 'same')(uconv2)
    
    uconv1 = concatenate([uconv2, conc1])
    uconv1 = conv2D(32, 3, activation = LeakyReLU(alpha=0.01), padding = 'same')(uconv1)
    
    # Segementation layers
    sconv3 = conv2D(1, 3, activation = LeakyReLU(alpha=0.01), padding = 'same')(lconv3)
    sconv3 = UpSampling2D(size = (2,2))(sconv3) 
    sconv2 = conv2D(1, 3, activation = LeakyReLU(alpha=0.01), padding = 'same')(lconv2) 
    usconv2 = Add()([sconv2, sconv3])
    usconv2 = UpSampling2D(size = (2,2))(usconv2) 
    sconv1 = conv2D(1, 3, activation = LeakyReLU(alpha=0.01), padding = 'same')(uconv1) 
    sconv1 = Add()([sconv1, usconv2])
    output = conv2D(1, 1, activation = 'sigmoid')(sconv1)
    model = Model(inputs = inputs, outputs = output)
    return model


# In[ ]:




