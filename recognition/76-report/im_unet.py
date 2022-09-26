#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import tensorflow 
from tensorflow import keras
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input,BatchNormalization
from tensorflow.keras.layers import Dropout, Lambda
from tensorflow.keras.layers import Conv2D, Conv2DTranspose,UpSampling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import concatenate
#from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam

from dice import *

def unet(input_size = (256,256,3)):
    inputs = Input(input_size)
    #3x3 conv with 16 fliters
    c1 =  Conv2D(16, 3, activation='relu', kernel_initializer='he_normal',padding='same') (inputs)
    #context module
    c1 =  Conv2D(16, 3, activation='relu',kernel_initializer='he_normal',  padding='same') (c1)
    d1 = Dropout(0.3)(c1)
    c1 =  Conv2D(16, 3, activation='relu',kernel_initializer='he_normal',  padding='same') (d1)
    # add two parts above
    c1 = c1+c1
    #upsampling
    p2 = MaxPooling2D((2, 2)) (c1)
    c2 =  Conv2D(32, 3, activation='relu',kernel_initializer='he_normal',  padding='same') (p2)
    d2 = Dropout(0.3)(c2)
    c2 =  Conv2D(32, 3, activation='relu',kernel_initializer='he_normal',  padding='same') (d2)
    c2 = c2+c2

    p3 = MaxPooling2D((2, 2)) (c2)
    c3 =  Conv2D(64, 3, activation='relu',kernel_initializer='he_normal',  padding='same') (p3)
    d3 = Dropout(0.3)(c3)
    c3 =  Conv2D(64, 3, activation='relu',kernel_initializer='he_normal',  padding='same') (d3)
    c3 = c3+c3
    
    p4 = MaxPooling2D((2, 2)) (c3)
    c4 =  Conv2D(128, 3, activation='relu',kernel_initializer='he_normal',  padding='same') (p4)
    d4 = Dropout(0.4)(c4)
    c4 =  Conv2D(128, 3, activation='relu',kernel_initializer='he_normal',  padding='same') (d4)
    c4 = c4+c4
    
    p5 = MaxPooling2D((2, 2))(c4)
    c5 =  Conv2D(256, 3, activation='relu',kernel_initializer='he_normal',  padding='same') (p5)
    d5 = Dropout(0.4)(c5)
    c5 =  Conv2D(256, 3, activation='relu',kernel_initializer='he_normal',  padding='same') (d5)
    c5 = c5+c5


    u6 = UpSampling2D((2, 2)) (c5)

    #merge downsampling and up sampling 
    merge6 = concatenate([c4,u6], axis = 3)
    c7 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    c7 = Conv2D(128, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(c7)
    u7 = UpSampling2D((2, 2)) (c7)

    merge7 = concatenate([c3,u7], axis = 3)
    c8 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    c8 = Conv2D(64, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(c8)
    # segmentation layer and upscale in side branch
    s8 = Conv2D(1, (1, 1))(c8)    
    s8 = UpSampling2D((2, 2))(s8)
    #upsampling
    u8 = UpSampling2D((2, 2))(c8)

    merge8 = concatenate([c2,u8], axis = 3)
    c9 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    c9 = Conv2D(32, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(c9)
    s9 = Conv2D(1, (1, 1))(c9)
    s9 = s8+s9    
    s9 = UpSampling2D((2,2))(s9)
    u9 = UpSampling2D((2, 2)) (c9)

    merge9 = concatenate([c1,u9], axis = 3)
    c10 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    s10 = Conv2D(1, (1, 1))(c10)
    s10 = s9+s10
    s10 = UpSampling2D((2,2))(s10)
    outputs = Conv2D(1, (1, 1), activation = 'sigmoid')(c10)

    model = Model(inputs, outputs)

    model.compile(optimizer = Adam(lr = 5e-4,decay=10e-5), loss='binary_crossentropy', metrics=[dice_coef])
    #model.summary()


    return model

