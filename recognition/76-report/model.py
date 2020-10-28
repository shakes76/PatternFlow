#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
from tensorflow import keras
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input,BatchNormalization
from tensorflow.keras.layers import Dropout, Lambda
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import concatenate
#from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam



def unet(pretrained_weights = None,input_size = (256,256,3)):
    inputs = Input(input_size)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal',padding='same') (inputs)
    c1 = Dropout(0.3) (c1)
    c1 = Conv2D(16, (3, 3), activation='relu',kernel_initializer='he_normal',  padding='same') (c1)
    p1 = MaxPooling2D((2, 2)) (c1)
    #b1 = BatchNormalization()(p1)

    c2 = Conv2D(32, (3, 3), activation='relu',kernel_initializer='he_normal', padding='same') (p1)
    c2 = Dropout(0.4) (c2)
    c2 = Conv2D(32, (3, 3), activation='relu',kernel_initializer='he_normal', padding='same') (c2)
    p2 = MaxPooling2D((2, 2)) (c2)
    #b2 = BatchNormalization()(p2)

    c3 = Conv2D(64, (3, 3), activation='relu',kernel_initializer='he_normal', padding='same') (p2)
    c3 = Dropout(0.5) (c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c3)
    p3 = MaxPooling2D((2, 2)) (c3)
    #b3 = BatchNormalization()(p3)

    c4 = Conv2D(128, (3, 3), activation='relu',kernel_initializer='he_normal', padding='same') (p3)
    c4 = Dropout(0.4) (c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
    #b4 = BatchNormalization()(p4)

    c5 = Conv2D(256, (3, 3), activation='relu',kernel_initializer='he_normal',  padding='same') (p4)
    c5 = Dropout(0.4) (c5)
    c5 = Conv2D(256, (3, 3), activation='relu',kernel_initializer='he_normal',  padding='same') (c5)

    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), kernel_initializer='he_normal',padding='same') (c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u6)
    c6 = Dropout(0.4) (c6)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), kernel_initializer='he_normal',padding='same') (c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu',  kernel_initializer='he_normal',padding='same') (u7)
    c7 = Dropout(0.4) (c7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), kernel_initializer='he_normal',padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='elu',kernel_initializer='he_normal', padding='same') (u8)
    c8 = Dropout(0.4) (c8)
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), kernel_initializer='he_normal',padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='elu',kernel_initializer='he_normal', padding='same') (u9)
    c9 = Dropout(0.4) (c9)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal',padding='same') (c9)

    outputs = Conv2D(2, (1, 1), activation='softmax') (c9)

    model = Model(input = inputs, output = outputs)
    
    adam=Adam(lr=0.001, decay=0.9)

    model.compile(optimizer = adam, loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    #model.summary()

    return model

