import numpy as np
import tensorflow 
from tensorflow import keras
from tensorflow.keras import layers, Sequential,Model
from tensorflow.keras.layers import Conv2D,UpSampling2D,concatenate,MaxPooling2D,Conv2DTranspose,LeakyReLU




def MODEL_implementation():
    inputs= tensorflow.keras.Input(shape=(256, 256, 3))
    conv1 = Conv2D(64, 3, activation=LeakyReLU(0.05), padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation=LeakyReLU(0.05), padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation=LeakyReLU(0.05), padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation=LeakyReLU(0.05), padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation=LeakyReLU(0.05), padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation=LeakyReLU(0.05), padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation=LeakyReLU(0.05), padding='same')(pool3)
    conv4 = Conv2D(512, 3, activation=LeakyReLU(0.05), padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(1024, 3, activation=LeakyReLU(0.05), padding='same')(pool4)
    conv5 = Conv2D(1024, 3, activation=LeakyReLU(0.05), padding='same')(conv5)

    up6 = Conv2D(512, 2, activation=LeakyReLU(0.05), padding='same')(UpSampling2D(size=(2, 2))(conv5))
    merge6 = concatenate([conv4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation=LeakyReLU(0.05), padding='same')(merge6)
    conv6 = Conv2D(512, 3, activation=LeakyReLU(0.05), padding='same')(conv6)

    up7 = Conv2D(256, 2, activation=LeakyReLU(0.05), padding='same')(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation=LeakyReLU(0.05), padding='same')(merge7)
    conv7 = Conv2D(256, 3, activation=LeakyReLU(0.05), padding='same')(conv7)

    up8 = Conv2D(128, 2, activation=LeakyReLU(0.05), padding='same')(UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation=LeakyReLU(0.05), padding='same')(merge8)
    conv8 = Conv2D(128, 3, activation=LeakyReLU(0.05), padding='same')(conv8)

    up9 = Conv2D(64, 2, activation=LeakyReLU(0.05), padding='same')(UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation=LeakyReLU(0.05), padding='same')(merge9)
    conv9 = Conv2D(64, 3, activation=LeakyReLU(0.05), padding='same')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
    model = Model(inputs=inputs, outputs=conv10)
    return model
