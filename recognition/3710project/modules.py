
# Some machine may need the following import statement to import (with .python.)
# May because of some version issue, however in most of the machine, (with .python.) will have a wrong result
# from tensorflow.python.keras.layers import Conv2D, MaxPool2D, Dense, Concatenate, UpSampling2D, Input
# from tensorflow.python.keras.models import Model

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Concatenate, UpSampling2D, Input
from tensorflow.keras.models import Model
from keras import backend as k


def DSC (y_true, y_pred):

    y_true_f = k.flatten(y_true)
    y_pred_f = k.flatten(y_pred) 
    
    intersection1 = k.sum(y_true_f*y_pred_f)
    coeff = (2.0 * intersection1) / (k.sum(k.square(y_true_f)) + k.sum(k.square(y_pred_f)))
    return coeff

def DSC_loss (y_true, y_pred):

    return 1 - DSC(y_true, y_pred)

def down(x, filters, kernel_size=(3, 3), padding="same", strides=1):

    c = Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
    c = Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    p = MaxPool2D((2, 2), (2, 2))(c)
    return c, p

def up(x, skip, filters, kernel_size=(3, 3), padding="same", strides=1):

    us = UpSampling2D((2, 2))(x)
    concat = Concatenate()([us, skip])
    c = Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(concat)
    c = Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    return c

def UNet():

    f = 16
    inputs=Input((256,256,3))
    p0 = inputs
    c1, p1 = down(p0, f) 
    c2, p2 = down(p1, f*2) 
    c3, p3 = down(p2, f*4) 
    c4, p4 = down(p3, f*8) 
    
    c = Conv2D(f*16, kernel_size=(3, 3), padding="same", strides=1, activation="relu")(p4)
    c = Conv2D(f*16, kernel_size=(3, 3), padding="same", strides=1, activation="relu")(c)
    
    u1 = up(c, c4, f*8) 
    u2 = up(u1, c3, f*4) 
    u3 = up(u2, c2, f*2) 
    u4 = up(u3, c1, f) 
    
    outputs = Conv2D(3, (1, 1), padding="same", activation="sigmoid")(u4)
    model = Model(inputs, outputs)
    return model