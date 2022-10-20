
# Some machine may need the following import statement to import (with .python.)
# May because of some version issue, however in most of the machine, (with .python.) will have a wrong result
# from tensorflow.python.keras.layers import Conv2D, MaxPool2D, Dense, Concatenate, UpSampling2D, Input
# from tensorflow.python.keras.models import Model

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Concatenate
from tensorflow.keras.layers import UpSampling2D, Input, LeakyReLU, Add, Dropout
from tensorflow.keras.models import Model

import tensorflow_addons as tfa
from keras import backend as k


def DSC (y_true, y_pred):
    """Dice similarity coefficient function, this should achieve 0.8 on the test set when testing

    Args:
        y_true (numpy.ndarray): true mask of the data
        y_pred (numpy.ndarray): predict mask of the data

    Returns:
        int: return dice similarity coefficient value between two image
    """
    y_true_f = k.flatten(y_true)
    y_pred_f = k.flatten(y_pred) 
    
    intersection1 = k.sum(y_true_f*y_pred_f)
    coeff = (2.0 * intersection1) / (k.sum(k.square(y_true_f)) + k.sum(k.square(y_pred_f)))
    return coeff

def DSC_loss (y_true, y_pred):
    """ Loss function of dice similarity coefficient, 
        in other word this is for the not matching part

    Args:
        y_true (numpy.ndarray): true mask of the data
        y_pred (numpy.ndarray): predict mask of the data

    Returns:
        int: return the loss value of dice similarity coefficient, 
             which is (1 - dice similarity coefficient)
    """
    return 1 - DSC(y_true, y_pred)


#-- normal UNet --#

def down(x, filters):
    conv = Conv2D(filters, kernel_size=(3, 3), padding="same", strides=1, activation="relu")(x)
    conv = Conv2D(filters, kernel_size=(3, 3), padding="same", strides=1, activation="relu")(conv)
    conv = Concatenate()([x, conv])
    out = MaxPool2D((2, 2), (2, 2))(conv)
    return conv, out

def up(x, skip, filters):
    us = UpSampling2D((2, 2))(x)
    concat = Concatenate()([us, skip])
    conv = Conv2D(filters, kernel_size=(3, 3), padding="same", strides=1, activation="relu")(concat)
    out = Conv2D(filters, kernel_size=(3, 3), padding="same", strides=1, activation="relu")(conv)
    return out

def UNet():
    f = 16
    inputs=Input((256,256,3))
    p0 = inputs
    c1, p1 = down(p0, f) 
    c2, p2 = down(p1, f*2) 
    c3, p3 = down(p2, f*4) 
    c4, p4 = down(p3, f*8) 
    
    conv = Conv2D(f*16, kernel_size=(3, 3), padding="same", strides=1, activation="relu")(p4)
    conv = Conv2D(f*16, kernel_size=(3, 3), padding="same", strides=1, activation="relu")(conv)
    
    u1 = up(conv, c4, f*8) 
    u2 = up(u1, c3, f*4) 
    u3 = up(u2, c2, f*2) 
    u4 = up(u3, c1, f) 
    
    outputs = Conv2D(3, (1, 1), padding="same", activation="sigmoid")(u4)
    model = Model(inputs, outputs)
    return model


#-- improved UNet --#

act = LeakyReLU(alpha = 0.01)

def down_context_module(input, filters):
    """
        A context module consisting of two 3x3 convolutions with a dropout of 0.3 between them
    """
    conv1 = tfa.layers.InstanceNormalization()(input)
    conv1 = Conv2D(filters, (3, 3), padding = "same", activation = act)(conv1)
    dropout = Dropout(0.3) (conv1)
    conv2 = tfa.layers.InstanceNormalization()(dropout)
    conv2 = Conv2D(filters, (3, 3), padding = "same", activation = act)(conv2)
    return conv2

def down_imp(input, filters, stride):
    """
        encode module with  3x3 convolution
    """
    conv = Conv2D(filters, (3, 3), strides = stride, padding = "same")(input)
    conv_module = down_context_module(conv, filters)
    add = Add()([conv, conv_module])
    return add
	

def localization_module(input, filter1, filter2):
    """
        A localization module consists of a 3x3 convolution 
        and a 1x1 convolution that halves the number of features
    """
    conv1 = Conv2D(filter1, (3, 3), padding = "same", activation = act)(input)
    conv2 = Conv2D(filter2, (1, 1), padding = "same", activation = act)(conv1)
    return conv2

def up_imp(input, adds, filter1, filter2):
    """
        decode module with 3x3 convolution
    """
    # upsampling module
    up = UpSampling2D((2, 2))(input)
    up = Conv2D(filter1, (3, 3), padding = "same", activation = act)(up)
    concat = Concatenate()([up, adds])

    # localization module
    local = localization_module(concat, filter2, filter2)
    return local

def UNet_imp():
    
    inputs = Input(shape=(256, 256, 3))
    
    # Encoder (down)
    add1 = down_imp(inputs, 32, (1, 1)) # only first one with strides = (1, 1)
    add2 = down_imp(add1, 32, (2, 2))
    add3 = down_imp(add2, 64, (2, 2))
    add4 = down_imp(add3, 128, (2, 2))
    add5 = down_imp(add4, 256, (2, 2))
    

	# Decoder (up)
    local1 = up_imp(add5, add4, 128, 128)

    local2 = up_imp(local1, add3, 64, 64)
    seg1 = Conv2D(3, (1, 1), padding = "same")(local2)
    seg1 = UpSampling2D(size = (2, 2))(seg1)

    local3 = up_imp(local2, add2, 32, 32)
    seg2 = Conv2D(3, (1, 1), padding = "same")(local3)
    seg_add_1 = Add()([seg1, seg2])
    seg_add_1 = UpSampling2D(size = (2, 2))(seg_add_1)

    up = UpSampling2D((2, 2))(local3) # up-sampling
    up = Conv2D(16, (3, 3), padding = "same", activation = act)(up)

    concat4 = Concatenate()([up, add1])
    conv4 = Conv2D(32, (3, 3), padding = "same")(concat4)
    seg3 = Conv2D(3, (1, 1), padding = "same")(conv4)
    seg_add_2 = Add()([seg_add_1, seg3])
    
    outputs = Conv2D(3, (1, 1), activation = "sigmoid")(seg_add_2)
    model = tf.keras.Model(inputs = inputs, outputs = outputs)
    
    return model

