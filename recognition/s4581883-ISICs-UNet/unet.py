import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras
from tensorflow import keras
from keras import layers, preprocessing
from tensorflow.keras.layers import Conv2D, UpSampling2D, LeakyReLU, Dropout, BatchNormalization
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

def down_sampling(init, filter, kernel_size = (3,3), padding = "same", strides = 1):
    conv = Conv2D(filter, kernel_size, padding = padding, strides = strides) (init)
    conv = BatchNormalization() (conv)
    conv = LeakyReLU(0.01) (conv)
    conv = Dropout(0.3) (conv)
    conv = Conv2D(filter, kernel_size, padding = padding, strides = strides) (conv)
    conv = BatchNormalization() (conv)
    conv = LeakyReLU(0.01) (conv)
    #pool = MaxPool2D((2,2), (2,2)) (conv)
    return conv #, pool

def upsampling(init, filters, kernel_size = (3,3), padding = "same", strides = 1):
    upsample = UpSampling2D((2,2)) (init)
    upsample = Conv2D(filters, kernel_size, padding, strides = strides) (upsample)
    upsample = BatchNormalization() (upsample)
    upsample = LeakyReLU(0.01) (upsample)
    return upsample

def localize(init, filters, kernel_size = (3,3), padding = "same", strides = 1):
    local = Conv2D(filters, kernel_size, padding, strides = strides) (init)
    local = BatchNormalization() (local)
    local = LeakyReLU(0.01) (local)
    local = Conv2D(filters, kernel_size = (1,1), padding = padding) (init)
    local = BatchNormalization() (local)
    local = LeakyReLU(0.01) (local)
    return local
