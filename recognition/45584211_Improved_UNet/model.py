import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, Dropout, Conv2DTranspose, Input, concatenate, UpSampling2D, LeakyReLU, Softmax
from tensorflow.python.keras.engine.input_layer import InputLayer

import tensorflow_addons as tfa
from tensorflow_addons.layers import InstanceNormalization

FILTER_NUM = 16
DROP = 0.3
STRIDE = 2
KER_SIZE = 3
PADDING = 'same'


def contract(input, filter_mul):        
    conv = Conv2D(filter_mul*FILTER_NUM, padding=PADDING, strides=STRIDE, kernel_size=KER_SIZE)(input)
    cont = Conv2D(filter_mul*FILTER_NUM, padding=PADDING, strides=STRIDE, kernel_size=KER_SIZE)(conv_1)
    cont = Dropout(DROP)(cont)
    concat = concatenate([conv, cont])
    return concat

def expand(input, concat, filter_mul):  
    concat_n = concatenate([input, concat])      
    conv = Conv2D(filter_mul*FILTER_NUM, padding=PADDING, kernel_size=KER_SIZE)(concat_n)
    conv = Conv2D(filter_mul*FILTER_NUM, padding=PADDING, kernel_size=1)(conv)
    up_samp = UpSampling2D()(conv)
    return up_samp
    
    
def improv_unet():    
    input = InputLayer(input_shape=(256, 256, 1))
    layers = {input}
    for i in range(1, 5):
        layers[i + 1] = contract(input, 2 ** (i + 1))

    up_samp = UpSampling2D()(layers[-1])

    for i in range(3, 0, -1):
        layers[8 - i] = contract(input, layers[6 - i], 2 ** (i))


    concat_n = concatenate([input, layers[0]])      
    layers[8] = Conv2D(filter_mul, padding=PADDING, kernel_size=KER_SIZE)(concat_n)

    return Model(layers[0], layers[-1])







