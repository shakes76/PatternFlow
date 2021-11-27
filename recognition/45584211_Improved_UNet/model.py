import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Dropout, concatenate, UpSampling2D, Softmax
from tensorflow.python.keras.engine.input_layer import InputLayer

# convolution constants
FILTER_NUM = 16
DROP = 0.3
STRIDE = 2
KER_SIZE = 3
PADDING = 'same'


def contract(input, filter_mul):
    """
    One contraction step
    Args:
        input: array from previous step
        filter_mul: filter multiply number

    Returns: processed array

    """
    conv = Conv2D(filter_mul*FILTER_NUM, activation="relu", padding=PADDING, strides=STRIDE, kernel_size=KER_SIZE)(input)
    cont = Conv2D(filter_mul*FILTER_NUM, activation="relu", padding=PADDING, strides=STRIDE, kernel_size=KER_SIZE)(conv_1)
    cont = Dropout(DROP)(cont)
    concat = concatenate([conv, cont])
    return concat


def expand(input, concat, filter_mul):
    """
    One expansion step
    Args:
        input: array from previous step
        concat: array from previous steps
        filter_mul: filter multiply number

    Returns: processed array

    """
    concat_n = concatenate([input, concat])      
    conv = Conv2D(filter_mul*FILTER_NUM, activation="relu", padding=PADDING, kernel_size=KER_SIZE)(concat_n)
    conv = Conv2D(filter_mul*FILTER_NUM, activation="relu", padding=PADDING, kernel_size=1)(conv)
    up_samp = UpSampling2D()(conv)
    return up_samp
    
    
def improv_unet():
    """

    Returns: model for pattern recognition

    """
    input = InputLayer(input_shape=(256, 256, 1))
    layers = [input]

    # contraction
    for i in range(1, 5):
        layers[i + 1] = contract(layers[-1], 2 ** (i + 1))

    layers[-1] = UpSampling2D()(layers[-1])

    # expansion
    for i in range(2, 0, -1):
        layers[8 - i] = expand(layers[-1], layers[6 - i], 2 ** (i))

    # finishing up
    concat_n = concatenate([input, layers[0]])      
    layer_temp = Conv2D(FILTER_NUM, activation="relu", padding=PADDING, kernel_size=KER_SIZE)(concat_n)
    layers[8] = Conv2D(1, activation='sigmoid', kernel_size=1)(layer_temp)
    return tf.keras.Model(layers[0], layers[-1])







