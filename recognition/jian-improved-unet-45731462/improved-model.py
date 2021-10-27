"""
Model Architecture of the Improved UNEt

@author Jian Yang Lee
@email jianyang.lee@uqconnect.edu.au
"""

import tensorflow as tf
from tensorflow.python.eager.context import context

from tensorflow.python.keras.layers import Dropout, Input, Conv2D, MaxPooling2D, Conv2DTranspose, Concatenate
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
from tensorflow.python.keras.models import Model

def context_module(input, filters):
    """[summary]

    Args:
        input ([type]): [description]
        filters ([type]): [description]

    Returns:
        [type]: [description]
    """
    conv_one = Conv2D(filters, kernel_size=3, strides=(2, 2), padding="same", activation="relu")(input)
    drop_one = Dropout(0.3)(conv_one)
    conv_two = Conv2D(filters, kernel_size=3, strides=(2, 2), padding="same", activation="relu")(drop_one)

    return conv_two

def localisation_module(input, filters):
    """[summary]

    Args:
        input ([type]): [description]
        filters ([type]): [description]

    Returns:
        [type]: [description]
    """
    conv_one = Conv2D(filters, 3, padding="same")(input)



    return input





def model(height, width, input_channel):
    """
    [Add docstrings]
    """
    input = Input((height, width, input_channel))

    # encoding
    conv1 = Conv2D(16, kernel_size=3, padding="same", activation=LeakyReLU(alpha=0.1))(input)
    context1 = context_module(conv1, 16)

    conv2 = Conv2D(32, kernel_size=3, strides=(2, 2), padding="same", activation=LeakyReLU(alpha=0.1))(context1)
    context2 = context_module(conv2, 32)

    conv3 = Conv2D(64, kernel_size=3, strides=(2, 2), padding="same", activation=LeakyReLU(alpha=0.1))(context2)
    context3 = context_module(conv3, 64)

    conv4 = Conv2D(128, kernel_size=3, strides=(2, 2), padding="same", activation=LeakyReLU(alpha=0.1))(context3)
    context4 = context_module(conv4, 128)

    # bridge
    conv_bridge = Conv2D(filters=256, kernel_size=3, strides=(2, 2), activation=LeakyReLU(alpha=0.1))(context4)
    context_bridge = context_module(conv_bridge, 256)
    upsample_bridge = Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=2)(context_bridge)

    # decoding
    concat1 = Concatenate()([upsample_bridge, context4])






    # segmentation with filter size 2 (for channel) and kernel size of 1 (removes the height and width )

