from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras import layers
import tensorflow_addons as tfa
import tensorflow as tf
import matplotlib.pyplot as plt

# context module function
def res_net_block(input_data, conv_size):
    x = tfa.layers.InstanceNormalization()(input_data)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.Conv2D(conv_size, kernel_size = 3, padding='same')(x) 
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.Conv2D(conv_size, kernel_size = 3, padding='same')(x) 
    return x

# add segmentation layers
def segmentation_layer(x):
    seg = tf.keras.layers.Conv2D(1, (1,1), activation = 'sigmoid')(x)
    return seg