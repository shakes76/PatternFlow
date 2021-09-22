
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, InputLayer, Flatten, Dense, Reshape, BatchNormalization, Dropout

def encoder():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(256,256,1)))

def decoder():
    model = tf.keras.models.Sequential()

def vector_quantizer():
    #discrete instead of continuous
    pass