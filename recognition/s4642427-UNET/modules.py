from keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, BatchNormalization, Conv2DTranspose, Concatenate
from keras.layers import Input, Activation, BatchNormalization
from keras.layers.convolutional import Conv2D, Conv2DTranspose
import tensorflow as tf


def Unet():
  
    inputs = Input((256,256,3))

    """ Encoder """
    s1, x1 = contraction(inputs, 64)
    s2, x2 = contraction(x1, 128)
    s3, x3 = contraction(x2, 256)
    s4, x4 = contraction(x3, 512)

    """ Bridge """
    x5 = conv_block(x4, 1024)

    """ Decoder """
    x6 = expansion(x5, s4, 512)
    x7 = expansion(x6, s3, 256)
    x8 = expansion(x7, s2, 128)
    x9 = expansion(x8, s1, 64)

    """ Outputs """
    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(x9)

    """ Model """
    model = Model(inputs, outputs)
    return model

def conv_block(inputs, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

# Downsample the resolution of the images
def contraction(inputs, num_filters):
    x = conv_block(inputs, num_filters)
    p = MaxPool2D((2, 2))(x)
    return x, p

# Upsample the resolution of the images, but decrease the feature maps
def expansion(inputs, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(inputs)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x