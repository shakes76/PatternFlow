from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras import layers
import tensorflow_addons as tfa
import tensorflow as tf
import matplotlib.pyplot as plt

# context module function
def res_net_block(input_data, conv_size):
    x = tfa.layers.InstanceNormalization()(input_data)
    x = tf.keras.layers.LeakyReLU(alpha=0.01)(x)
    x = tf.keras.layers.Conv2D(conv_size, kernel_size = 3, padding='same')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.01)(x)
    x = tf.keras.layers.Conv2D(conv_size, kernel_size = 3, padding='same')(x)
    return x

# add segmentation layers
def segmentation_layer(x):
    seg = tf.keras.layers.Conv2D(1, (1,1), activation = 'sigmoid')(x)
    return seg


img_height =256
img_width = 256
imag_channels = 3

def model(img_height, img_width, imag_channels):
    inputs = tf.keras.layers.Input((img_height, img_width, imag_channels))
    activation_function = tf.keras.layers.LeakyReLU(alpha=0.01)
    # encoding
    a1 = tf.keras.layers.Conv2D(16, (3,3), activation = activation_function, padding ='same')(inputs)
    b1 = res_net_block(a1, 16)
    c1 = layers.Add()([b1, a1])
    a2 = tf.keras.layers.Conv2D(32, (3,3), activation = activation_function, padding ='same', strides=2)(c1)
    b2 = res_net_block(a2, 32)
    c2 = layers.Add()([b2, a2])
    a3 = tf.keras.layers.Conv2D(64, (3,3), activation = activation_function, padding ='same', strides=2)(c2)
    b3 = res_net_block(a3, 64)
    c3 = layers.Add()([b3, a3])
    a4 = tf.keras.layers.Conv2D(128, (3,3), activation = activation_function, padding ='same', strides=2)(c3)
    b4 = res_net_block(a4, 128)
    c4 = layers.Add()([b4, a4])
    a5 = tf.keras.layers.Conv2D(256, (3,3), activation = activation_function, padding ='same', strides=2)(c4)
    b5 = res_net_block(a5, 256)
    c5 = layers.Add()([b5, a5])

    # decoding
    d1 = tf.keras.layers.UpSampling2D( size=(2, 2) )(c5)
    d1 = tf.keras.layers.Conv2D(128, (3,3), activation = activation_function, padding ='same')(d1)
    e1 = tf.keras.layers.concatenate([c4,d1])
    f1 = tf.keras.layers.Conv2D(128, (3,3), activation = activation_function, padding ='same')(e1)
    f1 = tf.keras.layers.Conv2D(128, (1,1), activation = activation_function, padding ='same')(f1)
    d2 = tf.keras.layers.UpSampling2D( size=(2, 2) )(f1)
    d2 = tf.keras.layers.Conv2D(64, (3,3), activation = activation_function, padding ='same')(d2)
    e2 = tf.keras.layers.concatenate([c3,d2])
    f2 = tf.keras.layers.Conv2D(64, (3,3), activation = activation_function, padding ='same')(e2)
    f2 = tf.keras.layers.Conv2D(64, (1,1), activation = activation_function, padding ='same')(f2)
    seg1 = segmentation_layer(f2)
    seg1 = tf.keras.layers.UpSampling2D( size=(2, 2) )(seg1)
    d3 = tf.keras.layers.UpSampling2D( size=(2, 2) )(f2)
    d3 = tf.keras.layers.Conv2D(32, (3,3), activation = activation_function, padding ='same')(d3)
    e3 = tf.keras.layers.concatenate([c2,d3])
    f3 = tf.keras.layers.Conv2D(32, (3,3), activation = activation_function, padding ='same')(e3)
    f3 = tf.keras.layers.Conv2D(32, (1,1), activation = activation_function, padding ='same')(f3)
    seg2 = segmentation_layer(f3)
    c6 = layers.Add()([seg1, seg2])
    c6 = tf.keras.layers.UpSampling2D( size=(2, 2) )(c6)
    d4 = tf.keras.layers.UpSampling2D( size=(2, 2))(f3)
    d4 = tf.keras.layers.Conv2D(16, (3,3), activation = activation_function, padding ='same')(d4)
    e4 = tf.keras.layers.concatenate([c1,d4])
    f4 = tf.keras.layers.Conv2D(32, (3,3), activation = activation_function, padding ='same')(e4)
    seg3 = segmentation_layer(f4)
    c7 = layers.Add()([c6, seg3])
    outputs = tf.keras.layers.Conv2D(1, (1,1),  activation = 'sigmoid')(c7)
    model = tf.keras.Model(inputs = [inputs], outputs = [outputs])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 0.0001),
                loss=["binary_crossentropy"],
                metrics=["accuracy"])
    return model
