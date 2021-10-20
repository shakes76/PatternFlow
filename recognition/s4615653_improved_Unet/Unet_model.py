import tensorflow as tf

def convolution(inputs, filters):
    conv = tf.keras.layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(inputs)
    conv = tf.keras.layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(conv)
    return conv

def pool(conv):
    pool = tf.keras.layers.MaxPool2D((2, 2))(conv)
    return pool

def Unet():

    inputs = tf.keras.layers.Input(shape=(512, 512, 3))

    conv_1 = convolution(inputs,32)
    pool_1 = pool(conv_1)

    conv_2 = convolution(pool_1,64)
    pool_2 = pool(conv_2)

    conv_3 = convolution(pool_2,128)
    pool_3 = pool(conv_3)

    conv_4 = convolution(pool_3,256)
    pool_4 = pool(conv_4)

    conv_5 = convolution(pool_4)
