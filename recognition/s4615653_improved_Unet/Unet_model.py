import tensorflow as tf

def convolution(inputs, filters):
    # function to implement convolution layer
    conv = tf.keras.layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(inputs)
    conv = tf.keras.layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(conv)
    return conv

def pool(conv):
    # function to implement pool layer.
    pool = tf.keras.layers.MaxPool2D((2, 2))(conv)
    return pool

def upsample(inputs,filter):
    #  function to implement unsample layer.
    conv = tf.keras.layers.UpSampling2D(size=(2, 2))(inputs)
    conv = tf.keras.layers.Conv2D(filter, (2, 2), padding="same")(conv)
    return conv

def Unet():
    # build the Unet model.

    inputs = tf.keras.layers.Input(shape=(512, 512, 3))

    # downsampling

    conv_1 = convolution(inputs, 4)
    pool_1 = pool(conv_1)

    conv_2 = convolution(pool_1, 8)
    pool_2 = pool(conv_2)

    conv_3 = convolution(pool_2, 16)
    pool_3 = pool(conv_3)

    conv_4 = convolution(pool_3, 32)
    pool_4 = pool(conv_4)

    conv_5 = convolution(pool_4, 64)

    # upsampling

    conv_6 = upsample(conv_5, 32)
    conv_6 = tf.keras.layers.concatenate([conv_6, conv_4])
    conv_6 = convolution(conv_6, 32)

    conv_7 = upsample(conv_6, 16)
    conv_7 = tf.keras.layers.concatenate([conv_7, conv_3])
    conv_7 = convolution(conv_7, 16)

    conv_8 = upsample(conv_7, 8)
    conv_8 = tf.keras.layers.concatenate([conv_8, conv_2])
    conv_8 = convolution(conv_8, 8)

    conv_9 = upsample(conv_8, 4)
    conv_9 = tf.keras.layers.concatenate([conv_9, conv_1])
    conv_9 = convolution(conv_9, 4)

    outputs = tf.keras.layers.Conv2D(2, (1, 1), activation='sigmoid', padding="same")(conv_9)

    return tf.keras.Model(inputs=inputs, outputs=outputs)



