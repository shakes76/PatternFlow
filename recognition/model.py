import tensorflow as tf
import numpy as np
def improved_unet(output_channels, filters=64, input_shape=(256, 256, 1)):
    modelInput = tf.keras.layers.Input(shape=(256, 256, 1))

    conv2d64_1 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(modelInput)
    conv2d64_2 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(conv2d64_1)

    maxPool1 = tf.keras.layers.MaxPooling2D((2,2))(conv2d64_2)

    conv2d128_1 = tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same")(maxPool1)
    conv2d128_2 = tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same")(conv2d128_1)

    maxPool2 = tf.keras.layers.MaxPooling2D((2, 2))(conv2d128_2)

    conv2d256_1 = tf.keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same")(maxPool2)
    conv2d256_2 = tf.keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same")(conv2d256_1)

    maxPool3 = tf.keras.layers.MaxPooling2D((2, 2))(conv2d256_2)

    conv2d512_1 = tf.keras.layers.Conv2D(512, (3, 3), activation="relu", padding="same")(maxPool3)
    conv2d512_2 = tf.keras.layers.Conv2D(512, (3, 3), activation="relu", padding="same")(conv2d512_1)

    maxPool4 = tf.keras.layers.MaxPooling2D((2, 2))(conv2d512_2)

    conv2d1024_1 = tf.keras.layers.Conv2D(1024, (3, 3), activation="relu", padding="same")(maxPool4)
    conv2d1024_2 = tf.keras.layers.Conv2D(1024, (3, 3), activation="relu", padding="same")(conv2d1024_1)

    drop = tf.keras.layers.Dropout(0.5)(conv2d1024_2)

    upConv2d512 = tf.keras.layers.Conv2DTranspose(512, (1, 1), strides=(2, 2), activation="relu", padding="same")(drop)

    concatenate512 = tf.keras.layers.Concatenate()([upConv2d512, conv2d512_2])

    conv2d512_3 = tf.keras.layers.Conv2D(512, (3, 3), activation="relu", padding="same")(concatenate512)
    conv2d512_4 = tf.keras.layers.Conv2D(512, (3, 3), activation="relu", padding="same")(conv2d512_3)

    upConv2d256 = tf.keras.layers.Conv2DTranspose(256, (1, 1), strides=(2, 2), activation="relu", padding="same")(conv2d512_4)

    concatenate256 = tf.keras.layers.Concatenate()([upConv2d256, conv2d256_2])

    conv2d256_3 = tf.keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same")(concatenate256)
    conv2d256_4 = tf.keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same")(conv2d256_3)

    upConv2d128 = tf.keras.layers.Conv2DTranspose(128, (1, 1), strides=(2, 2), activation="relu", padding="same")(conv2d256_4)

    concatenate128 = tf.keras.layers.Concatenate()([upConv2d128, conv2d128_2])

    conv2d128_3 = tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same")(concatenate128)
    conv2d128_4 = tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same")(conv2d128_3)

    upConv2d64 = tf.keras.layers.Conv2DTranspose(64, (1, 1), strides=(2, 2), activation="relu", padding="same")(conv2d128_4)

    concatenate64 = tf.keras.layers.Concatenate()([upConv2d64, conv2d64_2])

    conv2d64_3 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(concatenate64)
    conv2d64_4 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(conv2d64_3)

    #conv2d2_1 = tf.keras.layers.Conv2D(2, (3, 3), activation="relu", padding="same")(conv2d64_4)

    output = tf.keras.layers.Conv2D(4, (1, 1), padding="same", activation="softmax")(conv2d64_4)

    model = tf.keras.Model(inputs=modelInput, outputs=output)
    return model

