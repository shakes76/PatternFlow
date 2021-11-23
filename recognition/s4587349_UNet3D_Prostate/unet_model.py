import tensorflow as tf

import driver as drv


def unet3d(inputsize= (256,256,128,1), kernelSize=3):
    inputs = tf.keras.layers.Input(inputsize)

    # remove lot of BN
    c1 = tf.keras.layers.Conv3D(8, kernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    c1 = tf.keras.layers.BatchNormalization()(c1)
    c1 = tf.keras.layers.Conv3D(16, kernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(c1)
    # c1 = tf.keras.layers.BatchNormalization()(c1)
    p1 = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(c1)  # padding="same"?

    c2 = tf.keras.layers.Conv3D(16, kernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(p1)
    c2 = tf.keras.layers.BatchNormalization()(c2)
    c2 = tf.keras.layers.Conv3D(32, kernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(c2)
    # c2 = tf.keras.layers.BatchNormalization()(c2)
    p2 = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(c2)

    c3 = tf.keras.layers.Conv3D(32, kernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(p2)
    c3 = tf.keras.layers.BatchNormalization()(c3)
    c3 = tf.keras.layers.Conv3D(64, kernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(c3)
    # c3 = tf.keras.layers.BatchNormalization()(c3)
    p3 = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(c3)

    c4 = tf.keras.layers.Conv3D(64, kernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(p3)
    c4 = tf.keras.layers.BatchNormalization()(c4)
    c4 = tf.keras.layers.Conv3D(128, kernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(c4)
    # c4 = tf.keras.layers.BatchNormalization()(c4)

    u5 = tf.keras.layers.Conv3DTranspose(128, kernelSize, strides=(2, 2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(c4)
    concat5 = tf.keras.layers.Concatenate(axis=4)([c3,u5])
    c5 = tf.keras.layers.Conv3D(192, kernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(concat5)
    c5 = tf.keras.layers.BatchNormalization()(c5)
    c5 = tf.keras.layers.Conv3D(64, kernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(c5)
    # c5 = tf.keras.layers.BatchNormalization()(c5)

    u6 = tf.keras.layers.Conv3DTranspose(64, kernelSize, strides=(2, 2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(c5)
    concat6 = tf.keras.layers.Concatenate(axis=4)([c2,u6])
    c6 = tf.keras.layers.Conv3D(96, kernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(concat6)
    c6 = tf.keras.layers.BatchNormalization()(c6)
    c6 = tf.keras.layers.Conv3D(32, kernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(c6)
    # c6 = tf.keras.layers.BatchNormalization()(c6)

    u7 = tf.keras.layers.Conv3DTranspose(32, kernelSize, strides=(2, 2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(c6)
    concat7 = tf.keras.layers.Concatenate(axis=4)([c1,u7])
    c7 = tf.keras.layers.Conv3D(48, kernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(concat7)
    c7 = tf.keras.layers.BatchNormalization()(c7)
    c7 = tf.keras.layers.Conv3D(16, kernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(c7)
    # c7 = tf.keras.layers.BatchNormalization()(c7)
    c7 = tf.keras.layers.Conv3D(8, kernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(c7)

    outputs = tf.keras.layers.Conv3D(6, (1,1,1), activation="softmax")(c7)

    model = tf.keras.Model(inputs=[inputs], outputs = [outputs])
    return model

