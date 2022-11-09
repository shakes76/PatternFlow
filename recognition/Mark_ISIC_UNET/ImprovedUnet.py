import tensorflow as tf
import tensorflow.keras.layers as layers

# Improved U-Net based off https://arxiv.org/pdf/1802.10508v1.pdf
def IUNET_model():
    inputs = tf.keras.Input(shape=[256, 256, 1])

    # go down the U
    # step 1
    conv1 = tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding="same", activation=layers.LeakyReLU())(inputs)

    batch1 = layers.BatchNormalization()(conv1)
    context1 = tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding="same", activation=layers.LeakyReLU())(batch1)
    drop1 = tf.keras.layers.Dropout(0.3)(context1)
    context1 = tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding="same", activation=layers.LeakyReLU())(drop1)

    sum1 = tf.math.add(conv1, context1)

    # step 2
    stride2 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", strides=2, activation=layers.LeakyReLU())(sum1)
    batch2 = layers.BatchNormalization()(stride2)
    context2 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation=layers.LeakyReLU())(batch2)
    drop2 = tf.keras.layers.Dropout(0.3)(context2)
    context2 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation=layers.LeakyReLU())(drop2)

    sum2 = tf.math.add(stride2, context2)

    # step 3
    stride3 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", strides=2, activation=layers.LeakyReLU())(sum2)
    batch3 = layers.BatchNormalization()(stride3)
    context3 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation=layers.LeakyReLU())(batch3)
    drop3 = tf.keras.layers.Dropout(0.3)(context3)
    context3 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation=layers.LeakyReLU())(drop3)

    sum3 = tf.math.add(stride3, context3)

    # step 4
    stride4 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same", strides=2, activation=layers.LeakyReLU())(sum3)
    batch4 = layers.BatchNormalization()(stride4)
    context4 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same", activation=layers.LeakyReLU())(batch4)
    drop4 = tf.keras.layers.Dropout(0.3)(context4)
    context4 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same", activation=layers.LeakyReLU())(drop4)

    sum4 = tf.math.add(stride4, context4)

    # step 5 lowest one, bottom of the U
    stride5 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding="same", strides=2, activation=layers.LeakyReLU())(sum4)
    batch5 = layers.BatchNormalization()(stride5)
    context5 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding="same", activation=layers.LeakyReLU())(batch5)
    drop5 = tf.keras.layers.Dropout(0.3)(context5)
    context5 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding="same", activation=layers.LeakyReLU())(drop5)

    sum5 = tf.math.add(stride5, context5)

    up5 = tf.keras.layers.UpSampling2D(size=(2, 2))(sum5)

    # Go back up the U
    # step 6 up 1
    conc6 = tf.keras.layers.concatenate([up5, sum4])
    loc6 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same", activation=layers.LeakyReLU())(conc6)
    loc6 = tf.keras.layers.Conv2D(filters=128, kernel_size=1, padding="same", activation=layers.LeakyReLU())(loc6)
    up6 = tf.keras.layers.UpSampling2D(size=(2, 2))(loc6)

    # step 7 up 2
    conc7 = tf.keras.layers.concatenate([up6, sum3])
    loc7 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation=layers.LeakyReLU())(conc7)
    loc7 = tf.keras.layers.Conv2D(filters=64, kernel_size=1, padding="same", activation=layers.LeakyReLU())(loc7)
    up7 = tf.keras.layers.UpSampling2D(size=(2, 2))(loc7)

    uploc7 = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, padding='same')(up7)  # tf.keras.layers.UpSampling2D(size=(2, 2))(loc7)

    # step 8 up 3
    conc8 = tf.keras.layers.concatenate([up7, sum2])
    loc8 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation=layers.LeakyReLU())(conc8)
    loc8 = tf.keras.layers.Conv2D(filters=32, kernel_size=1, padding="same", activation=layers.LeakyReLU())(loc8)
    up8 = tf.keras.layers.UpSampling2D(size=(2, 2))(loc8)

    seg8 = tf.math.add(uploc7, loc8)
    upseg8 = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding="same")(seg8)

    # step 9 up 4
    conc9 = tf.keras.layers.concatenate([up8, sum1])
    conv9 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation=layers.LeakyReLU())(conc9)

    seg9 = tf.math.add(upseg8, conv9)

    # do a sigmoid activation instead of softmax
    sig = tf.keras.layers.Conv2D(filters=1, kernel_size=1, activation='sigmoid')(seg9)

    # create the model
    finalModel = tf.keras.Model(inputs, sig)
    return finalModel


