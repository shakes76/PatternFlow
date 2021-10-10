import tensorflow as tf

IMG_HEIGHT = 572
IMG_WIDTH = 572
IMG_CHANNELS = 1


def unet_conv2d(filters: int, name: str) -> tf.keras.layers.Conv2D:
    return tf.keras.layers.Conv2D(filters=filters, kernel_size=(3, 3),
                                  padding="same",
                                  activation=tf.keras.activations.relu,
                                  name=name)


def unet_maxpool2d(name: str) -> tf.keras.layers.MaxPool2D:
    return tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2,
                                     name=name)


def unet_upconv(filters: int, name: str) -> tf.keras.layers.Conv2DTranspose:
    return tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=(3, 3),
                                           name=name, strides=(2, 2))


unet_input = tf.keras.Input(
    shape=(IMG_WIDTH, IMG_WIDTH, IMG_CHANNELS),
    name="input")

conv_1a = unet_conv2d(filters=64, name="conv_1a")(unet_input)
conv_1b = unet_conv2d(filters=64, name="conv_1b")(conv_1a)

pool_1 = unet_maxpool2d(name="pool_1")(conv_1b)

conv_2a = unet_conv2d(128, "conv_2a")(pool_1)
conv_2b = unet_conv2d(128, "conv_2b")(conv_2a)

pool_2 = unet_maxpool2d("pool2")(conv_2b)

conv_3a = unet_conv2d(filters=256, name="conv_3a")(pool_2)
conv_3b = unet_conv2d(filters=256, name="conv_3b")(conv_3a)

pool_3 = unet_maxpool2d(name="pool_3")(conv_3b)

conv_4a = unet_conv2d(filters=512, name="conv_4a")(pool_3)
conv_4b = unet_conv2d(filters=512, name="conv_4b")(conv_4a)

pool_4 = unet_maxpool2d(name="pool_4")(conv_4b)

conv_5a = unet_conv2d(filters=1024, name="conv_5a")(pool_4)
conv_5b = unet_conv2d(filters=1024, name="conv_5b")(conv_5a)

up_conv4 = unet_upconv(filters=512, name="up_conv4")(conv_5b)

concat_4 = tf.keras.layers.Concatenate(name="concat_4")([up_conv4, conv_4b])
conv_4c = unet_conv2d(filters=512, name="conv_4c")(concat_4)
conv_4d = unet_conv2d(filters=512, name="conv_4d")(conv_4c)

up_conv3 = unet_upconv(filters=256, name="up_conv3")(conv_4d)

concat_3 = tf.keras.layers.Concatenate(name="concat_3")([up_conv3, conv_3b])
conv_3c = unet_conv2d(filters=256, name="conv_3c")(concat_3)
conv_3d = unet_conv2d(filters=256, name="conv_3d")(conv_3c)

up_conv2 = unet_upconv(filters=128, name="up_conv2")(conv_3d)

concat_2 = tf.keras.layers.Concatenate(name="concat_2")([up_conv2, conv_2b])
conv_2c = unet_conv2d(filters=128, name="conv_2c")(concat_2)
conv_2d = unet_conv2d(filters=128, name="conv_2d")(conv_2c)

up_conv1 = unet_upconv(filters=64, name="up_conv1")(conv_2d)

concat_1 = tf.keras.layers.Concatenate(name="concat_1")([up_conv1, conv_1b])
conv_1c = unet_conv2d(filters=64, name="conv_1c")(concat_1)
conv_1d = unet_conv2d(filters=64, name="conv_1d")(conv_1c)

unet_output = tf.keras.layers.Conv2D(filters=2, kernel_size=(1, 1),
                                 padding="same",
                                 activation=tf.keras.activations.relu,
                                 name="output")(conv_1d)

model = tf.keras.Model(inputs=unet_input, outputs=unet_output)
model.summary()
