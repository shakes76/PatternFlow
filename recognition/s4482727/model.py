import tensorflow as tf

IMG_HEIGHT = 572
IMG_WIDTH = 572
IMG_CHANNELS = 1

unet_input = tf.keras.Input(
    shape=(IMG_WIDTH, IMG_WIDTH, IMG_CHANNELS),
    name="input")

conv_1a = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3),
                                 padding="valid",
                                 activation=tf.keras.activations.relu,
                                 name="conv_1a")(unet_input)
conv_1b = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3),
                                 padding="valid",
                                 activation=tf.keras.activations.relu,
                                 name="conv_1b")(conv_1a)

pool_1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2,
                                   name="pool_1")(conv_1b)

conv_2a = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3),
                                 padding="valid",
                                 activation=tf.keras.activations.relu,
                                 name="conv_2a")(pool_1)
conv_2b = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3),
                                 padding="valid",
                                 activation=tf.keras.activations.relu,
                                 name="conv_2b")(conv_2a)

pool_2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2,
                                   name="pool_2")(conv_2b)

conv_3a = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3),
                                 padding="valid",
                                 activation=tf.keras.activations.relu,
                                 name="conv_3a")(pool_2)
conv_3b = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3),
                                 padding="valid",
                                 activation=tf.keras.activations.relu,
                                 name="conv_3b")(conv_3a)

pool_3 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2,
                                   name="pool_3")(conv_3b)

conv_4a = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3),
                                 padding="valid",
                                 activation=tf.keras.activations.relu,
                                 name="conv_4a")(pool_3)
conv_4b = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3),
                                 padding="valid",
                                 activation=tf.keras.activations.relu,
                                 name="conv_4b")(conv_4a)

pool_4 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2,
                                   name="pool_4")(conv_4b)

conv_5a = tf.keras.layers.Conv2D(filters=1024, kernel_size=(3, 3),
                                 padding="valid",
                                 activation=tf.keras.activations.relu,
                                 name="conv_5a")(pool_4)
conv_5b = tf.keras.layers.Conv2D(filters=1024, kernel_size=(3, 3),
                                 padding="valid",
                                 activation=tf.keras.activations.relu,
                                 name="conv_5b")(conv_5a)

up_conv4 = tf.keras.layers.Conv2DTranspose(filters=512, kernel_size=(2, 2),
                                           name="up_conv4")(conv_5b)

concat_4 = tf.keras.layers.Concatenate(name="concat_4")([up_conv4, conv_4b])
conv_4c = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3),
                                 padding="valid",
                                 activation=tf.keras.activations.relu,
                                 name="conv_4c")(concat_4)
conv_4d = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3),
                                 padding="valid",
                                 activation=tf.keras.activations.relu,
                                 name="conv_4d")(conv_4c)

up_conv3 = tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=(2, 2),
                                           name="up_conv3")(conv_4d)

concat_3 = tf.keras.layers.Concatenate(name="concat_3")([up_conv3, conv_3b])
conv_3c = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3),
                                 padding="valid",
                                 activation=tf.keras.activations.relu,
                                 name="conv_3c")(concat_3)
conv_3d = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3),
                                 padding="valid",
                                 activation=tf.keras.activations.relu,
                                 name="conv_3d")(conv_3c)

up_conv2 = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=(2, 2),
                                           name="up_conv2")(conv_3d)

concat_2 = tf.keras.layers.Concatenate(name="concat_2")([up_conv2, conv_2b])
conv_2c = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3),
                                 padding="valid",
                                 activation=tf.keras.activations.relu,
                                 name="conv_2c")(concat_2)
conv_2d = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3),
                                 padding="valid",
                                 activation=tf.keras.activations.relu,
                                 name="conv_2d")(conv_2c)

up_conv1 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=(2, 2),
                                           name="up_conv1")(conv_2d)

concat_1 = tf.keras.layers.Concatenate(name="concat_1")([up_conv1, conv_1b])
conv_1c = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3),
                                 padding="valid",
                                 activation=tf.keras.activations.relu,
                                 name="conv_1c")(concat_1)
conv_1d = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3),
                                 padding="valid",
                                 activation=tf.keras.activations.relu,
                                 name="conv_2d")(conv_1c)

unet_output = tf.keras.layers.Conv2D(filters=2, kernel_size=(1, 1),
                                 padding="valid",
                                 activation=tf.keras.activations.relu,
                                 name="output")(conv_1d)

model = tf.keras.Model(inputs=unet_input, outputs=unet_output)
model.summary()
