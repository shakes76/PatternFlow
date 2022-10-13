import tensorflow as tf
from keras import layers


def double_conv_block(x, n_filters):

    x = layers.Conv2D(n_filters, 3, padding="same", activation="relu", kernel_initializer="he_normal")(x)
    
    x = layers.Conv2D(n_filters, 3, padding="same", activation="relu", kernel_initializer="he_normal")(x)
    return x


def down_sample_block(x, n_filters):
    f = double_conv_block(x, n_filters)
    p = layers.MaxPool2D(2, 2)(f)
    p = layers.Dropout(0.3)(p)

    return f, p


def up_sample_block(x, conv_features, n_filters):
    # upsample
    x = layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
    # concatenate
    x = layers.concatenate([x, conv_features])
    # dropout
    x = layers.Dropout(0.3)(x)
    # Conv2D twice with ReLU activation
    x = double_conv_block(x, n_filters)

    return x


def build_unet_model():
    # inputs
    inputs = layers.Input(shape=(256, 256, 3))

    f1, p1 = down_sample_block(inputs, 32)

    f2, p2 = down_sample_block(p1, 64)

    f3, p3 = down_sample_block(p2, 128)

    f4, p4 = down_sample_block(p3, 256)


    bottleneck = double_conv_block(p4, 512)

    u6 = up_sample_block(bottleneck, f4, 256)

    u7 = up_sample_block(u6, f3, 128)

    u8 = up_sample_block(u7, f2, 64)

    u9 = up_sample_block(u8, f1, 32)


    outputs = layers.Conv2D(1, 1, padding="same", activation="softmax")(u9)

    
    unet_model = tf.keras.Model(inputs, outputs, name="U-Net")

    return unet_model


