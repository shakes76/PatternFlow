import tensorflow as tf
from keras import layers


"""
    # This block contains all the convolutional layers with n_filters
    # Kernel Size is set to 3
    # Padding is set tp same so that image size is same
"""

def double_conv_block(x, n_filters):
    # Conv2D then ReLU activation
    x = layers.Conv2D(n_filters, 3, padding="same", activation="relu", kernel_initializer="he_normal")(x)
    # Conv2D then ReLU activation
    x = layers.Conv2D(n_filters, 3, padding="same", activation="relu", kernel_initializer="he_normal")(x)
    return x


"""
    # This block is implemented for down sampling the data
    # It contains pooling of layer as 2x2
"""

def down_sample_block(x, n_filters):
    f = double_conv_block(x, n_filters)
    p = layers.MaxPool2D(2, 2)(f)
    p = layers.Dropout(0.3)(p)

    return f, p


"""
    # This block is used for upsampling the data which was earlier down sampled
    # It contains various features to combine the data and minimise the loss
"""
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


"""
    # It is the building model for creating the UNet architecture
    # This block calls all the other 3 blocks for the processing
"""

def build_unet_model():
    # inputs
    inputs = layers.Input(shape=(256, 256, 3))

    # encoder: contracting path - downsample
    # 1 - downsample
    f1, p1 = down_sample_block(inputs, 64)
    # 2 - downsample
    f2, p2 = down_sample_block(p1, 128)
    # 3 - downsample
    f3, p3 = down_sample_block(p2, 256)
    # 4 - downsample
    f4, p4 = down_sample_block(p3, 512)

    # 5 - bottleneck
    bottleneck = double_conv_block(p4, 1024)

    # decoder: expanding path - upsample
    # 6 - upsample
    u6 = up_sample_block(bottleneck, f4, 512)
    # 7 - upsample
    u7 = up_sample_block(u6, f3, 256)
    # 8 - upsample
    u8 = up_sample_block(u7, f2, 128)
    # 9 - upsample
    u9 = up_sample_block(u8, f1, 64)

    # outputs
    outputs = layers.Conv2D(1, 1, padding="same", activation="softmax")(u9)

    # unet model with Keras Functional API
    unet_model = tf.keras.Model(inputs, outputs, name="U-Net")

    return unet_model


