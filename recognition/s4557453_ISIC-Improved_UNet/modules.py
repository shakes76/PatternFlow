from keras.layers import BatchNormalization, Conv2D, UpSampling2D, LeakyReLU, Dropout


def contextModule(inputs, filters):
    """
    Implemented based on: https://arxiv.org/abs/1802.10508v1
    Two 3x3x3 convolutional layers and a dropout layer (p = 0.3) in between.
    Use leaky ReLU nonlinearities with a negative slope of 10^−2
    """

    conv1 = Conv2D(filters, (3, 3), strides=1, padding="same")(inputs)
    leak1 = LeakyReLU(0.01)(conv1)
    norm1 = BatchNormalization()(leak1)

    drop = Dropout(0.3)(norm1)

    conv2 = Conv2D(filters, (3, 3), strides=1, padding="same")(drop)
    leak2 = LeakyReLU(0.01)(conv2)
    norm2 = BatchNormalization()(leak2)

    return norm2


def localizationModule(inputs, filters):
    """
    Implemented based on: https://arxiv.org/abs/1802.10508v1
    Consists of a 3x3x3 convolution followed by a 1x1x1 convolution that halves the number of feature maps.
    """

    conv1 = Conv2D(filters, (3, 3), strides=1, padding="same")(inputs)
    leak1 = LeakyReLU(0.01)(conv1)
    norm1 = BatchNormalization()(leak1)

    conv2 = Conv2D(filters/2, (1, 1), strides=1, padding="same")(norm1)
    leak2 = LeakyReLU(0.01)(conv2)
    norm2 = BatchNormalization()(leak2)

    return norm2


def upsamplingModule(inputs, filters):
    """
    Implemented based on: https://arxiv.org/abs/1802.10508v1
    upsampling the low resolution feature maps, which is done by means of a simple upscale that repeats the
    feature voxels twice in each spatial dimension, followed by a 3x3x3 convolution that halves the number
    of feature maps.
    """
    upsample = UpSampling2D((2, 2))(inputs)
    conv1 = Conv2D(filters, (3, 3), strides=1, padding="same")(upsample)
    leak1 = LeakyReLU(0.01)(conv1)
    norm1 = BatchNormalization()(leak1)

    return norm1
