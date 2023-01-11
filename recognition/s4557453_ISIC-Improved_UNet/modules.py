from keras.layers import BatchNormalization, Conv2D, UpSampling2D, LeakyReLU, Dropout, Add, Concatenate
from keras import Input, Model


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


def getModel(input_size):
    """
    Implement the Improved UNet as per: https://arxiv.org/abs/1802.10508v1
    """

    # Entry block
    inputs = Input(shape=input_size)

    # Layer 1 (16 filter) - Context Pathway
    conv1 = Conv2D(16, (3, 3), strides=1, padding="same")(inputs)
    leak1 = LeakyReLU(0.01)(conv1)

    cont1 = contextModule(leak1, 16)

    add1 = Add()([leak1, cont1])

    # Layer 2 (32 filter) - Context Pathway
    conv2 = Conv2D(32, (3, 3), strides=2, padding="same")(add1)
    leak2 = LeakyReLU(0.01)(conv2)

    cont2 = contextModule(leak2, 32)

    add2 = Add()([leak2, cont2])

    # Layer 3 (64 filter) - Context Pathway
    conv3 = Conv2D(64, (3, 3), strides=2, padding="same")(add2)
    leak3 = LeakyReLU(0.01)(conv3)

    cont3 = contextModule(leak3, 64)

    add3 = Add()([leak3, cont3])

    # Layer 4 (128 filter) - Context Pathway
    conv4 = Conv2D(128, (3, 3), strides=2, padding="same")(add3)
    leak4 = LeakyReLU(0.01)(conv4)

    cont4 = contextModule(leak4, 128)

    add4 = Add()([leak4, cont4])

    # Layer 5 (256 filter) - Bottleneck
    conv5 = Conv2D(256, (3, 3), strides=2, padding="same")(add4)
    leak5 = LeakyReLU(0.01)(conv5)

    cont5 = contextModule(leak5, 256)

    add5 = Add()([leak5, cont5])

    # Layer 5 (128 filter) - Bottleneck
    upsamp1 = upsamplingModule(add5, 128)

    concat1 = Concatenate()([upsamp1, add4])

    # Layer 4 (128 and 64 filters) - Localization Pathway
    local1 = localizationModule(concat1, 128)
    upsamp2 = upsamplingModule(local1, 64)

    concat2 = Concatenate()([upsamp2, add3])

    # Layer 3 (64 and 32 filters) - Localization Pathway
    local2 = localizationModule(concat2, 64)
    upsamp3 = upsamplingModule(local2, 32)

    concat3 = Concatenate()([upsamp3, add2])

    seg1 = Conv2D(2, (3, 3), strides=1, padding="same")(local2)
    seg1 = LeakyReLU(0.01)(seg1)

    # Layer 2 (32 and 16 filters) - Localization Pathway
    local3 = localizationModule(concat3, 32)
    upsamp4 = upsamplingModule(local3, 16)

    concat4 = Concatenate()([upsamp4, add1])

    seg2 = Conv2D(2, (3, 3), strides=1, padding="same")(local3)
    seg2 = LeakyReLU(0.01)(seg2)

    segUp1 = UpSampling2D((2, 2))(seg1)
    add6 = Add()([segUp1, seg2])

    # Layer 1 (32 filter) - Localization Pathway
    conv6 = Conv2D(32, (3, 3), strides=1, padding="same")(concat4)

    seg3 = Conv2D(2, (3, 3), strides=1, padding="same")(conv6)
    seg3 = LeakyReLU(0.01)(seg3)

    segUp2 = UpSampling2D((2, 2))(seg2)
    add7 = Add()([segUp2, seg3])

    output = Conv2D(2, (3, 3), strides=1, padding="same", activation="softmax")(add7)
    model = Model(inputs, output)

    return model


def main():
    model = get_model((256, 256, 1))
    print(model.summary())


if __name__ == "__main__":
    main()
