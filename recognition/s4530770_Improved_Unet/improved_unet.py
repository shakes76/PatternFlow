import tensorflow as tf
from tensorflow.keras.layers import Conv2D, UpSampling2D, Concatenate, Input, LeakyReLU, SpatialDropout2D, Softmax, Flatten
from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.models import Model



def encoder_block(input_layer, filters, stride):
    """
    Set of layers which encode the images to latent space, consists of
    Consists of a convolution layer of given stride followed by a context module
    which has to convolution layers and a dropout layer.

    :param input_layer: Layer to to connect to encoder block.
    :param filters: Filter size of all convolution layers.
    :param stride: Stride of first convolution layer, should be 1 for every first layer and 2 for the rest.
    :return: Layer after encoder block has been connected to input layer.
    """
    # Level 1 context pathway
    conv_layer = Conv2D(filters, 3, strides=stride, padding="same", activation="relu")(input_layer)
    residual = conv_layer
    # Context module
    c_module = LeakyReLU()(conv_layer)
    conv_layer = Conv2D(filters, 3, strides=1, padding="same", activation="relu")(c_module)
    dropout = SpatialDropout2D(0.6)(conv_layer)
    c_module = Conv2D(filters, 3, strides=1, padding="same", activation="relu")(dropout)
    # Element wise summation of convolution and context module
    sum = tf.math.add(c_module, residual)
    norm_layer = InstanceNormalization()(sum)
    return LeakyReLU()(norm_layer)


def upsample_module(input_layer, filters):
    """
    Upsample module as defined by report, consists of upsample layer and convolution layer.

    :param input_layer: Layer to connect to upsample modules.
    :param filters: filter size for convolution layer.
    :return: Layer after input_layer has been connected to upsample layer.
    """
    norm = InstanceNormalization()(input_layer)
    norm = LeakyReLU()(norm)
    upsample = UpSampling2D(2, interpolation="nearest")(norm)
    conv_layer = Conv2D(filters, 3, strides=1, padding="same", activation="relu")(upsample)
    norm = InstanceNormalization()(conv_layer)
    return LeakyReLU()(norm)

def localisation_module(input_layer, context, filters):
    """
    Localisation module as described by report. Consists of concatenation of context and a convolution layer that halves
    the number of filters.

    :param input_layer: Layer to connect to localisation modules.
    :param context: A layer which is a skip connection from the encoder section of model.
    :param filters: Filter size of convolution layers.
    :return: Layer after input_layer has been connected to localisation layer.
    """
    cat = Concatenate()([input_layer, context])
    conv_layer = Conv2D(filters, 3, strides=1, padding="same", activation="relu")(cat)
    norm = InstanceNormalization()(conv_layer)
    norm = LeakyReLU()(norm)
    # Segmentation layer for deep supervision
    return Conv2D(filters/2, 1, strides=1, padding="same", activation="relu")(norm)

def build_model(input_size, filter_size):
    """
    Builds Modified Unet model based of report: https://arxiv.org/abs/1802.10508v1

    Reference:
        [1] https://github.com/pykao/Modified-3D-UNet-Pytorch/blob/63f0489e8d1fdd7ec6a203bcff095f12ea030824/model.py#L70

    :param input_size: Tuple of input size, in format (image width, image height, number of channels).
    :param filter_size: Base number of filters.
    :return: Modified Unet model
    """
    # Level 1 context pathway
    input_layer = Input(input_size)
    out = encoder_block(input_layer, filter_size, 1)
    # Skip connection
    context_1 = out

    # Level 2 context pathway
    out = encoder_block(out, filter_size*2, 2)
    context_2 = out

    # Level 3 context gateway
    out = encoder_block(out, filter_size*4, 2)
    context_3 = out

    # Level 4 context gateway
    out = encoder_block(out, filter_size*8, 2)
    context_4 = out

    # Level 5
    out = encoder_block(out, filter_size*16, 2)
    out = upsample_module(out, filter_size*8)

    # Level 1 localisation pathway
    out = localisation_module(out, context_4, filter_size*8)
    out = upsample_module(out, filter_size*4)

    # Level 2 localisation pathway - Save localisation for deep supervision
    seg_1 = localisation_module(out, context_3, filter_size*4)
    out = upsample_module(seg_1, filter_size*2)

    # Level 3 localisation pathway- Save localisation for deep supervision
    seg_2 = localisation_module(out, context_2, filter_size*2)
    out = upsample_module(seg_2, filter_size)

    # Level 4 localisation pathway
    cat = Concatenate()([out, context_1])
    conv_layer = Conv2D(filter_size, 3, strides=1, padding="same", activation="relu")(cat)
    norm = InstanceNormalization()(conv_layer)
    out = LeakyReLU()(norm)

    # Element wise summation of deep supervision layers as described in report
    seg_1 = Conv2D(filter_size, 1, strides=1, padding="same", activation="relu")(seg_1)
    seg_1 = UpSampling2D(2, interpolation="nearest")(seg_1)
    seg_2 = Conv2D(filter_size, 1, strides=1, padding="same", activation="relu")(seg_2)
    seg_layer = seg_1 + seg_2
    seg_layer = UpSampling2D(2, interpolation="nearest")(seg_layer)
    out = out + seg_layer
    # reduce output to mask of channel 1
    out = Conv2D(1, 3, strides=1, padding='same', activation="sigmoid")(out)
    return Model(inputs=input_layer, outputs=out)
