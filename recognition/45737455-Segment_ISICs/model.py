from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow_addons.layers import InstanceNormalization

INIT_FILTERS = 16


def improved_UNet(input_shape=(256, 256, 3), n_classes=1, dropout_rate=0.3, leaky_slope=1e-2):
    """
    An improved U-Net model.
    Read more in https://arxiv.org/abs/1802.10508v1.

    Parameters
    ----------
    input_shape : tuple, default=(256, 256, 3)
        The shape of input data.
    n_classes : int, default=1
        The number of output classes.
    dropout_rate : float, default=0.3
        The dropout rate used in all dropout layers in the model.
    leaky_slope : float, default=1e-2
        The leaky slope used in all leaky ReLU activation in the model.

    Returns
    -------
    improved_UNet : tensorflow.keras.models.Model
        Returns the improved U-Net model.
    """

    input_layer = Input(shape=input_shape)

    # context pathway 1 (top)
    con1_conv = Conv2D(INIT_FILTERS*1, kernel_size=3, strides=1, padding='same')(input_layer)
    con1_context = context_module(con1_conv, INIT_FILTERS*1, leaky_slope, dropout_rate)
    con1_context += con1_conv

    # context pathway 2
    con2_conv = Conv2D(INIT_FILTERS*2, kernel_size=3, strides=2, padding='same')(con1_context)
    con2_context = context_module(con2_conv, INIT_FILTERS*2, leaky_slope, dropout_rate)
    con2_context += con2_conv

    # context pathway 3 
    con3_conv = Conv2D(INIT_FILTERS*4, kernel_size=3, strides=2, padding='same')(con2_context)
    con3_context = context_module(con3_conv, INIT_FILTERS*4, leaky_slope, dropout_rate)
    con3_context += con3_conv

    # context pathway 4
    con4_conv = Conv2D(INIT_FILTERS*8, kernel_size=3, strides=2, padding='same')(con3_context)
    con4_context = context_module(con4_conv, INIT_FILTERS*8, leaky_slope, dropout_rate)
    con4_context += con4_conv

    # context pathway 5 (bottom)
    con5_conv = Conv2D(INIT_FILTERS*16, kernel_size=3, strides=2, padding='same')(con4_context)
    con5_context = context_module(con5_conv, INIT_FILTERS*16, leaky_slope, dropout_rate)
    con5_context += con5_conv

    # localization pathway 1 (bottom)
    local1_up = UpSampling2D()(con5_context)
    local1_conv = Conv2D(INIT_FILTERS*8, kernel_size=3, strides=1, padding='same',
                         activation=LeakyReLU(leaky_slope))(local1_up)

    # localization pathway 2
    local2_concat = concatenate([local1_conv, con4_context])
    _, local2_localization = localization_module(local2_concat, INIT_FILTERS*8, leaky_slope)
    local2_up = UpSampling2D()(local2_localization)
    local2_conv = Conv2D(INIT_FILTERS*4, kernel_size=3, strides=1, padding='same',
                         activation=LeakyReLU(leaky_slope))(local2_up)

    # localization pathway 3
    local3_concat = concatenate([local2_conv, con3_context])
    tmp_1, local3_localization = localization_module(local3_concat, INIT_FILTERS*4, leaky_slope)
    local3_up = UpSampling2D()(local3_localization)
    local3_conv = Conv2D(INIT_FILTERS*2, kernel_size=3, strides=1, padding='same',
                         activation=LeakyReLU(leaky_slope))(local3_up)

    # localization pathway 4
    local4_concat = concatenate([local3_conv, con2_context])
    tmp_2, local4_localization = localization_module(local4_concat, INIT_FILTERS*2, leaky_slope)
    local4_up = UpSampling2D()(local4_localization)
    local4_conv = Conv2D(INIT_FILTERS*1, kernel_size=3, strides=1, padding='same',
                         activation=LeakyReLU(leaky_slope))(local4_up)

    # final (top)
    final_concat = concatenate([local4_conv, con1_context])
    final_conv = Conv2D(INIT_FILTERS*2, kernel_size=3, strides=1, padding='same',
                        activation=LeakyReLU(leaky_slope))(final_concat)

    # segmentations
    seg_1 = Conv2D(n_classes, kernel_size=1, strides=1)(tmp_1)
    seg_1_up = UpSampling2D(interpolation='bilinear')(seg_1)

    seg_2 = Conv2D(n_classes, kernel_size=1, strides=1)(tmp_2)
    seg_2_up = UpSampling2D(interpolation='bilinear')(seg_2+seg_1_up)

    seg_3 = Conv2D(n_classes, kernel_size=1, strides=1)(final_conv)
    final_add = seg_2_up + seg_3

    if n_classes == 1:
        # sigmoid for binary classification
        pre = Activation('sigmoid')(final_add)
    else:
        # softmax for multi-class classification
        pre = Activation('softmax')(final_add)

    return Model(inputs=input_layer, outputs=pre)


def context_module(pre_layer, n_filters, leaky_slope=1e-2, dropout_rate=0.3):
    """
    A context module in the improved U-Net model.
    Read more in https://arxiv.org/abs/1802.10508v1.

    Parameters
    ----------
    pre_layer : tensorflow.keras.layers.Layer
        The last layer before this context module.
    n_filters : int
        The number of filters of convolutional layers in the module.
    leaky_slope : float, default=1e-2
        The leaky slope used in all leaky ReLU activation in the module.
    dropout_rate : float, default=0.3
        The dropout rate used in the dropout layer in the module.

    Returns
    -------
    conv2 : tensorflow.keras.layers.Layer
        Returns the last layer of the context module.
    """

    # 1st
    norm1 = InstanceNormalization()(pre_layer)
    leakyRelu1 = LeakyReLU(alpha=leaky_slope)(norm1)
    conv1 = Conv2D(filters=n_filters, kernel_size=3, strides=1, padding='same', use_bias=False)(leakyRelu1)

    dropout = Dropout(dropout_rate)(conv1)

    # 2nd
    norm2 = InstanceNormalization()(dropout)
    leakyRelu2 = LeakyReLU(alpha=leaky_slope)(norm2)
    conv2 = Conv2D(filters=n_filters, kernel_size=3, strides=1, padding='same', use_bias=False)(leakyRelu2)

    return conv2


def localization_module(pre_layer, n_filters, leaky_slope=1e-2):
    """
    A localization module in the improved U-Net model.
    Read more in https://arxiv.org/abs/1802.10508v1.

    Parameters
    ----------
    pre_layer : tensorflow.keras.layers.Layer
        The last layer before this localization module.
    n_filters : int
        The number of filters of the first convolutional layer.
    leaky_slope : float, default=1e-2
        The leaky slope used in all leaky ReLU activation in the module.

    Returns
    -------
    norm1 : tensorflow.keras.layers.Layer
        The layer in the middle of the localization module.
    norm2 : tensorflow.keras.layers.Layer
        The last layer of the localization module.
    """

    conv1 = Conv2D(filters=n_filters, kernel_size=3, strides=1, padding='same',
                    activation=LeakyReLU(leaky_slope))(pre_layer)
    norm1 = InstanceNormalization()(conv1)

    conv2 = Conv2D(filters=n_filters/2, kernel_size=1, strides=1, padding='same',
                    activation=LeakyReLU(leaky_slope))(norm1)
    norm2 = InstanceNormalization()(conv2)

    return norm1, norm2

