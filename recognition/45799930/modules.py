import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Input, Conv2D, Add, concatenate, UpSampling2D, Dropout, BatchNormalization

# Dense, Flatten,
DROPOUT_PROB = 0.3
INPUT_SHAPE = (256, 256, 1)


def context_module(layer, num_filters):
    """
    Creates the context module as defined in the improved UNet. It normalises the model between every
    Conv2D call. This module includes two 2D 3x3 conv layers with a dropout layer inbetween.

    :param layer: the input layer within this module.
    :param num_filters: the number of filters to pass into Conv2D.
    :return: the final layer of this context module.
    """
    norm = BatchNormalization()(layer)
    encoder = Conv2D(num_filters, 3, activation='relu', padding='same')(norm)
    dropout = Dropout(DROPOUT_PROB)(encoder)
    norm = BatchNormalization()(dropout)
    return Conv2D(num_filters, 3, activation='relu', padding='same')(norm)


def upsample_module(layer, num_filters):
    """

    :param layer:
    :param num_filters:
    :return:
    """
    up = UpSampling2D(2)(layer)
    return Conv2D(num_filters, 3, activation='relu', padding='same')(up)


def create_model(input_shape=INPUT_SHAPE):
    """
    Create the improved unet model. This will include all the convolutions, max pooling,
    and upsampling.

    :param input_shape: The shape of  the images that we will be passing into the model.
                        As we are using the one type of image we set the default to (256, 256, 1)
                        but added this feature so that it could be easily adapted to another model
                        if required in the future.
    :return:
    """

    # Create the input for the model.
    input = Input(shape=input_shape)

    # Create the encoder:
    # First layer : 2 x 2D Convolutions, filter size 16,  with a 3x3 kernel size and a stride size of (2,2).
    encoder_layer1 = context_module(input, 16)

    # element wise sum of pre context module and post context module.
    sum1 = Add()([input, encoder_layer1])

    # Do conv with stride of 2 to reduce the size of the model.
    pool1 = Conv2D(32, 3, 2, activation='relu', padding='same')(sum1)

    # Second layer : 2 x 2D Convolutions, filter size 32,  with a 3x3 kernel size and a stride size of (2,2).
    encoder_layer2 = context_module(pool1, 32)

    # element wise sum of pre context module and post context module.
    sum2 = Add()([pool1, encoder_layer2])

    # Do conv with stride of 2 to reduce the size of the model.
    pool2 = Conv2D(64, 3, 2, activation='relu', padding='same')(sum2)

    # Third layer : 2 x 2D Convolutions, filter size 64,  with a 3x3 kernel size and a stride size of (2,2).
    encoder_layer3 = context_module(pool2, 64)

    # element wise sum of pre context module and post context module.
    sum3 = Add()([pool2, encoder_layer3])

    # Do conv with stride of 2 to reduce the size of the model.
    pool3 = Conv2D(128, 3, 2, activation='relu', padding='same')(sum3)

    # Forth layer : 2 x 2D Convolutions, filter size 128,  with a 3x3 kernel size and a stride size of (2,2).
    encoder_layer4 = context_module(pool3, 128)

    # element wise sum of pre context module and post context module.
    sum4 = Add()([pool3, encoder_layer4])

    # Do conv with stride of 2 to reduce the size of the model.
    pool4 = Conv2D(256, 3, 2, activation='relu', padding='same')(sum4)

    # Fifth layer : 2 x 2D Convolutions, filter size 256,  with a 3x3 kernel size and a stride size of (2,2).
    encoder_layer5 = context_module(pool4, 256)

    # element wise sum of pre context module and post context module.
    sum5 = Add()([pool4, encoder_layer5])

    # Create the decoder:

    # up sample layer 5 to get to layer 4
    up4 = upsample_module(encoder_layer5, 128)
    # concatenate the output of encoder_layer4 with  the up sample
    con4 = concatenate([up4, encoder_layer4])
    # Forth layer : 2 x 2D Convolutions, filter size 128,  with a 3x3 kernel size for the first one and 1x1 for the
    # second one and a stride size of (2,2).
    decoder_layer4 = Conv2D(128, 3, activation='relu', padding='same')(con4)
    # decoder_layer4 = Dropout(DROPOUT_PROB)(encoder_layer5)
    # decoder_layer4 = BatchNormalization()(decoder_layer4)
    decoder_layer4 = Conv2D(128, 1, activation='relu', padding='same')(decoder_layer4)

    # up sample layer 4 to get to layer 3
    up3 = upsample_module(decoder_layer4, 64)
    # concatenate the output of encoder_layer4 with  the up sample
    con3 = concatenate([up3, encoder_layer3])
    # Third layer : 2 x 2D Convolutions, filter size 64,  with a 3x3 kernel size for the first one and 1x1 for the
    # second one and a stride size of (2,2).
    decoder_layer3 = Conv2D(64, 3, activation='relu', padding='same')(con3)
    decoder_layer3 = Conv2D(64, 1, activation='relu', padding='same')(decoder_layer3)

    # up sample layer 3 to get to layer 2
    up2 = upsample_module(decoder_layer3, 32)
    # concatenate the output of encoder_layer4 with  the up sample
    con2 = concatenate([up2, encoder_layer2])
    # Second layer : 2 x 2D Convolutions, filter size 32,  with a 3x3 kernel size for the first one and 1x1 for the
    # second one and a stride size of (2,2).
    decoder_layer2 = Conv2D(32, 3, activation='relu', padding='same')(con2)
    decoder_layer2 = Conv2D(32, 1, activation='relu', padding='same')(decoder_layer2)

    # up sample layer 2 to get to layer 1
    up1 = upsample_module(decoder_layer2, 16)
    # concatenate the output of encoder_layer4 with  the up sample
    con1 = concatenate([up1, encoder_layer1])
    # Third layer : 1 x 2D Convolutions, filter size 32,  with a 3x3 kernel size for the first one and 1x1 for the
    # second one and a stride size of (2,2).
    decoder_layer1 = Conv2D(32, 3, activation='relu', padding='same')(con1)

    # Now do all the segmentation layers.

    output = None

    return Model(inputs=input, outputs=output)
