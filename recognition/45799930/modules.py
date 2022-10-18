import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, concatenate, UpSampling2D, Dropout, BatchNormalization

# Dense, Flatten,
DROPOUT_PROB = 0.3
INPUT_SHAPE = (256, 256, 1)


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
    # norm1 = InstanceNormalisation
    # Create the encoder:
    # First layer : 2 x 2D Convolutions, filter size 16,  with a 3x3 kernel size and a stride size of (2,2).
    encoder_layer1 = Conv2D(16, 3, activation='relu', padding='same')(input)
    dropout1 = Dropout(DROPOUT_PROB)(encoder_layer1)
    norm1 = BatchNormalization()(dropout1)
    encoder_layer1 = Conv2D(16, 3, activation='relu', padding='same')(norm1)

    # max pooling for the first layer to get to the second one
    pool1 = MaxPooling2D()(encoder_layer1)

    # Second layer : 2 x 2D Convolutions, filter size 32,  with a 3x3 kernel size and a stride size of (2,2).
    encoder_layer2 = Conv2D(32, 3, activation='relu', padding='same')(pool1)
    dropout2 = Dropout(DROPOUT_PROB)(encoder_layer2)
    norm2 = BatchNormalization()(dropout2)
    encoder_layer2 = Conv2D(32, 3, activation='relu', padding='same')(norm2)

    # max pooling for the second layer to get to the third one
    pool2 = MaxPooling2D()(encoder_layer2)

    # Third layer : 2 x 2D Convolutions, filter size 64,  with a 3x3 kernel size and a stride size of (2,2).
    encoder_layer3 = Conv2D(64, 3, activation='relu', padding='same')(pool2)
    dropout3 = Dropout(DROPOUT_PROB)(encoder_layer3)
    norm3 = BatchNormalization()(dropout3)
    encoder_layer3 = Conv2D(64, 3, activation='relu', padding='same')(norm3)

    # max pooling for the third layer to get to the forth one
    pool3 = MaxPooling2D()(encoder_layer3)

    # Forth layer : 2 x 2D Convolutions, filter size 128,  with a 3x3 kernel size and a stride size of (2,2).
    encoder_layer4 = Conv2D(128, 3, activation='relu', padding='same')(pool3)
    dropout4 = Dropout(DROPOUT_PROB)(encoder_layer4)
    norm4 = BatchNormalization()(dropout4)
    encoder_layer4 = Conv2D(128, 3, activation='relu', padding='same')(norm4)

    # max pooling for the forth layer to get to the fifth one
    pool4 = MaxPooling2D()(encoder_layer4)

    # Fifth layer : 2 x 2D Convolutions, filter size 256,  with a 3x3 kernel size and a stride size of (2,2).
    encoder_layer5 = Conv2D(256, 3, activation='relu', padding='same')(pool4)
    dropout5 = Dropout(DROPOUT_PROB)(encoder_layer5)
    norm5 = BatchNormalization()(dropout5)
    encoder_layer5 = Conv2D(256, 3, activation='relu', padding='same')(norm5)

    # Create the decoder:

    # up sample layer 5 to get to layer 4
    up4 = UpSampling2D()(encoder_layer5)
    # concatenate the output of encoder_layer4 with  the up sample
    con4 = concatenate([up4, encoder_layer4])
    # Forth layer : 2 x 2D Convolutions, filter size 128,  with a 3x3 kernel size for the first one and 1x1 for the
    # second one and a stride size of (2,2).
    decoder_layer4 = Conv2D(128, 3, activation='relu', padding='same')(con4)
    # decoder_layer4 = Dropout(DROPOUT_PROB)(encoder_layer5)
    # decoder_layer4 = BatchNormalization()(decoder_layer4)
    decoder_layer4 = Conv2D(128, 1, activation='relu', padding='same')(decoder_layer4)

    # up sample layer 4 to get to layer 3
    up3 = UpSampling2D()(decoder_layer4)
    # concatenate the output of encoder_layer4 with  the up sample
    con3 = concatenate([up3, encoder_layer3])
    # Third layer : 2 x 2D Convolutions, filter size 64,  with a 3x3 kernel size for the first one and 1x1 for the
    # second one and a stride size of (2,2).
    decoder_layer3 = Conv2D(64, 3, activation='relu', padding='same')(con3)
    decoder_layer3 = Conv2D(64, 1, activation='relu', padding='same')(decoder_layer3)

    # up sample layer 3 to get to layer 2
    up2 = UpSampling2D()(decoder_layer3)
    # concatenate the output of encoder_layer4 with  the up sample
    con2 = concatenate([up2, encoder_layer2])
    # Second layer : 2 x 2D Convolutions, filter size 32,  with a 3x3 kernel size for the first one and 1x1 for the
    # second one and a stride size of (2,2).
    decoder_layer2 = Conv2D(32, 3, activation='relu', padding='same')(con2)
    decoder_layer2 = Conv2D(32, 1, activation='relu', padding='same')(decoder_layer2)

    # up sample layer 2 to get to layer 1
    up1 = UpSampling2D()(decoder_layer2)
    # concatenate the output of encoder_layer4 with  the up sample
    con1 = concatenate([up1, encoder_layer1])
    # Third layer : 1 x 2D Convolutions, filter size 32,  with a 3x3 kernel size for the first one and 1x1 for the
    # second one and a stride size of (2,2).
    decoder_layer1 = Conv2D(32, 3, activation='relu', padding='same')(con1)

    output = None

    return Model(inputs=input, outputs=output)
