from keras.layers import (
    Input,
    Conv2D,
    Add,
    UpSampling2D,
    Concatenate,
    LeakyReLU,
    Concatenate,
    Dropout,
)
from tensorflow_addons.layers import InstanceNormalization
from keras.models import Model


# fmt: off
def ConvBlock(filters: int):
    """ 
    A convolution block consisting of two 2D convolutional layers with instance normalised inputs that are joined by a dropout to prevent overfitting
    Emulates the keras functional API
    """
    def _create(inputs):
        normalized = InstanceNormalization()(inputs)
        conv_layer_1 = Conv2D(filters, (3, 3), padding="same", activation=LeakyReLU(0.01))(normalized)

        dropout = Dropout(0.3)(conv_layer_1)

        normalized = InstanceNormalization()(dropout)
        conv_layer_2 = Conv2D(filters, (3, 3), padding="same", activation=LeakyReLU(0.01))(normalized)
 
        return conv_layer_2

    return _create


def Encoder(filters: int, strides=(2,2)):
    """ 
    The main encoder block which returns the sum of a regular 2D convolutional layer and the above `ConvBlock`
    Emulates the keras functional API
    """
    def _create(inputs):
        conv = Conv2D(filters, (3, 3), padding="same", strides=strides)(inputs)
        encoded = ConvBlock(filters)(conv)
        element_sum = Add()([conv, encoded])
        return element_sum

    return _create


def UpSample(filters: int):
    """
    A wrapper for the `UpSampling2D` layer which also includes a convolutional layer to extract any upsampled features
    Emulates the keras functional API
    """
    def _create(inputs):
        upsampled = UpSampling2D(size=(2, 2))(inputs)
        conv = Conv2D(filters, (3, 3), padding="same", activation=LeakyReLU(0.01))(upsampled)
        return conv

    return _create

def Segmentation(filters: int):
    """
    A wrapper for a regular `Conv2D` layer that creates a one-to-one projection which pools feature maps
    Emulates the keras functional API
    """
    def _create(inputs):
        return Conv2D(filters, (1, 1), padding="same", activation=LeakyReLU(0.01))(inputs)

    return _create

def Localisation(filters: int):
    """
    A localisation module which recombines the upsampled features with the corresponding contextual features
    Emulates the keras functional API
    """
    def _create(inputs):
        conv = Conv2D(filters, (3, 3), padding="same", activation=LeakyReLU(0.01))(inputs)
        conv = Conv2D(filters, (1, 1), padding="same", activation=LeakyReLU(0.01))(conv)
        return conv

    return _create

# fmt: on

from constants import IMG_DIM


def UNet():
    """
    Builds and returns a version of an improved UNet which can be used to segment the ISIC challenge dataset
    """
    base_filters = 16
    inputs = Input(shape=(IMG_DIM, IMG_DIM, 3))

    # Feature extraction and aggregation of high level characteristics
    encoded_256 = Encoder(base_filters, strides=(1, 1))(inputs)
    encoded_128 = Encoder(base_filters * 2)(encoded_256)
    encoded_64 = Encoder(base_filters * 4)(encoded_128)
    encoded_32 = Encoder(base_filters * 8)(encoded_64)
    encoded_16 = Encoder(base_filters * 16)(encoded_32)

    # Upsampling of features and recombination with aggregated features
    upsample_32 = UpSample(base_filters * 8)(encoded_16)
    concat_32 = Concatenate()([upsample_32, encoded_32])
    localisation_32 = Localisation(base_filters * 8)(concat_32)

    upsample_64 = UpSample(base_filters * 4)(localisation_32)
    concat_64 = Concatenate()([upsample_64, encoded_64])
    localisation_64 = Localisation(base_filters * 4)(concat_64)

    # Balance model performance / training time with segmentation layerse
    segmentation_64 = Segmentation(base_filters)(localisation_64)
    segmentation_64 = UpSampling2D()(segmentation_64)

    upsample_128 = UpSample(base_filters * 2)(localisation_64)
    concat_128 = Concatenate()([upsample_128, encoded_128])
    localisation_128 = Localisation(base_filters * 2)(concat_128)

    segmentation_128 = Segmentation(base_filters)(localisation_128)
    segmentation_128 = Add()([segmentation_128, segmentation_64])
    segmentation_128 = UpSampling2D()(segmentation_128)

    upsample_256 = UpSample(base_filters)(localisation_128)
    concat_256 = Concatenate()([upsample_256, encoded_256])
    conv_256 = Conv2D(base_filters, (3, 3), padding="same", activation=LeakyReLU(0.01))(
        concat_256
    )

    segmentation_256 = Segmentation(base_filters)(conv_256)
    segmentation_256 = Add()([segmentation_256, segmentation_128])

    outputs = Conv2D(3, (3, 3), padding="same", activation="sigmoid")(segmentation_256)
    model = Model(inputs, outputs)

    return model
