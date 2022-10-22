from keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    UpSampling2D,
    BatchNormalizationV2,
    Concatenate,
    LeakyReLU,
    Concatenate,
    Dropout,
)
from keras.models import Model


def ConvBlock(filters: int):
    def _create(inputs):
        conv_layer_1 = Conv2D(filters, (3, 3), padding="same")(inputs)
        batch_norm_1 = BatchNormalizationV2()(conv_layer_1)
        activation_1 = LeakyReLU()(batch_norm_1)

        dropout = Dropout(0.2)(activation_1)

        conv_layer_2 = Conv2D(filters, (3, 3), padding="same")(dropout)
        batch_norm_2 = BatchNormalizationV2()(conv_layer_2)
        activation_2 = LeakyReLU()(batch_norm_2)

        return activation_2

    return _create


def Encoder(filters: int):
    def _create(inputs):
        encoded = ConvBlock(filters)(inputs)
        downsampled = MaxPooling2D(pool_size=(2, 2))(encoded)
        return encoded, downsampled

    return _create


def Decoder(filters: int):
    def _create(inputs, skip):
        upsampled = UpSampling2D(size=(2, 2))(inputs)
        concatenated = Concatenate()([upsampled, skip])
        conv = ConvBlock(filters)(concatenated)
        return conv

    return _create


class UNet:
    def __init__(self, image_shape=(256, 256, 1), base_filters=32):
        self.image_shape = image_shape
        self.base_filters = base_filters

    def __call__(self):
        inputs = Input(self.image_shape)

        encoded_256, downsampled_128 = Encoder(self.base_filters)(inputs)
        encoded_128, downsampled_64 = Encoder(self.base_filters * 2)(downsampled_128)
        encoded_64, downsampled_32 = Encoder(self.base_filters * 4)(downsampled_64)
        encoded_32, downsampled_16 = Encoder(self.base_filters * 8)(downsampled_32)

        encoded_16 = ConvBlock(self.base_filters * 16)(downsampled_16)

        decoded_32 = Decoder(self.base_filters * 8)(encoded_16, encoded_32)
        decoded_64 = Decoder(self.base_filters * 4)(decoded_32, encoded_64)
        decoded_128 = Decoder(self.base_filters * 2)(decoded_64, encoded_128)
        decoded_256 = Decoder(self.base_filters)(decoded_128, encoded_256)

        outputs = Conv2D(1, (1, 1))(decoded_256)

        model = Model(inputs, outputs)

        return model
