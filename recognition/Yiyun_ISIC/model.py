"""Improved UNet implementation (2D version)

Reference: https://arxiv.org/abs/1802.10508v1
"""
from tensorflow.keras import layers, models, optimizers
import tensorflow_addons as tfa

from metrics import dice_coef, dice_loss


def __encoder_module(input, num_filters, strides=(1, 1)):
    """Encoder module for the Improved UNet

    Args:
        input (layers.Layer): Input layer to the encoder module
        num_filters (int): Number of filters
        strides (tuple, optional): Strides for the convolution. Defaults to (1, 1).

    Returns:
        layers.Layer: Output layer of the encoder module
    """
    conv = layers.Conv2D(num_filters, (3, 3), strides,
                         padding="same", activation=layers.LeakyReLU(0.01))(input)

    # context module (pre-activation residual blocks)
    ctx1 = tfa.layers.InstanceNormalization()(conv)
    ctx1 = layers.Activation(layers.LeakyReLU(0.01))(ctx1)
    ctx1 = layers.Conv2D(num_filters, (3, 3), padding="same")(ctx1)
    ctx_drop = layers.Dropout(0.3)(ctx1)
    ctx2 = tfa.layers.InstanceNormalization()(ctx_drop)
    ctx2 = layers.Activation(layers.LeakyReLU(0.01))(ctx2)
    ctx2 = layers.Conv2D(num_filters, (3, 3), padding="same")(ctx2)

    # element-wise sum
    sum = layers.Add()([conv, ctx2])
    return sum


def __decoder_module(input, encode_output, num_filters, localization_module=True):
    """Decoder module for the Improved UNet

    Args:
        input (layers.Layer): Input layer to the decoder module
        encode_output (layers.Layer): Output layer of the encoder module
        num_filters (int): Number of filters
        localization_module (bool, optional): Whether to use the localization module. Defaults to True.

    Returns:
        layers.Layer: Output layer of the decoder module
    """
    # upsampling module
    up = layers.UpSampling2D((2, 2))(input)
    conv1 = layers.Conv2D(num_filters, (3, 3), padding="same",
                          activation=layers.LeakyReLU(0.01))(up)
    concat = layers.Concatenate()([conv1, encode_output])

    if not localization_module:
        return concat

    # localization module
    conv2 = layers.Conv2D(num_filters, (3, 3), padding="same",
                          activation=layers.LeakyReLU(0.01))(concat)
    conv2 = layers.Conv2D(num_filters, (1, 1), padding="same",
                          activation=layers.LeakyReLU(0.01))(conv2)
    return conv2


def build_model(input_shape):
    """Builds the Improved UNet model

    Args:
        input_shape (tuple): Shape of the input image

    Returns:
        models.Model: Model of the Improved UNet
    """
    inputs = layers.Input(input_shape)

    # downsampling
    down1 = __encoder_module(inputs, 16)
    down2 = __encoder_module(down1, 32, strides=(2, 2))
    down3 = __encoder_module(down2, 64, strides=(2, 2))
    down4 = __encoder_module(down3, 128, strides=(2, 2))
    down5 = __encoder_module(down4, 256, strides=(2, 2))

    # upsampling
    up1 = __decoder_module(down5, down4, 128)
    up2 = __decoder_module(up1, down3, 64)
    up3 = __decoder_module(up2, down2, 32)
    up4 = __decoder_module(up3, down1, 16, localization_module=False)
    conv = layers.Conv2D(32, (3, 3), padding="same",
                         activation=layers.LeakyReLU(0.01))(up4)

    # segmentation layers
    seg1 = layers.Conv2D(1, (1, 1), padding="same",
                         activation=layers.LeakyReLU(0.01))(up2)
    seg1 = layers.UpSampling2D((2, 2), interpolation="bilinear")(seg1)
    seg2 = layers.Conv2D(1, (1, 1), padding="same",
                         activation=layers.LeakyReLU(0.01))(up3)
    seg2 = layers.Add()([seg2, seg1])
    seg2 = layers.UpSampling2D((2, 2), interpolation="bilinear")(seg2)
    seg3 = layers.Conv2D(1, (1, 1), padding="same",
                         activation=layers.LeakyReLU(0.01))(conv)
    seg3 = layers.Add()([seg3, seg2])

    outputs = layers.Activation("sigmoid")(seg3)
    model = models.Model(inputs, outputs, name="AdvUNet")
    return model


class AdvUNet:
    """The Improved UNet model
    """
    def __init__(self, input_shape):
        self.model = build_model(input_shape)

    def compile(self):
        """Compiles the model
        """
        self.model.compile(optimizer=optimizers.Adam(learning_rate=5e-4),
                           loss=dice_loss, metrics=["accuracy", dice_coef])

    def fit(self, train_dataset, val_dataset, batch_size, epochs):
        """Fits the model

        Args:
            train_dataset (tf.data.Dataset): Training dataset
            val_dataset (tf.data.Dataset): Validation dataset
            batch_size (int): Batch size
            epochs (int): Number of epochs

        Returns:
            History: Training history
        """
        return self.model.fit(train_dataset.batch(batch_size), validation_data=val_dataset.batch(batch_size),
                              epochs=epochs, verbose=1)

    def evaluate(self, dataset, batch_size):
        """Evaluates the model

        Args:
            dataset (tf.data.Dataset): Test dataset
            batch_size (int): Batch size

        Returns:
            tuple: Loss, accuracy and dice coefficient
        """
        return self.model.evaluate(dataset.batch(batch_size), verbose=1)
