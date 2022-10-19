import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from constants import image_shape, n_channels, n_residual_blocks, n_pixel_cnn_layers

class Encoder(tf.keras.Model):
    def __init__(self, latent_dimensions, **kwargs):
        super().__init__(**kwargs)
        self.encoder = Sequential(name="encoder")
        # filters, kernel_size
        self.encoder.add(Conv2D(16, 3, activation="relu", strides=2, padding="same",\
                        input_shape=(image_shape[0], image_shape[0], n_channels)))
        self.encoder.add(Conv2D(32, 3, activation="relu", strides=2, padding="same"))
        self.encoder.add(Conv2D(64, 3, activation="relu", strides=2, padding="same"))
        self.encoder.add(Conv2D(128, 3, activation="relu", strides=2, padding="same"))
        self.encoder.add(Conv2D(latent_dimensions, 1, padding="same"))

        self.out_shape = (1, 16, 16, 16)

    def call(self, x):
        return self.encoder(x)

class Decoder(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.decoder = Sequential(name="decoder")
        self.decoder.add(Conv2DTranspose(128, 3, activation="relu", strides=2, padding="same"))
        self.decoder.add(Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same"))
        self.decoder.add(Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same"))
        self.decoder.add(Conv2DTranspose(16, 3, activation="relu", strides=2, padding="same"))
        self.decoder.add(Conv2DTranspose(1, 3, padding="same"))

    def call(self, x):
        return self.decoder(x)
