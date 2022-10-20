import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import InputLayer, Input, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
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


class VectorQuantiser(keras.layers.Layer):
    def __init__(self, num_embeddings, embedding_dimensions, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dimensions = embedding_dimensions
        self.num_embeddings = num_embeddings
        # beta training parameter
        self.commitment_cost = 1

        random_uniform_initialiser = tf.random_uniform_initializer()
        self.embeddings = tf.Variable(
            initial_value = random_uniform_initialiser(
                shape=(embedding_dimensions, num_embeddings),
                dtype="float32"),
                trainable=True,
                name="embeddings_vqvae",
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_embeddings": self.num_embeddings,
            "embedding_dimensions": self.embedding_dimensions
        })
        return config

    def get_code_indices(self, flattened):
        distances = tf.reduce_sum(flattened ** 2, axis=1, keepdims=True) \
                    + tf.reduce_sum(self.embeddings ** 2, axis=0) \
                    - 2 * tf.matmul(flattened, self.embeddings)
        return tf.argmin(distances, axis=1)

    def call(self, x):
        input_shape = tf.shape(x)
        flattened = tf.reshape(x, [-1, self.embedding_dimensions])
        encoding_indices = self.get_code_indices(flattened)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)
        quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)
        unflattened = tf.reshape(quantized, input_shape)

        commitment_loss = self.commitment_cost * tf.reduce_mean((tf.stop_gradient(unflattened) - x) ** 2)
        codebook_loss = tf.reduce_mean((unflattened - tf.stop_gradient(x)) ** 2)
        self.add_loss(commitment_loss + codebook_loss)

        return x + tf.stop_gradient(unflattened - x)


class VQVAE(keras.models.Sequential):
    def __init__(self, variance, latent_dimensions, num_embeddings, **kwargs):
        super(VQVAE, self).__init__(**kwargs)
        self.total_loss_list = []
        self.reconstruction_loss_list = []
        self.vq_loss_list = []
        self.variance = variance
        self.latent_dimensions = latent_dimensions
        self.num_embeddings = num_embeddings

        # Create the Sequential model
        vector_quantiser = VectorQuantiser(num_embeddings, latent_dimensions, name="quantiser")
        encoder = Encoder(latent_dimensions)
        decoder = Decoder()

        # Add the components of the model
        self.add(encoder)
        self.add(vector_quantiser)
        self.add(decoder)

        # Initialise the loss metrics
        self.loss_total = keras.metrics.Mean()
        self.loss_reconstruction = keras.metrics.Mean()
        self.loss_vq = keras.metrics.Mean()


    @property
    def metrics(self):
        return [self.loss_total, self.loss_reconstruction, self.loss_vq]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            reconstructions = self.call(data)
            reconstruction_loss = tf.reduce_mean((data - reconstructions) ** 2) / self.variance
            total_loss = reconstruction_loss + sum(self.losses)

            gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.loss_total.update_state(total_loss)
        self.loss_reconstruction.update_state(reconstruction_loss)
        self.loss_vq.update_state(sum(self.losses))

        losses = {
            "loss": self.loss_total.result(),
            "reconstruction_loss": self.loss_reconstruction.result(),
            "vqvae_loss": self.loss_vq.result(),
        }
        self.total_loss_list.append(losses["loss"])
        self.reconstruction_loss_list.append(losses["reconstruction_loss"])
        self.vq_loss_list.append(losses["vqvae_loss"])

        return losses


class ResidualBlock(keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.conv1 = Conv2D(
            filters=filters, kernel_size=1, activation="relu"
        )
        self.pixel_conv = Conv2D(
            filters = filters // 2,
            kernel_size=3,
            activation="relu",
            padding="same"
        )
        self.conv2 = keras.layers.Conv2D(
            filters=filters, kernel_size=1, activation="relu"
        )

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pixel_conv(x)
        x = self.conv2(x)
        return keras.layers.add([inputs, x])

x = keras.Input(shape=10, dtype=tf.int32)

class OneHotLayer(tf.keras.layers.Layer):
    def __init__(self, n_embeddings, **kwargs):
        super(OneHotLayer, self).__init__(**kwargs)
        self.n_embeddings = n_embeddings

    def call(self, inputs):
        return tf.one_hot(inputs, self.n_embeddings)


class PixelCNN(tf.keras.Model):
    def __init__(self, in_shape, n_embeddings, **kwargs):
        super().__init__(**kwargs)

        self.in_shape = in_shape
        self.n_embeddings = n_embeddings

        self.model = Sequential()
        self.model.add(OneHotLayer(n_embeddings, name="OneHot"))

        self.model.add(
            Conv2D(
                filters=128, kernel_size=7, activation="relu", padding="same"
            )
        )

        for i in range(n_residual_blocks):
            self.model.add(ResidualBlock(filters=128, name=f"ResidualBlock{i}"))

        for i in range(n_pixel_cnn_layers):
            Conv2D(
                filters=128, kernel_size=1, strides=1, activation="relu", padding="valid"
            )

        self.model.add(keras.layers.Conv2D(filters=n_embeddings, kernel_size=1,
                strides=1, padding="valid", name="Conv2D"))

    def get_config(self):
        config = super().get_config()
        config.update({
            "in_shape": self.in_shape,
            "n_embeddings": self.n_embeddings
        })
        return config

    def call(self, x):
        return self.model(x)
