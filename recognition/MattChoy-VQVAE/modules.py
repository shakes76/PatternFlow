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
