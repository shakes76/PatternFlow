import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras import layers
import tensorflow as tf

import load_oasis_data

img_length = 256

def create_encoder(latent_dimensions):
    encoder = Sequential(name="Encoder")
    encoder.add(Conv2D(32, 3, activation="relu", strides=2, padding="same", input_shape=(img_length, img_length, 1)))
    encoder.add(Conv2D(64, 3, activation="relu", strides=2, padding="same"))
    encoder.add(Conv2D(latent_dimensions, 1, padding="same"))
    return encoder

def create_decoder(encoder: Sequential, latent_dimensions):
    decoder = Sequential(name="Decoder")
    input_shape = (32, 32, 16)
    decoder.add(Conv2DTranspose(64, 3, activation="relu", strides="2", padding="same", pe=input_shape))
    decoder.add(Conv2DTranspose(32, 3, activation="relu", strides="2", padding="same"))
    decoder.add(Conv2DTranspose(1, 3, padding="same"))
    return decoder

class VectorQuantiser(keras.layers.Layer):
    def __init__(self, num_embeddings, embedding_dimensions, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dimensions = embedding_dimensions
        self.num_embeddings = num_embeddings
        self.beta = 0.25

        random_uniform_initialiser = tf.random_uniform_initializer()
        self.embeddings = tf.Variable(
            initial_value= random_uniform_initialiser(shape=(embedding_dimensions, num_embeddings), dtype="float32"),
            trainable=True,
            name="embeddings_vqvae",
        )

    def get_code_indices(self, inputs):
        similarity = tf.matmul(inputs, self.embeddings)
        distances = (
            tf.reduce_sum(inputs ** 2, axis=1, keepdims=True)
            + tf.reduce_sum(self.embedding_dimensions ** 2, axis=0)
            - 2 * similarity
        )
        return tf.argmin(distances, axis=1)

    def call(self, x):
        input_shape = tf.shape(x)
        flattened = tf.reshape(x, [-1, self.embedding_dimensions])

        encoding_indices = self.get_code_indices(flattened)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)
        quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)
        unflattened = tf.reshape(quantized, input_shape)

        commitment_loss = self.beta * tf.reduce_mean(
            (tf.stop_gradient(unflattened) - x) ** 2
        )
        codebook_loss = tf.reduce_mean((unflattened - tf.stop_gradient(x)) ** 2)
        self.add_loss(commitment_loss + codebook_loss)

        return x + tf.stop_gradient(unflattened - x)



class VQVae(keras.models.Sequential):
    def __init__(self, variance, latent_dimensions, num_embeddings, **kwargs):

        super(VQVae, self).__init__(**kwargs)
        self.variance = variance
        self.latent_dimensions = latent_dimensions
        self.num_embeddings = num_embeddings

        # Create the Sequential model
        vector_quantiser = VectorQuantiser(num_embeddings, latent_dimensions, name="vector_quantiser")
        encoder = create_encoder(latent_dimensions)
        decoder = create_decoder()

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
            reconstruction_loss = (
                    tf.reduce_mean((data - reconstructions) ** 2) / self.variance
            )
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

        return losses

def train_vqvae(vqvae: VQVae):
    x_train, x_test = load_oasis_data.get_data()

    x_train = np.expand_dims(x_train, -1)
    x_train_scaled = (x_train / 255.0) - 0.5

    data_variance = np.var(x_train / 255.0)
    vqvae = VQVae(data_variance, latent_dimensions=32, num_embeddings=64)
    vqvae.compile(optimizer=keras.optimizers.Adam())
    vqvae.fit(x_train_scaled, epochs=2, batch_size=128)

