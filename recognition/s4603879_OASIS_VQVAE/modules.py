import numpy as np

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp
import tensorflow as tf


class Encoder():
    def __init__(self, latent_dim):
        self.latent_dim = latent_dim
        self.model = self.gen_encoder()

    def gen_encoder(self):
        input = keras.Input(shape=(256, 256, 1))
        x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(input)
        x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.Conv2D(128, 3, activation="relu", strides=2, padding="same")(x)
        encoder_outputs = layers.Conv2D(self.latent_dim, 1, padding="same")(x)
        return keras.Model(input, encoder_outputs, name="encoder")

    def get_model(self):
        return self.model

class Decoder():
    def __init__(self, latent_dim):
        self.latent_dim = latent_dim
        self.model = self.gen_decoder()

    def gen_decoder(self):
        input = keras.Input(shape=(32, 32, 30))
        x = layers.Conv2DTranspose(128, 3, activation="relu", strides=2, padding="same")(input)
        x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
        # x = layers.Conv2DTranspose(16, 3, activation="relu", strides=2, padding="same")(x)
        decoder_outputs = layers.Conv2DTranspose(1, 3, padding="same")(x)
        return keras.Model(input, decoder_outputs, name="decoder")

    def get_model(self):
        return self.model


class VQ_VAE():
    def __init__(self, input_shape, latent_dim, embedding_num):
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.embedding_num = embedding_num
        self.encoder = Encoder(latent_dim=latent_dim).get_model()
        self.decoder = Decoder(latent_dim=latent_dim).get_model()
        self.model = self.gen_vq_vae()
        print(self.encoder.summary())
        print(self.decoder.summary())

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def get_model(self):
        return self.model
        
    def gen_vq_vae(self):
        input = keras.Input(shape=self.input_shape)
        encoder_output = self.encoder(input)
        latent_vector = VectorQuantizer(self.embedding_num, self.latent_dim, 0.5)(encoder_output)
        decoder_output = self.decoder(latent_vector)
        return keras.Model(input, decoder_output, name='vq_vae')

class Trainer(keras.models.Model):
    def __init__(self, img_shape, latent_dim, num_embeddings, variance, **kwargs):
        super(Trainer, self).__init__(**kwargs)     
        self.img_shape = img_shape
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings
        self.variance = variance
        self.vq_vae = VQ_VAE(self.img_shape, self.latent_dim, self.num_embeddings).get_model()
        self.total_loss = keras.metrics.Mean(name="loss")
        self.reconstruction_loss = keras.metrics.Mean(name="reconstruction_loss")
        self.vq_loss = keras.metrics.Mean(name="vq_loss")  

    @property
    def metrics(self):
        return [
            self.total_loss,
            self.reconstruction_loss,
            self.vq_loss,
        ]

    def train_step(self, x):
        with tf.GradientTape() as tape:
            # Outputs from the VQ-VAE.
            reconstructions = self.vq_vae(x)

            # Calculate the losses.
            reconstruction_loss = (
                tf.reduce_mean((x - reconstructions) ** 2) / self.variance
            )
            total_loss = reconstruction_loss + sum(self.vq_vae.losses)

        print('gradient calculated')
        # Backpropagation.
        grads = tape.gradient(total_loss, self.vq_vae.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.vq_vae.trainable_variables))

        # Update losses.
        self.total_loss.update_state(total_loss)
        self.reconstruction_loss.update_state(reconstruction_loss)
        self.vq_loss.update_state(sum(self.vq_vae.losses))
        print('updated')

        # Log results.
        return {
            "loss": self.total_loss.result(),
            "reconstruction_loss": self.reconstruction_loss.result(),
            "vqvae_loss": self.vq_loss.result(),
        }


class VectorQuantizer(layers.Layer):
    def __init__(self, num_embeddings, embedding_dim, beta, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.beta = beta  #should be within 0.25 to 2
        # Code book initialization
        codebook_init = tf.random_uniform_initializer()
        self.embeddings = tf.Variable(
            initial_value=codebook_init(shape=(self.embedding_dim, self.num_embeddings), dtype="float32"),
            trainable=True,
            name="codebook_embedding",
        )

    def call(self, x):
        shape = tf.shape(x)
        # reshape it to the shape (embedding_num, embedding_dim)
        flattened = tf.reshape(x, [-1, self.embedding_dim])

        '''Calculation to get quantization.'''
        # Get the index of the vectors in the embedding codebook.
        codebook_indices = self.get_codebook_indices(flattened)
        # One hot encoding to encode indices
        encodings = tf.one_hot(codebook_indices, self.num_embeddings)
        # Get the quantized vector
        quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)
        # Reshape back
        quantized = tf.reshape(quantized, shape)


        # The loss is norm2(stop_gradient(encoder_vector) - decoder_vector) + beta * norm2(encoder_vector - stop_gradient(decoder_vector))
        commitment_loss = tf.reduce_mean((tf.stop_gradient(quantized) - x) ** 2)
        codebook_loss = tf.reduce_mean((quantized - tf.stop_gradient(x)) ** 2)
        self.add_loss(self.beta * commitment_loss + codebook_loss)

        # Estimator.
        quantized = x + tf.stop_gradient(quantized - x)

        return quantized

    def get_codebook_indices(self, input):
        # Calculate norm2 between the input vectors and the codebook vectors.
        similarity = tf.matmul(input, self.embeddings)
        distances = (
            tf.reduce_sum(input ** 2, axis=1, keepdims=True)
            + tf.reduce_sum(self.embeddings ** 2, axis=0) - 2 * similarity
        )

        # Get the closest indices
        closest_indices = tf.argmin(distances, axis=1)
        return closest_indices