from pickletools import optimize
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras import layers
import tensorflow_probability as tfp
import tensorflow as tf

class VectorQuantizer(layers.Layer):
    """
    Create the custom VectorQuantizer class layer, which is inbetween the encoder and decoder
    """
    def __init__(self, num_embeddings, embedding_dim, beta=0.25, **kwargs):
        """
        Initialise the layer

        Parameters:
        num_embeddings: number of embeddings in the codebook
        embedding_dim: The dimensionality of the embedding vector
        beta: Used to calculate loss, initiates between [0.25, 2] per paper
        """
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.beta = beta

        #Initialise embeddings to be quantized
        w_init = tf.random_uniform_initializer()
        self.embeddings = tf.Variable(
            initial_value=w_init(
                shape=(self.embedding_dim, self.num_embeddings), dtype="float32"
            ),
            trainable=True,
            name="embeddings_vqvae",
        )
    
    def call(self, given):
        # Calculate the shape of the given input
        input_shape = tf.shape(given)
        # flatten the given input keeping dimensionality intact.
        flattened = tf.reshape(given, [-1, self.embedding_dim])

        # Quantization.
        encoding_indices = self.get_code_indices(flattened)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)
        quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)

        # Reshape the quantized values back to the original input shape
        quantized = tf.reshape(quantized, input_shape)

        # Calculate vector quantization loss and add that to the layer. 
        commitment_loss = tf.reduce_mean((tf.stop_gradient(quantized) - given) ** 2)
        codebook_loss = tf.reduce_mean((quantized - tf.stop_gradient(given)) ** 2)
        self.add_loss(self.beta * commitment_loss + codebook_loss)

        # Straight-through estimator.
        quantized = given + tf.stop_gradient(quantized - given)
        return quantized

    def get_code_indices(self, flattened_inputs):
        # Calculate L2-normalized distance between the inputs and the codes.
        similarity = tf.matmul(flattened_inputs, self.embeddings)
        distances = (
            tf.reduce_sum(flattened_inputs ** 2, axis=1, keepdims=True)
            + tf.reduce_sum(self.embeddings ** 2, axis=0)
            - 2 * similarity
        )
        # Derive the indices for minimum distances.
        encoding_indices = tf.argmin(distances, axis=1)
        return encoding_indices

def new_encoder(latent_dim):
    """
    Create the Structure for a typical CNN encoder
    """
    encoder = Sequential(name="encoder")
    encoder.add(Conv2D(32, 3, activation="relu", strides=2, padding="same", input_shape=(256, 256, 1)))
    encoder.add(Conv2D(64, 3, activation="relu", strides=2, padding="same"))
    encoder.add(Conv2D(128, 3, activation="relu", strides=2, padding="same"))
    encoder.add(Conv2D(256, 3, activation="relu", strides=2, padding="same"))
    encoder.add(Conv2D(latent_dim, 1, padding="same"))
    return encoder

def new_decoder():
    """
    Create the Structure for the decoder based on the inverse of the encoder created above
    """
    decoder = Sequential(name="decoder")
    decoder.add(Conv2DTranspose(256, 3, activation="relu", strides=2, padding="same"))
    decoder.add(Conv2DTranspose(128, 3, activation="relu", strides=2, padding="same"))
    decoder.add(Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same"))
    decoder.add(Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same"))
    decoder.add(Conv2DTranspose(1, 3, padding="same"))
    return decoder

class VQVAE(keras.models.Sequential):
    def __init__(self, train_variance, latent_dim=256, num_embeddings=256, **kwargs):
        super(VQVAE, self).__init__(**kwargs)
        self.train_variance = train_variance
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings
        #Create model sequence
        self.vqvae = VectorQuantizer(self.num_embeddings, self.latent_dim, name="vQuantiser")
        encoder = new_encoder(latent_dim, name="encoder")
        decoder = new_decoder(name="decoder")
        #Construct the components for the model
        self.add(encoder)
        self.add(self.vqvae)
        self.add(decoder)
        #Initialise loss components
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.vq_loss_tracker = keras.metrics.Mean(name="vq_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.vq_loss_tracker,
        ]

    def train_step(self, x):
        with tf.GradientTape() as tape:
            # Outputs from the VQ-VAE.
            reconstructions = self.vqvae(x)

            # Calculate the losses.
            reconstruction_loss = (
                tf.reduce_mean((x - reconstructions) ** 2) / self.train_variance
            )
            total_loss = reconstruction_loss + sum(self.vqvae.losses)

        # Backpropagation.
        grads = tape.gradient(total_loss, self.vqvae.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.vqvae.trainable_variables))

        # Loss tracking.
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.vq_loss_tracker.update_state(sum(self.vqvae.losses))

        # Log results.
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "vqvae_loss": self.vq_loss_tracker.result(),
        }

def train(model, data, num_epochs):
    model.compile(optimizer=keras.optimizers.Adam())
    model.get_layer("encoder").summary()
    model.get_layer("decoder").summary()
    model.fit(data, epochs=num_epochs, batch_size=128, validation_split=0.2)
