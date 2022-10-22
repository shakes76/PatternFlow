import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np

length = 256
depth = 16
kernel = 3

def create_encoder(latent_dim=16):
    """ Create a simple encoder """
    encoder = tf.keras.Sequential(name="encoder")
    encoder.add(layers.Conv2D(depth, kernel, activation="relu", strides=2, padding="same", input_shape=(length, length, 1)))
    encoder.add(layers.Conv2D(depth*2, kernel, activation="relu", strides=2, padding="same"))
    encoder.add(layers.Conv2D(depth*4, kernel, activation="relu", strides=2, padding="same"))
    encoder.add(layers.Conv2D(depth*8, kernel, activation="relu", strides=2, padding="same"))
    encoder.add(layers.Conv2D(latent_dim, 1, padding="same"))
    return encoder
        
def create_decoder():
    """ Create a simple decoder """
    decoder = tf.keras.Sequential(name="decoder")
    decoder.add(layers.Conv2D(depth*8, kernel, activation="relu", strides=2, padding="same"))
    decoder.add(layers.Conv2D(depth*4, kernel, activation="relu", strides=2, padding="same"))
    decoder.add(layers.Conv2D(depth*2, kernel, activation="relu", strides=2, padding="same"))
    decoder.add(layers.Conv2D(depth, kernel, activation="relu", strides=2, padding="same"))
    decoder.add(layers.Conv2D(1, kernel, padding="same"))
    return decoder


class VQLayer(layers.Layer):
    def __init__(self, n_embeddings, embedding_dim, beta=0.25, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.n_embeddings = n_embeddings
        self.beta = beta

        # Initialise embeddings
        w_init = tf.random_uniform_initializer()
        self.embeddings = tf.Variable(
            initial_value=w_init(shape=(self.embedding_dim, self.n_embeddings), 
            dtype="float32"), trainable=True, name="vqvae_embeddings"
        )

    def call(self, x):
        # Calc then flatten inputs, not incl embedding dimension
        input_shape = tf.shape(x)
        flattened = tf.reshape(x, [-1, self.embedding_dim])

        # Perform quantisation then reshape quantised values to orig shape
        encoding_indices = self.get_code_indices(flattened)
        encodings = tf.one_hot(encoding_indices, self.n_embeddings)
        quantised = tf.matmul(encodings, self.embeddings, transponse_b=True)
        quantised = tf.reshape(quantised, input_shape)
        
        # Vector quntisation loss is added to the layer
        commitment_loss = tf.reduce_mean((tf.stop_gradient(quantised) - x) ** 2)
        codebook_loss = tf.reduce_mean((quantised - tf.stop_gradient(x)) ** 2)
        self.add_loss(self.beta * commitment_loss - codebook_loss)
        
        # Straight-through estimator 
        return x + tf.stop_gradient(quantised - x)

    def get_code_indices(self, flattened_inputs):
        # Calc L2-normalised dist between inputs and codes
        similarity = tf.matmul(flattened_inputs, self.embeddings)
        distances = (
            tf.reduce_sum(flattened_inputs ** 2, axis=1, keepdims=True) 
            + tf.reduce_sum(self.embeddings ** 2, axis=0) - 2 * similarity
        )

        encoding_indices = tf.argmin(distances, axis=1)
        return encoding_indices


class VQVAEModel(tf.keras.Sequential):
    def __init__(self, variance, latent_dim, n_embeddings, **kwargs):
        super(VQVAEModel, self).__init__(**kwargs)
        self.variance = variance
        self.latent_dim = latent_dim
        self.n_embeddings = n_embeddings

        # Build our model
        self.add(VQLayer(n_embeddings, latent_dim, name="vector quantiser")) 
        self.add(create_encoder(latent_dim))
        self.add(create_decoder())

        # Measure our losses
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="recontruction_loss")
        self.vq_loss_tracker = keras.metrics.Mean(name="vq_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.reconstruction_loss_tracker, self.vq_loss_tracker]

    def train_step(self, x):
        with tf.GradientTape() as tape:
            reconstructions = self.call(x)

            reconstruction_loss = (tf.reduce_mean((x - reconstructions) ** 2) / self.variance)
            total_loss = reconstruction_loss + sum(self.losses)

        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.vq_loss_tracker.update_state(sum(self.losses))

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "vqvae_loss": self.vq_loss_tracker.result()
        }
