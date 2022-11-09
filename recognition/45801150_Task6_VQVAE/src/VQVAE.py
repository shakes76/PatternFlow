from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras import layers
import tensorflow as tf


img_length = 256

def create_encoder(latent_dimensions):
    """
    Typical CNN encoder
    """
    encoder = Sequential(name="encoder")
    encoder.add(Conv2D(16, 3, activation="relu", strides=2, padding="same", input_shape=(img_length, img_length, 1)))
    encoder.add(Conv2D(32, 3, activation="relu", strides=2, padding="same"))
    encoder.add(Conv2D(64, 3, activation="relu", strides=2, padding="same"))
    encoder.add(Conv2D(128, 3, activation="relu", strides=2, padding="same"))
    encoder.add(Conv2D(latent_dimensions, 1, padding="same"))
    return encoder

def create_decoder():
    """
    Typical CNN decoder
    """
    decoder = Sequential(name="decoder")
    decoder.add(Conv2DTranspose(128, 3, activation="relu", strides=2, padding="same"))
    decoder.add(Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same"))
    decoder.add(Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same"))
    decoder.add(Conv2DTranspose(16, 3, activation="relu", strides=2, padding="same"))
    decoder.add(Conv2DTranspose(1, 3, padding="same"))
    return decoder


class VectorQuantiser(keras.layers.Layer):
    def __init__(self, num_embeddings, embedding_dimensions, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dimensions = embedding_dimensions
        self.num_embeddings = num_embeddings
        self.commitment_cost = 0.25

        # Iniitialise random embeddings
        random_uniform_initialiser = tf.random_uniform_initializer()
        self.embeddings = tf.Variable(
            initial_value = random_uniform_initialiser(shape=(embedding_dimensions, num_embeddings), dtype="float32"),
            trainable=True,
            name="embeddings_vqvae",
        )

    def get_code_indices(self, flattened):
        """
        Get L2 normalised distance between flattened input and code
        """
        distances = tf.reduce_sum(flattened ** 2, axis=1, keepdims=True) \
                    + tf.reduce_sum(self.embeddings ** 2, axis=0) \
                    - 2 * tf.matmul(flattened, self.embeddings)
        return tf.argmin(distances, axis=1)

    def call(self, x):
        input_shape = tf.shape(x)
        flattened = tf.reshape(x, [-1, self.embedding_dimensions])

        # Quantise the vectors
        encoding_indices = self.get_code_indices(flattened)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)
        quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)
        unflattened = tf.reshape(quantized, input_shape)

        commitment_loss = self.commitment_cost * tf.reduce_mean((tf.stop_gradient(unflattened) - x) ** 2)
        codebook_loss = tf.reduce_mean((unflattened - tf.stop_gradient(x)) ** 2)
        self.add_loss(commitment_loss + codebook_loss)

        # Straight through loss estimator
        return x + tf.stop_gradient(unflattened - x)



class VQVae(keras.models.Sequential):
    def __init__(self, variance, latent_dimensions, num_embeddings, **kwargs):

        super(VQVae, self).__init__(**kwargs)
        self.total_loss_list = []
        self.reconstruction_loss_list = []
        self.vq_loss_list = []
        self.variance = variance
        self.latent_dimensions = latent_dimensions
        self.num_embeddings = num_embeddings

        # Create the Sequential model
        vector_quantiser = VectorQuantiser(num_embeddings, latent_dimensions, name="quantiser")
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
            # Forward propagate
            reconstructions = self.call(data)
            reconstruction_loss = tf.reduce_mean((data - reconstructions) ** 2) / self.variance
            total_loss = reconstruction_loss + sum(self.losses)

        # Backpropagate
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

def train_vqvae(vqvae, x_train_normalised, x_val_normalised, n_epochs):
    vqvae.compile(optimizer=keras.optimizers.Adam())
    vqvae.get_layer("encoder").summary()
    vqvae.get_layer("decoder").summary()
    vqvae.fit(x_train_normalised, validation_split=0.1, epochs=n_epochs, batch_size=128)




