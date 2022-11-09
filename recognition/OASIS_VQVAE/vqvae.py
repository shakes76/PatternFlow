import tensorflow as tf


def get_closest_embedding_indices(embeddings, inputs):
    # Calculate L2 distance between the embeddings and inputs.
    distances = (tf.reduce_sum(inputs ** 2, axis=1, keepdims=True)
                 + tf.reduce_sum(embeddings ** 2, axis=0)
                 - 2 * tf.matmul(inputs, embeddings))

    # Find and return closest indices based on distances computed above
    return tf.argmin(distances, axis=1)


class VectorQuantizerLayer(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, num_embeddings, beta, name="vq"):
        super().__init__(name=name)

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._beta = beta

        # uniform prior for the embeddings, as explained in the VQVAE paper
        initialiser = tf.random_uniform_initializer()
        embeddings_shape = (self._embedding_dim, self._num_embeddings)
        self._embeddings = tf.Variable(
            initial_value=initialiser(
                embeddings_shape, tf.float32))

    def embeddings(self):
        return self._embeddings

    def call(self, x):
        # Flatten the input and find the current closest embeddings
        flattened_x = tf.reshape(x, [-1, self._embedding_dim])
        encoding_indices = get_closest_embedding_indices(
            self._embeddings, flattened_x)

        # Turn the indices found above into one-hot vectors.
        # This lines up with equation (1) in the VQVAE paper.
        encodings = tf.one_hot(encoding_indices, self._num_embeddings)

        # Equation (2) in the VQVAE paper.
        quantized = tf.reshape(
            tf.matmul(
                encodings,
                self._embeddings,
                transpose_b=True),
            tf.shape(x))

        # Calculate the 2nd and 3rd terms in the loss in equation (3) in the paper.
        # Both of these are just squared-L2 norms.
        quantizer_loss = tf.reduce_mean((quantized - tf.stop_gradient(x)) ** 2)
        commitment_loss = tf.reduce_mean(
            (tf.stop_gradient(quantized) - x) ** 2)

        # Calculate the total loss and use add_loss so the VQVAE class can
        # access it in training loop.
        self.add_loss(quantizer_loss + self._beta * commitment_loss)

        # Return the straight-through estimator
        return x + tf.stop_gradient(quantized - x)


class VQVAE(tf.keras.Model):
    def __init__(
            self,
            img_size,
            latent_dim,
            num_embeddings,
            beta,
            data_variance,
            **kwargs):
        super(VQVAE, self).__init__(**kwargs)

        self._encoder = self._create_encoder(img_size, latent_dim)
        self._decoder = self._create_decoder(
            self._encoder.output.shape[1:], latent_dim)

        self._latent_dim = latent_dim
        self._num_embeddings = num_embeddings
        self._data_variance = data_variance

        self._vq = VectorQuantizerLayer(
            latent_dim, num_embeddings, beta, name="vq")

        self._total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self._reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss")
        self._vq_loss_tracker = tf.keras.metrics.Mean(name="vq_loss")

    @staticmethod
    def _create_encoder(input_shape, latent_dim, name="encoder"):
        return tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=input_shape),
            tf.keras.layers.Conv2D(32, 3, activation="relu", strides=2, padding="same"),
            tf.keras.layers.Conv2D(64, 3, activation="relu", strides=2, padding="same"),
            tf.keras.layers.Conv2D(latent_dim, 1, padding="same")
        ], name=name)

    @staticmethod
    def _create_decoder(input_shape, latent_dim, name="decoder"):
        return tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=input_shape),
            tf.keras.layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same"),
            tf.keras.layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same"),
            tf.keras.layers.Conv2DTranspose(1, 3, padding="same")
        ], name=name)

    def encoder(self):
        return self._encoder

    def decoder(self):
        return self._decoder

    def quantizer(self):
        return self._vq

    def call(self, x):
        # Pass image through the encoder, quantizer, and back out the decoder
        x = self._encoder(x)
        x = self._vq(x)
        x = self._decoder(x)
        return x

    def train_step(self, x):
        with tf.GradientTape() as tape:
            # Reconstruct the images through the autoencoder
            reconstructions = self(x)

            reconstruction_loss = tf.reduce_mean(
                (x - reconstructions) ** 2) / self._data_variance
            total_loss = reconstruction_loss + sum(self._vq.losses)

        # Backpropagate gradients
        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # Update metrics
        self._total_loss_tracker.update_state(total_loss)
        self._reconstruction_loss_tracker.update_state(reconstruction_loss)
        self._vq_loss_tracker.update_state(sum(self._vq.losses))

        # Log results.
        return {
            "loss": self._total_loss_tracker.result(),
            "reconstruction_loss": self._reconstruction_loss_tracker.result(),
            "vq_loss": self._vq_loss_tracker.result()
        }

    @property
    def metrics(self):
        return [
            self._total_loss_tracker,
            self._reconstruction_loss_tracker,
            self._vq_loss_tracker
        ]
