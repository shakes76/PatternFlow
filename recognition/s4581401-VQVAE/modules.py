import tensorflow as tf
import numpy as np
from tensorflow import keras

class VectorQuantizerLayer(tf.keras.layers.Layer):
    """
    Custom layer to represent the discrete latent space used within the VQVAE model
    """
    def __init__(self, embedding_num, embedding_dim, beta, **kwargs):
        super().__init__(**kwargs)
        # Embedding parameters
        self._embedding_num = embedding_num
        self._embedding_dim = embedding_dim

        # Scales commitment loss for encoder picking an embedding
        self._commitment_cost = beta

        # Initialise the embeddings which we are going to quantise using a uniform distribution
        initial_embedding = tf.random_uniform_initializer()
        self._embedding_shape = (self._embedding_dim, self._embedding_num)
        self._embedding = tf.Variable(initial_value=initial_embedding(shape=self._embedding_shape),
                                                                      dtype="float32",
                                                                      trainable=True)

    def call(self, x):
        # We need to flatten all of the dimensions so that we get a total number of vectors to be quantized independently
        input_shape = tf.shape(x)

        flattened_vector = tf.reshape(x, [-1, self._embedding_dim])
        # Now we need to find the embedding indices for each of the flattened vectors based on minimising the L2 normalised distance
        encoding_indices = self.get_closest_index(flattened_vector)

        # Need to one-hot encode them so that quantisation can occur
        one_hot_encoding = tf.one_hot(encoding_indices, self._embedding_num)

        # Calculate quantised values and return the flattened vector into the original input shape
        quantized_vectors = tf.matmul(one_hot_encoding, self._embedding, transpose_b=True)
        quantized_vectors = tf.reshape(quantized_vectors, input_shape)

        # Calculate the quantized vector loss and add it to the layer. For losses we use mean squared error.
        commitment_loss = tf.reduce_mean((x - tf.stop_gradient(quantized_vectors) ) ** 2)
        # Embeddings optimised by codebook loss
        codebook_loss = tf.reduce_mean((tf.stop_gradient(x) - quantized_vectors) ** 2)
        self.add_loss(self._commitment_cost * commitment_loss + codebook_loss)
        # Note we use a stopgradient operator. Defined as identity during a forward computation time, and has zero partial
        # derivatives. Therefore, the operand it is applied to becomes a non-updated constant.

        # Using stop_gradient, during backpropagation the gradients of the output are given back to the inputs
        quantized_vectors = x + tf.stop_gradient(quantized_vectors - x)
        return quantized_vectors

    def get_closest_index(self, flattened_vector):
        """
        Returns the indices of the closest embeddings inside the codebook for the given flattened pixel vector

        params:
            flattened_vector - Flattened image vector

        Returns:
            Index of the closest embedding in the codebook
        """

        pixel_vector_len = tf.reduce_sum(flattened_vector ** 2, axis=1, keepdims=True) #(Height * Width, 1)
        embedding_len = tf.reduce_sum(self._embedding ** 2, axis=0, keepdims=True) #(1, #embeddings)
        subtract_term = tf.matmul(flattened_vector, self._embedding) # (Height*Width, Channel) x (Channel, #embeddings) -> (Height*Width, #n_embeddings)
        distances = pixel_vector_len + embedding_len - 2*subtract_term
        return tf.argmin(distances, axis = 1)


class VQVAEModel(tf.keras.Model):
    """
    Custom trainable keras model to implement the VQ-VAE Model.
    Model is able to be compiled with self.compile() and trained with self.fit()

    Adapted from https://keras.io/examples/generative/vq_vae/
    """
    def __init__(self, img_shape, embedding_num, embedding_dim, beta, data_variance, **kwargs):
        super(VQVAEModel, self).__init__(**kwargs)

        # Parameters
        self._img_shape = img_shape
        self._embedding_num = embedding_num
        self._embedding_dim = embedding_dim
        self._beta = beta #Commitment loss in Vector Quantized Layer
        self._data_variance = data_variance

        # Model components
        self._encoder = self.create_encoder(self._embedding_dim, self._img_shape)
        self._vq = VectorQuantizerLayer(
            self._embedding_num, self._embedding_dim, self._beta)
        self._decoder = self.create_decoder(self._img_shape)

        # Loss metrics
        self._total_loss = keras.metrics.Mean(name="total_loss")
        self._reconstruction_loss = keras.metrics.Mean(name="reconstruction_loss")
        self._vq_loss = keras.metrics.Mean(name="vq_loss")
        self._mean_ssim = keras.metrics.Mean(name="mean_ssim")

    def get_vq(self):
        """
        Returns the Vector Quantized Layer within the model. If the VQ-VAE model is trained, returns the trained vector
        quantized layer
        """
        return self._vq

    def create_encoder(self, embedding_dim, img_shape):
        """
        Method initialises the encoder for the VQ-VAE model. Should only be called inside the class, not externally

        params:
            embedding_dim - Number of embeddings in the codebook
            img_shape - Shape of the image being encoded
        Returns:
            A Sequential Keras model which is the encoder
        """
        # My encoder implementation
        encoder_model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=img_shape),
            tf.keras.layers.Conv2D(32, 3, activation="relu", strides=2, padding="same"),
            tf.keras.layers.Conv2D(64, 3, activation="relu", strides=2, padding="same"),
            tf.keras.layers.Conv2D(64, 3, activation="relu", strides=2, padding="same"),
            tf.keras.layers.Conv2D(embedding_dim, 1, padding="same")
        ], name="encoder")

        return encoder_model

    def get_encoder(self):
        """
        Returns the Encoder Model within the model. If the VQ-VAE model is trained, returns the trained encoder
        """
        return self._encoder

    def create_decoder(self, img_shape):
        """
        Method initialises the encoder for the VQ-VAE model. Should only be called inside the class, not externally

        params:
            embedding_dim - Number of embeddings in the codebook
            img_shape - Shape of the image being encoded
        Returns:
            A Sequential Keras model which is the decoder
        """
        decoder_model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=self.get_encoder().output.shape[1:]),
            tf.keras.layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same"),
            tf.keras.layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same"),
            tf.keras.layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same"),
            tf.keras.layers.Conv2DTranspose(3, 3, activation="relu", padding="same")
        ], name="decoder")
        return decoder_model

    def get_decoder(self):
        """
        Returns the Decoder Model within the model. If the VQ-VAE model is trained, returns the trained decoder
        """
        return self._decoder

    def create_vqvae_model(self):
        """
        Method to create the VQ-VAE model. Only done to be able to run the model summary to see the number
        of parameters and the structure of the model. Implemented for flexibility of implementation, but not actually
        implemented / used in the project

        Returns:
        A keras model representing the VQ-VAE model used for looking at the summary of the model structure
        """
        initial_layer = tf.keras.Input(shape=self._img_shape)
        encoder_layer = self._encoder(initial_layer)
        vq_layer = self._vq(encoder_layer)
        reconstruction_layer = self._decoder(vq_layer)
        return tf.keras.Model(initial_layer, reconstruction_layer, name="vqvae")

    def call(self, x):
        x = self._encoder(x)
        x = self._vq(x)
        x = self._decoder(x)
        return x

    @property
    def metrics(self):
        """
        Returns: A list of the metrics used for training the model
        """
        return [
            self._total_loss,
            self._reconstruction_loss,
            self._vq_loss
        ]

    def train_step(self, x):
        """
        The method contains the mathematical logic for one step of training.
        Includes the forward pass, loss calculation, backpropagation, and metric updates.
        """

        with tf.GradientTape() as tape:
            reconstructed_img = self(x)

            reconstruction_loss = (
                    tf.reduce_mean((x - reconstructed_img) ** 2) / self._data_variance
            )
            vq_loss = sum(self._vq.losses)
            total_loss = reconstruction_loss + vq_loss
        #    mean_ssim_loss = tf.reduce_mean(tf.image.ssim(x, reconstructed_img, 1.0))

        # Backpropagation step
        # Find the gradients and then apply them
        gradients = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        # Track the new losses
        self._total_loss.update_state(total_loss)
        self._reconstruction_loss.update_state(reconstruction_loss)
        self._vq_loss.update_state(vq_loss)
        #   self._mean_ssim.update_state(mean_ssim_loss)

        return {
            "loss": self._total_loss.result(),
            "reconstruction_loss": self._reconstruction_loss.result(),
            "vq loss": self._vq_loss.result()
        }

    def test_step(self, x):
        """
        The method contains the mathematical logic for one step validation step.
        Includes the forward pass, loss calculation, backpropagation, and metric updates.
        """
        with tf.GradientTape() as tape:
            reconstructed_img = self(x)

            reconstruction_loss = (
                    tf.reduce_mean((x - reconstructed_img) ** 2) / self._data_variance
            )
            vq_loss = sum(self._vq.losses)
            total_loss = reconstruction_loss + vq_loss
            mean_ssim_loss = tf.reduce_mean(tf.image.ssim(x, reconstructed_img, 1.0))

        # Backpropagation step
        # Find the gradients and then apply them
        gradients = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        # Track the new losses
        self._total_loss.update_state(total_loss)
        self._reconstruction_loss.update_state(reconstruction_loss)
        self._vq_loss.update_state(vq_loss)
        self._mean_ssim.update_state(mean_ssim_loss)

        return {
            "loss": self._total_loss.result(),
            "reconstruction_loss": self._reconstruction_loss.result(),
            "vq loss": self._vq_loss.result(),
            "mean ssim": self._mean_ssim.result()
        }


class PixelConvLayer(tf.keras.layers.Layer):
    """
    Custom keras layer representing a single Pixel Convolutional Layer
    """
    def __init__(self, mask_type, **kwargs):
        super(PixelConvLayer, self).__init__()
        self._mask_type = mask_type
        self._convolution = tf.keras.layers.Conv2D(**kwargs)

    def build(self, img_shape):
        """
        Builds the layer based on a convolutional layer, with appropriate based implemented
        params:
            img_shape - Shape of the input image
        """
        self._convolution.build(input_shape=img_shape)
        self._conv_kernel_shape = self._convolution.kernel.get_shape()
        self.mask = np.zeros(shape=self._conv_kernel_shape)
        # Reference for setting up the mask propertly: https://keras.io/examples/generative/pixelcnn/
        self.mask[: self._conv_kernel_shape[0] // 2, ...] = 1.0
        self.mask[self._conv_kernel_shape[0] // 2, : self._conv_kernel_shape[1] // 2, ...] = 1.0

        # Mask type B referes to the mask that allows connections to predicted colours in the current pixel images
        # Mask type A applied to the first convolutional layer, restricting connections to colors in current pixels that have
        # already been predicted.
        if self._mask_type == "B":
            self.mask[self._conv_kernel_shape[0] // 2, self._conv_kernel_shape[1] // 2, ...] = 1.0

    def call(self, x):
        self._convolution.kernel.assign(self._convolution.kernel * self.mask)
        return self._convolution(x)


class ResidualBlock(tf.keras.layers.Layer):
    """
    Custom keras layer representing a single residual block
    Layer consists of a Pixel Convolution Layer between two 2D Convolutional layers
    """
    def __init__(self, input_filters, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)

        self._conv_layer_1 = tf.keras.layers.Conv2D(input_filters, kernel_size=1, activation="relu")
        self._pixel_conv_layer_2 = PixelConvLayer(mask_type="B",
                                                  filters=input_filters // 2,
                                                  kernel_size=3,
                                                  padding="same",
                                                  activation="relu"
                                                  )
        self._conv_layer_3 = tf.keras.layers.Conv2D(input_filters, kernel_size=1, activation="relu")

    def call(self, input_arg):
        x = self._conv_layer_1(input_arg)
        x = self._pixel_conv_layer_2(x)
        x = self._conv_layer_3(x)
        return tf.keras.layers.add([input_arg, x])


class PixelCNNModel(tf.keras.Model):
    """
    Custom trainable keras model to implement the PixelCNN Model.
    Model is able to be compiled with self.compile() and trained with self.fit()

    Adapted from https://keras.io/examples/generative/vq_vae/
    """
    def __init__(self, input_shape, embedding_num, input_filters, num_res_layer, num_pixel_layer, **kwargs):
        super(PixelCNNModel, self).__init__(**kwargs)

        # Parameters
        self._input_shape = input_shape
        self._embedding_num = embedding_num
        self._filters = input_filters
        self._num_res_layer = num_res_layer
        self._num_pixel_layer = num_pixel_layer

        # Initialising the model
        input_layer = tf.keras.Input(shape=self._input_shape, dtype=tf.int32)
        one_hot_input = tf.one_hot(input_layer, self._embedding_num)
        x = PixelConvLayer(mask_type="A", filters=self._filters, kernel_size=7, activation="relu", padding="same")(
            one_hot_input)
        # Adding residual blocks
        for _ in range(self._num_res_layer):
            x = ResidualBlock(self._filters)(x)

        # Adding Pixel Convolution Layers
        for _ in range(self._num_pixel_layer):
            x = PixelConvLayer(mask_type="B", filters=self._filters, kernel_size=1, strides=1, activation="relu",
                               padding="valid")(x)

        final_output = tf.keras.layers.Conv2D(filters=self._embedding_num, kernel_size=1, strides=1, padding="valid")(x)

        self._pixel_cnn_model = tf.keras.Model(input_layer, final_output, name="pixel_cnn")
        self._total_loss = tf.keras.metrics.Mean(name="total_loss")

    def call(self, x):
        return self._pixel_cnn_model(x)

    def train_step(self, x):
        """
        The method contains the mathematical logic for one step of training.
        Includes the forward pass, loss calculation, backpropagation, and metric updates.
        """

        with tf.GradientTape() as tape:
            predicted_output = self(x)
            total_loss = self.compiled_loss(x, predicted_output)

        gradients = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        self._total_loss.update_state(total_loss)
        return {
            "loss": self._total_loss.result()
        }

    def test_step(self, x):
        """
        The method contains the mathematical logic for one step validation step.
        Includes the forward pass, loss calculation, backpropagation, and metric updates.
        """

        with tf.GradientTape() as tape:
            predicted_output = self(x)
            total_loss = self.compiled_loss(x, predicted_output)

        gradients = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        self._total_loss.update_state(total_loss)
        return {
            "loss": self._total_loss.result()
        }
