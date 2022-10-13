"""
    Defines all modules and components of the VQ-VAE and PixelCNN.

    Author: Adrian Rahul Kamal Rajkamal
    Student Number: 45811935
"""
import tensorflow as tf
from tensorflow import keras
from keras import layers, Model, Input

import numpy as np

""" VQ-VAE """


class VQ(layers.Layer):
    """
        Defines custom Vector-Quantisation (VQ) Layer for VQ-VAE.
    """

    def __init__(self, num_encoded, latent_dim, beta=0.25, name="vq"):
        super().__init__(name=name)

        self._latent_dim = latent_dim
        self._num_encoded = num_encoded
        self._beta = beta

        # Provide a Uniform Distribution Prior for encoded vectors
        runif_initialiser = tf.random_uniform_initializer()
        encoded_shape = self._latent_dim, self._num_encoded
        self._encoded = tf.Variable(initial_value=runif_initialiser(shape=encoded_shape,
                                                                    dtype="float32"))

    def get_encoded(self):
        """ Returns encoded vectors """
        return self._encoded

    def get_codebook_indices(self, inputs):
        """
            Returns the codebook indices of the encodings 'closest' (defined by a normalised
            Euclidean norm) to the input.

            Args:
                inputs: Input (flattened to vectors)

            Returns: indices of the closest encodings, with respect to normalised Euclidean norm
        """
        # Calculate 'closeness' between the (flattened) inputs and the encodings.
        norms = (
                tf.reduce_sum(inputs ** 2, axis=1, keepdims=True) +
                tf.reduce_sum(self._encoded ** 2, axis=0) -
                2 * tf.linalg.matmul(inputs, self._encoded)
        )
        return tf.argmin(norms, axis=1)

    def call(self, inputs):
        """
            Forward computation of this layer.

            Args:
                inputs: inputs of this layer

            Returns: outputs of this layer

        """
        # Flatten input and store original dimensions for reshaping later
        original_shape = tf.shape(inputs)
        flattened_input = tf.reshape(inputs, [-1, self._latent_dim])

        # Perform quantization (i.e. compression)
        encoded_indices = self.get_codebook_indices(flattened_input)
        onehot_indices = tf.one_hot(encoded_indices, self._num_encoded)
        quantized = tf.reshape(
            tf.linalg.matmul(onehot_indices, self._encoded, transpose_b=True),
            original_shape
        )

        # Calculate vector quantization loss from [1] in README.md. The stop_gradient function
        # treats its input as a constant during forward computation, i.e. stopping the computation
        # of its gradient, as it would effectively be 0, hence detaching it from the computational
        # graph.
        quantized_loss = tf.reduce_mean((quantized - tf.stop_gradient(inputs)) ** 2)
        commitment_loss = tf.reduce_mean((tf.stop_gradient(quantized) - inputs) ** 2)
        self.add_loss(self._beta * commitment_loss + quantized_loss)  # Total loss in training loop

        # Return the (straight-through) estimator (i.e. an estimator that is treated like an
        # identity function with respect to gradients during backprop, and is detached from the
        # computational graph as a result)
        return inputs + tf.stop_gradient(quantized - inputs)


class Encoder(Model):
    """
        Defines Encoder for VQ-VAE.
    """

    def __init__(self, latent_dim=16, name="encoder"):
        super(Encoder, self).__init__(name=name)
        self._latent_dim = latent_dim

        # self.input1 = Input(input_shape=self._img_size)
        self.conv1 = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")
        self.conv2 = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")
        self.conv3 = layers.Conv2D(self._latent_dim, 1, padding="same")

    def call(self, inputs):
        """
            Forward computation of this block.

            Args:
                inputs: inputs of this block

            Returns: outputs of this block
        """
        # inputs = self.input1(inputs)
        hidden = self.conv1(inputs)
        hidden = self.conv2(hidden)
        return self.conv3(hidden)


class Decoder(Model):
    """
        Defines Decoder for VQ-VAE.
    """

    def __init__(self, name="decoder", **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        # self.input1 = layers.InputLayer(input_shape=input_dim)
        self.conv_t1 = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")
        self.conv_t2 = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")
        self.conv_t3 = layers.Conv2DTranspose(1, 3, padding="same")

    def call(self, inputs):
        """
            Forward computation of this block.

            Args:
                inputs: inputs of this block

            Returns: outputs of this block
        """
        # inputs = self.input1(inputs)
        hidden = self.conv_t1(inputs)
        hidden = self.conv_t2(hidden)
        return self.conv_t3(hidden)


class VQVAE(Model):
    """
        Defines main VQ-VAE architecture.
    """

    def __init__(self, tr_var, img_size=28, num_encoded=64, latent_dim=16,
                 beta=0.25, name="vq_vae"):
        super(VQVAE, self).__init__(name=name)
        self._tr_var = tr_var
        self._img_size = img_size
        self._num_encoded = num_encoded
        self._latent_dim = latent_dim
        self._beta = beta

        self._encoder = Encoder(self._latent_dim)
        self._vq = VQ(self._num_encoded, self._latent_dim, self._beta)
        self._decoder = Decoder()

        self._total_loss = tf.keras.metrics.Mean(name="total_loss")
        self._vq_loss = tf.keras.metrics.Mean(name="vq_loss")
        self._reconstruction_loss = tf.keras.metrics.Mean(name="reconstruction_loss")

    def call(self, inputs):
        """
            Forward computation of this model.

            Args:
                inputs: inputs of this model

            Returns: outputs of this model
        """
        encoded = self._encoder(inputs)
        quantised = self._vq(encoded)
        decoded = self._decoder(quantised)
        return decoded

    def train_step(self, data):
        """
            Performs one iteration of training and return loss values.

            Args:
                data: input data

            Returns: loss values

        """
        with tf.GradientTape() as tape:
            # Get reconstructions
            recon = self.vqvae(data)

            # Obtain loss values
            reconstruction_loss_val = (
                    tf.reduce_mean((data - recon) ** 2) / self._tr_var
            )
            total_loss_val = reconstruction_loss_val + sum(self.get_vq().losses)

        # Perform Backpropagation
        grads = tape.gradient(total_loss_val, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # Update loss values
        self.total_loss.update_state(total_loss_val)
        self.reconstruction_loss.update_state(reconstruction_loss_val)
        self.vq_loss.update_state(sum(self.get_vq().losses))

        # Log results.
        return {
            "loss": self.total_loss.result(),
            "vq_loss": self.vq_loss.result(),
            "reconstruction_loss": self.reconstruction_loss.result(),
        }

    @property
    def metrics(self):
        """ Returns metrics """
        return [
            self.total_loss,
            self.vq_loss,
            self.reconstruction_loss,
        ]

    def get_encoder(self):
        """ Returns encoder """
        return self._encoder

    def get_vq(self):
        """ Returns VQ layer """
        return self._vq

    def get_decoder(self):
        """ Returns decoder """
        return self._decoder


# vqvae = VQVAE(.3)
# vqvae.build((None, 28, 28, 1))
# vqvae.summary()

""" PixelCNN """


class PixelConv(layers.Layer):
    """
        Represents a custom Conv layer with masking of certain sections of filters for
        autoregressive learning.
    """

    def __init__(self, kernel_mask_type, name="pixel_conv", **kwargs):
        super(PixelConv, self).__init__(name=name)
        self._main_conv = layers.Conv2D(**kwargs)

        # Either "A" or "B" - see self.build() for further details
        self._kernel_mask_type = kernel_mask_type

    def build(self, input_shape):
        """
            Define layer variables, etc.

            Args:
                input_shape: Shape of layer's input
        """
        self._main_conv.build(input_shape)
        kernel_shape = self._main_conv.kernel.get_shape()

        # Creates a way to 'mask out' unseen bits to achieve autoregressive behaviour. There are
        # two types of masks, namely, type "A" and "B".

        # Mask "A" zeroes out all pixels above and left of the pixel in the middle of the mask.
        # Mask "B" is similar to Mask "A", but does not zero out the middle pixel itself.
        self._kernel_mask = np.zeros(shape=kernel_shape)
        self._kernel_mask[:kernel_shape[0] // 2, ...] = 1.0
        self._kernel_mask[kernel_shape[0] // 2, :kernel_shape[1] // 2, ...] = 1.0
        if self._kernel_mask_type == "B":
            # Include the middle pixel as well
            self._kernel_mask[kernel_shape[0] // 2, kernel_shape[1] // 2, ...] = 1.0

    def call(self, inputs):
        """
            Forward computation of this layer.

            Args:
                inputs: inputs of this layer

            Returns: outputs of this layer

        """
        self._main_conv.kernel.assign(self._main_conv.kernel * self._kernel_mask)
        return self._main_conv(inputs)


class PixelResidualBlock(layers.Layer):
    """
        Defines a typical residual block, but using the PixelConv layer.
    """

    def __init__(self, num_filters, name="pixel_res_block"):
        super(PixelResidualBlock, self).__init__(name=name)
        self._num_filters = num_filters

        self._conv1 = layers.Conv2D(filters=self._num_filters, kernel_size=1, activation="relu",
                                    padding="same")
        self._pixel_conv = PixelConv(filters=self._num_filters // 2,
                                     kernel_size=3, activation="relu",
                                     padding="same", kernel_mask_type="B")
        self._conv2 = layers.Conv2D(filters=self._num_filters, kernel_size=1, activation="relu",
                                    padding="same")

    def call(self, inputs):
        """
            Forward computation of this layer.

            Args:
                inputs: inputs of this layer

            Returns: outputs of this layer

        """
        hidden = self._conv1(inputs)
        hidden = self._pixel_conv(hidden)
        hidden = self._conv2(hidden)

        # Residual connection
        return inputs + hidden


class PixelCNN(Model):
    """
        Defines the main PixelCNN generative model.
    """

    def __init__(self, num_res=2, num_pixel_B=2, num_encoded=128, num_filters=128,
                 kernel_size=7, activation="relu", name="pixel_cnn"):
        super(PixelCNN, self).__init__(name=name)
        self._num_res = num_res
        self._num_pixel_B = num_pixel_B
        self._num_encoded = num_encoded
        self._num_filters = num_filters
        self._kernel_size = kernel_size
        self._activation = activation
        self._total_loss = tf.keras.metrics.Mean(name="total_loss")

        self._pixel_A = PixelConv(kernel_mask_type="A", filters=self._num_filters,
                                  kernel_size=self._kernel_size, activation=self._activation)
        self._pixel_resids = [PixelResidualBlock(num_filters=self._num_filters)
                              for _ in range(self._num_res)]
        self._pixel_Bs = [PixelConv(kernel_mask_type="B", filters=self._num_filters, kernel_size=1,
                                    activation=self._activation)
                          for _ in range(self._num_pixel_B)]
        self._conv = layers.Conv2D(filters=self._num_encoded, kernel_size=1, activation="sigmoid")

    def call(self, inputs):
        """
            Forward computation of this model.

            Args:
                inputs: inputs of this model

            Returns: outputs of this model

        """
        inputs = tf.cast(inputs, dtype=tf.int32)
        inputs = tf.one_hot(inputs, self._num_encoded)
        hidden = self._pixel_A(inputs)
        for i in range(self._num_res):
            hidden = self._pixel_resids[i](hidden)
        for i in range(self._num_pixel_B):
            hidden = self._pixel_Bs[i](hidden)
        return self._conv(hidden)

    def train_step(self, data):
        """
            Performs one iteration of training and return loss values.

            Args:
                data: input data

            Returns: loss values

        """
        with tf.GradientTape() as tape:
            # Calculate loss value
            loss_val = self.compiled_loss(data, self(data))

        # Perform backpropagation
        grads = tape.gradient(loss_val, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self._total_loss.update_state(loss_val)
        return {
            "loss": self._total_loss.result()
        }


# pixel_cnn = PixelCNN()
# pixel_cnn.build((None, 7, 7))
# pixel_cnn.summary()
