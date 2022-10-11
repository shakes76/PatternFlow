import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow import keras

###
#   References:
# - https://keras.io/examples/generative/vq_vae/
# - https://www.kaggle.com/code/ameroyer/keras-vq-vae-for-image-generation/notebook
# - https://www.youtube.com/watch?v=VZFVUrYcig0
# - https://arxiv.org/pdf/1711.00937.pdf
# - https://keras.io/examples/generative/pixelcnn/
# - https://arxiv.org/pdf/1606.05328.pdf
# - https://arxiv.org/pdf/1601.06759v3.pdf
# - https://towardsdatascience.com/auto-regressive-generative-models-pixelrnn-pixelcnn-32d192911173
##


class VectorQuantizerLayer(tf.keras.layers.Layer):
    def __init__(self, embedding_num, embedding_dim, beta, **kwargs):
        super().__init__(**kwargs)
        self._embedding_num = embedding_num
        self._embedding_dim = embedding_dim
        self._commitment_cost = beta

        #Initialise the embeddings which we are going to quantise using a uniform distribution
        initial_embedding = tf.random_uniform_initializer()
        self._embedding_shape = (self._embedding_dim, self._embedding_num)
        self._embedding = tf.Variable(initial_value=initial_embedding(shape=self._embedding_shape),
                                                                      dtype="float32",
                                                                      trainable=True)

    def call(self, x):
        #We need to flatten all of the dimensions so that we get a total number of vectors to be quantized independently
        input_shape = tf.shape(x)

        flattened_vector = tf.reshape(x, [-1, self._embedding_dim])
        #Now we need to find the embedding indices for each of the flattened vectors based on minimising the L2 normalised distance
        encoding_indices = self.get_closest_index(flattened_vector)

        #Need to one-hot encode them so that quantisation can occur
        one_hot_encoding = tf.one_hot(encoding_indices, self._embedding_num)

        #Calculate quantised values and return the flattened vector into the original input shape
        quantized_vectors = tf.matmul(one_hot_encoding, self._embedding, transpose_b=True)
        quantized_vectors = tf.reshape(quantized_vectors, input_shape)

        #Calculate the quantized vector loss and add it to the layer. For losses we use mean squared error.
        # Loss learnt by encoder
        commitment_loss = tf.reduce_mean((x - tf.stop_gradient(quantized_vectors) ) ** 2)
        # Embeddings optimised by codebook loss
        codebook_loss = tf.reduce_mean((tf.stop_gradient(x) - quantized_vectors) ** 2)
        self.add_loss(self._commitment_cost * commitment_loss + codebook_loss)
        #Note we use a stopgradient operator. Defined as identity during a forward computation time, and has zero partial
        #derivatives. Therefore, the operand it is applied to becomes a non-updated constant.

        #Using stop_gradient, during backpropagation the gradients of the output are given back to the inputs
        quantized_vectors = x + tf.stop_gradient(quantized_vectors - x)
        return quantized_vectors

    def get_closest_index(self, flattened_vector):

        pixel_vector_len = tf.reduce_sum(flattened_vector ** 2, axis=1, keepdims=True) #(Height * Width, 1)
        embedding_len = tf.reduce_sum(self._embedding ** 2, axis=0, keepdims=True) #(1, #embeddings)
        subtract_term = tf.matmul(flattened_vector, self._embedding) # (Height*Width, Channel) x (Channel, #embeddings) -> (Height*Width, #n_embeddings)
        distances = pixel_vector_len + embedding_len - 2*subtract_term
        return tf.argmin(distances, axis = 1)

class VQVAEModel(tf.keras.Model):
    def __init__(self, img_shape, embedding_num, embedding_dim, beta, data_variance, **kwargs):
        super(VQVAEModel, self).__init__(**kwargs)

        # Parameters
        self._img_shape = img_shape
        self._embedding_num = embedding_num
        self._embedding_dim = embedding_dim
        self._beta = beta
        self._data_variance = data_variance

        # Model components
        self._encoder = self.create_encoder(self._embedding_dim, self._img_shape)
        self._vq = VectorQuantizerLayer(
            self._embedding_num, self._embedding_dim, self._beta)
        self._decoder = self.create_decoder(self._img_shape)

        # Model itself
        # self._vqvae = self.create_vqvae_model()

        # Loss metrics
        self._total_loss = keras.metrics.Mean(name="total_loss")
        self._reconstruction_loss = keras.metrics.Mean(name="reconstruction_loss")
        self._vq_loss = keras.metrics.Mean(name="vq_loss")
        self._mean_ssim = keras.metrics.Mean(name="mean_ssim")

    def get_vq(self):
        return self._vq

    def create_encoder(self, embedding_dim, img_shape):
        # My encoder implementation
        encoder_model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=img_shape),
            tf.keras.layers.Conv2D(32, 3, activation="relu", strides=2, padding="same"),
            tf.keras.layers.Conv2D(64, 3, activation="relu", strides=2, padding="same"),
            tf.keras.layers.Conv2D(embedding_dim, 1, padding="same")
        ], name="encoder")

        return encoder_model

    def get_encoder(self):
        return self._encoder

    def create_decoder(self, img_shape):
        # My decoder implementation
        decoder_model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=self.get_encoder().output.shape[1:]),
            tf.keras.layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same"),
            tf.keras.layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same"),
            tf.keras.layers.Conv2DTranspose(3, 3, activation="relu", padding="same")
        ], name="decoder")
        return decoder_model

    def get_decoder(self):
        return self._decoder

    def create_vqvae_model(self):
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

    def get_vqvae(self):
        return self._vqvae

    @property
    def metrics(self):
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
            "vq loss": self._vq_loss.result()  # ,
            # "mean ssim": self._mean_ssim.result()
        }

    def test_step(self, x):
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


# Adapted from https://keras.io/examples/generative/pixelcnn/
class PixelConvLayer(tf.keras.layers.Layer):
    def __init__(self, mask_type, **kwargs):
        super(PixelConvLayer, self).__init__()
        self._mask_type = mask_type
        self._convolution = tf.keras.layers.Conv2D(**kwargs)

    def build(self, img_shape):
        # Initialise the kernel variables
        self._convolution.build(input_shape=img_shape)
        self._conv_kernel_shape = self._convolution.kernel.get_shape()
        self.mask = np.zeros(shape=self._conv_kernel_shape)
        # I couldn't get this working myself, used reference.
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


# Represents a single residual block within the model
class ResidualBlock(tf.keras.layers.Layer):
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
    def __init__(self, input_shape, embedding_num, input_filters, num_res_layer, num_pixel_layer, **kwargs):
        super(PixelCNNModel, self).__init__(**kwargs)
        self._input_shape = input_shape
        self._embedding_num = embedding_num
        self._filters = input_filters
        self._num_res_layer = num_res_layer
        self._num_pixel_layer = num_pixel_layer

        input_layer = tf.keras.Input(shape=self._input_shape, dtype=tf.int32)
        one_hot_input = tf.one_hot(input_layer, self._embedding_num)
        x = PixelConvLayer(mask_type="A", filters=self._filters, kernel_size=7, activation="relu", padding="same")(
            one_hot_input)

        for _ in range(self._num_res_layer):
            x = ResidualBlock(self._filters)(x)

        for _ in range(self._num_pixel_layer):
            x = PixelConvLayer(mask_type="A", filters=self._filters, kernel_size=1, strides=1, activation="relu",
                               padding="valid")(x)

        final_output = tf.keras.layers.Conv2D(filters=self._embedding_num, kernel_size=1, strides=1, padding="valid")(x)

        self._pixel_cnn_model = tf.keras.Model(input_layer, final_output, name="pixel_cnn")
        self._total_loss = tf.keras.metrics.Mean(name="total_loss")

    def call(self, x):
        return self._pixel_cnn_model(x)

    def train_step(self, x):
        # https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit
        with tf.GradientTape() as tape:
            predicted_output = self(x)
            total_loss = self.compiled_loss(x, predicted_output)

        gradients = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        self._total_loss.update_state(total_loss)
        return {
            "loss": self._total_loss.result()
        }
