from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

"""
    =========================================================
    Reference to 
        [-]: https://keras.io/examples/generative/vq_vae/
        [-]: https://towardsdatascience.com/autoregressive-models-pixelcnn-e30734ede0c1
    =========================================================
"""


class VectorQuantizer(layers.Layer):
    def __init__(self, num_embeddings, embedding_dim, beta=0.2, **kwargs):
        """
          num_embeddings: The number of embeddings
          embedding_dim: The dimension of the embedding vector
          beta: Loss factor 
        """
        super().__init__(**kwargs)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta

        initialiser = tf.random_uniform_initializer()
        self.embeddings = tf.Variable(
            initial_value=initialiser(
                shape=(self.embedding_dim,
                       self.num_embeddings), dtype="float32"
            ),
            trainable=True,
            name="VQVAE_EMBEDDINGS",
        )

    def call(self, input):
        """
            input: The input to the VQ Layer (output of Encoder)
        """
        # Flattening
        shape = tf.shape(input)
        flattened_input = tf.reshape(input, [-1, self.embedding_dim])

        # Quantizing
        encoding_indices = self.get_code_indices(flattened_input)
        encodings = tf.one_hot(
            encoding_indices, self.num_embeddings, dtype="float32")
        quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)
        quantized = tf.reshape(quantized, shape)

        # Losses
        commitment_loss = tf.reduce_mean(
            (tf.stop_gradient(quantized) - input) ** 2)
        codebook_loss = tf.reduce_mean(
            (quantized - tf.stop_gradient(input)) ** 2)
        self.add_loss(self.beta * commitment_loss + codebook_loss)

        # Straight-through estimator.
        quantized = input + tf.stop_gradient(quantized - input)
        return quantized

    def get_code_indices(self, flattened_input):
        """
            flattened_input: A flattened input vector
        """
        similarity = tf.matmul(flattened_input, self.embeddings)
        distances = (tf.reduce_sum(flattened_input**2, axis=1, keepdims=True) +
                     tf.reduce_sum(self.embeddings**2, axis=0) - 2*similarity)

        encoding_indices = tf.argmin(distances, axis=1)
        return encoding_indices


class Encoder(keras.models.Model):
    def __init__(self, latent_dim=256, **kwargs):
        """
          latent_dim: Dimension of the latent output space
        """
        super().__init__(**kwargs)
        self.intermediate_layers = [
            layers.Conv2D(32, 3, activation="relu", strides=2, padding="same"),
            layers.Conv2D(64, 3, activation="relu", strides=2, padding="same"),
            layers.Conv2D(128, 3, activation="relu",
                          strides=2, padding="same"),
        ]
        self.final_layer = layers.Conv2D(latent_dim, 1, padding="same")

    def call(self, input):
        """
            input: A brain image for the encoder to encode
        """
        payload = input
        for layer in self.intermediate_layers:
            payload = layer(payload)
        final_result = self.final_layer(payload)
        return final_result


class Decoder(keras.models.Model):
    def __init__(self, latent_dim=256, **kwargs):
        """
          latent_dim: Dimension of the latent input space
        """
        super().__init__(**kwargs)
        self.intermediate_layers = [
            layers.Conv2DTranspose(latent_dim, 3, activation="relu",
                                   strides=2, padding="same"),
            layers.Conv2DTranspose(latent_dim // 2, 3, activation="relu",
                                   strides=2, padding="same"),
            layers.Conv2DTranspose(latent_dim // 4, 3, activation="relu",
                                   strides=2, padding="same"),
        ]
        self.final_layer = layers.Conv2DTranspose(1, 3, padding="same")

    def call(self, input):
        """
            input: A vector from the discrete latent space (output of VQ Layer)
        """
        payload = input
        for layer in self.intermediate_layers:
            payload = layer(payload)
        final_result = self.final_layer(payload)
        return final_result


class VQVAE(keras.models.Model):
    def __init__(self, num_embeddings=256, latent_dim=256, **kwargs):
        """
          num_embeddings: Number of embeddings
          latent_dim: Dimensionality of the output of the Encoder
        """
        super().__init__(**kwargs)
        self.vq_layer = VectorQuantizer(
            num_embeddings=num_embeddings, embedding_dim=latent_dim, name="VQ_Layer")
        self.encoder = Encoder(latent_dim=latent_dim, name="ENCODER")
        self.decoder = Decoder(name="DECODER")
        self.total_loss_cum = []
        self.reconstructive_loss_cum = []
        self.vq_loss_cum = []
        self.total_loss = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss = tf.keras.metrics.Mean(
            name="reconstruction_loss")
        self.vq_loss = tf.keras.metrics.Mean(name="vq_loss")

    def call(self, input, training=False):
        """
            input: A brain image
        """
        encoder_outputs = self.encoder(input, training=training)
        quantized_latents = self.vq_layer(encoder_outputs, training=training)
        reconstructions = self.decoder(quantized_latents, training=training)
        return reconstructions

    @property
    def metrics(self):
        """
        Model metrics

        Returns:
            the losses (total loss, reconstruction loss and the vq_loss)
        """
        return [self.total_loss, self.reconstruction_loss, self.vq_loss]

    def train_step(self, input):
        """
            input: A brain image
        """
        with tf.GradientTape() as tape:
            # Output from the VQ-VAE.
            reconstructions = self(input, training=True)

            # Calculate the losses.
            reconstruction_loss = tf.reduce_mean(
                (input - reconstructions) ** 2)
            total_loss = reconstruction_loss + sum(self.vq_layer.losses)

        # Compute the gradients
        trainable_vars = self.trainable_variables
        grads = tape.gradient(total_loss, trainable_vars)

        # Update the weights
        self.optimizer.apply_gradients(zip(grads, trainable_vars))

        # Update the metrics
        self.total_loss.update_state(total_loss)
        self.reconstruction_loss.update_state(reconstruction_loss)
        self.vq_loss.update_state(sum(self.vq_layer.losses))

        self.total_loss_cum.append(self.total_loss.result())
        self.reconstructive_loss_cum.append(
            self.reconstruction_loss.result())
        self.vq_loss_cum.append(self.vq_loss.result())

        # Log results.
        return {metric.name: metric.result() for metric in self.metrics}

    def show_subplot(self, original, reconstructed):
        """
            original: An original image of a brain from a dataset
            reconstructed: A reconstructed brain that has come from the output of the VQVAE
        """
        plt.subplot(1, 2, 1)
        plt.imshow(original.squeeze() + 0.5)
        plt.title("Original")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(reconstructed.squeeze() + 0.5)
        plt.title("Reconstructed")
        plt.axis("off")

        plt.show()
        print("SSIM: ", tf.image.ssim(
            reconstructed * 255, original*255, max_val=255))

    def plot(self, num, test_data):
        """
            num: Number of images to plot
            test_data: The test dataset
        """
        test_imgs = test_data.take(1)
        for elem in test_imgs:
            test_imgs = elem.numpy()
        idx = np.random.choice(len(test_imgs), num)
        test_images = test_imgs[idx]
        reconstructions_test = self.predict(test_images)

        for test_image, reconstructed_image in zip(test_images, reconstructions_test):
            self.show_subplot(test_image, reconstructed_image)

        self.plot_codes(test_images)

    def plot_codes(self, test_images):
        """
            test_images: The images which we wish to construct codebooks from
        """
        encoded_outputs = self.encoder.predict(test_images)
        print(encoded_outputs.shape[1:-1])
        flat_enc_outputs = encoded_outputs.reshape(
            -1, encoded_outputs.shape[-1])
        codebook_indices = self.vq_layer.get_code_indices(flat_enc_outputs)
        codebook_indices = codebook_indices.numpy().reshape(
            encoded_outputs.shape[:-1])

        # Plot the Original Images vs their codebooks
        for i in range(len(test_images)):
            plt.subplot(1, 2, 1)
            plt.imshow(test_images[i].squeeze() + 0.5, cmap="gray")
            plt.title("Original")
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.imshow(codebook_indices[i], cmap="gray")
            plt.title("Code")
            plt.axis("off")
            plt.show()


class PixelConvLayer(layers.Layer):
    def __init__(self, mask_type, **kwargs):
        """
            mask_type: The type of mask, "A" or "B"
        """
        super(PixelConvLayer, self).__init__()
        self.mask_type = mask_type
        self.conv = layers.Conv2D(**kwargs)

    def build(self, input_shape):
        """
            input_shape: The shape the first Conv Layer should accept as input
        """
        # Build the conv2d layer to initialize kernel variables
        self.conv.build(input_shape)
        # Use the initialized kernel to create the mask
        kernel_shape = self.conv.kernel.get_shape()
        self.mask = np.zeros(shape=kernel_shape)
        self.mask[: kernel_shape[0] // 2, ...] = 1.0
        self.mask[kernel_shape[0] // 2, : kernel_shape[1] // 2, ...] = 1.0
        if self.mask_type == "B":
            self.mask[kernel_shape[0] // 2, kernel_shape[1] // 2, ...] = 1.0

    def call(self, inputs):
        """
            inputs: A tensor to input to the PixelConvLayer
        """
        self.conv.kernel.assign(self.conv.kernel * self.mask)
        return self.conv(inputs)


class ResidualBlock(keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        """
            filters: The number of filters
        """
        super(ResidualBlock, self).__init__(**kwargs)
        self.conv1 = keras.layers.Conv2D(
            filters=filters, kernel_size=1, activation="relu")
        self.norm1 = keras.layers.BatchNormalization()
        self.pixel_conv = PixelConvLayer(
            mask_type="B",
            filters=filters // 2,
            kernel_size=3,
            activation="relu",
            padding="same",
        )
        self.norm2 = keras.layers.BatchNormalization()
        self.conv2 = keras.layers.Conv2D(
            filters=filters, kernel_size=1, activation="relu")
        self.norm3 = keras.layers.BatchNormalization()

    def call(self, inputs):
        """
            inputs: the input to the Residual Block
        """
        x = self.conv1(inputs)
        x = self.norm1(x)
        x = self.pixel_conv(x)
        x = self.norm2(x)
        x = self.conv2(x)
        x = self.norm3(x)
        return keras.layers.add([inputs, x])


class PixelCNN(keras.models.Model):
    def __init__(self, num_blocks, num_layers, num_embeddings, **kwargs):
        """
            num_blocks: The number of Residual Blocks
            num_layers: The number of PixelConvLayers
            num_embeddings: The number of embeddings
        """
        super().__init__(**kwargs)
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.first_layer = PixelConvLayer(
            mask_type="A", filters=128, kernel_size=7, activation="relu", strides=1, padding="same")
        self.norm1 = keras.layers.BatchNormalization()

        self.block_layers = []

        for _ in range(num_blocks):
            self.block_layers.append(ResidualBlock(filters=128))

        self.pixel_layers = []
        self.norms = []
        for _ in range(num_layers):
            self.pixel_layers.append(PixelConvLayer(
                mask_type="B", filters=128, kernel_size=1, strides=1, activation="relu", padding="valid"))
            self.norms.append(keras.layers.BatchNormalization())

        self.final_layer = keras.layers.Conv2D(
            filters=num_embeddings, kernel_size=1, strides=1, padding="valid")

    def call(self, input):
        """
            input: The input to the PixelCNN (a generated codebook)
        """
        input = self.first_layer(input)
        input = self.norm1(input)

        for i in range(self.num_blocks):
            input = self.block_layers[i](input)

        for i in range(self.num_layers):
            input = self.pixel_layers[i](input)
            input = self.norms[i](input)

        output = self.final_layer(input)
        return output
