import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp
import tensorflow as tf

# vector quantizer class
class VQ(layers.Layer):
    def __init__(self, embed_n, embed_d, beta=0.25, **kwargs):
        super().__init__(**kwargs)
        self.embed_d = embed_d
        self.embed_n = embed_n
        self.beta = beta

        # fnitialise embeddings to be quantized
        w_init = tf.random_uniform_initializer()
        self.embeds = tf.Variable(
            initial_value=w_init(shape=(self.embed_d, self.embed_n), dtype="float32"),
            trainable=True,
            name="vqvae-embeddings",
        )

    def call(self, x):
        # flatten inputs while maintaining embed_d then quantize
        shape = tf.shape(x)
        flattened = tf.reshape(x, [-1, self.embed_d])
        encoding_indices = self.get_code_indices(flattened)
        encodings = tf.one_hot(encoding_indices, self.embed_n)
        quantized = tf.matmul(encodings, self.embeds, transpose_b=True)

        # get back original shape
        quantized = tf.reshape(quantized, shape)

        # loss
        c_loss = tf.reduce_mean((tf.stop_gradient(quantized) - x) ** 2)
        cb_loss = tf.reduce_mean((quantized - tf.stop_gradient(x)) ** 2)
        self.add_loss(self.beta * c_loss + cb_loss)

        quantized = x + tf.stop_gradient(quantized - x)
        return quantized

    def get_code_indices(self, flattened):
        # l2-normalised distance between input and codes
        similarity = tf.matmul(flattened, self.embeds)
        dists = (
            tf.reduce_sum(flattened**2, axis=1, keepdims=True)
            + tf.reduce_sum(self.embeds**2, axis=0)
            - 2*similarity
        )

        # get best indices
        encode_indices = tf.argmin(dists, axis=1)
        return encode_indices

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_d": self.embed_d,
            "embed_n": self.embed_n,
            "beta": self.beta,
        })
        return config

class Train_VQVAE(keras.models.Model):
    def __init__(self, train_variance, dim=32, embed_n=128, **kwargs):
        super(Train_VQVAE, self).__init__(**kwargs)
        self.train_variance = train_variance
        self.dim = dim
        self.embed_n = embed_n
        self.vqvae = get_vqvae(self.dim, self.embed_n)
        self.total_loss = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss = keras.metrics.Mean(
            name="reconstruction_loss"
        )
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
            reconstructions = self.vqvae(x)

            # calculate loss
            reconstruction_loss = (
                tf.reduce_mean((x - reconstructions) ** 2) / self.train_variance
            )
            total_loss = reconstruction_loss + sum(self.vqvae.losses)

        # backpropagate
        grads = tape.gradient(total_loss, self.vqvae.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.vqvae.trainable_variables))

        # track loss
        self.total_loss.update_state(total_loss)
        self.reconstruction_loss.update_state(reconstruction_loss)
        self.vq_loss.update_state(sum(self.vqvae.losses))
        return {
            "loss": self.total_loss.result(),
            "reconstruction_loss": self.reconstruction_loss.result(),
            "vqvae_loss": self.vq_loss.result(),
        }

def encoder(dim=16):
    inputs = keras.Input(shape=(80, 80, 1))
    x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(
        inputs
    )
    x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    out = layers.Conv2D(dim, 1, padding="same")(x)
    return keras.Model(inputs, out, name="encoder")

def decoder(dim=16):
    inputs = keras.Input(shape=encoder(dim).output.shape[1:])
    x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(
        inputs
    )
    x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    out = layers.Conv2DTranspose(1, 3, padding="same")(x)
    return keras.Model(inputs, out, name="decoder")

def get_vqvae(dim=16, embed_n=64):
    vq_layer = VQ(embed_n, dim, name="vector_quantizer")
    enc = encoder(dim)
    dec = decoder(dim)
    inputs = keras.Input(shape=(80, 80, 1))
    out = enc(inputs)
    quantized_latents = vq_layer(out)
    reconstructions = dec(quantized_latents)
    return keras.Model(inputs, reconstructions, name="vq_vae")

# builds on the 2D convolutional layer, but includes masking
class PixelConvLayer(layers.Layer):
    def __init__(self, mask_type, **kwargs):
        super(PixelConvLayer, self).__init__()
        self.mask_type = mask_type
        self.conv = layers.Conv2D(**kwargs)

    def build(self, input):
        # initialize kernel variables
        self.conv.build(input)
        # create the mask
        kernel_shape = self.conv.kernel.get_shape()
        self.mask = np.zeros(shape=kernel_shape)
        self.mask[: kernel_shape[0] // 2, ...] = 1.0
        self.mask[kernel_shape[0] // 2, : kernel_shape[1] // 2, ...] = 1.0
        if self.mask_type == "B":
            self.mask[kernel_shape[0] // 2, kernel_shape[1] // 2, ...] = 1.0

    def call(self, inputs):
        self.conv.kernel.assign(self.conv.kernel * self.mask)
        return self.conv(inputs)

    def get_config(self):
        config = super().get_config()
        config.update({
            "mask_type": self.mask_type,
        })
        return config

# residual block layer
class ResBlock(keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super(ResBlock, self).__init__(**kwargs)
        self.filters = filters
        self.conv_1 = keras.layers.Conv2D(filters=self.filters, kernel_size=1, activation="relu")
        self.pixel_conv = PixelConvLayer(mask_type="B", filters=self.filters // 2, kernel_size=3,
            activation="relu", padding="same",)
        self.conv_2 = keras.layers.Conv2D(filters=self.filters, kernel_size=1, activation="relu")

    def call(self, inputs):
        conv = self.conv_1(inputs)
        conv = self.pixel_conv(conv)
        conv = self.conv_2(conv)
        return keras.layers.add([inputs, conv])

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters" : self.filters
        })
        return config

def get_pixelcnn(vqvae_trainer, encoded_outputs):
    """
    Builds and returns the PixelCNN model.
    """
    
    # Initialise number of PixelCNN blocks
    num_residual_blocks = 2
    num_pixelcnn_layers = 2
    pixelcnn_input_shape = encoded_outputs.shape[1:-1]
    print(f"Input shape of the PixelCNN: {pixelcnn_input_shape}")
    
    # Initialise inputs to PixelCNN
    pixelcnn_inputs = keras.Input(shape=pixelcnn_input_shape, dtype=tf.int32)
    ohe = tf.one_hot(pixelcnn_inputs, vqvae_trainer.embed_n)
    x = PixelConvLayer(mask_type="A", filters=128, kernel_size=7, activation="relu", padding="same")(ohe)

    # Build PixelCNN model
    for _ in range(num_residual_blocks):
        x = ResBlock(filters=128)(x)

    for _ in range(num_pixelcnn_layers):
        x = PixelConvLayer(
            mask_type="B",
            filters=128,
            kernel_size=1,
            strides=1,
            activation="relu",
            padding="valid",
        )(x)

    # Outputs from PixelCNN    
    out = keras.layers.Conv2D(filters=vqvae_trainer.embed_n, kernel_size=1, strides=1, padding="valid")(x)

    pixel_cnn = keras.Model(pixelcnn_inputs, out, name="pixel_cnn")
    
    return pixel_cnn