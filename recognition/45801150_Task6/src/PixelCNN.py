import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, InputLayer, Input, Lambda
from tensorflow.keras.models import Sequential
from VQVAE import VQVae, VectorQuantiser
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import tensorflow_probability as tfp
import matplotlib.pyplot as plt


class PixelConvLayer(keras.layers.Layer):
    def __init__(self, mask_type=None, **kwargs):
        super(PixelConvLayer, self).__init__()
        self.mask_type = mask_type
        self.conv = Conv2D(**kwargs)

    def build(self, input_shape):
        self.conv.build(input_shape)
        kernel_shape = self.conv.kernel.get_shape()
        self.mask = np.zeros(shape=kernel_shape)
        self.mask[: kernel_shape[0] // 2, ...] = 1.0
        self.mask[kernel_shape[0] // 2, : kernel_shape[1] // 2, ...] = 1.0
        if self.mask_type == "B":
            self.mask[kernel_shape[0] // 2, kernel_shape[1] // 2, ...] = 1.0

    def call(self, inputs):
        self.conv.kernel.assign(self.conv.kernel * self.mask)
        return self.conv(inputs)


class ResidualBlock(keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.conv1 = Conv2D(
            filters=filters, kernel_size=1, activation="relu"
        )
        self.pixel_conv = PixelConvLayer(
            mask_type="B",
            filters=filters // 2,
            kernel_size=3,
            activation="relu",
            padding="same",
        )
        self.conv2 = keras.layers.Conv2D(
            filters=filters, kernel_size=1, activation="relu"
        )

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pixel_conv(x)
        x = self.conv2(x)
        return keras.layers.add([inputs, x])

pixel_layer = PixelConvLayer(filters=128, kernel_size=1, strides=1, mask_type='A')

## params
n_residual_blocks = 2
n_pixel_cnn_layers = 2

x = keras.Input(shape=10, dtype=tf.int32)
def create_pixel_cnn(input_shape, n_embeddings):
    model = Sequential()
    model.add(Lambda(lambda inputs: tf.one_hot(inputs, n_embeddings)))
    model.add(
        PixelConvLayer(
            mask_type="A", filters=128, kernel_size=7, activation="relu", padding="same"
        )
    )

    for i in range(n_residual_blocks):
        model.add(ResidualBlock(filters=128))

    for i in range(n_pixel_cnn_layers):
        model.add(
            PixelConvLayer(
                mask_type="B",
                filters=128,
                kernel_size=1,
                strides=1,
                activation="relu",
                padding="valid",
            )
        )

    model.add(
        keras.layers.Conv2D(
            filters=n_embeddings, kernel_size=1, strides=1, padding="valid"
        )
    )

    return model

def train_pixel_cnn(pixel_cnn, vqvae: VQVae, x_train_normalised, n_epochs):
    encoder = vqvae.get_layer("encoder")
    quantiser: VectorQuantiser = vqvae.get_layer("quantiser")

    outputs = encoder.predict(x_train_normalised)
    flattened = outputs.reshape(-1, outputs.shape[-1])

    code_indices = quantiser.get_code_indices(flattened)

    code_indices = tf.reshape(code_indices, outputs.shape[:-1])


    pixel_cnn.compile(
        optimizer=Adam(learning_rate=(0.0003)),
        loss=SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    pixel_cnn.fit(x=code_indices, y=code_indices, batch_size=64, epochs=n_epochs, validation_split=0.1)




def generate_image(vqvae, pixel_cnn, input_shape, output_shape):
    n_priors = 10
    priors = tf.zeros(shape=(n_priors,) + pixel_cnn.input_shape[1:], dtype=tf.int32)

    _, rows, cols = priors.shape

    for row in range(rows):
        for col in range(cols):
            print(f"\rrow: {row}, col: {col}", end="")
            dist = tfp.distributions.Categorical(logits=pixel_cnn(priors, training=False))
            probs = dist.sample()

            indices = tf.constant(tf.expand_dims(tf.range(n_priors), axis=1))
            tf.tensor_scatter_nd_update(priors, indices, probs)

    quantiser = vqvae.get_layer("quantiser")

    embeddings = quantiser.embeddings
    priors = tf.cast(priors, tf.int32)
    priors_one_hot = tf.one_hot(priors, vqvae.num_embeddings)
    priors_one_hot = tf.cast(priors_one_hot, tf.float32)
    quantised = tf.matmul(priors_one_hot, embeddings, transpose_b=True)
    quantised = tf.reshape(quantised, (-1, *(output_shape[1:])))

    # Generate novel images.
    decoder = vqvae.get_layer("decoder")
    generated_samples = decoder.predict(quantised)

    for i in range(n_priors):
        plt.subplot(1, 2, 1)
        plt.imshow(priors[i])
        plt.title("Code")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(generated_samples[i].squeeze(), vmin=0, vmax=1)
        plt.title("Generated Sample")
        plt.axis("off")
        plt.savefig(f"generated_{i}.png")
        plt.close()







