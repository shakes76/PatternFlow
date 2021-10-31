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

        self.mask = tf.Variable(tf.zeros(shape=kernel_shape, dtype=tf.float32), dtype=tf.float32)

        shape = self.mask[: kernel_shape[0] // 2, ...].shape
        self.mask = self.mask[: kernel_shape[0] // 2, ...].assign(tf.ones(shape=shape))

        shape = self.mask[kernel_shape[0] // 2, : kernel_shape[1] // 2, ...].shape
        self.mask = self.mask[kernel_shape[0] // 2, : kernel_shape[1] // 2, ...].assign(tf.ones(shape=shape))
        if self.mask_type == "B":
            shape = self.mask[kernel_shape[0] // 2, kernel_shape[1] // 2, ...].shape
            self.mask = self.mask[kernel_shape[0] // 2, kernel_shape[1] // 2, ...].assign(tf.ones(shape=shape))
        self.mask = tf.constant(self.mask)


    def call(self, inputs):
        x = self.mask * self.conv.kernel
        self.conv.kernel.assign(x)
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

class OneHotLayer(tf.keras.layers.Layer):
    def __init__(self, n_embeddings):
        super(OneHotLayer, self).__init__()
        self.n_embeddings = n_embeddings

    def call(self, inputs):
        return tf.one_hot(inputs, self.n_embeddings)

def create_pixel_cnn(input_shape, n_embeddings):
    model = Sequential()
    model.add(OneHotLayer(n_embeddings))
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




def generate_images(vqvae, pixel_cnn, n_images, output_shape):
    n_priors = n_images
    priors = tf.Variable(tf.zeros(shape=(n_priors,) + pixel_cnn.input_shape[1:], dtype=tf.int32))

    _, rows, cols = priors.shape

    for row in range(rows):
        for col in range(cols):
            print(f"\rrow: {row}, col: {col}", end="")
            dist = tfp.distributions.Categorical(logits=pixel_cnn(priors, training=False))
            probs = dist.sample()

            priors = priors[:, row, col].assign(probs[:, row, col])

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

    return priors, generated_samples.squeeze()








