"""
Based on PixelCNN by ADMoreau available at
https://keras.io/examples/generative/pixelcnn/

PixelCNN to mimic latent space of the encoders output. For generation
"""

import keras
import numpy as np
from keras import layers
from keras import ops
import tensorflow as tf
import tensorflow_probability as tfp

# for displaying
import matplotlib.pyplot as plt
import math

import importlib
vqvae = importlib.import_module("vq-vae")

class PixelConvLayer(layers.Layer):

    """
    a convolution layer that masks the kernel to only influence pixels behind
    the current pixel - allows for conditional generation of the output
    """

    def __init__(
            self,
            mask_type,  # A includes the pixel itself, B does not
            **kwargs,  # arguments for convolution layer
    ):
        super().__init__()
        self.mask_type=mask_type
        self.conv = layers.Conv2D(**kwargs)

    def build(
            self, input_shape
    ):
        # initialise kernels
        self.conv.build(input_shape)

        # now mask the kernel
        kernel_shape = ops.shape(self.conv.kernel)
        self.mask = np.zeros(shape=kernel_shape) # initialise to zero

        self.mask[: kernel_shape[0] // 2, ...] = 1.0 # above pixels

        # same row
        self.mask[kernel_shape[0] // 2, : kernel_shape[1] // 2, ...] = 1.0

        # center pixel (if type B)
        if self.mask_type == "B":
            self.mask[kernel_shape[0] // 2, kernel_shape[1] // 2, ...] = 1.0

    def call(
            self, x
    ):
        # mask the kernel with the mask
        self.conv.kernel.assign(self.conv.kernel * self.mask)

        # return
        return self.conv(x)


def residual_pixel_layer(
        x,
        num_filters,
):

    """
    a residual layer with 2 1x1 convolutions around a 3x3 pixel convolution.
    """
    start = x

    x = layers.Conv2D(
        num_filters, 1, 1
    )(x)
    x = keras.activations.relu(x)

    x = PixelConvLayer(
        "B", filters=num_filters//2, kernel_size=3, padding="same"
    )(x)

    x = keras.activations.relu(x)


    x = layers.Conv2D(
        num_filters, 1, 1
    )(x)

    x = keras.layers.add([x, start])
    x = keras.activations.relu(x)  # tutorial does relu THEN add

    return x


def pixel_cnn_model(
        input_shape,
        num_residuals,
        latent_dim,
        num_filters=128
):
    """
    returns a pixelCNN model
    """

    inputs = layers.Input(shape=input_shape)

    # initial convolution
    x = PixelConvLayer(
        "A",
        filters=num_filters,
        kernel_size=7,
        padding="same"
    )(inputs)

    x = keras.activations.relu(x)

    for _ in range(num_residuals):
        x = residual_pixel_layer(
            x, num_filters
        )

    # end on 2 more pixelconvolutions
    for _ in range(2):
        x = PixelConvLayer(
            "B", filters=num_filters, kernel_size=1, strides=1, padding='valid'
        )(x)

        x = keras.activations.relu(x)

    output = layers.Conv2D(
        filters=latent_dim,
        kernel_size=1,
        strides=1,
        activation="sigmoid",
        padding="valid"
    )(x)

    return keras.Model(inputs=inputs, outputs=output)


class CategoricalLayer(layers.Layer):
    """
    Wraps tfp.distributions.Categorical for use in sampler
    """
    def call(self, logits):
        dist = tfp.distributions.Categorical(logits=logits)
        return dist.sample()


class sampler:

    """
    Wraps pixel_cnn model in Distribution object that allows random sampling
    """

    def __init__(self, pixel_cnn, quantizer, decoder):

        self.quantizer = quantizer
        self.decoder = decoder
        self.pixel_cnn = pixel_cnn

        self._build_sampler()

    def _build_sampler(self):
        inputs = layers.Input(shape=self.pixel_cnn.input_shape[1:])
        logits = self.pixel_cnn(inputs, training=False)

        outputs = CategoricalLayer()(logits)

        self.sampler = keras.Model(inputs, outputs)

    def sample(self, batch_size):
        priors = np.zeros(
            shape=(batch_size,) + (self.pixel_cnn.input_shape)[1:-1]
        )

        batch, rows, cols = priors.shape

        for row in range(rows):
            for col in range(cols):
                probabilities = self.sampler.predict(priors, verbose=0)
                priors[:, row, col] = probabilities[:, row, col]

        # convert to one hot to extract from codebook
        one_hot = tf.one_hot(priors.astype("int32"), quantizer.num_embeddings)
        codebook_vectors = tf.matmul(
            tf.cast(one_hot, tf.float32), # convert bc we are mult by floats
            quantizer.codebook, transpose_b=True
        )
        codebook_vectors = tf.reshape(
            codebook_vectors, (-1, *(self.decoder.input.shape[1:]))
        )

        # actually generate now
        generated = self.decoder.predict(
            codebook_vectors
        )

        return generated


 # pass image dataset through new pipeline
def dataset_pipeline(img):
    # add batch and channel dimension
    img = tf.expand_dims(img, axis=0) # batch
    img = tf.expand_dims(img, axis=-1) # channel

    prediction = encoder(img, training=False)
    flattened = tf.reshape(prediction, (-1, prediction.shape[-1]))

    # quantize
    indices = quantizer.get_code_indices(flattened)

    indices = tf.reshape(indices, prediction.shape[:-1])
    indices = tf.expand_dims(indices, -1) # channel dim
    return (indices, indices)


def load_model(path):
    shape = (
        vqvae.get_dataset(vqvae.TRAIN_FOLDER)
        .map(dataset_pipeline)
        .element_spec[0].shape[1:]
    )

    new_pixel = pixel_cnn_model(
        shape,
        4,
        vqvae.CODEBOOK_SIZE,
        num_filters=256
    ) # new model with same parameters

    new_pixel.compile("adam")
    new_pixel.build(shape)
    new_pixel.load_weights(path)

    return new_pixel


def show_generated_images(batch, title="Generated Images"):
    """
    Display a batch of generated grayscale images as a square grid.

    Args:
        batch: np.ndarray or tf.Tensor of shape (N, H, W, 1) or (N, H, W)
               N = number of images
        title: optional string for figure title
    """

    # Convert to numpy
    batch = np.array(batch)

    # Drop channel dimension if present
    if batch.ndim == 4 and batch.shape[-1] == 1:
        batch = batch[..., 0]

    n = batch.shape[0]
    grid_size = math.ceil(math.sqrt(n))

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 2, grid_size * 2))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        ax.axis("off")
        if i < n:
            ax.imshow(batch[i], cmap="gray")
    plt.suptitle(title)
    plt.tight_layout()
    # plt.show()  # show doesnt work on ranpur
    plt.savefig("novel_generation.jpg")


if __name__ == "__main__":

    vae_model = vqvae.load_model(
        "trained_models/trainer.weights.h5"
    )

    encoder = vae_model.get_layer("encoder")
    decoder = vae_model.get_layer("decoder")
    quantizer = vae_model.get_layer("quantizer")

    image_dataset = vqvae.get_dataset(
        vqvae.TRAIN_FOLDER
    )

    pixelcnn_dataset = image_dataset.map(dataset_pipeline)
    io_shape = pixelcnn_dataset.element_spec[0].shape[1:] # it adds batch dim auto
    pcnn = pixel_cnn_model(
        io_shape, 4, vqvae.CODEBOOK_SIZE, num_filters=256
    )

    pcnn.compile(
        "adam",
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    )

    pcnn.fit(
        pixelcnn_dataset,
        epochs=10
    )

    # now use it to predict
    slice_sampler = sampler(
        pcnn, quantizer, decoder
    )

    generated_outputs = slice_sampler.sample(
        9
    )

    show_generated_images(generated_outputs, title="Novel Generated Outputs")

    # save model
    pcnn.save_weights(
        "trained_models/pixel.weights.h5"
    )

    # # reload and test
    # pcnn = load_model("trained_models/pixel.weights.h5")
    # slice_sampler = sampler(pcnn, quantizer, decoder)
    # generated_outputs = slice_sampler.sample(4)
    # show_generated_images(
    #     generated_outputs, "reloaded, novel generation"
    # )


