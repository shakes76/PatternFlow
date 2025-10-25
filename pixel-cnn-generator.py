"""
Based on PixelCNN by ADMoreau available at
https://keras.io/examples/generative/pixelcnn/

PixelCNN to mimic latent space of the encoders output. For generation
"""

import keras
import numpy as np
from keras import layers

import tensorflow as tf
# from keras import mixed_precision
# mixed_precision.set_global_policy("mixed_float16") # speed up


# for displaying
import matplotlib.pyplot as plt
import math

import importlib
vqvae = importlib.import_module("vq-vae")

class MaskConstraint(keras.constraints.Constraint):
    def __init__(self, mask): self.mask = tf.constant(mask, dtype=tf.float32)
    def __call__(self, w):    return w * self.mask

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
        self.conv.build(input_shape)
        k = self.conv.kernel

        # get dimensions
        kh, kw, cin, cout = k.shape

        mask = np.zeros((kh, kw, cin, cout), dtype=np.float32)
        mask[: kh // 2, :, :, :] = 1.0                       # rows above
        mask[kh // 2, : kw // 2, :, :] = 1.0                  # same row, left

        if self.mask_type == "B":
            mask[kh // 2, kw // 2, ...] = 1.0                # center (only for B)

        self.mask = tf.constant(mask, dtype=tf.float32)
        # Hard-apply once at init to eliminate any initial leakage
        self.conv.kernel.assign(self.conv.kernel * self.mask)
        # Keep enforcing after each optimizer step
        self.conv.kernel_constraint = MaskConstraint(self.mask)

        print(self.mask_type, self.mask[..., 0, 0][k.shape[0]//2])
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
        # "B", filters=num_filters//2, kernel_size=3, padding="same"
        "B", filters=num_filters, kernel_size=3, padding="same"
    )(x)

    x = keras.activations.relu(x)


    x = layers.Conv2D(
        num_filters, 1, 1
    )(x)

    x = keras.layers.add([x, start])
    x = keras.activations.relu(x)

    return x


def pixel_cnn_model(
        input_shape,
        num_residuals,
        num_embeddings,
        num_filters=128
):
    """
    returns a pixelCNN model
    """

    inputs = layers.Input(shape=input_shape, dtype=tf.int32)
    # x = tf.cast(inputs, tf.float32) / 511
    # one hot
    # x = layers.Embedding(num_embeddings, 3)(tf.squeeze(inputs, -1))
    x = tf.one_hot(tf.squeeze(inputs, -1), num_embeddings)

    # initial convolution
    x = PixelConvLayer(
        "A",
        filters=num_filters,
        kernel_size=7,
        padding="same"
    )(x)

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

    output = layers.Conv2D(
        filters=num_embeddings,
        kernel_size=1,
        strides=1,
        padding="valid"
    )(x)

    return keras.Model(inputs=inputs, outputs=output)

def generate_images(
    pixelcnn,
    quantizer,
    decoder,
    image_shape,          # (rows, cols)
    num_samples=16,
):

    rows, cols = image_shape
    batch_size = num_samples
    num_embeddings = quantizer.num_embeddings

    # initialise latent index grid
    # priors = np.random.random_integers(0, num_embeddings-1, (batch_size, rows, cols, 1))
    priors = np.zeros((batch_size, rows, cols, 1))

    for r in range(rows):
        for c in range(cols):
            logits = pixelcnn(priors)
            logits_rc = logits[:, r, c, :] / 0.7
            sample = tf.random.categorical(logits_rc, 1)
            priors[:, r, c, 0] = tf.squeeze(sample, -1).numpy().astype(np.int32)

    # back to one-hot
    priors = priors.squeeze(-1)
    indices = tf.one_hot(priors, num_embeddings)

    codebook = tf.convert_to_tensor(quantizer.codebook, dtype=tf.float32)
    quantized = tf.matmul(indices, tf.transpose(codebook))

    # decode to image space
    generated = decoder(quantized, training=False)

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
    indices = tf.cast(indices, tf.int32)
    indices = tf.reshape(indices, prediction.shape[1:-1])
    indices = tf.expand_dims(indices, -1) # channel dim

    return (indices, indices)


def load_model(path):
    shape = (
        vqvae.get_dataset(vqvae.TRAIN_FOLDER)
        .map(dataset_pipeline)
        .element_spec[0].shape#[1:]
    )

    new_pixel = pixel_cnn_model(
        shape,
        4,
        vqvae.CODEBOOK_SIZE,
        num_filters=32
    ) # new model with same parameters


    new_pixel.compile("adam")
    new_pixel.build(shape)
    new_pixel.load_weights(path)

    return new_pixel


def show_generated_images(batch, title="Generated_Images"):
    """
    Display a batch of generated grayscale images as a square grid.

    Args:
        batch: np.ndarray or tf.Tensor of shape (N, H, W, 1) or (N, H, W)
               N = number of images
    title: optional string for figure title + save file name
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
    plt.savefig(title+".jpg")


if __name__ == "__main__":
    # load existing vqvae model
    vae_model = vqvae.load_model(
        "trainer.weights.h5"
    )

    # get components
    encoder = vae_model.get_layer("encoder")
    decoder = vae_model.get_layer("decoder")
    quantizer = vae_model.get_layer("quantizer")

    print("Codebook shape:", quantizer.codebook.shape)


    # load the dataset if it already exists
    dataset_save_path = "pixelcnn_dataset"
    try:
        pixelcnn_dataset = tf.data.Dataset.load(dataset_save_path)
    except:
        print("Dataset not found, creating now...")
        # create pixelcnn dataset
        image_dataset = vqvae.get_dataset(
            vqvae.TRAIN_FOLDER
        )

        pixelcnn_dataset = image_dataset.map(dataset_pipeline)
        pixelcnn_dataset = pixelcnn_dataset.cache().batch(64)

        # save so it isnt used again
        pixelcnn_dataset.save(dataset_save_path)

    # actually need to remove channel dim on Y
    pixelcnn_dataset = pixelcnn_dataset.map(lambda x, y: (x, tf.squeeze(y, -1)))
    pixelcnn_dataset = pixelcnn_dataset.shuffle(pixelcnn_dataset.cardinality())

    # initialise pixelcnn object
    io_shape = pixelcnn_dataset.element_spec[0].shape[1:] # it adds batch dim auto
    pcnn = pixel_cnn_model(
        io_shape, 8, vqvae.CODEBOOK_SIZE, num_filters=256
    )

    pcnn.compile(
        keras.optimizers.Adam(
            # dtype_policy="mixed_float16",
            learning_rate=0.001,
            clipnorm=1.0
        ),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy()]
    )

    # load in existing weights
    # pcnn = load_model("pixel.weights.h5")

    pcnn.fit(
        pixelcnn_dataset,
        epochs=200
    )

    generated_outputs = generate_images(
        pcnn, quantizer, decoder, io_shape[:-1]
    )

    show_generated_images(generated_outputs, title="Novel_Generated_Outputs_3")

    # save model
    pcnn.save_weights(
        "pixel_round_2.weights.h5"
    )


