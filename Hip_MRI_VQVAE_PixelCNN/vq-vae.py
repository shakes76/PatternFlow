import tensorflow as tf
import keras
from keras import layers

# for data
import os
import nibabel as nib
import numpy as np

# testing
import matplotlib.pyplot as plt
from skimage.transform import resize
import json

"""
Based on example: 'Vector-Quantized Variational Autoencoders' by Sayak Paul
href: https://keras.io/examples/generative/vq_vae/
"""

"""
IDEAS: Could increase the number of hidden units across the board
"""

# TRAIN_FOLDER = "data/keras_slices_data/keras_slices_train/"
TRAIN_FOLDER = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_train"
HISTORY_PATH = "history.json"
LATENT_DIM = 3
CODEBOOK_SIZE = 512

class VectorQuantizerLayer(layers.Layer):

    def __init__(
            self,
            num_embeddings,  # number of vectors in codebook
            embedding_dim,
            commitment_weight=0.25,  # coefficient of commitment loss
            **kwargs
        ):
        super().__init__(**kwargs)

        self.commitment_weight=commitment_weight
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        # Initialise codebook
        self.codebook = self.add_weight(
            name="VQ-CODEBOOK",
            shape=(self.embedding_dim, self.num_embeddings),
            initializer=tf.random_uniform_initializer(),
            trainable=True
        )


    def call(self, x):
        """
        Given a layer of network, replace each vector in the layer with a
        vector from the codebook - quantize the layer
        """

        # store starting shape
        start_shape = tf.shape(x)

        # flatten to single row of embedding_dim vectors
        flattened = tf.reshape(x, [-1, self.embedding_dim])

        # actually quantize
        # indices of closest code vectors
        indices = self.get_code_indices(flattened)

        # one-hot the needed vector
        one_hot_indices = tf.one_hot(indices, self.num_embeddings)

        # extract the given index from the codebook - one hot selects correct
        # codebook vector
        quantized = tf.matmul(one_hot_indices, self.codebook, transpose_b=True)
        quantized = tf.reshape(quantized, start_shape)

        # loss definitions
        loss_commitment = tf.reduce_mean(
            # dont let this loss flow to the codebook - only to update encoder
            # so detach with stop_gradient

            # MSE to get encoder to give vectors close to codebook vectors
            (tf.stop_gradient(quantized) - x)**2
        )

        loss_codebook = tf.reduce_mean(
            # for codebook only, stop gradient from flowing to encoder
            (quantized - tf.stop_gradient(x)) ** 2
        )

        # add to model
        loss_total = self.commitment_weight * loss_commitment + loss_codebook
        self.add_loss(loss_total)

        # trick to stop gradient flowing from rest of model going through
        # the codebook - quantized is gradient stopped but we're still passing
        # the quantized output -- gradient flows through x
        gradient_skipped_quantized = x + tf.stop_gradient(quantized - x)
        return gradient_skipped_quantized

    def get_code_indices(self, flattened):
        """
        calculate L2 distance to all codebook vectors and return the index
        of the closest one for each vector in flattened
        """

        ab = tf.matmul(
            flattened, self.codebook
        )

        distances = (#L2
            tf.reduce_sum(flattened ** 2, axis=1, keepdims=True)
            + tf.reduce_sum(self.codebook ** 2, axis=0)
            -2*ab
        )

        smallest_indices = tf.argmin(
            distances, axis=1
        )

        return smallest_indices


def residual_block(x, num_filters):
        """
        standard residual block. used for encoder and decoder both
        """
        start = x
        x = keras.layers.Conv2D(
            num_filters, 3, 1, padding="same"
        )(x)

        x = keras.activations.relu(x)

        x = keras.layers.Conv2D(
            num_filters, 1, 1, padding="same"
        )(x)

        x = keras.layers.Add()([x, start])
        x = keras.activations.relu(x)

        return x


def get_encoder(
        input_shape,
        layers_filters,
        latent_dim=64,
    ):
    """
    returns an encoder

    layers: list of kernal sizes to iterate over
    """

    inputs = keras.Input(shape=input_shape)

    x = inputs
    for num_filters in layers_filters:
        x = keras.layers.Conv2D(
            num_filters,
            4,
            strides=2,
            padding="same"
        )(x)

        x = keras.activations.relu(x)

    # down sampling followed by residual blocks
    x = residual_block(x, num_filters)
    x = residual_block(x, num_filters)

    # use 1x1 kernel to convert to latent dim size
    outputs = keras.layers.Conv2D(
        latent_dim, 1, padding="same"
    )(x)

    return keras.Model(inputs, outputs, name='encoder')

def get_decoder(
    layers_filters,
    input_shape, # shape of encoder output
    latent_dim = 64
):
    """
    layers_filters should be the same as the encoder but reversed.

    Returns decoder object.
    """

    inputs = keras.Input(
        shape=input_shape
    )
    x = inputs
    for num_filters in layers_filters:

        x = keras.layers.Conv2DTranspose(
            num_filters, 4, strides=2, padding="same"
        )(x)

        x = keras.activations.relu(x)

    x = residual_block(x, num_filters)
    x = residual_block(x, num_filters)

    # 1x1 kernel to bring to a single layer
    x = layers.Conv2DTranspose(1, 3, padding="same")(x)
    outputs = keras.activations.sigmoid(x) # scale to [0,1]
    return keras.Model(
        inputs, outputs, name="decoder"
    )

def vq_vae(
    input_shape,
    latent_dim=64,
    num_embeddings=128,
):
    quantize_layer = VectorQuantizerLayer(
        num_embeddings, latent_dim, name="quantizer"
    )

    filter_layers = [
        128, 256, 512
    ]

    encoder = get_encoder(
        latent_dim=latent_dim,
        input_shape=input_shape,
        layers_filters=filter_layers
    )

    decoder = get_decoder(
        filter_layers[::-1], # reverse order to encoder to end in same shape
        encoder.output.shape[1:],
        latent_dim=latent_dim
    )

    inputs = keras.Input(shape=input_shape)
    encoder_out = encoder(inputs)
    quantized_out = quantize_layer(encoder_out)
    reconstruction = decoder(quantized_out)

    return keras.Model(
        inputs, reconstruction, name="vq-vae"
    )

class trainer(keras.models.Model):
    """
    Class to train a vq-vae
    """
    def __init__(
        self,
        train_variance,
        latent_dim,
        codebook_size,
        input_shape,
        **kwargs
    ):
        # init
        super().__init__(**kwargs)
        self.train_variance = train_variance
        self.latent_dim = latent_dim
        self.codebook_size = codebook_size


        # actual model
        self.vqvae = vq_vae(
            input_shape, latent_dim, codebook_size
        )

        # initialise losses - all mean to average over batch
        self.total_loss = keras.metrics.Mean("total_loss")
        self.reconstruction_loss = keras.metrics.Mean("reconstruction_loss")
        self.vqvae_loss = keras.metrics.Mean("vqvae_loss")

    def call(self, x):
        return self.vqvae(x)

    # override metrics so our custom metrics are reset on epochs/tracked
    @property
    def metrics(self):
        return [
            self.total_loss, self.reconstruction_loss, self.vqvae_loss
        ]

    def train_step(self, x):

        with tf.GradientTape() as tape:
            # get output
            reconstruction = self.vqvae(x)

            # calculate loss - reconstrution normalised to dataset variance
            recon_loss = tf.reduce_mean((x-reconstruction)**2) #/ self.train_variance

            total_loss = (
                recon_loss
                + sum(self.vqvae.losses)
            )

        # now backpropegate
        grad = tape.gradient(
            # derivative of total_loss w.r.t all trainable variables
            total_loss, self.vqvae.trainable_variables
        )
        self.optimizer.apply_gradients(  # run gradient descent
            zip(grad, self.vqvae.trainable_variables)
        )

        # track losses
        self.total_loss.update_state(total_loss)
        self.reconstruction_loss.update_state(recon_loss)
        self.vqvae_loss.update_state(sum(self.vqvae.losses))

        # return result
        return {
            "loss" : self.total_loss.result(),
            "reconstruction loss": self.reconstruction_loss.result(),
            "vqvae loss": self.vqvae_loss.result()
        }

def __NUMPY__load_image_to_tensor(filepath: str):
    """
    loads the image at filepath and returns it as a greyscale numpy array
    with values between [0,1]
    """
    filepath = filepath.decode("utf-8")
    # load into greyscale
    img = nib.load(filepath)

    # as np array
    data = img.get_fdata().astype(np.float32)
    zoom = img.header.get_zooms()

    # Normalize to [0,1]
    data = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)

    data_resized = resize(
        data, (256, 128), order=1, preserve_range=True, anti_aliasing=True
    ).astype(np.float32)

    # expand channels on end to 1
    data = np.expand_dims(data_resized, axis=-1)

    return data  # returns as numpy

def load_image_to_tensor(filepath):
    data = tf.numpy_function(  # execute and cast to tensor
        __NUMPY__load_image_to_tensor, [filepath], tf.float32
    )
    # define the shape
    data.set_shape([256, 128, 1])
    print(data)

    return data


def get_dataset(folder):
    train_files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.endswith(".nii.gz")
    ]

    train_data = tf.data.Dataset.from_tensor_slices(
    train_files
    )

    train_files = train_data.map(load_image_to_tensor)

    return train_files

def get_variance(dataset):
    """
    Compute variance of all elements in a tf.data.Dataset.
    Assumes dataset yields numeric tensors (scalars or arrays).
    """
    # First pass: compute mean
    count = 0
    total = 0.0
    for x in dataset:
        total += tf.reduce_sum(tf.cast(x, tf.float32))
        count += tf.size(x).numpy()
    mean = total / count

    # Second pass: compute squared differences
    sq_diff_sum = 0.0
    for x in dataset:
        sq_diff_sum += tf.reduce_sum((tf.cast(x, tf.float32) - mean) ** 2)

    variance = sq_diff_sum / count
    return variance.numpy()


def get_image_shape():
    """
    returns the shape of an image in the train dataset
    """
    # fetch the shape of the output
    for batch in get_dataset(TRAIN_FOLDER):
        img_shape = batch.shape
        return img_shape


def plot_reconstruction(original, reconstruction):
    """
    Plots side by side the two greyscale tensors
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(original)
    axes[1].imshow(reconstruction)

    plt.tight_layout()
    plt.savefig("reconstruction.jpg")


def main():

    train_variance = get_variance(
        get_dataset(TRAIN_FOLDER)
    )

    train_dataset = get_dataset(TRAIN_FOLDER).batch(128)

    data_augmentation = keras.Sequential([
        layers.RandomRotation(0.05),        # 10% rotation
        layers.RandomZoom(0.1),            # zoom in/out
        layers.RandomTranslation(0.05, 0.05), # shift
        layers.RandomContrast(0.1),
    ])

    train_dataset.map(lambda x: data_augmentation(x, training=True))

    img_shape = get_image_shape()
    print(f"Image shape: {img_shape}")

    # actuall train the model

    vq_trainer = trainer(
        train_variance, LATENT_DIM, CODEBOOK_SIZE, img_shape
    )

    vq_trainer.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
    )

    history = vq_trainer.fit(
        train_dataset,
        epochs=1,

    )

    model = vq_trainer.vqvae

    ### Test save/load
    weights_path = "trainer.weights.h5"
    model.save_weights(weights_path)
    model = load_model(weights_path)


    ####### PLOT RECONSTRUCTION

    for batch in get_dataset(TRAIN_FOLDER):
        img = batch # just fetch one image
        break

    img_batched = tf.expand_dims(img, axis=0) # add batch dim
    img_batched = tf.expand_dims(img_batched, axis=-1) # add channel dim
    reconstruction = tf.squeeze(model(img_batched), axis=0)

    plot_reconstruction(
        img, reconstruction
    )

    # save loss
    with open(HISTORY_PATH, 'w+') as f:
        json.dump(history.history['loss'], f)

def load_model(path):
    """
    loads the vqvae weights stored at the path into a new vqvae model

    to save weights for use by this helper. trainer.vqvae.save_weights(path)
    """

    new_vq_vae = vq_vae(
        get_image_shape(),
        LATENT_DIM,
        CODEBOOK_SIZE
    )

    new_vq_vae.compile(optimizer="adam")
    new_vq_vae.build(get_image_shape())
    new_vq_vae.load_weights(path)

    return new_vq_vae

def test_load():
    model = load_model("trained_models/trainer.weights.h5")
    model.summary()

if __name__ == "__main__":
    # main()
    ####### PLOT RECONSTRUCTION

    model = load_model("trainer.weights.h5")

    i = 0
    for batch in get_dataset(TRAIN_FOLDER):
        img = batch # just fetch one image
        if i > 25:
            break
        i += 1

    img_batched = tf.expand_dims(img, axis=0) # add batch dim
    img_batched = tf.expand_dims(img_batched, axis=-1) # add channel dim
    reconstruction = tf.squeeze(model(img_batched), axis=0)

    plot_reconstruction(
        img, reconstruction
    )