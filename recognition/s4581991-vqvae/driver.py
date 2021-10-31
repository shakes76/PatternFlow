import os
from typing import Tuple

from keras.datasets import mnist
from keras.callbacks import History
import keras.models
import keras.preprocessing.image
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim
import tensorflow as tf
import tensorflow_probability as tfp

import model

# Constants and Parameters

OASIS_TRAIN_DIR = os.path.join(os.curdir, "keras_png_slices_data", "keras_png_slices_train")
OASIS_TEST_DIR = os.path.join(os.curdir, "keras_png_slices_data", "keras_png_slices_test")

IMAGE_SHAPE = (256, 256, 3)

NUM_LATENT_DIMS = 16
NUM_EMBEDDINGS = 128

# Functions for OASIS

def load_oasis_data(path):
    '''
    Loads preprocessed OASIS Brain data
    '''
    files = os.listdir(path)
    oasis_images = []
    for filename in files:
        file_path = os.path.join(path, filename)
        image = keras.preprocessing.image.load_img(file_path)
        oasis_images.append(keras.preprocessing.image.img_to_array(image))

    return np.array(oasis_images)

def run_oasis_vqvae():
    train_data = load_oasis_data(OASIS_TRAIN_DIR)
    test_data = load_oasis_data(OASIS_TEST_DIR)

    history, vqvae = train_vqvae(train_data, IMAGE_SHAPE,
        NUM_LATENT_DIMS, NUM_EMBEDDINGS)

    recreated, recreated_ssim = test_vqvae(vqvae, test_data[0:20])
    print(f"The SSIM is {recreated_ssim}")
    for i in range(1, 21):
        plt.subplot(4, 5, i)
        plt.imshow(recreated[i - 1])
    plt.show()

# Functions for MNIST

def load_and_preprocess_mnist_data():
    '''
    Loads and preprocesses MNIST data
    '''
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    train_X = np.expand_dims(train_X, -1) / 255.0
    test_X = np.expand_dims(test_X, -1) / 255.0
    return (train_X, train_y), (test_X, test_y)

def run_mnist_vqvae():
    '''
    Train and test the VQ VAE model on MNIST data
    '''
    (train_X, train_y), (test_X, test_y) = load_and_preprocess_mnist_data()
    history, vqvae = train_vqvae(train_X, (28, 28, 1), 16, 128)

    # recreated, recreated_ssim = test_vqvae(vqvae, test_X[0:20])
    # print(f"The SSIM is {recreated_ssim}")
    # for i in range(1, 21):
    #     plt.subplot(4, 5, i)
    #     plt.imshow(recreated[i - 1])
    # plt.show()

    generate_and_plot_with_pixel_cnn(vqvae, train_X)

# VQ VAE Model functions

def generate_and_plot_with_pixel_cnn(vqvae_model: keras.models.Model, input):
    encoder: keras.layers.Layer = vqvae_model.get_layer("encoder")
    decoder: keras.layers.Layer = vqvae_model.get_layer("decoder")
    vector_quantizer = vqvae_model.get_layer("vector_quantizer")

    encoded = encoder.predict(input)
    flattened_encoded = encoded.reshape(-1, encoded.shape[-1])
    quantized = vector_quantizer.encode(flattened_encoded)
    quantized = quantized.numpy().reshape(encoded.shape[:-1])

    latent_height = quantized.shape[1]
    latent_width = quantized.shape[2]
    pixel_cnn = train_pixel_cnn(quantized, quantized,
            latent_width, latent_height,
            vector_quantizer.number_of_embeddings)
    generated_samples = generate_using_pixel_cnn(
            pixel_cnn, decoder, vector_quantizer, 10, encoded.shape[1:])
    plot_generated_samples(generated_samples, 10)

def train_vqvae(data: np.array, sample_shape: Tuple[int, int, int],
        latent_dimensions: int, number_of_embeddings: int)\
        -> Tuple[History, keras.models.Sequential]:
    '''
    Train the VQ VAE model and return relavent stats
    '''
    vqvae = model.create_vqvae_model(
            latent_dimensions, number_of_embeddings, sample_shape)
    vqvae.compile(loss=vq_vae_loss(np.var(data)), optimizer="adam")
    return vqvae.fit(
            data, data, validation_split=0.1, batch_size=128, epochs=10), vqvae

def test_vqvae(vqvae_model: keras.models.Sequential, test_data):
    '''
    Test the VQ VAE model and return relavent stats
    '''
    recreated = vqvae_model.predict(test_data, use_multiprocessing=True)

    total_ssim = 0
    for i in range(test_data.shape[0]):
        total_ssim += ssim(recreated[i], test_data[i], multichannel=True)
    recreated_ssim = total_ssim / test_data.shape[0]

    return recreated, recreated_ssim

def train_pixel_cnn(x, y, latent_width, latent_height, number_of_embeddings):
    pixel_cnn = model.create_pixel_cnn(latent_width, latent_height, number_of_embeddings)
    pixel_cnn.compile(
            optimizer="adam",
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"])
    pixel_cnn.fit(x, y, batch_size=128, epochs=10, validation_split=0.1)

    return pixel_cnn

def generate_using_pixel_cnn(pixel_cnn_model, vqvae_decoder_model,
        vector_quantizer, number_of_images, quantized_shape):
    # Create priors by sampling the PixelCNN model
    priors = np.zeros(shape=(number_of_images,)
            + (pixel_cnn_model.input_shape)[1:])
    batch, rows, cols = priors.shape
    for row in range(rows):
        for col in range(cols):
            logits = pixel_cnn_model(priors, training=False)
            next_sample = tfp.distributions.Categorical(logits=logits).sample()
            priors[:, row, col] = next_sample.numpy()[:, row, col]

    # Use the priors to create latent space encodings
    one_hot_priors = tf.one_hot(priors.astype("int32"),
            vector_quantizer.number_of_embeddings).numpy()
    quantized = tf.matmul(one_hot_priors.astype("float32"),
        vector_quantizer.embeddings, transpose_b=True)
    quantized = tf.reshape(quantized, (-1, *(quantized_shape)))

    # Generate samples from the latent space
    generated_samples = vqvae_decoder_model.predict(quantized)

    return generated_samples

def plot_generated_samples(generated_samples, number_of_images):
    for i in range(number_of_images):
        plt.subplot(5, number_of_images // 5 , i + 1)
        plt.imshow(generated_samples[i].squeeze())
        plt.axis("off")

    plt.savefig("generated_samples.png")
    plt.show()

# Loss Calculation Helpers

def vq_vae_loss(variance):
    def calc_loss(x1, x2,):
        return tf.reduce_mean((x1 - x2) ** 2) / variance

    return calc_loss

def main():
    run_oasis_vqvae()

if __name__ == "__main__":
    main()