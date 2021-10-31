from math import pi
from typing import Tuple

from keras.datasets import mnist
from keras.callbacks import History
import keras.models
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim
import tensorflow as tf
import tensorflow_probability as tfp

import model

# The model will be optimised using MNIST before moving on to the

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

    generate_mnist_with_pixel_cnn(vqvae, train_X)

def generate_mnist_with_pixel_cnn(vqvae_model: keras.models.Model, input):
    encoder: keras.layers.Layer = vqvae_model.get_layer("encoder")
    decoder: keras.layers.Layer = vqvae_model.get_layer("decoder")
    vector_quantizer = vqvae_model.get_layer("vector_quantizer")

    encoded = encoder.predict(input)
    flattened_encoded = encoded.reshape(-1, encoded.shape[-1])
    quantized = vector_quantizer.encode(flattened_encoded)
    quantized = quantized.numpy().reshape(encoded.shape[:-1])

    print(quantized.shape)

    pixel_cnn = train_pixel_cnn(quantized, quantized, vector_quantizer.number_of_embeddings)
    pixel_cnn.summary()
    generate_using_pixel_cnn(pixel_cnn, decoder, vector_quantizer, 10, encoded.shape[1:])

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
            data, data, validation_split=0.1, batch_size=64, epochs=10), vqvae

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

def train_pixel_cnn(x, y, number_of_embeddings):
    pixel_cnn = model.create_pixel_cnn(7, 7, number_of_embeddings)
    pixel_cnn.compile(
            optimizer="adam",
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"])
    pixel_cnn.fit(x, y, batch_size=128, epochs=10, validation_split=0.1)

    return pixel_cnn

def generate_using_pixel_cnn(pixel_cnn_model, vqvae_decoder_model, vector_quantizer, number_of_images, quantized_shape):
    priors = np.zeros(shape=(10,) + (pixel_cnn_model.input_shape)[1:])
    batch, rows, cols = priors.shape

    # Iterate over the priors because generation has to be done sequentially pixel by pixel.
    for row in range(rows):
        for col in range(cols):
            logits = pixel_cnn_model(priors, training=False)
            next_sample = tfp.distributions.Categorical(logits=logits).sample()
            print(next_sample[:, row, col])
            print(priors[:, row, col])
            priors[:, row, col] = next_sample.numpy()[:, row, col]

    pretrained_embeddings = vector_quantizer.embeddings
    priors_ohe = tf.one_hot(priors.astype("int32"),
            vector_quantizer.number_of_embeddings).numpy()
    quantized = tf.matmul(
        priors_ohe.astype("float32"), pretrained_embeddings, transpose_b=True
    )
    quantized = tf.reshape(quantized, (-1, *(quantized_shape)))

    generated_samples = vqvae_decoder_model.predict(quantized)

    for i in range(batch):
        plt.subplot(1, 2, 1)
        plt.imshow(priors[i])
        plt.title("Code")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(generated_samples[i].squeeze() + 0.5)
        plt.title("Generated Sample")
        plt.axis("off")
        plt.show()

# Loss Calculation Helpers

def vq_vae_loss(variance):
    def calc_loss(x1, x2,):
        return tf.reduce_mean((x1 - x2) ** 2) / variance

    return calc_loss

def main():
    run_mnist_vqvae()

if __name__ == "__main__":
    main()