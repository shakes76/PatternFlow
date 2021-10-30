from typing import Tuple

from keras.datasets import mnist
from keras.callbacks import History
import keras.models
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim
import tensorflow as tf

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
    history, vqvae = train(train_X, (28, 28, 1), 16, 128)

    recreated, recreated_ssim = test(vqvae, test_X[0:20])
    print(f"The SSIM is {recreated_ssim}")
    for i in range(1, 21):
        plt.subplot(4, 5, i)
        plt.imshow(recreated[i - 1])
    plt.show()

def generate_mnist_with_pixel_cnn(vqvae_model):
    pass

def train(data: np.array, sample_shape: Tuple[int, int, int],
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

def test(vqvae_model: keras.models.Sequential, test_data):
    '''
    Test the VQ VAE model and return relavent stats
    '''
    recreated = vqvae_model.predict(test_data, use_multiprocessing=True)

    total_ssim = 0
    for i in range(test_data.shape[0]):
        total_ssim += ssim(recreated[i], test_data[i], multichannel=True)
    recreated_ssim = total_ssim / test_data.shape[0]

    return recreated, recreated_ssim

# Loss Calculation Helpers

def vq_vae_loss(variance):
    def calc_loss(x1, x2,):
        return tf.reduce_mean((x1 - x2) ** 2) / variance

    return calc_loss

def main():
    run_mnist_vqvae()

if __name__ == "__main__":
    main()