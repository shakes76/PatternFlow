import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from VQVAE import VQVae

def compare_reconstructions(vqvae: VQVae, x_test_normalised, n_images):
    indices = np.random.choice(len(x_test_normalised), n_images)
    test_samples = x_test_normalised[indices]

    reconstructed = vqvae.predict(test_samples)
    calculate_ssim(test_samples, reconstructed)
    return test_samples, reconstructed


def calculate_ssim(original_images, reconstructed_images):
    similarity = tf.image.ssim(original_images, reconstructed_images, max_val=1)
    print("Structured similarity is:", similarity)

def show_reconstructions(n_images, test_samples, reconstructed):
    for i in range(n_images):
        original_image = test_samples[i].squeeze()
        reconstructed_image = reconstructed[i].squeeze()

        plt.subplot(1, 2, 1)
        plt.imshow(original_image, vmin=0, vmax=1)
        plt.title("Original")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(reconstructed_image, vmin=0, vmax=1)
        plt.title("Reconstructed")
        plt.axis("off")

        plt.savefig(f"reconstructions_{i}.png")
        plt.close()

def show_generated_images(n_images, priors, generated):
    for i in range(n_images):
        plt.subplot(1, 2, 1)
        plt.imshow(priors[i])
        plt.title("Code")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(generated[i].squeeze(), vmin=0, vmax=1)
        plt.title("Generated Sample")
        plt.axis("off")
        plt.savefig(f"generated_{i}.png")
        plt.close()

