import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from VQVAE import VQVae

def show_vqvae_training_loss(vqvae: VQVae):
    """
    Shows graph of VQVAE training loss
    """
    plt.plot(vqvae.total_loss_list)
    plt.plot(vqvae.reconstruction_loss_list)
    plt.plot(vqvae.vq_loss_list)
    plt.savefig("loss_graph.png")
    plt.title("VQ-VAE training losses")
    plt.legend(["Total loss", "Reconstruction loss", "VQ loss"])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.close()

def compare_reconstructions(vqvae: VQVae, x_test_normalised, n_images):
    """
    Returns a list of sample images, and a list of corresponding VQVAE reconstructions
    
    Variables
        vqvae: Vector quantised autoencoder
        x_test_normalised: List of normalised test images
        n_images: Number of images to return
        
    Return value
        (test_samples, reconstructed): list of images from the test set, and a list of their
                                       respective reconstructions
    """
    indices = np.random.choice(len(x_test_normalised), n_images)
    test_samples = x_test_normalised[indices]

    reconstructed = vqvae.predict(test_samples)
    calculate_ssim(test_samples, reconstructed)
    return test_samples, reconstructed


def calculate_ssim(original_images, reconstructed_images):
    """
    Calculate and print the average structured similarity between original and reconstructed images
    """
    similarity = tf.reduce_mean(tf.image.ssim(original_images, reconstructed_images, max_val=1))
    print("Structured similarity is:", similarity)

def show_reconstructions(n_images, test_samples, reconstructed):
    """
    Create and save an image comparing a number of test samples and their reconstructions
    """
    for i in range(n_images):
        original_image = test_samples[i].squeeze()
        reconstructed_image = reconstructed[i].squeeze()

        plt.subplot(1, 2, 1)
        plt.imshow(original_image, vmin=0, vmax=1, cmap="gray")
        plt.title("Original")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(reconstructed_image, vmin=0, vmax=1, cmap="gray")
        plt.title("Reconstructed")
        plt.axis("off")

        plt.savefig(f"reconstructions_{i}.png")
        plt.close()

def show_generated_images(n_images, priors, generated):
    """
    Create and save an image containing a number of priors and generated images
    """
    for i in range(n_images):
        plt.subplot(1, 2, 1)
        plt.imshow(priors[i], cmap="gray")
        plt.title("Code")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(generated[i].squeeze(), vmin=0, vmax=1, cmap="gray")
        plt.title("Generated Sample")
        plt.axis("off")
        plt.savefig(f"generated_{i}.png")
        plt.close()

