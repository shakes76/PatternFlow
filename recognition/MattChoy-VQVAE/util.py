import tensorflow as tf
from modules import VQVAE
import matplotlib.pyplot as plt

def compare_reconstructions(vqvae, dataset, n_images):
    # Run n_images from the dataset through the vqvae
    # Calculate SSIM
    # Return samples used, and reconstructed images.
    test_samples = [im for im in dataset.unbatch().take(n_images)]
    test_samples = tf.convert_to_tensor(test_samples)

    reconstructed = vqvae.predict(test_samples)
    test_samples = [tf.image.rgb_to_grayscale(im) for im in test_samples]
    ssim(test_samples, reconstructed)
    return test_samples, reconstructed

def ssim(original_images, reconstructed_images):
    # https://en.wikipedia.org/wiki/Structural_similarity
    similarity = tf.reduce_mean(tf.image.ssim(original_images, reconstructed_images, max_val=1))
    print("SSIM is:", similarity)
