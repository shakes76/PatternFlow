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
    print("SSIM is:", float(similarity))

def show_reconstructions(n_images, test_samples, reconstructed):
    for i in range(n_images):
        original_image = tf.squeeze(test_samples[i], axis=2)
        reconstructed_image = tf.squeeze(reconstructed[i], axis=2)

        plt.subplot(1, 2, 1)
        plt.imshow(original_image, vmin=0, vmax=1, cmap="gray")
        plt.title("Original")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(reconstructed_image, vmin=0, vmax=1, cmap="gray")
        plt.title("Reconstructed")
        plt.axis("off")

        plt.show()
