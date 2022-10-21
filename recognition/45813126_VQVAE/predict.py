"""
File: predict.py
Author: Georgia Spanevello
Student ID: 45813126
Description: Displays example usage of the trained model
"""

from train import data, trained_model
import matplotlib.pyplot as plt
import tensorflow as tf

# Plot the original and reconstructed image on the same plot
def show_subplot(original, reconstructed):
    plt.subplot(1, 2, 1)
    plt.imshow(original.squeeze() + 0.5)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed.squeeze() + 0.5)
    plt.title("Reconstructed")
    plt.axis("off")

    plt.show()


# Use the trained model to predict the test images
reconstructions = trained_model.predict(data.test_data)

# Calculate the average ssim
total_ssim_val = 0
for test_image, reconstructed_image in zip(data.test_data, reconstructions):
    ssim = tf.image.ssim(test_image, reconstructed_image, max_val=1.0)
    total_ssim_val += ssim.numpy()
    show_subplot(test_image, reconstructed_image)
average_ssim = total_ssim_val / len(data.test_data)
print(average_ssim)