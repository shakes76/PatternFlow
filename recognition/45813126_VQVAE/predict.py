from train import data, trained_model
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

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


num_test_images = 100
idx = np.random.choice(len(data.test_data), num_test_images)
test_images = data.test_data[idx]
reconstructions = trained_model.predict(test_images)

total_ssim_val = 0
for test_image, reconstructed_image in zip(test_images, reconstructions):
    ssim = tf.image.ssim(test_image, reconstructed_image, max_val=1.0)
    total_ssim_val += ssim.numpy()
    #show_subplot(test_image, reconstructed_image)

average_ssim = total_ssim_val / num_test_images
print(average_ssim)