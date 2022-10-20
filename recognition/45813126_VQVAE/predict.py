from train import data, trained_model
import numpy as np
import matplotlib.pyplot as plt

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


idx = np.random.choice(len(data.test_data), 10)
test_images = data.test_data[idx]
reconstructions = trained_model.predict(test_images)

for test_image, reconstructed_image in zip(test_images, reconstructions):
    show_subplot(test_image, reconstructed_image)