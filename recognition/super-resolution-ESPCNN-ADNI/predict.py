"""predict.py

Showing example usage of the trained super-resolution model.
"""

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

def display_predictions(
    test_ds: tf.data.Dataset,
    model: keras.Model,
    num_images: int = 9,
) -> None:
    """Display some predictions from the super-resolution model.

    Predictions will be displayed against the low and high resolution versions
    of each image.

    Args:
        test_ds (tf.data.Dataset): Testing dataset to retrieve full-sized images
            from
        model (keras.Model): Model to retrieve predictions on downsized images
        num_images (int, optional): Number of images to show. Defaults to 9.
    """
    for images, labels in test_ds.take(1):
        for i in range(num_images):  # For each image show low, high, pred res
            plt.figure()
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))

            ax1.imshow(images[i].numpy())
            ax1.set_label("Low res")

            ax2.imshow(images[i].numpy())
            ax2.set_label("High res")

            ax3.imshow(model.predict(images[i]))
            ax3.set_label("Prediction")
            plt.axis("off")
