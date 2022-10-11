"""predict.py

Showing example usage of the trained super-resolution model.
"""

from typing import Any

from dataset import downsample_image

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
    for images, _ in test_ds.take(1):
        for i in range(num_images):  # For each image show low, high, pred res
            display_prediction(images[i], model)


def display_prediction(
    test_image: Any,
    model: keras.Model,
    title: str | None = None,
) -> None:
    """Display the low res, original, and predicted form of test_image

    Args:
        test_image (Any): Full-resolution image to test on
        model (keras.Model): Model to predict this image's super-sampling
        title (str | None): Optional title for the plot
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))
    if title:
        fig.suptitle(title)
    down_img = downsample_image(test_image[tf.newaxis, ...])

    for ax in (ax1, ax2, ax3):
        ax.axis("off")

    ax1.imshow(down_img[0].numpy())
    ax1.set_title("Low res")

    ax2.imshow(test_image.numpy())
    ax2.set_title("High res")

    ax3.imshow(model.predict(down_img)[0])
    ax3.set_title("Prediction")

    fig.show()
