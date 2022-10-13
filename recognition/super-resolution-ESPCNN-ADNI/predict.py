"""predict.py

Showing example usage of the trained super-resolution model.
"""

from typing import Any

from dataset import downsample_image
from constants import IMG_ORIG_SIZE

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
    for _, targets in test_ds.take(1):
        for i in range(num_images):  # For each image show low, high, pred res
            display_prediction(targets[i], model)


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
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 20))
    if title:
        fig.suptitle(title)

    down_img = downsample_image(test_image[tf.newaxis, ...])
    down_img_resized = tf.image.resize(down_img[0], IMG_ORIG_SIZE)
    pred_img = model(down_img)[0]

    lowres_psnr = tf.image.psnr(down_img_resized, test_image, max_val=1.0)
    pred_psnr = tf.image.psnr(pred_img, test_image, max_val=1.0)

    ax1.imshow(down_img_resized)
    ax1.set_title(f"Low res - psnr={lowres_psnr:.2f}")

    ax2.imshow(test_image.numpy())
    ax2.set_title("High res")

    ax3.imshow(pred_img)
    ax3.set_title(f"Prediction - psnr={pred_psnr:.2f}")

    fig.show()
