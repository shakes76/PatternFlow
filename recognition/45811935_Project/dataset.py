"""
    Load and pre-process all data.

    Author: Adrian Rahul Kamal Rajkamal
"""
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

FILE_PATH = "./ADNI_AD_NC_2D/"
IMG_DIMENSION = 256
BATCH_SIZE = 32


def load_preprocess_image_data(path, img_dim, batch_size, shift):
    """
    Load and preprocess image data (in our case, the ADNI dataset).

    Args:
        path: absolute path to image data (unzipped).
        img_dim: Size of images (assume square)
        batch_size: Size of each batch

    Returns:
        A tf.data.Dataset of the image files.

    """
    img_data = image_dataset_from_directory(path,
                                        label_mode=None,
                                        image_size=(img_dim, img_dim),
                                        color_mode="rgb",
                                        batch_size=batch_size,
                                        shuffle=True)

    return img_data.map(lambda x: (x / 255.0) - shift)
