"""
This script contains various functions required for image data processing for the UNet model. 
@author: Mujibul Islam Dipto
"""
import tensorflow as tf # # for DL functionalities
from tensorflow.image import resize, decode_jpeg, decode_png # for handling jpg and png images
from tensorflow.io import read_file # for reading image files


def process_image(image, x, y):
    """Proccesses the image of a scan. The image is decoded to a uint8 tensor.

    Args:
        image (Tensor): Tensor representing the image
        x (float): x dimension for resizing
        y (float): y dimension for resizing

    Returns:
        uint8 tensor: decoded image
    """
    image = decode_jpeg(image, channels=1)
    image = resize(image, (x, y))
    image = tf.cast(image, tf.float32) / 255.0
    return image


def process_mask(mask, x, y):
    """Processes the image of a mask. The image is decoded to a uint8 or uint16 tensor.

    Args:
        mask (Tensor): Tensor represting the mask
        x (float): x dimension for resizing
        y (float): y dimension for resizing

    Returns:
        uint8 or uint16 tensor: decoded mask
    """
    mask = decode_png(mask)
    mask = resize(mask, (x, y))
    return mask


def process_data(image_path, mask_path):
    """Processes an image and mask tuple. 

    Args:
        image_path (str): location of the image
        mask_path (str): location of the mask

    Returns:
        tuple: decoded image and mask
    """
    # handle images of scans
    image = read_file(image_path)
    image = process_image(image, 256, 256)

    # handle images of masks
    mask = read_file(mask_path)
    mask = process_mask(mask, 256, 256)
    valid_mask = mask == [0, 255]
    return image, valid_mask
