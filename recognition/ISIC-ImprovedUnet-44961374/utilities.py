"""
This script contains utility functions for processing data, displaying images
and calculating dice similarity coefficient.
@author: Mujibul Islam Dipto
@date: 31/10/2021
@license: Attribution-NonCommercial 4.0 International. See license.txt for more details.
"""
import tensorflow as tf 
import matplotlib.pyplot as plt
from tensorflow.image import resize, decode_jpeg, decode_png
from tensorflow.io import read_file


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
    mask = decode_png(mask, channels=1)
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

def display_images(data, figsize, cmap):
    """Displays scan, mask and predicted mask of a lesion from given data.

    Args:
        data (list): list containing scan, mask and predicted mask Tensors
        figsize (tuple): size of the figure to be displayed
        cmap (str): colormap for the plot
    """
    plt.figure(figsize=figsize)
    for i, img in enumerate(data):
        plt.subplot(1, len(data), i + 1)
        plt.imshow(img, cmap=cmap)
        plt.axis("off")
    plt.show()


def dice_similarity(x, y):
    """Calculates the Dice Similarity Coefficient between two images, where:
    DSC = (2 * |x intersection y|) / |x| + |y|
    More information about DSC can be found here:
    https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    Function adapted from:
    https://medium.com/@karan_jakhar/100-days-of-code-day-7-84e4918cb72c
    Args:
        x (Tensor): image 1
        y (Tensor): image 2

    Returns:
        float: calculated DSC between these two images
    """
    intersection = tf.reduce_sum(x * y) # |x intersection y|
    numerator = 2 * intersection #  (2 * |x intersection y|)
    denominator = (tf.reduce_sum(x) + tf.reduce_sum(y)) #  |x| + |y|
    dsc = numerator / denominator
    return dsc

def plot_accuracy(history):
    """Plots accuracy graph showing training and validation accuracy for 
    a given training history of a model
    Adapted from: 
    https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/

    Args:
        history (keras.callbacks.History): training history of a model
    """
    plt.plot(history.history['dice_similarity'])
    plt.plot(history.history['val_dice_similarity'])
    plt.title('Accuracy Graph')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()