"""
This script contains functions required for dsiplaying images and calculating DSC.
@author: Mujibul Islam Dipto
"""
import tensorflow as tf
import matplotlib.pyplot as plt


def display(data, figsize, cmap):
    """Displays a scan, mask and predicted mask of a lesion from given data.

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
    More information about DSC can be found here: https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    Function adapted from: https://medium.com/@karan_jakhar/100-days-of-code-day-7-84e4918cb72c
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