"""Metrics for training the UNet model
   Author: Yiyun Zhang (s4513350)
"""
import tensorflow.keras.backend as K
from tensorflow.keras import models, optimizers


def dice_coef(y_true, y_pred) -> float:
    """Calculates the dice coefficient for a batch of images

    Args:
        y_true (tf.Tensor): The ground truth
        y_pred (tf.Tensor): The predicted output

    Returns:
        float: The dice coefficient
    """
    # flatten array for faster computation
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)

    intersect = K.sum(K.abs(y_true * y_pred))
    total = K.sum(K.square(y_true)) + K.sum(K.square(y_pred))
    return (2. * intersect + 1.) / (total + 1.)


def dice_loss(y_true, y_pred) -> float:
    """Calculates the dice coefficient loss for a batch of images

    Args:
        y_true (tf.Tensor): The ground truth
        y_pred (tf.Tensor): The predicted output

    Returns:
        float: The dice coefficient loss
    """
    return 1 - dice_coef(y_true, y_pred)
