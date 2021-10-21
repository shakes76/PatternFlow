"""
Metrics functions to perform on segmentation tasks

@author: Jeng-Chung Lien
@student id: 46232050
@email: jengchung.lien@uqconnect.edu.au
"""
import os
# Suppress the INFO message
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf


def dice_coef(y_true, y_pred):
    """
    Function to calculate the dice coefficient between the true masks and the predicted masks

    Parameters
    ----------
    y_true : array, tensor
      The tensor of the true masks
    y_pred : array, tensor
      The tensor of the predicted masks

    Returns
    -------
    coef : float32 tensor
      The dice coefficient value
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    coef = (2 * tf.reduce_sum(y_true * y_pred)) / tf.reduce_sum(y_true + y_pred)

    return coef


def dice_loss(y_true, y_pred):
    """
    Function to calculate the dice loss between the true masks and the predicted masks

    Parameters
    ----------
    y_true : array, tensor
      The tensor of the true masks
    y_pred : array, tensor
      The tensor of the predicted masks

    Returns
    -------
    loss : float32 tensor
      The dice loss value
    """
    loss = 1. - dice_coef(y_true, y_pred)

    return loss