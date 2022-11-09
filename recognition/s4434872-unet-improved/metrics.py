"""
Metrics module.

@author Dhilan Singh (44348724)

Created: 07/11/2020
"""
import tensorflow as tf
from tensorflow.keras import backend as K

def dice_coefficient(y_true, y_pred, smooth=0.1):
    """
    The Dice Similarity Coefficient (DSC) is a statistical tool used to measure
    the similarities between two sets of data. Most broadly used tool in the 
    validation of image segmentation algorithms.

    Given two sets X and Y,
        DSC = (2*|X intersection with Y|) / (|X| + |Y|)
        
        Where the vertical bars refers to the cardinality of the set, i.e. the 
        number of elements in that set, e.g. |X| means the number of elements in set X.
        The intersection of two sets is just the multiplication of them. So this is
        really twice the number of elements common to both sets divided by the sum 
        of the number of elements in each set.
    
    @param y_true:
        A one-hot encoding of the ground truth segmentation map. Ground truth (actual)
        image segmentation.
    @param y_pred:
        Softmax output of the network. Predicted image segmentation.
    @param smooth:
        Dummy number to prevent division by zero.
    @returns:
        Dice coefficient, between 0 and 1. Used as a metric.
    """
    # Change the dimension to one (pixels in image)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    # Calculation for the loss function
    intersection = K.sum(tf.cast(y_true_f, tf.float32) * tf.cast(y_pred_f, tf.float32))
    return (tf.cast(2., tf.float32) * tf.cast(intersection + smooth, tf.float32)) / (K.sum(tf.cast(y_true_f, tf.float32)) + K.sum(tf.cast(y_pred_f, tf.float32)) + smooth)

def dice_coefficient_loss(y_true, y_pred):
    """
    Dice similarity coefficient loss. Returns the distance between the ground
    truth segmentation and the predicted segmentation. Used as a loss function 
    for training.

        Loss = 1 - DSC
    
    @param y_true:
        A one-hot encoding of the ground truth segmentation map. Ground truth (actual)
        image segmentation.
    @param y_pred:
        Softmax output of the network. Predicted image segmentation.
    @returns:
        Dice coefficient loss (distance), between 0 and 1. Used as a loss function.
    """
    return 1. - dice_coefficient(y_true, y_pred)