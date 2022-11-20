# Model metrics
from tensorflow.keras import backend as K
import tensorflow as tf


def dice_coefficient(y_true, y_pred):
    """
    Dice coefficient calculation between two tensor samples.
    
    Derived from Wikipedia formula, this function can be used
    to gauge the similiarity between two samples.
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    numerator = 2 * K.sum(y_true_f * y_pred_f)
    denominator = K.sum(y_true_f) + K.sum(y_pred_f)
    
    dice = numerator / denominator

    return dice


def IoU(y_true, y_pred):
    """
    Intersection over union between two tensor samples.
    
    Also known as Jaccard index, this function can be used to
    express the similiarity between samples.
    """
    y_pred_f = K.flatten(y_pred)
    y_true_f = K.flatten(y_true)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return intersection / union
