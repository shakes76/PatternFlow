import tensorflow as tf
from tensorflow.keras import backend

def DSC(y_true, y_pred, smooth = 1.):
    '''
    Function used to calculate dice similarity coefficient
    :param y_true:
        actual y value
    :param y_pred:
        Predicted y value
    :param smooth:
    :return:
    '''
    # Cast the input true value to the float number for calculation
    y_true = tf.cast(y_true, tf.float32)
    # Flatten the true y value
    y_true = backend.flatten(y_true)
    # Flatten the predicted y value
    y_pred = backend.flatten(y_pred)
    # Get the intersection of the predicted y value and true y value
    intersection = backend.sum(y_true * y_pred)
    # Get the intersection of the predicted y value and true y value
    union = backend.sum(y_true) + backend.sum(y_pred)
    return (2. * intersection + smooth) / (union + smooth)

def DSC_LOSS(y_true, y_pred):
    """Function used to Calculate the DSC LOSS"""
    return 1. - DSC(y_true, y_pred)

