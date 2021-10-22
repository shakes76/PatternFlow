import tensorflow as tf
def dice_coefficient(truth, pred, smooth=1):
    """
    Code for this function written by Karan Jakhar (2019). Retrieved from:
    https://medium.com/@karan_jakhar/100-days-of-code-day-7-84e4918cb72c
    """
    truth = tf.keras.backend.flatten(truth)
    pred = tf.keras.backend.flatten(pred)
    intersection = tf.keras.backend.sum(truth * pred)
    dice_coef = (2. * intersection + smooth) / (tf.keras.backend.sum(truth)
                                           + tf.keras.backend.sum(pred)
                                           + smooth)
    return dice_coef