
import tensorflow as tf
import tensorflow.keras.backend as K 


def diceCoefficient(y_true, y_pred, smooth=1.):
    y_true = tf.convert_to_tensor(y_true, dtype='float32')
    y_pred = tf.convert_to_tensor(y_pred, dtype='float32')
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    #avoid the result equal to 0
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)