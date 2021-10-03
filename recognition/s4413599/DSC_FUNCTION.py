import tensorflow as tf


def DSC(y_pred, y_true):
    '''
    :param y_pred:
        Predicted y value
    :param y_true:
        Actual y value
    :return:
        Print the dice coefficient
    '''
    logical_and = tf.logical_and(y_pred, y_true)
    logical_and = tf.cast(logical_and, tf.float32)
    intersection = tf.reduce_sum(logical_and, axis= (1, 2))
    union = tf.reduce_sum(tf.cast(y_pred, tf.float32), axis= (1, 2)) + tf.reduce_sum(tf.cast(y_true, tf.float32), axis= (1, 2))
    dice = (2 * intersection)/union
    dice = tf.reduce_mean(dice)
    print("THE Dice coefficient is {0:.2f}%".format(dice * 100))