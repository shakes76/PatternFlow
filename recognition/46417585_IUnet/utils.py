import tensorflow as tf


def DSC(tensor_1, tensor_2):
    "Calculates the Dice similarity coefficient for two given tensors"
    intersection = tf.reduce_sum((tensor_1 * tensor_2))
    union = tf.reduce_sum(tensor_1) + tf.reduce_sum(tensor_2)
    return 2 * intersection / union


DSC_loss = lambda t1, t2: 1 - DSC(t1, t2)
