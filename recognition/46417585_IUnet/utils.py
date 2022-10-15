import tensorflow as tf


def DSC(tensor_1: tf.Tensor, tensor_2: tf.Tensor):
    "Calculates the Dice similarity coefficient for two given tensors"
    matched_elements = tf.math.reduce_sum(tf.cast(tensor_1 == tensor_2, tf.uint8))
    total_elements = tf.size(tensor_1) + tf.size(tensor_2)
    return 2 * matched_elements.numpy() / total_elements.numpy()
