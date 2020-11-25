import tensorflow as tf


def adjust_log(image, gain=1, inv=False):
    def log2(x):
        """
        create a method to calculate the log2 value by equivalent:
        log2(x) = log(x) / log(2)
        """
        numerator = tf.log(x)
        denominator = tf.log(tf.constant(2, dtype=tf.float32))
        return numerator / denominator

    image = tf.constant(image, tf.float32)
    " To make sure the input is non-negative "
    with tf.control_dependencies([tf.assert_non_negative(image)]):
        scale = tf.reduce_max(image) - tf.reduce_min(image)
        with tf.Session() as sess:
            if inv:
                output = (2 ** (image / scale) - 1) * scale * gain
                output = tf.to_int32(output)
                return output.eval(session=sess)
            else:
                cal = log2(1 + image / scale) * scale * gain
                out = tf.to_int32(cal)
                return out.eval(session=sess)
