import tensorflow as tf

def dice_similarity(a, b):
    """
    Implementation of dice similarity from wikipedia
    """
    a_flat = tf.keras.backend.flatten(a)
    b_flat = tf.keras.backend.flatten(b)
    top = 2 * (tf.keras.backend.sum(a_flat * b_flat))
    bot = tf.keras.backend.sum(a_flat) + tf.keras.backend.sum(b_flat)
    dsc = top/bot
    return dsc