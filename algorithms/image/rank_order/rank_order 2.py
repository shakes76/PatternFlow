# Author: Askar Jaboldinov

import tensorflow as tf

def rank_order(file_path):
    """Returns an image of the same shape where each pixel is the
    index of the pixel value in the ascending order of the unique
    values of "image".

    Parameters:
    image - path to the image file
    
    Returns a tuple containing:
    labels - an array where each pixel has the rank-order value of the
        corresponding pixel in decoded image from file path. 
    originals - 3-D array of [width, height, channels] shape"""
    try: 
        img = tf.io.read_file(file_path)
        decoded = tf.image.decode_jpeg(img)
        # flattening file, to use tf.unique
        decoded_flat = tf.reshape(decoded, [-1])
        ranks, idx = tf.unique(decoded_flat)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            shape = decoded.eval().shape
            rank_idx = tf.reshape(idx, shape)
            labels = rank_idx.eval() 
            originals = decoded.eval()
            return (labels, originals)
    except AttributeError:
        print("input image must be of array type!")
