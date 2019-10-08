# Tony Meng, Student No: 443298999
# ported from https://github.com/scikit-image/scikit-image/blob/v0.15.0/skimage/transform/radon_transform.py#L12

import tensorflow as tf

def radon(image, theta = None, circle = True):
    if tf.rank(image) != 2:
        raise ValueError('The input image must be 2D')
    if theta is None:
        theta = range(180)
        #theta = tf.range(0, 180, 1)
    
    if circle:
        radius = min(image.shape.as_list()) // 2
        c = [list(range(image.shape.as_list()[0]))]
        c0 = tf.transpose(tf.constant(c))
        c1 = tf.constant(c)
        reconstruction_circle = ((c0 - image.shape.as_list()[0] // 2) ** 2
                                 + (c1 - image.shape.as_list()[1] // 2) ** 2)
        reconstruction_circle = reconstruction_circle <= radius ** 2
        pass
    else:
        pass
    
    #placeholder return
    return image