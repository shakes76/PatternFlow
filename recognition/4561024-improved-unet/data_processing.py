'''
Load and process OASIS brain data set

@author Aghnia Prawira (45610240)
'''

import tensorflow as tf

def decode_image(filename):
    '''
    Load and resize an image.
    '''
    # Load and resize image
    image = tf.io.read_file(filename)
    image = tf.image.decode_png(image, channels=1)
    image = tf.image.resize(image, (256, 256))
    return image

def process_image(image, seg):
    '''
    Load and normalize input images.
    Load and one-hot encode segmentation images.
    '''
    image = decode_image(image)
    image = image/255.0
    
    seg = decode_image(seg)
    # One-hot encode image
    seg = tf.cast(seg == [0.0, 85.0, 170.0, 255.0], tf.float32)
    return image, seg