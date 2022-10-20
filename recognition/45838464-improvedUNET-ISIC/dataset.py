import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import numpy as np
import os
import glob

def preprocess_data(path):
    """
    Returns an array of the image data
    """

    images = []
    image_locations = sorted(glob.glob(path))

    for file in image_locations:
        
        # load image
        image = tf.io.read_file(file)
        image = tf.io.decode_jpeg(image, channels=3)

        # resize and normalize
        image = tf.image.resize_with_pad(image, 128, 128)
        image = image / 255.0
        images.append(image)
    
    images = np.array(images)
    return images    

def preprocess_masks(path):

    masks = []
    mask_locations = sorted(glob.glob(path))

    for file in mask_locations:
        
        # load mask
        mask = tf.io.read_file(file)
        mask = tf.io.decode_png(mask, channels=1)

        # resize and normalize
        mask = tf.image.resize_with_pad(mask, 128, 128)
        mask = mask / 255.0
        masks.append(mask)

    masks = np.array(masks) 
    masks = to_categorical(masks)  
    return masks
