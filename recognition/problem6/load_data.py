import tensorflow as tf
import os
import random

def load_paths(directory):
    training_paths = {}

    
    for filename in sorted(os.listdir(directory)):  
        path = os.path.join(directory,filename)
        training_paths[path] = 1
    
        
      
    return sorted(training_paths.keys())


def load_images(filename):
    img = tf.io.read_file(filename)
    img = tf.io.decode_png(img, channels = 3)
    #img = tf.image.resize(img, size = (256, 256))
    img = tf.dtypes.cast(img, tf.float32)
    img = img / 255.0
    #img = img.numpy().astype('float32')
    return img
