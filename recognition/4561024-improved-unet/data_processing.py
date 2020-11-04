'''
Load and process OASIS brain data set

@author Aghnia Prawira (45610240)
'''

import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def test():
    print("Testing data processing.")
    
def load_image(path):
    image_array = []
    for name in os.listdir(path)[:100]:
        # Loading and resizing image
        filename = path + name
        image = load_img(filename, color_mode='rgba', target_size=(256, 256))
        # Convert image pixels to array
        image = img_to_array(image)
        image = image/255
        image_array.append(image)
    return image_array

def get_palette(image):
    palette = []
    for i in image:
        for j in i:
            palette.append(j)
    return list(sorted(set([tuple(x) for x in palette]), reverse=True))

def get_one_hot(palette, image):
    one_hot = []
    for color in palette:
        class_map = tf.reduce_all(tf.equal(image, color), axis=-1)
        one_hot.append(class_map)
    one_hot = tf.stack(one_hot, axis=-1)
    one_hot = tf.cast(one_hot, tf.float32)
    return one_hot

def one_hot_encode(dataset):
    images = []
    for image in dataset:
        palette = get_palette(image)
        one_hot = get_one_hot(palette, image)
        images.append(one_hot)
    return images