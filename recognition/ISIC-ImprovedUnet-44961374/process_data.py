"""
This script contains various functions required for data processing for the UNet model. 
@author: Mujibul Islam Dipto
"""
import os  
from sklearn.utils import shuffle, validation  
import math 
import tensorflow as tf
from tensorflow.image import resize, decode_jpeg, decode_png
from tensorflow.io import read_file


def normalize_img(img):
    return tf.cast(img, tf.float32) / 255.0


def process_data(image_path, mask_path):
    # process the image
    image = read_file(image_path)
    image = decode_jpeg(image, channels = 1)
    image = resize(image, (256, 256))
    image = normalize_img(image)
    
    # process the mask
    mask = read_file(mask_path)
    mask = decode_png(mask, channels = 1)
    mask = resize(mask, (256, 256))
    mask = mask == [0, 255]
    return image, mask