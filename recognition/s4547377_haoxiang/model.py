from matplotlib import image
from pathlib import Path
from PIL import Image
from numpy import asarray
import glob
import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

'''
This section aims to pre-process the training images and the ground truth images
Function Explanations:
1.tf.image.decode_jpeg(): Decode a JPEG-encoded image to a uint8 tensor.
2.tf.image.resize(): Resize images to the given size. (second parameter)
3.tf.cast(): Casts a tensor to a new type.
4.tf.image.decode_png: Decode a PNG-encoded image to a uint8 or uint16 tensor.
5.tf.round: Rounds the values of a tensor to the nearest integer, element-wise.
6.tf.equal(): Returns the truth value of (x == y) element-wise.
'''
'''
process_training(): This function takes training images as input, and pre-processes them.
1. Convert to the tensor 
2. Resize it
3. Normalize it
'''
def process_training(inputs):
    #Change the input image into tensor
    inputs=tf.image.decode_jpeg(inputs,channels=3)
    # resize the image 256*256 
    inputs=tf.image.resize(inputs,[height,width])
    # Standardise values to be in the [0, 1] range.
    inputs=tf.cast(inputs,tf.float32)/255.0   
    return inputs

'''
process_groundtruth(): This function takes ground truth images as input, and pre-processes them.
1. Convert to the tensor 
(This part is different because I find out that the ground truth images are .png)
2. Resize it
3. Normalize it
'''    
def process_groundtruth(inputs):
    inputs=tf.image.decode_png(inputs,channels=1)
    inputs=tf.image.resize(inputs,[height,width])
    inputs=tf.round(inputs/255.0)
    inputs=tf.cast(inputs,tf.float32)
    return inputs

'''
This function simply uses the function above to process all the image data
'''
def process_images(training, groundtruth):
    training = tf.io.read_file(training)
    training = process_training(training)  
    groundtruth = tf.io.read_file(groundtruth)
    groundtruth = process_groundtruth(groundtruth)    
    return training, groundtruth
