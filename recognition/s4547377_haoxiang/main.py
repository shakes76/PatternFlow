from matplotlib import image
from pathlib import Path
from PIL import Image
from numpy import asarray
import glob
import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

training_images_path='H:/githubcomp3710/PatternFlow/recognition/s4547377_haoxiang/ISBI2016_ISIC_Part1_Training_Data/ISBI2016_ISIC_Part1_Training_Data/*.jpg'
groundTruth_path='H:/githubcomp3710/PatternFlow/recognition/s4547377_haoxiang/ISBI2016_ISIC_Part1_Training_GroundTruth/ISBI2016_ISIC_Part1_Training_GroundTruth/*.png'


'''
Function explanations:
1.sorted(): It sorts the elements of a given iterable in a specific order (ascending or descending) and returns it as a list.
2.glob(): Return a possibly-empty list of path names that match pathname, 
    which must be a string containing a path specification. 
'''
training_images=sorted(glob.glob(training_images_path))
groundTruth_images=sorted(glob.glob(groundTruth_path))
'''
Define the batch size, image height, image width, image channels and the size of data set
'''
s_dataset=len(training_images)
s_batch=32
height=192
width=256
number_channel=4

'''
Allocate the sizes of three data set: trainging set(0.7), validation set(0.15), test size(0.15)
'''
s_train=int(0.7*s_dataset)
s_validation=int(0.15*s_dataset)
s_test=int(0.15*s_dataset)

#Generate training, validation and test datasets.
'''
Function explanations:
1.from_tensor_slices():Creates a Dataset whose elements are slices of the given tensors.
2.shuffle():Randomly shuffles the elements of this dataset.
3.take():Creates a Dataset with at most count elements from this dataset.
4.skip():Creates a Dataset that skips count elements from this dataset.
'''
complete_dataset=tf.data.Dataset.from_tensor_slices((training_images, groundTruth_images))
complete_dataset=complete_dataset.shuffle(s_dataset, reshuffle_each_iteration=False)
training_dataset=complete_dataset.take(s_train)
# skip the dataset for train
test_dataset=complete_dataset.skip(s_train)
validation_dataset=complete_dataset.skip(s_validation)
test_dataset=complete_dataset.take(s_test)


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
