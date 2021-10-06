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
