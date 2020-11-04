'''
Build, train and test a TensorFlow based model to segment ISICs 2018 dermatology data
'''

# Import modules
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import numpy as np
from math import ceil
from glob import glob
from IUNet_sequence import iunet_sequence

# Define global variables
X_DATA_LOCATION = 'C:/Users/match/Downloads/ISIC2018_Task1-2_Training_Data/ISIC2018_Task1-2_Training_Input_x2'
Y_DATA_LOCATION = 'C:/Users/match/Downloads/ISIC2018_Task1-2_Training_Data/ISIC2018_Task1_Training_GroundTruth_x2'
TRAIN_SIZE = 0.8
VALIDATE_SIZE = 0.1
TRAIN_BATCH_SIZE = 500
VALIDATE_BATCH_SIZE = 100


# Import data
x_images = glob(X_DATA_LOCATION + '/*.jpg')
y_images = glob(Y_DATA_LOCATION + '/*.png')

x_images.sort()
y_images.sort()
x_images, y_images = shuffle(x_images, y_images)

train_index = ceil(len(x_images) * TRAIN_SIZE)
validate_index = ceil(len(x_images) * (TRAIN_SIZE + VALIDATE_SIZE))

train_seq = iunet_sequence(x_images[:train_index], y_images[:train_index], TRAIN_BATCH_SIZE)
validate_seq = iunet_sequence(x_images[train_index:validate_index], y_images[train_index:validate_index], VALIDATE_BATCH_SIZE)
test_seq = iunet_sequence(x_images[validate_index:], y_images[validate_index:], VALIDATE_BATCH_SIZE)

# Create model



# Create model

# Compile model

# Train model

# Evaluate model

