import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

## Load dataset

# Data paths: Change path variable to direct program to location of data.
path = "C:/Users/joshu/COMP3710-project/keras_png_slices_data/"
train_path = path + "train/"
validation_path = path + "validate/"
test_path = path + 'test/'

# Variables
input_shape = (256, 256, 3)
batch_size = 50
depth = 32

# Data Generator - normalise data with image data generator
train_data = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,)

# Training images
train_batches = train_data.flow_from_directory(train_path, batch_size=batch_size)
X, y = train_batches.next()
data_variance = np.var(X)

# Validation images
validation_batches = train_data.flow_from_directory(validation_path, batch_size=batch_size)
X_validate, y_validate = train_batches.next()

# Test images
test_batches = train_data.flow_from_directory(test_path, batch_size=batch_size)
X_test, y_test = train_batches.next()