import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy 
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator

## Load dataset

# data paths
train_path = "/Users/joshu/COMP3710-project/keras_png_slices_data/train/"
validation_path = "/Users/joshu/COMP3710-project/keras_png_slices_data/validate/"
test_path = '/Users/joshu/COMP3710-project/keras_png_slices_data/test/'

# Variables
input_shape = (256, 256, 3)
#latent_dim = 16
epochs = 10
batch_size = 50
depth = 32

# Data Generator
train_data = ImageDataGenerator(rescale=1./255,)

train_batches = train_data.flow_from_directory(train_path, batch_size=batch_size)
X, y = train_batches.next()
#data_variance = np.var(X)

validation_batches = train_data.flow_from_directory(validation_path, batch_size=batch_size)
X_validate, y_validate = train_batches.next()

test_batches = train_data.flow_from_directory(test_path, batch_size=batch_size)
X_test, y_test = train_batches.next()