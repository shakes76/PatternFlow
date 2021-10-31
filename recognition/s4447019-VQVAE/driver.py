import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import glob
import matplotlib.pyplot as plt
import tensorflow_probability as tfp


### PRE-PROCESSING FROM DEMO 2 ################################

# Download the entire OASIS dataset and save to local drive
dataset_url = "https://cloudstor.aarnet.edu.au/plus/s/tByzSZzvvVh0hZA/download"
data_dir = tf.keras.utils.get_file(origin=dataset_url, fname='oasis_data', extract=True)

# Use the glob library to separate and save the train, test and validation datasets to local drive
train_images = glob.glob('C:\\Users\\mmene\\.keras\\datasets\\keras_png_slices_data\\keras_png_slices_train' + '/*.png')
validate_images = glob.glob('C:\\Users\\mmene\\.keras\\datasets\\keras_png_slices_data\\keras_png_slices_validate' + '/*.png')
test_images = glob.glob('C:\\Users\\mmene\\.keras\\datasets\\keras_png_slices_data\\keras_png_slices_test' + '/*.png')

# Print the number of images available within each dataset
print('Number of training images:', len(train_images))
print('Number of validation images:', len(validate_images))
print('Number of testing images:', len(test_images))

# Transform the image data into tensors for preprocessing
train_ds = tf.data.Dataset.from_tensor_slices(train_images)
validate_ds = tf.data.Dataset.from_tensor_slices(validate_images)
test_ds = tf.data.Dataset.from_tensor_slices(test_images)

# Reshuffle the data every epoch so that there are different batches for each epoch
# Note: buffer needs to be greater than, or equal to the size of the data set for effective shuffling
train_ds = train_ds.shuffle(len(train_images))
validate_ds = validate_ds.shuffle(len(validate_images))
test_ds = validate_ds.shuffle(len(test_images))

# Mapping the filenames to data arrays
#---Reference for function used below: 
#---COMP3710 lecture, 24th October 2020---#
def map_fn(filename):
    img = tf.io.read_file(filename) #Open the file
    img = tf.image.decode_png(img, channels=1) #Defining the number of channels (1 for B&W images)
    img = tf.image.resize(img, (128, 128)) #Resize the images to feed into the network
    img = tf.cast(img, tf.float32) / 255.0 #Normalise the data
    return img

# Update the datasets by passing them in map_fn()
train_ds = train_ds.map(map_fn)
validate_ds = validate_ds.map(map_fn)
test_ds = test_ds.map(map_fn)

# Print a sample of images from the test dataset
plt.figure(figsize=(10,10))
plt.title("Sample of images from OASIS test dataset")
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(next(iter(train_ds.batch(9)))[i])
    plt.axis('off')
plt.show()
###############################################################




