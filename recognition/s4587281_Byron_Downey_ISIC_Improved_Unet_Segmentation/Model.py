import sys
import os
import tensorflow as tf
from PIL import Image

num_epochs = argv[1]
batch_size = argv[2]
train_split = argv[3
val_split = argv[4]
test_split = argv[5]
dataset_size = argv[6]
x_data_location = argv[7]
y_data_location = argv[8]

if len(argv) != 8 or not math.isclose(float(train_split) + float(validation_split) + float(test_split), 1):
    print("Usage: Model.py [num_epochs] [batch_size] [train_split] [validation_split] [test_split] [x_data_location] [y_data_location].\nPlease ensure train, validation and test split add up to 1 (e.g. 0.7, 0.15, 0.15)")
script arguments

#Load images into Tensorflow Datasets
x_dataset = tf.keras.utils.image_dataset_from_directory(x_data_location, labels=None)
y_dataset = tf.keras.utils.image_dataset_from_directory(y_data_location, labels=None)

#contains training data
x_train = x_dataset.take(math.floor(train_split * dataset_size))
y_train = y_dataset.take(math.floor(train_split * dataset_size))

#contains validation and test data. Is used simply to extract the validation and test parts.
val_and_test = x_dataset.skip(math.floor(train_split * dataset_size))
val_and_test = y_dataset.skip(math.floor(train_split * dataset_size))

#contains validation data
x_val = x_dataset.take(math.floor(val_split * dataset_size))
y_val = y_dataset.take(math.floor(val_split * dataset_size))

#contains test data
x_test = x_dataset.skip(math.floor(val_split * dataset_size))
y_test = y_dataset.skip(math.floor(val_split * dataset_size))