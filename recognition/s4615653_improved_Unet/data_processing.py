import tensorflow as tf
import numpy as np
import glob

def load_data(self):
    # Getting the file paths of data and sort paths
    inputs = sorted(glob.glob("../../ISIC2018_Task1-2_Training_Input_x2/*.jpg"))
    trues = sorted(glob.glob("../../ISIC2018_Task1_Training_GroundTruth_x2/*.png"))

def split_data(inputs,trues,ratio_train,ratio_validation,ratio_test):
    num_inputs = len(inputs)

    # The number of images in our validation and test set
    val_size = int(num_inputs * ratio_validation)
    test_size = int(num_inputs * ratio_test)
    train_size = int(num_inputs * ratio_train)

    # array of inputs for the training images
    train_inputs = inputs[:train_size]
    # array of inputs for validation images
    val_inputs = inputs[train_size:train_size + val_size]
    # array of inputs for test images
    test_inputs = inputs[test_size:]

    # array of trues for the training images
    train_trues = trues[:train_size]
    # array of trues for validation images
    val_trues = trues[train_size:train_size + val_size]
    # array of trues for test images
    test_trues = trues[test_size:]

    return train_inputs, train_trues, val_inputs, val_trues, test_inputs, test_trues

# create TensorFlow Datasets and shuffle them
def tensor_data(train_inputs, train_trues, val_inputs, val_trues, test_inputs, test_trues):
    train_ds = tf.data.Dataset.from_tensor_slices((train_inputs, train_trues))
    val_ds = tf.data.Dataset.from_tensor_slices((val_inputs, val_trues))
    test_ds = tf.data.Dataset.from_tensor_slices((test_inputs, test_trues))

def shuffle_data(train_ds,val_ds,test_ds,train_inputs,val_inputs,test_inputs):
    train_ds = train_ds.shuffle(len(train_inputs))
    val_ds = val_ds.shuffle(len(val_inputs))
    test_ds = test_ds.shuffle(len(test_inputs))

# map data to data arrays
def map_data(train_ds,val_ds,test_ds):
    train_ds = train_ds.map()
    val_ds = val_ds.map()
    test_ds = test_ds.map()

