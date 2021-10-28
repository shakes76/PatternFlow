from tensorflow.keras.datasets import mnist
import numpy as np
import tensorflow as tf

import pathlib
import math

class OASISSeq(tf.keras.utils.Sequence):
    """
    Sequence to load OASIS dataset

    Based on this: https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence
    and my own Demo 2 part 3 code
    """

    def __init__(self, x_set, y_set, batch_size):
        """
        Initialises a data loader for a set of data
        x_set, y_set: set of file paths for x and y files
        batch_size: number of images in each batch
        """
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.x_img_size=256
        self.y_img_size=256

    def __len__(self):
        """ Returns length of batch set"""
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        """ Returns one batch of data, X and y as a tuple """
        # select set of file names that corrospond to index idx
        X_train_files = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        y_train_files = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        # open each file and load them into the list as an image
        X_train = []
        y_train = []
        for file in X_train_files:
            img = mpimg.imread(file)
            X_train.append(img)
        for file in y_train_files:
            img = mpimg.imread(file)
            y_train.append(img)
            
        # change label names from floats to ints
        labels_o = np.unique(y_train)

        # add an extra dimention 
        X_train = np.array(X_train).reshape(-1, self.x_img_size, self.y_img_size, 1)
        y_train = np.array(y_train).reshape(-1, self.x_img_size, self.y_img_size, 1)
        
        # rename labels from floats to integers
        y_train[np.where(y_train==labels_o[0])] = 0
        y_train[np.where(y_train==labels_o[-1])] = 3
        y_train[np.where(y_train==labels_o[1])] = 1
        y_train[np.where(y_train==labels_o[2])] = 2 
        
        # remove extra dimention in y now that labels are correct and cast as int
        y_train = np.array(y_train).reshape(-1, self.x_img_size, self.y_img_size).astype(np.int32)
        # print(y_train.dtype)
        # should i return tensors or numpy arrays
        return tf.constant(X_train), tf.constant(y_train)

def load_oasis_data(path):
    """ 
    loads oasis data in default structure at path
    returns three sequences: train, valid, test
    """
    data_dir = pathlib.Path(path)
    # actually realised don't need y. remove those
    X_train_files = list(data_dir.glob('./keras_png_slices_train/*'))
    y_train_files = list(data_dir.glob('./keras_png_slices_seg_train/*'))

    X_test_files = list(data_dir.glob('./keras_png_slices_test/*'))
    y_test_files = list(data_dir.glob('./keras_png_slices_seg_test/*'))

    X_valid_files = list(data_dir.glob('./keras_png_slices_validate/*'))
    y_valid_files = list(data_dir.glob('./keras_png_slices_seg_validate/*'))

    train_seq = OASISSeq(sorted(X_train_files),sorted( y_train_files), 10)
    valid_seq = OASISSeq(sorted(X_valid_files),sorted( y_valid_files), 5)
    test_seq = OASISSeq(sorted(X_test_files),sorted( y_test_files), 5)
    return train_seq, valid_seq, test_seq

def load_minst_data(batch_size):
    """
    loads minst dataset to test with
    return three batched sequences: train, valid
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train = tf.reshape(x_train, (60000, 28, 28, 1))
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_data_loader = train_dataset.batch(batch_size)

    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_data_loader = test_dataset.batch(batch_size)

    return train_data_loader, test_data_loader