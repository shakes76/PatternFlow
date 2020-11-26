'''
Custom sequence class and functions for accessing data in a Keras model
'''

import tensorflow as tf
import numpy as np
from matplotlib.pyplot import imread
from skimage.transform import resize
from sklearn.utils import shuffle
from glob import glob
from math import ceil


def process_segs(seg_image):
    '''
    Process segmentation images into one-hot encoded tensors
    '''
    seg_image = resize(seg_image, (256, 256, 1))
    seg_image = tf.dtypes.cast(tf.math.ceil(seg_image), dtype=tf.uint8)
    return tf.one_hot(seg_image, 2, axis=2)[:, :, :, 0]


class iunet_sequence(tf.keras.utils.Sequence):
    '''
    Custom sequence class for processing and supplying ISIC image data to a 
    Keras model as tensors
    '''
    def __init__(self, x_data, y_data, batch_size):
        self.x_data, self.y_data = x_data, y_data
        self.batch_size = batch_size

    def __len__(self):
        return int(tf.math.ceil(len(self.x_data) / self.batch_size))

    def __getitem__(self, idx):
        i = idx * self.batch_size
        x_batch_paths = self.x_data[i : i + self.batch_size]
        y_batch_paths = self.y_data[i : i + self.batch_size]
        x = np.zeros((self.batch_size, 256, 256, 3), dtype="float32")
        for j, path in enumerate(x_batch_paths):
            x[j] = tf.constant(resize(imread(path), (256, 256, 3)))
        y = np.zeros((self.batch_size, 256, 256, 2), dtype="uint8")
        for j, path in enumerate(y_batch_paths):
            y[j] = process_segs(imread(path))
        return tf.constant(x), tf.constant(y)


def split_data(x_data_location, y_data_location, x_filetype, y_filetype, train_size, val_test_size, batch_size):
    '''
    Split the data into train, validate and test Keras sequences randomly
    '''
    x_images = glob(x_data_location + '/*.' + x_filetype)
    y_images = glob(y_data_location + '/*.' + y_filetype)

    x_images.sort()
    y_images.sort()
    x_images, y_images = shuffle(x_images, y_images)

    train_index = ceil(len(x_images) * train_size)
    validate_index = ceil(len(x_images) * (train_size + val_test_size))

    train_seq = iunet_sequence(x_images[:train_index], y_images[:train_index], batch_size)
    validate_seq = iunet_sequence(x_images[train_index:validate_index], y_images[train_index:validate_index], batch_size)
    test_seq = iunet_sequence(x_images[validate_index:], y_images[validate_index:], batch_size)
    return train_seq, validate_seq, test_seq

