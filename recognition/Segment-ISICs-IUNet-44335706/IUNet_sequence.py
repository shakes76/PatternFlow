'''
Custom sequence class and functions for accessing data in a Keras model
'''

import tensorflow as tf
import numpy as np
from matplotlib.pyplot import imread
from skimage.transform import resize

# One hot encode segmented images
def process_segs(seg_image):
    seg_image = resize(seg_image, (256, 256, 1))
    seg_image = tf.dtypes.cast(tf.math.ceil(seg_image), dtype=tf.uint8)
    return tf.one_hot(seg_image, 2, axis=2)[:, :, :, 0]


class iunet_sequence(tf.keras.utils.Sequence):
    
    def __init__(self, x_data, y_data, batch_size):
        self.x_data, self.y_data = x_data, y_data
        self.batch_size = batch_size

    def __len__(self):
        return int(tf.math.ceil(len(self.x_data) / self.batch_size))

    def __getitem__(self, idx):
        i = idx * self.batch_size
        x_batch_paths = self.x_data[i : i + self.batch_size]
        y_batch_paths = self.y_data[i : i + self.batch_size]
        x = np.zeros((self.batch_size, 256, 256, 1), dtype="float32")
        for j, path in enumerate(x_batch_paths):
            x[j] = tf.constant(resize(imread(path), (256, 256, 1)))
        y = np.zeros((self.batch_size, 256, 256, 2), dtype="uint8")
        for j, path in enumerate(y_batch_paths):
            y[j] = process_segs(imread(path))
        return tf.constant(x), tf.constant(y)

