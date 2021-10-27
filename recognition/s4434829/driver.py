"""
Driver script to show an example usage of the model

TODO: unsure if training can be here or needs to be separate
or maybe training needs to be in the class?
can I have other files?
"""
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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

def main():
    print("hello")

    ### load data
    data_dir = pathlib.Path("./keras_png_slices_data/keras_png_slices_data")
    X_train_files = list(data_dir.glob('./keras_png_slices_train/*'))
    y_train_files = list(data_dir.glob('./keras_png_slices_seg_train/*'))

    X_test_files = list(data_dir.glob('./keras_png_slices_test/*'))
    y_test_files = list(data_dir.glob('./keras_png_slices_seg_test/*'))

    X_valid_files = list(data_dir.glob('./keras_png_slices_validate/*'))
    y_valid_files = list(data_dir.glob('./keras_png_slices_seg_validate/*'))

    train_seq = OASISSeq(sorted(X_train_files),sorted( y_train_files), 10)
    valid_seq = OASISSeq(sorted(X_valid_files),sorted( y_valid_files), 5)
    test_seq = OASISSeq(sorted(X_test_files),sorted( y_test_files), 5)
    X_train, y_train = train_seq.__getitem__(1)
    # print(X_train.__len__())

    plt.imshow(X_train[0], cmap='gray')
    plt.figure()
    plt.imshow(y_train[0], cmap='gray')
    plt.show()

    ### initilise model
    ### make optimiser and loss

    ### train model

    ### plot results

    print("Done")


if __name__ == "__main__":
    main()
