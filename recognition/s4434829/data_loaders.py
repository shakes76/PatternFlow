from tensorflow.keras.datasets import mnist
import numpy as np
import tensorflow as tf
import matplotlib.image as mpimg
import pathlib
import math

class OASISSeq(tf.keras.utils.Sequence):
    """
    Sequence to load OASIS dataset

    Based on this: https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence
    and my own Demo 2 part 3 code
    """

    def __init__(self, x_set, y_set, batch_size, downsample=None):
        """
        Initialises a data loader for a set of data
        x_set, y_set: set of file paths for x and y files
        batch_size: number of images in each batch
        """
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.x_img_size=256
        self.y_img_size=256
        self.downsample = downsample

    def __len__(self):
        """ Returns length of batch set"""
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        """ Returns one batch of data, two copies of X as a tuple """
        # select set of file names that corrospond to index idx
        X_train_files = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]

        # open each file and load them into the list as an image
        X_train = []
        for file in X_train_files:
            img = mpimg.imread(file)
            X_train.append(img)

        # add an extra dimention 
        X_train = np.array(X_train).reshape(-1, self.x_img_size, self.y_img_size, 1)
        if self.downsample:
            X_train = tf.image.resize(X_train, [self.downsample, self.downsample])
        return tf.constant(X_train), tf.constant(X_train)


def load_oasis_data(path, batch_size):
    """ 
    loads oasis data in default structure at path
    returns three sequences: train, valid, test
    """
    data_dir = pathlib.Path(path)
    X_train_files = list(data_dir.glob('./keras_png_slices_train/*'))
    y_train_files = list(data_dir.glob('./keras_png_slices_seg_train/*'))

    X_test_files = list(data_dir.glob('./keras_png_slices_test/*'))
    y_test_files = list(data_dir.glob('./keras_png_slices_seg_test/*'))

    X_valid_files = list(data_dir.glob('./keras_png_slices_validate/*'))
    y_valid_files = list(data_dir.glob('./keras_png_slices_seg_validate/*'))

    train_seq = OASISSeq(sorted(X_train_files),sorted( y_train_files), batch_size)
    valid_seq = OASISSeq(sorted(X_valid_files),sorted( y_valid_files), batch_size)
    test_seq = OASISSeq(sorted(X_test_files),sorted( y_test_files), 20)
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

def make_indices(X_reshaped, emb):
    """
    Copy of how indices are made in VQ to be used when generating data for
    pixelCNN
    """
    distances = tf.math.reduce_sum(X_reshaped**2, axis=1, keepdims=True) +\
                tf.math.reduce_sum(emb**2, axis=0, keepdims=True) -\
                2*tf.linalg.matmul(X_reshaped, emb)

    indices = tf.argmin(distances, axis=1)
    return indices

def make_indices_dataset(train_seq, encoder, emb, save_folder ="./encoded_data/"):
    for i, X_data in enumerate(train_seq):
        enc_outputs = encoder.predict(X_data)
        reashaped_enc_outputs = enc_outputs.reshape(-1, enc_outputs.shape[-1])
        indices = make_indices(reashaped_enc_outputs, emb)

        indices = indices.numpy().reshape(enc_outputs.shape[:-1])
        np.save(save_folder+"cbindices_{}".format(i), indices)

class CodeBookSeq(tf.keras.utils.Sequence):
    """
    Sequence to load saved embedding indices from file
    """

    def __init__(self, x_set):
        """
        Initialises a data loader for a set of data
        x_set:  set of file paths for x  files
        batch_size: number of images in each batch, since the codebook files are
        already batched I set this to 1
        """
        self.x = x_set
        self.batch_size = 1

    def __len__(self):
        """ Returns length of batch set, should be multiplied by 64"""
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        """ Returns one batch of data, X and X as a tuple """
        # files are already batched so just want to return one
        X_train_files = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        X_train = np.load(X_train_files[0])

        return tf.constant(X_train), tf.constant(X_train)#, tf.constant(y_train)
