"""
Author: Zachary Oar
Student Number: 45314669
Course: COMP3710 Semester 2
Date: November 2020

Driver script for an Improved UNet model, which will conduct image segmentation
on the ISIC 2018 Challenge dataset.
"""

from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from PIL import Image
from os import listdir
from os.path import isfile, join
import math
from model import make_model


def get_filenames_from_dir(directory):
    """
    Returns a list of all file names within a given directory.

    Parameters
    ----------
    directory: string
        The name of the directory having its file names returned.

    Returns
    ----------
    A list of all file names within the directory with the given name.
    """
    return [f for f in listdir(directory) 
    if isfile(join(directory, f)) and f not in ("ATTRIBUTION.txt", "LICENSE.txt")]


def encode_y(y):
    """
    One-hot encodes a label image's numpy array to prepare for
    binary cross-entropy.

    Parameters
    ----------
    y: numpy array of floats
        The numpy array of a label image to be encoded.

    Returns
    ----------
    y: numpy array of floats
        The input array, now one-hot encoded for binary cross-entropy.
    """
    y = np.where(y < 0.5, 0, y)
    y = np.where(y > 0.5, 1, y)

    y = keras.utils.to_categorical(y, num_classes=2)
    return y


class SequenceGenerator(keras.utils.Sequence):
    """
    A keras Sequence to be used as an image generator for the model.
    """

    def __init__(self, x, y, batchsize):
        """
        Creates a new SequenceGenerator instance.

        Parameters
        ----------
        x: list of strings
            A list of file names for preprocessed images.
        y: list of strings
            A list of file names for corresponding label images.
        batchsize: int
            The set batch size for this generator.
        """
        self.x, self.y, self.batchsize = x, y, batchsize

    def __len__(self):
        """
        Returns the total number of unique batches that can be generated.

        Returns
        ----------
        The total number of unique batches that can be generated.
        """
        return math.ceil(len(self.x) / self.batchsize)

    def __getitem__(self, idx):
        """
        Returns a batch preprocessed image data and label image data, 
        corresponding to the given id.

        Parameters
        ----------
        idx: int
            The id of the batch to be returned.

        Returns
        ----------
        batch_x: numpy array of image data
            A batch of image data for preprocessed images.
        batch_x: numpy array of image data
            A batch of image data for corresponding label images.
        """
        x_names = self.x[idx * self.batchsize:(idx + 1) * self.batchsize]
        y_names = self.y[idx * self.batchsize:(idx + 1) * self.batchsize]
        
        # open x image names, resize, normalise and make a numpy array
        batch_x = np.array([np.asarray(Image.open("ISIC2018_Task1-2_Training_Input_x2/" 
                + file_name).resize((256, 192))) for file_name in x_names]) / 255.0

        # open y image names, resize, normalise, encode to one-hot and make a numpy array
        batch_y = np.array([np.asarray(Image.open("ISIC2018_Task1_Training_GroundTruth_x2/" 
                + file_name).resize((256, 192))) for file_name in y_names]) / 255.0
        batch_y = encode_y(batch_y)

        return batch_x, batch_y

if __name__ == "__main__":
    # makes arrays of the images and label names
    x_names = get_filenames_from_dir("ISIC2018_Task1-2_Training_Input_x2")
    y_names = get_filenames_from_dir("ISIC2018_Task1_Training_GroundTruth_x2")

    # 15% of all the images are set aside as the test set
    x_train_val, x_test, y_train_val, y_test = train_test_split(x_names, y_names, test_size=0.15, random_state=42)

    # 17% of the non-test images are set aside as the validation set
    x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.17, random_state=42)

    # make generators with batch size 4 for each set
    train_gen = SequenceGenerator(x_train, y_train, 4)
    val_gen = SequenceGenerator(x_val, y_val, 4)
    test_gen = SequenceGenerator(x_test, y_test, 4)

    # train the model
    model = make_model()
    model.fit(train_gen, validation_data=val_gen, epochs=15)

    # evaluate the model on the test set
    model.evaluate(test_gen)

    # show 4 generated images from the test set and compare with expected output
    test_images_x, test_images_y = test_gen.__getitem__(0)
    prediction = model.predict(test_images_x)
    plt.figure(figsize=(10, 10))
    for i in range(4):
        plt.subplot(4, 3, i*3+1)
        plt.imshow(test_images_x[i])
        plt.axis('off')
        plt.title("Original", size=12)
        plt.subplot(4, 3, i*3+2)
        plt.imshow(tf.argmax(prediction[i], axis=2))
        plt.axis('off')
        plt.title("Predicted", size=12)
        plt.subplot(4, 3, i*3+3)
        plt.imshow(tf.argmax(test_images_y[i], axis=2))
        plt.axis('off')
        plt.title("Expected", size=12)
    plt.show()
