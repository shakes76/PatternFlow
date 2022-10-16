"""
COMP3170
Jialiang Hou
45996216
Preprocess the image data, use images to create Siamese network train and test set.
"""

# load necessary libraries
import random
import tensorflow as tf
import os
import tensorflow.keras.backend as K
import numpy as np


def img_shuffle(x, y):
    """
    shuffle the image.
    The given data are separated by AD and NC
    It needs to shuffle when combine to together
    :param x: the images [image1, image2, image3, ... imagen]
    :param y: the labels [label1, label2, label3, ... labeln]
    :return: images and labels after shuffle
    """
    pack = list(zip(x, y))
    random.shuffle(pack)
    x[:], y[:] = zip(*pack)
    return x, y


def load_path(path):
    """
    give a path of a folder return all the path of files in this folder
    :param path: a path of a folder
    :return: all the path of files in this folder
    """
    all_path = os.listdir("./" + path)
    for i in range(len(all_path)):
        all_path[i] = "./" + path + '/' + all_path[i]
    return all_path


def process_image(x, y):
    """
    input the image path return the path and process it
    :param x: the path of images
    :param y: the labels
    :return: processed images and labels
    """
    image = tf.io.read_file(x)
    image = tf.image.decode_png(image, channels=1)
    image = tf.cast(image, dtype=tf.float32)
    image = tf.image.resize(image, (60, 64))
    image = image / 255.0
    y = tf.cast(y, dtype=tf.uint8)
    return image, y


def get_dataset():
    """ Input the images """
    # The path of images
    x_train_true = "train/AD"
    x_train_false = "train/NC"
    x_test_true = "test/AD"
    x_test_false = "test/NC"
    # load path of images
    x_train_true = load_path(x_train_true)
    # create array for label, set AD for true
    y_train_true = np.ones(len(x_train_true))

    x_train_false = load_path(x_train_false)
    y_train_false = np.zeros(len(x_train_false))

    x_test_true = load_path(x_test_true)
    y_test_true = np.ones(len(x_test_true))

    x_test_false = load_path(x_test_false)
    y_test_false = np.zeros(len(x_test_false))

    # shuffle the images
    x_train, y_train = img_shuffle \
        (np.concatenate((x_train_true, x_train_false)), np.append(y_train_true, y_train_false))
    x_test, y_test = img_shuffle \
        (np.concatenate((x_test_true, x_test_false)), np.append(y_test_true, y_test_false))

    # create tensorflow dataset
    # currently the dataset contains paths of image and their labels
    train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    # use map to read the images
    train = train.map(process_image)
    test = test.map(process_image)

    """ create train and test dataset """
    # convert the tensorflow dataset from numpy array
    train = np.array(list(train.as_numpy_iterator()))
    test = np.array(list(test.as_numpy_iterator()))

    import numpy.random as random

    # lists for train and test
    train_x1 = list()
    train_x2 = list()
    train_y = list()
    test_x1 = list()
    test_x2 = list()
    test_y = list()

    # create train dataset
    for i in range(21520):  # there are 21520 samples in training set
        # choose 21520 random numbers from 0 to 21520
        index = random.randint(0, 21519, size=21520)
        # the image choose by index
        train_x1.append(train[i][0])
        # the random image
        train_x2.append(train[index[i]][0])
        # define the label
        # if two images have the same label, the new label should be 1. 0 otherwise.
        if train[i][1] == train[index[i]][1]:
            train_y.append(1)
        else:
            train_y.append(0)

    for i in range(563):
        index = random.randint(0, 21519, size=21520)
        test_x1.append(train[i][0])
        test_x2.append(train[index[i]][0])
        if train[i][1] == train[index[i]][1]:
            test_y.append(1)
        else:
            test_y.append(0)

    # finish the creating of train and test set
    # input numpy array to network
    train_x1 = np.array(train_x1)
    train_x2 = np.array(train_x2)
    train_y = np.array(train_y)
    test_x1 = np.array(test_x1)
    test_x2 = np.array(test_x2)
    test_y = np.array(test_y)
    return train_x1,train_x2,train_y,test_x1,test_x2,test_y
