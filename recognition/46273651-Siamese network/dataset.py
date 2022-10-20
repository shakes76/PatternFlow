"""
This is the script for the data preprocessing, which is used to load the data, 
make pairs, split the data into train and test, and visualize the data.
"""


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt



def load_data(path, img_size, color_mode):
    """
    Load the data from the path and return the normalized dataset

    Args:
        path: the path of the data
        img_size: the size of the image
        color_mode: the color mode of the image (grayscale or rgb)

    Returns:
        dataset: the normalized dataset
    """
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        path, label_mode=None, color_mode = color_mode, image_size=img_size, batch_size=16
    )

    # normalize the data
    dataset = dataset.map(lambda x: x / 255.0)

    # unbatch the data
    dataset = dataset.unbatch()

    return dataset


def make_pair(datset1, dataset2):
    """
    Make pairs of the images from the two datasets.
    Label 0 means the two images are from the same class, 
    and label 1 means the two images are from different class.

    Args:
        dataset1: the normalized dataset of the first class
        dataset2: the normalized dataset of the second class

    Returns:
        pos_pair1: the positive pairs of the first class
        pos_pair2: the positive pairs of the second class
        neg_pair1: the negative pairs, which the first image is from the first class and the second image is from the second class
        neg_pair2: the negative pairs, which the first image is from the second class and the second image is from the first class
    """
    # zip the data from the two datasets
    pos_pair1 = tf.data.Dataset.zip((datset1, datset1))
    pos_pair2 = tf.data.Dataset.zip((dataset2, dataset2))
    neg_pair1 = tf.data.Dataset.zip((datset1, dataset2))
    neg_pair2 = tf.data.Dataset.zip((dataset2, datset1))

    # label the data
    pos_pair1 = pos_pair1.map(lambda x, y: (x, y, 0.0))
    pos_pair2 = pos_pair2.map(lambda x, y: (x, y, 0.0))
    neg_pair1 = neg_pair1.map(lambda x, y: (x, y, 1.0))
    neg_pair2 = neg_pair2.map(lambda x, y: (x, y, 1.0))

    return pos_pair1, pos_pair2, neg_pair1, neg_pair2


def make_pair_test(test_datset1, test_dataset2, train_dataset1, train_dataset2):
    # zip the data and labels if their labels are the same (positive pairs), otherwise zip them with different labels (negative pairs)
    pos_pair1 = tf.data.Dataset.zip((test_datset1, train_dataset1))
    pos_pair2 = tf.data.Dataset.zip((test_dataset2, train_dataset2))
    neg_pair1 = tf.data.Dataset.zip((test_datset1, train_dataset2))
    neg_pair2 = tf.data.Dataset.zip((test_dataset2, train_dataset1))

    # label the data and labels
    pos_pair1 = pos_pair1.map(lambda x, y: (x, y, 0.0))
    pos_pair2 = pos_pair2.map(lambda x, y: (x, y, 0.0))
    neg_pair1 = neg_pair1.map(lambda x, y: (x, y, 1.0))
    neg_pair2 = neg_pair2.map(lambda x, y: (x, y, 1.0))

    return pos_pair1, pos_pair2, neg_pair1, neg_pair2


def shuffle(pos_pair1, pos_pair2, neg_pair1, neg_pair2):
    """
    shuffle the data to make the data more random

    Args:
        pos_pair1: the positive pairs of the first class
        pos_pair2: the positive pairs of the second class
        neg_pair1: the negative pairs, which the first image is from the first class and the second image is from the second class
        neg_pair2: the negative pairs, which the first image is from the second class and the second image is from the first class

    Returns:
        choice_dataset: the shuffled dataset
    """
    choice_dataset = tf.data.experimental.sample_from_datasets([pos_pair1, pos_pair2, neg_pair1, neg_pair2])

    return choice_dataset


def split_dataset(dataset, batch_size, train_size):
    """
    Batch the data and split the dataset into train and validation dataset

    Args:
        dataset: the dataset
        batch_size: the batch size
        train_size: the size of the train dataset (the rest is the validation dataset)

    Returns:
        train_dataset: the train dataset
        val_dataset: the validation dataset
    """
    # batch the data
    dataset = dataset.batch(batch_size)

    # split the data into train and test
    train_dataset = dataset.take(train_size)
    validation_dataset = dataset.skip(train_size)

    return train_dataset, validation_dataset


def visualize(img1, img2, labels, to_show=6, num_col=3, predictions=None, test=False):
    """
    Visualize the images and their true labels
    If the predictions are given, visualize their predictions label as well

    Args:
        img1: the first image
        img2: the second image
        labels: the true labels
        to_show: the number of images to show
        num_col: the number of images in each column
        predictions: the predictions of the images
        test: show the prediction labels or not
    """

    num_row = to_show // num_col if to_show // num_col != 0 else 1

    to_show = num_row * num_col

    # Plot the images
    fig, axes = plt.subplots(num_row, num_col, figsize=(10, 10))
    for i in range(to_show):

        # If the number of rows is 1, the axes array is one-dimensional
        if num_row == 1:
            ax = axes[i % num_col]
        else:
            ax = axes[i // num_col, i % num_col]

        ax.imshow((tf.concat([img1[i], img2[i]], axis=1).numpy()*255.0).astype("uint8"))
        ax.set_axis_off()
        if test:
            ax.set_title("True: {} | Pred: {:.5f}".format(labels[i], predictions[i][0]))
        else:
            ax.set_title("Label: {}".format(labels[i]))
    if test:
        plt.tight_layout(rect=(0, 0, 1.9, 1.9), w_pad=0.0)
    else:
        plt.tight_layout(rect=(0, 0, 1.5, 1.5))
    plt.show()


def main():
    AD_dataset = load_data('./AD_NC/train/AD', (224, 224))
    NC_dataset = load_data('./AD_NC/train/NC', (224, 224))

    pos_pair1, pos_pair2, neg_pair1, neg_pair2 = make_pair(AD_dataset, NC_dataset)

    choice_dataset = shuffle(pos_pair1, pos_pair2, neg_pair1, neg_pair2)

    train_dataset, validation_dataset = split_dataset(choice_dataset, 16, 100)

    for img1, img2, label in train_dataset.take(1):
        visualize(img1, img2, label)


if __name__ == '__main__':
    main()