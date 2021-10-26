"""Data Pre-processing
"""
import tensorflow as tf
import glob
import os


def get_filenames(isic_dir):
    feature_dir = os.path.join(
        isic_dir, 'ISIC2018_Task1-2_Training_Input', '*.jpg')

    # the dataset actually shuffles here due to arbitrary reading order
    features = glob.glob(feature_dir)
    # make sure the features and labels have the same order
    labels = [f.replace('ISIC2018_Task1-2_Training_Input',
                        'ISIC2018_Task1_Training_GroundTruth').replace('.jpg', '_segmentation.png') for f in features]

    print("Number of images loaded:", len(features), len(labels))
    return features, labels


def split_data(features, labels, training_split=0.6, validation_split=0.2, test_split=0.2):
    # calculate the split size
    num_train = int(training_split * len(features))
    num_val = int(validation_split * len(features))
    num_test = int(test_split * len(features))

    train_features, val_features, test_features = tf.split(
        features, [num_train, num_val, num_test])
    train_labels, val_labels, test_labels = tf.split(
        labels, [num_train, num_val, num_test])

    return train_features, train_labels, val_features, val_labels, test_features, test_labels
