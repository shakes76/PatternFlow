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
