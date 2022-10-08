from email.mime import base
from gettext import translation
import os
import tensorflow as tf
import keras as k
import numpy as np


AD_PATH = 'C:\\Users\\Wom\\Desktop\\COMP3710\\ADNI_AD_NC_2D\\AD_NC\\train\\AD'
CN_PATH = 'C:\\Users\\Wom\\Desktop\\COMP3710\\ADNI_AD_NC_2D\\AD_NC\\train\\NC'


def load_train_data():
    """ Load image data into tf Dataset, in the form of image pairs
    mapped to labels (0 for same, 1 for different)

    Returns:
        dataset: dataset for train and validation data
    """

     # Get the path to each image and mask
    ad_paths = [os.path.join(AD_PATH, path) for path in os.listdir(AD_PATH)]
    cn_paths = [os.path.join(CN_PATH, path) for path in os.listdir(CN_PATH)]

    

    # X pair bases, x/2 ad, x/2 cn to create pairs of images
    pair_base = cn_paths[0::2]  # every second path starting at 0

    num_pairs = len(pair_base)

    pair_ad = ad_paths[0:(num_pairs//2)]
    pair_cn = cn_paths[1::4][0:num_pairs]  # every 4th path starting at 1

    pair_compare = pair_cn + pair_ad

    # Create labels array 
    # first num_pairs/2 elements is 0 as pair_compare starts with cn images
    labels = np.concatenate([np.zeros([num_pairs//2]), np.ones([num_pairs//2])])

    
    base_ds = tf.data.Dataset.from_tensor_slices(pair_base) \
            .map(get_image)
    pair_ds = tf.data.Dataset.from_tensor_slices(pair_compare) \
            .map(get_image)
    labels_ds = tf.data.Dataset.from_tensor_slices(labels)
    
    
    dataset = tf.data.Dataset.zip(((base_ds, pair_ds), labels_ds))
    
    train, val = train_val_split(dataset, 0.8)

    return train, val

def load_test_data():
    """ Load testing image data, images with labels,
    0 for ad, 1 for cn

    Returns:
        dataset: dataset for testing
    """

    pass

def get_image(path):
    """ Get tf image from path

    Args:
        path (string): path to image

    Returns:
        tf.image: image at path
    """
     # Get the image
    image = tf.io.read_file(path)

    # Convert to jpeg
    image = tf.image.decode_jpeg(image, 1)

    # Scale imaage
    image = tf.image.resize(image, [256, 240])  # TODO: may not need this

    # May need to normalise
    return image

def train_val_split(dataset, ratio):
    train_num = int(round(len(dataset)*ratio, 1))

    train = dataset.take(train_num)
    val = dataset.skip(train_num)

    return train, val
    

