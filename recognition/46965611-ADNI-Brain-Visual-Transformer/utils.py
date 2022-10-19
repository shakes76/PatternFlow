"""
utils.py

Utilities for visual transformer.

Author: Joshua Wang (Student No. 46965611)
Date Created: 15 Oct 2022
"""
import matplotlib.pyplot as plt
import tensorflow as tf
from dataset import DataLoader
from modules import PatchLayer
from parameters import DATA_LOAD_PATH, IMAGE_SIZE, NUM_PATCHES, PATCH_SIZE, PROJECTION_DIM


def plot_image(dataset):
    """
    Plots an image from the given dataset.
    """
    plt.figure(figsize=(5, 5))
    dataset_it = dataset.__iter__()
    image = dataset_it.next()[0][0]
    plt.imshow(image.numpy().astype('uint8'))
    plt.axis('off')
    plt.show()

def plot_patches(dataset):
    """
    Plots an image from the given dataset after applying Patch_Layer to it.
    """
    dataset_it = dataset.__iter__()
    image = dataset_it.next()[0][0]

    resized_image = tf.image.resize(
        tf.convert_to_tensor([image]), size=(IMAGE_SIZE, IMAGE_SIZE)
    )

    (token, patch) = PatchLayer(IMAGE_SIZE, PATCH_SIZE, NUM_PATCHES, PROJECTION_DIM)(resized_image/255.0)
    (token, patch) = (token[0], patch[0])
    n = patch.shape[0]
    shifted_images = ["ORIGINAL", "LEFT-UP", "LEFT-DOWN", "RIGHT-UP", "RIGHT-DOWN"]

    for index, name in enumerate(shifted_images):
        count = 1
        plt.figure(figsize=(5, 5))
        plt.suptitle(name)

        for row in range(n):
            for col in range(n):
                plt.subplot(n, n, count)
                count = count + 1
                image= tf.reshape(patch[row][col], (PATCH_SIZE, PATCH_SIZE, 3 * 5))
                plt.imshow(image[..., 3 * index : 3 * index + 3])
                plt.axis('off')

    plt.show()

if __name__ == '__main__':
    loader = DataLoader(DATA_LOAD_PATH)
    train, val, test = loader.load_data()
    plot_image(train)
    plot_patches(train)