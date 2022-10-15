"""
utils.py

Utilities for visual transformer.

Author: Joshua Wang (Student No. 46965611)
Date Created: 15 Oct 2022
"""
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from dataset import DataLoader
from modules import PatchLayer
from train import IMAGE_SIZE, PATCH_SIZE

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

    patches = PatchLayer(patch_size=PATCH_SIZE)(resized_image)
    print(f"Image size: {IMAGE_SIZE} X {IMAGE_SIZE}")
    print(f"Patch size: {PATCH_SIZE} X {PATCH_SIZE}")
    print(f"Patches per image: {patches.shape[1]}")
    print(f"Elements per patch: {patches.shape[-1]}")

    n = int(np.sqrt(patches.shape[1]))
    plt.figure(figsize=(5, 5))
    for i, patch in enumerate(patches[0]):
        ax = plt.subplot(n, n, i+1)
        patch_img = tf.reshape(patch, (PATCH_SIZE, PATCH_SIZE, 3))
        plt.imshow(patch_img.numpy().astype("uint8"))
        plt.axis("off")
    plt.show()

if __name__ == '__main__':
    loader = DataLoader('C:/AD_NC')
    train, val, test = loader.load_data()
    plot_image(train)
    plot_patches(train)