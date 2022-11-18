from pickletools import optimize
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras import layers
import tensorflow_probability as tfp
import tensorflow as tf
import train
import modules
import dataset


"""
Some of the content for the pixelcnn should probably be in this file however i left it in train.py
because i didnt want to have to work out how to make sure things all connected with the different models

"""


def calculate_ssim(original_images, reconstructed_images):
    """
    Calculate and print the average structured similarity between original and reconstructed images
    """
    similarity = tf.reduce_mean(tf.image.ssim(original_images, reconstructed_images, max_val=1))
    #print("Structured similarity is:", similarity)
    return similarity

def average_ssim(sims):
  sum = 0
  for x in sims:
    sum = sum + x
  avg = sum / len(sims)
  return avg

def show_subplot(original, reconstructed):
    plt.subplot(1, 2, 1)
    plt.imshow(original.squeeze())
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed.squeeze())
    plt.title("Reconstructed")
    plt.axis("off")

    plt.show()
