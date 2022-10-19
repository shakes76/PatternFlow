"""
    Performs all training/validating/testing/saving of models and plotting of results (i.e.
    losses and metrics during training/validation).

    Author: Adrian Rahul Kamal Rajkamal
    Student Number: 45811935
"""
import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from dataset import *
from modules import *

""" APPROACH """

# Load data (which also pre-processes it)
train_data = load_preprocess_image_data(FILE_PATH + "train", IMG_DIMENSION, BATCH_SIZE, SHIFT)

# Calculate (unbiased) training variance (across all pixels of all images)
full_train_data = train_data.unbatch()
num_train_images = full_train_data.reduce(np.int32(0), lambda x, _: x + 1).numpy()
num_train_pixels = num_train_images * IMG_DIMENSION ** 2

image_sum = full_train_data.reduce(np.float32(0), lambda x, y: x + y).numpy().flatten()
total_pixel_sum = image_sum.sum()
mean_pixel_train = total_pixel_sum / num_train_pixels

image_sse = full_train_data.reduce(np.float32(0), lambda x, y: x + (y - mean_pixel_train) ** 2)\
                           .numpy().flatten()
var_pixel_train = image_sse.sum() / (num_train_pixels - 1)

# Create VQ-VAE model

# Compile & Fit (define SSIM callback)

# Save

# Plot

# Test/generate --> maybe load saved model above and put this in predict.py

# Get trained VQ-VAE codebooks

# Create PixelCNN model

# Compile & Fit (with trained VQ-VAE codebooks)

# Save

# Plot

# Test/generate --> maybe load saved model above and put this in predict.py
