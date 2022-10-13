"""
    Performs all training/validating/testing/saving of models and plotting of results (i.e.
    losses and metrics during training/validation).

    Author: Adrian Rahul Kamal Rajkamal
    Student Number: 45811935
"""
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from dataset import *
from modules import *

""" APPROACH """

# Load data (which also pre-processes it)

# Calculate training variance --> maybe goes into dataset.py or make a utils.py?

# Create VQ-VAE model

# Compile & Fit (define SSIM callback)

# Save

# Plot

# Test/generate --> maybe load saved model above and put this in predict.py

# Get VQ-VAE codebooks

# Create PixelCNN model

# Compile & Fit (with VQ-VAE codebooks)

# Save

# Plot

# Test/generate --> maybe load saved model above and put this in predict.py
