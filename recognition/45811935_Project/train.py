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

""" Main data-related constants """
FILE_PATH = "./ADNI_AD_NC_2D/AD_NC/"
IMG_DIMENSION = 256
SEED = 42

""" Hyper-parameters """
NUM_EMBEDDINGS = 256
BATCH_SIZE = 32
LATENT_DIM = 32
PIXEL_SHIFT = 0.5
NUM_EPOCHS = 30
VALIDATION_SPLIT = 0.3

# Check if GPU is available and if so use it, otherwise use CPU
gpu_used = len(tf.config.list_physical_devices('GPU'))
device = "/GPU:0" if gpu_used else "/CPU:0"

# Load and pre-process data
train_data = load_preprocess_image_data(path=FILE_PATH + "train", img_dim=IMG_DIMENSION,
                                        batch_size=BATCH_SIZE, validation_split=VALIDATION_SPLIT,
                                        subset="training", seed=SEED, shift=PIXEL_SHIFT)

val_data = load_preprocess_image_data(path=FILE_PATH + "train", img_dim=IMG_DIMENSION,
                                      batch_size=BATCH_SIZE, validation_split=VALIDATION_SPLIT,
                                      subset="validation", seed=SEED, shift=PIXEL_SHIFT)

test_data = load_preprocess_image_data(path=FILE_PATH + "test", img_dim=IMG_DIMENSION,
                                       batch_size=BATCH_SIZE, shift=PIXEL_SHIFT)

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
vqvae = VQVAE(tr_var=var_pixel_train, num_encoded=NUM_EMBEDDINGS, latent_dim=LATENT_DIM)

# Compile & Fit (define SSIM callback)
# vqvae.compile(...)
# with tf.device(device):
#     history = vqvae.fit(train_data, epochs=NUM_EPOCHS, callbacks=...)

# Save

# Plot

# Test/generate --> maybe load saved model above and put this in predict.py

# Get trained VQ-VAE codebooks

# Create PixelCNN model

# Compile & Fit (with trained VQ-VAE codebooks)

# Save

# Plot

# Test/generate --> maybe load saved model above and put this in predict.py
