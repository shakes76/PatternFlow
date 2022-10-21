import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from PIL import Image
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
seed = 123
batch_size = 64
img_height = 256
img_width = 256
image_shape = (img_height, img_width, 3)

from modules import *
from tools import *
from train import train_ds, val_ds


vqvae_trainer = VQVAETrainer(0.05, latent_dim=16, num_embeddings=128, image_shape = image_shape)
vqvae_trainer.vqvae = keras.models.load_model("saved_models")
vqvae_trainer.compile(optimizer=keras.optimizers.Adam())
vqvae = vqvae_trainer.vqvae
history = vqvae_trainer.vqvae.history

test_images = val_ds.take(1)
reconstructed_test_images = vqvae.predict(test_images)
reconstructed_test_images.shape

for test_image, reconstructed_image in zip(test_images, reconstructed_test_images):
    show_original_vs_reconstructed(test_image[0], reconstructed_image)