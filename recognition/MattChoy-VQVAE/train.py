import os
import random
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from modules import VQVAE, VectorQuantiser
from constants import batch_size, variance, latent_dimensions, num_embeddings, n_epochs, vqvae_epochs
from dataset import x_train, x_test
from util import compare_reconstructions

# Bug fix - my computer has some issue that is fixed using this one-liner
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

print(f"The training set contains {len(x_train) * batch_size} samples")
print(f"The testing set contains {len(x_test) * batch_size} samples")

# Train the VQVAE model.
vqvae = VQVAE(variance, latent_dimensions, num_embeddings)
vqvae.compile(optimizer=keras.optimizers.Adam())
vqvae.fit(x_train, epochs=vqvae_epochs, batch_size=batch_size)

# Evaluate
test_images, reconstructed = compare_reconstructions(vqvae, x_test, 10)
