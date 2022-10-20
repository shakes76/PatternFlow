import os

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from dataset import get_train_dataset, get_test_dataset
from modules import VQVAETrainer
from utils import models_directory, vqvae_weights_filename

# Load the datasets
test_ds = get_test_dataset()
train_ds = get_train_dataset()

# Create the model and load the weights
test_ds = get_test_dataset()
data_variance = np.var(train_ds)
vqvae_trainer = VQVAETrainer(data_variance, latent_dim=16, num_embeddings=128)
vqvae_trainer.load_weights(models_directory + vqvae_weights_filename)

# Find the SSIM between the original and decoded images
print("\nCalculating SSIM across test set...")
trained_vqvae_model = vqvae_trainer.vqvae
reconstructions_test = trained_vqvae_model.predict(test_ds)

ssim_values = tf.image.ssim(test_ds, reconstructions_test, 1)
avg_ssim = np.mean(ssim_values)
print("SSIM index:", avg_ssim)

# Show a histogram of the SSIM values
plt.hist(ssim_values)
plt.ylabel('Number of occurrences')
plt.xlabel('SSIM')
plt.title('Structural similarity index measure')
plt.annotate(f'Mean: %f' % avg_ssim, xy=(0.05, 0.95), xycoords='axes fraction')
plt.show()

