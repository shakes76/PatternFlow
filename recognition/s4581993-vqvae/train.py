import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf

from dataset import get_train_dataset, get_test_dataset
import modules
from modules import VQVAETrainer
from utils import models_directory, vqvae_weights_filename

# Get the datasets
train_ds = get_train_dataset()
test_ds = get_test_dataset()
data_variance = np.var(train_ds)

# Train the VQ-VAE model
vqvae_trainer = VQVAETrainer(data_variance, latent_dim=16, num_embeddings=128)
vqvae_trainer.compile(optimizer=keras.optimizers.Adam())
history = vqvae_trainer.fit(train_ds, epochs=30, batch_size=128)

# Save the model
vqvae_trainer.save_weights(models_directory + vqvae_weights_filename)

# Plot the training loss
loss = history.history["loss"]
r_loss = history.history["reconstruction_loss"]
v_loss = history.history["vqvae_loss"]

plt.figure(figsize=(8, 12))
plt.subplot(3, 1, 1)
plt.plot(loss)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.title('Loss')

plt.subplot(3, 1, 2)
plt.plot(r_loss)
plt.ylabel('Reconstruction Loss')
plt.title('Reconstruction Loss')
plt.xlabel('Epoch')

plt.subplot(3, 1, 3)
plt.plot(v_loss)
plt.ylabel('VQ-VAE Loss')
plt.title('VQ-VAE Loss')
plt.xlabel('Epoch')

plt.tight_layout()
plt.show()

# Find the SSIM between the original and decoded images
print("Calculating SSIM across test set...")
trained_vqvae_model = vqvae_trainer.vqvae
reconstructions_test = trained_vqvae_model.predict(test_ds)

ssim_values = np.zeros(len(test_ds))
for i in range(len(test_ds)):
    ssim_values[i] = tf.image.ssim(test_ds[i], reconstructions_test[i], 1)
avg_ssim = np.mean(ssim_values)
print("SSIM index:", avg_ssim)

# Show a histogram of the SSIM values
plt.hist(ssim_values)
plt.ylabel('Number of occurrences')
plt.xlabel('SSIM')
plt.title('Structural similarity index measure')
plt.annotate(f'Mean: %f' % avg_ssim, xy=(0.05, 0.95), xycoords='axes fraction')
plt.show()
