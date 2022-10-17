import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf

from dataset import get_train_dataset
import modules
from modules import VQVAETrainer
from utils import models_directory, vqvae_weights_filename

# Get the datasets
train_ds = get_train_dataset()
data_variance = np.var(train_ds)

# Train the VQ-VAE model
vqvae_trainer = VQVAETrainer(data_variance, latent_dim=16, num_embeddings=128)
vqvae_trainer.compile(optimizer=keras.optimizers.Adam())
vqvae_trainer.fit(train_ds, epochs=30, batch_size=128)
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
