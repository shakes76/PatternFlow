import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from dataset import get_test_dataset, get_train_dataset
from modules import VQVAETrainer
from utils import models_directory, vqvae_weights_filename

# Load testing dataset
test_ds = get_train_dataset()

# Create the model and load the weights
train_ds = get_train_dataset()
data_variance = np.var(train_ds)
vqvae_trainer = VQVAETrainer(data_variance, latent_dim=16, num_embeddings=128)
vqvae_trainer.load_weights(models_directory + vqvae_weights_filename)

# Visualise results of encoding and decoding
test_ds = get_test_dataset()
trained_vqvae_model = vqvae_trainer.vqvae
idx = np.random.choice(len(test_ds), 4)
test_images = test_ds[idx]
reconstructions_test = trained_vqvae_model.predict(test_images)

encoder = vqvae_trainer.vqvae.get_layer("encoder")
quantizer = vqvae_trainer.vqvae.get_layer("vector_quantizer")

encoded_outputs = encoder.predict(test_images)
flat_enc_outputs = encoded_outputs.reshape(-1, encoded_outputs.shape[-1])
codebook_indices = quantizer.get_code_indices(flat_enc_outputs)
codebook_indices = codebook_indices.numpy().reshape(encoded_outputs.shape[:-1])

plt.figure(figsize=(12, len(test_images) * 4))
plt.subplot(len(test_images), 3, 1)
plt.title("Original")
plt.subplot(len(test_images), 3, 2)
plt.title("Code")
plt.subplot(len(test_images), 3, 3)
plt.title("Decoded")

for i in range(len(test_images)):
    plt.subplot(len(test_images), 3, i * 3 + 1)
    plt.imshow(test_images[i].squeeze() + 0.5, cmap='gray')
    plt.axis("off")

    plt.subplot(len(test_images), 3, i * 3 + 2)
    plt.imshow(codebook_indices[i] + 0.5, cmap='gray')
    plt.axis("off")

    plt.subplot(len(test_images), 3, i * 3 + 3)
    plt.imshow(reconstructions_test[i].squeeze() + 0.5, cmap='gray')
    plt.axis("off")
plt.tight_layout()
plt.show()

